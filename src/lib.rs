#![feature(optin_builtin_traits)]
#![feature(arc_counts)]
#![feature(rc_counts)]
//#![feature(zero_one)]

extern crate cuda;
extern crate cuda_blas;
extern crate densearray;

extern crate libc;

use kernels::*;

use cuda::runtime::*;
use cuda_blas::*;
use densearray::prelude::*;

use std::cell::{RefCell};
use std::marker::{PhantomData};
use std::mem::{size_of};
//use std::num::{Zero};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc};
use std::sync::{Arc, Mutex};

pub mod kernels;
pub mod linalg;
pub mod prelude;

thread_local!(static DRIVER_CONTEXT: Rc<DriverContext> = Rc::new(DriverContext{}));

struct DriverContext {}

impl !Send for DriverContext {}
impl !Sync for DriverContext {}

pub struct Device {
  dev_idx:  usize,
  stream:   Arc<CudaStream>,
  cublas:   Rc<RefCell<Option<Rc<CublasHandle>>>>,
}

impl Device {
  pub fn count() -> usize {
    CudaDevice::count().unwrap()
  }

  pub fn new(dev_idx: usize) -> Device {
    DRIVER_CONTEXT.with(|driver| {
      let driver = driver.clone();
      assert!(Rc::strong_count(&driver) <= 2,
          //"DeviceConn requires exclusive reference to DriverContext!");
          "DriverContext does not support nesting");
      CudaDevice::set_current(dev_idx).unwrap();
      Device{
        dev_idx:  dev_idx,
        stream:   Arc::new(CudaStream::create().unwrap()),
        cublas:   Rc::new(RefCell::new(None)),
      }
    })
  }

  pub fn conn(&self) -> DeviceConn {
    DRIVER_CONTEXT.with(|driver| {
      let driver = driver.clone();
      assert!(Rc::strong_count(&driver) <= 2,
          //"DeviceConn requires exclusive reference to DriverContext!");
          "DriverContext does not support nesting");
      CudaDevice::set_current(self.dev_idx).unwrap();
      DeviceConn{
        driver:     driver,
        dev_idx:    self.dev_idx,
        stream:     self.stream.clone(),
        cublas:     self.cublas.clone(),
      }
    })
  }
}

#[derive(Clone)]
pub struct DeviceConn {
  driver:   Rc<DriverContext>,
  dev_idx:  usize,
  stream:   Arc<CudaStream>,
  cublas:   Rc<RefCell<Option<Rc<CublasHandle>>>>,
  //cudnn:    Rc<RefCell<Option<Rc<CudnnHandle>>>>,
}

impl DeviceConn {
  pub fn sync(&self) {
    self.stream.synchronize().unwrap();
  }

  pub fn device(&self) -> usize {
    self.dev_idx
  }

  pub fn stream(&self) -> Arc<CudaStream> {
    self.stream.clone()
  }

  pub fn cublas(&self) -> Rc<CublasHandle> {
    {
      let mut cublas = self.cublas.borrow_mut();
      if cublas.is_none() {
        let handle = CublasHandle::create().unwrap();
        handle.set_stream(&*self.stream).unwrap();
        //handle.set_atomics_mode(CublasAtomicsMode::Allowed).unwrap();
        *cublas = Some(Rc::new(handle));
      }
    }
    let cublas = self.cublas.borrow();
    cublas.as_ref().unwrap().clone()
  }
}

pub struct DeviceMemDependencyTracker {
  events:   Vec<(Arc<CudaStream>, Rc<CudaEvent>)>,
  posts:    Vec<Rc<CudaEvent>>,
}

impl Drop for DeviceMemDependencyTracker {
  fn drop(&mut self) {
    // FIXME(20161014): should we wait for outstanding posts?
  }
}

impl DeviceMemDependencyTracker {
  pub fn new() -> DeviceMemDependencyTracker {
    DeviceMemDependencyTracker{
      events:   vec![],
      posts:    vec![],
    }
  }

  pub fn post(&mut self, conn: &DeviceConn) {
    let posts_count = self.posts.len();
    if posts_count > 0 {
      println!("WARNING: DeviceMemDependencyTracker::post(): {} events have been posted! This is likely a bug.", posts_count);
    }
    let conn_id = conn.stream.ptr as usize;
    for &(ref stream, ref event) in self.events.iter() {
      if Arc::strong_count(stream) <= 1 {
        // FIXME(20160925): the stream is unreachable from outside, so drop it.
        continue;
      }
      let id = stream.ptr as usize;
      if conn_id == id {
        let event = event.clone();
        event.record(&conn.stream).unwrap();
        self.posts.push(event);
        return;
      }
    }
    let event = Rc::new(CudaEvent::create_fastest().unwrap());
    self.events.push((conn.stream.clone(), event.clone()));
    event.record(&conn.stream).unwrap();
    self.posts.push(event);
  }

  pub fn wait(&mut self, conn: &DeviceConn) {
    let posts_count = self.posts.len();
    if posts_count > 1 {
      println!("WARNING: DeviceMemDependencyTracker::wait(): {} events have been posted! This is likely a bug.", posts_count);
    }
    for post in self.posts.drain( .. ) {
      conn.stream.wait_event(&post).unwrap();
    }
  }
}

pub struct DeviceMem<T> where T: Copy {
  dev_idx:  usize,
  dptr:     *mut T,
  len:      usize,
  tracker:  Rc<RefCell<DeviceMemDependencyTracker>>,
}

pub trait ZeroBits: Copy {}

impl ZeroBits for f32 {}
impl ZeroBits for f64 {}
impl ZeroBits for u8 {}
impl ZeroBits for u32 {}

impl<T> DeviceMem<T> where T: ZeroBits {
  pub fn zeros(len: usize, conn: DeviceConn) -> DeviceMem<T> {
    let mut buf = unsafe { DeviceMem::alloc(len, conn.clone()) };
    unsafe { cuda_memset_async(buf.dptr as *mut u8, 0, buf.size_bytes(), &*conn.stream) }.unwrap();
    buf.tracker.borrow_mut().post(&conn);
    buf
  }
}

impl<T> DeviceMem<T> where T: Copy {
  pub unsafe fn alloc(len: usize, conn: DeviceConn) -> DeviceMem<T> {
    let dptr = match cuda_alloc_device(len) {
      Err(_) => panic!("DeviceMem allocation failed"),
      Ok(dptr) => dptr,
    };
    DeviceMem{
      dev_idx:  conn.dev_idx,
      dptr:     dptr,
      len:      len,
      tracker:  Rc::new(RefCell::new(DeviceMemDependencyTracker::new())),
    }
  }

  pub fn as_ptr(&self) -> *const T {
    self.dptr
  }

  pub fn as_mut_ptr(&self) -> *mut T {
    self.dptr
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn size_bytes(&self) -> usize {
    self.len * size_of::<T>()
  }

  pub fn as_ref<'a>(&'a self) -> DeviceMemRef<'a, T> {
    DeviceMemRef{
      mem:      self,
      offset:   0,
      len:      self.len,
    }
  }

  pub fn as_mut<'a>(&'a mut self) -> DeviceMemRefMut<'a, T> {
    let len = self.len;
    DeviceMemRefMut{
      mem:      self,
      offset:   0,
      len:      len,
    }
  }
}

#[derive(Clone)]
pub struct DeviceMemRef<'a, T> where T: 'a + Copy {
  mem:      &'a DeviceMem<T>,
  offset:   usize,
  len:      usize,
}

impl<'a, T> DeviceMemRef<'a, T> where T: 'a + Copy {
  pub fn as_ptr(&self) -> *const T {
    unsafe { self.mem.dptr.offset(self.offset as isize) }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn size_bytes(&self) -> usize {
    self.len * size_of::<T>()
  }

  pub fn post(&self, conn: &DeviceConn) {
    self.mem.tracker.borrow_mut().post(conn);
  }

  pub fn wait(&self, conn: &DeviceConn) {
    self.mem.tracker.borrow_mut().wait(conn);
  }

  pub fn slice(self, start: usize, end: usize) -> DeviceMemRef<'a, T> {
    let new_len = end - start;
    assert!(new_len <= self.len);
    DeviceMemRef{
      mem:      self.mem,
      offset:   self.offset + start,
      len:      new_len,
    }
  }

  pub fn store_sync(&mut self, output: &mut [T], conn: DeviceConn) {
    assert_eq!(self.len(), output.len());
    self.wait(&conn);
    let status = unsafe { cuda_memcpy_async(
        output.as_mut_ptr(),
        self.mem.dptr,
        self.len(),
        CudaMemcpyKind::DeviceToHost,
        &conn.stream,
    ) };
    self.post(&conn);
    self.wait(&conn);
    conn.sync();
  }
}

pub struct DeviceMemRefMut<'a, T> where T: 'a + Copy {
  mem:      &'a mut DeviceMem<T>,
  offset:   usize,
  len:      usize,
}

impl<'a, T> DeviceMemRefMut<'a, T> where T: 'a + Copy {
  pub fn as_ptr(&self) -> *const T {
    unsafe { self.mem.dptr.offset(self.offset as isize) }
  }

  pub fn as_mut_ptr(&self) -> *mut T {
    unsafe { self.mem.dptr.offset(self.offset as isize) }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn size_bytes(&self) -> usize {
    self.len * size_of::<T>()
  }

  pub fn post(&self, conn: &DeviceConn) {
    self.mem.tracker.borrow_mut().post(conn);
  }

  pub fn wait(&self, conn: &DeviceConn) {
    self.mem.tracker.borrow_mut().wait(conn);
  }

  pub fn slice_mut(self, start: usize, end: usize) -> DeviceMemRefMut<'a, T> {
    let new_len = end - start;
    assert!(new_len <= self.len);
    DeviceMemRefMut{
      mem:      self.mem,
      offset:   self.offset + start,
      len:      new_len,
    }
  }

  pub fn copy(&mut self, src: DeviceMemRef<'a, T>, conn: DeviceConn) {
    assert_eq!(self.len(), src.len());
    self.wait(&conn);
    let status = unsafe { cuda_memcpy_async(
        self.mem.dptr,
        src.as_ptr(),
        self.len(),
        CudaMemcpyKind::DeviceToDevice,
        &conn.stream,
    ) };
    self.post(&conn);
  }

  pub fn load_sync(&mut self, input: &[T], conn: DeviceConn) {
    assert_eq!(self.len(), input.len());
    self.wait(&conn);
    let status = unsafe { cuda_memcpy_async(
        self.mem.dptr,
        input.as_ptr(),
        self.len(),
        CudaMemcpyKind::HostToDevice,
        &conn.stream,
    ) };
    self.post(&conn);
    self.wait(&conn);
    conn.sync();
  }
}

pub struct DeviceArray1d<T> where T: Copy {
  buf:      DeviceMem<T>,
  dim:      usize,
  stride:   usize,
  //_marker:  PhantomData<T>,
}

impl<T> DeviceArray1d<T> where T: ZeroBits {
  pub fn zeros(dim: usize, conn: DeviceConn) -> DeviceArray1d<T> {
    let mut buf = unsafe { DeviceMem::alloc(dim, conn.clone()) };
    unsafe { cuda_memset_async(buf.dptr as *mut _, 0, buf.size_bytes(), &*conn.stream) }.unwrap();
    buf.tracker.borrow_mut().post(&conn);
    DeviceArray1d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
      //_marker:  PhantomData,
    }
  }
}

impl<T> DeviceArray1d<T> where T: Copy {
  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn stride(&self) -> usize {
    self.stride
  }

  /*pub fn as_slice(&self) -> &[T] {
    &*self.buf
  }

  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut *self.buf
  }*/
}

impl<'a, T> AsView<'a, DeviceArray1dView<'a, T>> for DeviceArray1d<T> where T: Copy {
  fn as_view(&'a self) -> DeviceArray1dView<'a, T> {
    DeviceArray1dView{
      buf:      self.buf.as_ref(),
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T> AsViewMut<'a, DeviceArray1dViewMut<'a, T>> for DeviceArray1d<T> where T: Copy {
  fn as_view_mut(&'a mut self) -> DeviceArray1dViewMut<'a, T> {
    DeviceArray1dViewMut{
      buf:      self.buf.as_mut(),
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T> Reshape<'a, usize, DeviceArray1dView<'a, T>> for DeviceMemRef<'a, T> where T: Copy {
  fn reshape(self, dim: usize) -> DeviceArray1dView<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim);
    DeviceArray1dView{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, usize, DeviceArray1dViewMut<'a, T>> for DeviceMemRefMut<'a, T> where T: Copy {
  fn reshape_mut(self, dim: usize) -> DeviceArray1dViewMut<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim);
    DeviceArray1dViewMut{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize), DeviceArray2dView<'a, T>> for DeviceMemRef<'a, T> where T: Copy {
  fn reshape(self, dim: (usize, usize)) -> DeviceArray2dView<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim.flat_len());
    DeviceArray2dView{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize), DeviceArray2dViewMut<'a, T>> for DeviceMemRefMut<'a, T> where T: Copy {
  fn reshape_mut(self, dim: (usize, usize)) -> DeviceArray2dViewMut<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim.flat_len());
    DeviceArray2dViewMut{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

#[derive(Clone)]
pub struct DeviceArray1dView<'a, T> where T: 'a + Copy {
  buf:      DeviceMemRef<'a, T>,
  dim:      usize,
  stride:   usize,
}

impl<'a, T> DeviceArray1dView<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn stride(&self) -> usize {
    self.stride
  }

  pub fn as_ptr(&self) -> *const T {
    self.buf.as_ptr()
  }

  pub fn post(&self, conn: &DeviceConn) {
    self.buf.post(conn);
  }

  pub fn wait(&self, conn: &DeviceConn) {
    self.buf.wait(conn);
  }

  pub fn store_sync(&mut self, mut output: Array1dViewMut<'a, T>, conn: DeviceConn) {
    assert_eq!(self.dim(), output.dim());
    if self.stride() == self.dim().least_stride() && self.stride() == output.stride() {
      self.buf.wait(&conn);
      let status = unsafe { cuda_memcpy_async(
          output.as_mut_ptr(),
          self.as_ptr(),
          self.dim().flat_len(),
          CudaMemcpyKind::DeviceToHost,
          &conn.stream,
      ) };
      self.buf.post(&conn);
      self.buf.wait(&conn);
      conn.sync();
    } else {
      unimplemented!();
    }
  }
}

impl<'a> DeviceArray1dView<'a, f32> {
}

impl<'a, T> View<'a, usize, DeviceArray1dView<'a, T>> for DeviceArray1dView<'a, T> where T: 'a + Copy {
  fn view(self, lo: usize, hi: usize) -> DeviceArray1dView<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    DeviceArray1dView{
      buf:      self.buf.slice(new_offset, new_offset_end),
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize), DeviceArray2dView<'a, T>> for DeviceArray1dView<'a, T> where T: Copy {
  fn reshape(self, dim: (usize, usize)) -> DeviceArray2dView<'a, T> {
    assert!(dim == (self.dim, 1) || dim == (1, self.dim));
    if dim.1 == 1 {
      DeviceArray2dView{
        buf:      self.buf,
        dim:      dim,
        stride:   (self.stride, self.stride * self.dim),
      }
    } else if dim.0 == 1 {
      DeviceArray2dView{
        buf:      self.buf,
        dim:      dim,
        stride:   (1, self.stride),
      }
    } else {
      unreachable!();
    }
  }
}

pub struct DeviceArray1dViewMut<'a, T> where T: 'a + Copy {
  buf:      DeviceMemRefMut<'a, T>,
  dim:      usize,
  stride:   usize,
}

impl<'a, T> DeviceArray1dViewMut<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn stride(&self) -> usize {
    self.stride
  }

  pub fn as_mut_ptr(&self) -> *mut T {
    self.buf.as_mut_ptr()
  }

  pub fn load_sync(&mut self, input: Array1dView<'a, T>, conn: DeviceConn) {
    assert_eq!(self.dim(), input.dim());
    if self.stride() == self.dim().least_stride() && self.stride() == input.stride() {
      self.buf.wait(&conn);
      let status = unsafe { cuda_memcpy_async(
          self.as_mut_ptr(),
          input.as_ptr(),
          self.dim().flat_len(),
          CudaMemcpyKind::HostToDevice,
          &conn.stream,
      ) };
      self.buf.post(&conn);
      self.buf.wait(&conn);
      conn.sync();
    } else {
      unimplemented!();
    }
  }
}

impl<'a, T> ViewMut<'a, usize, DeviceArray1dViewMut<'a, T>> for DeviceArray1dViewMut<'a, T> where T: 'a + Copy {
  fn view_mut(self, lo: usize, hi: usize) -> DeviceArray1dViewMut<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    DeviceArray1dViewMut{
      buf:      self.buf.slice_mut(new_offset, new_offset_end),
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize), DeviceArray2dViewMut<'a, T>> for DeviceArray1dViewMut<'a, T> where T: Copy {
  fn reshape_mut(self, dim: (usize, usize)) -> DeviceArray2dViewMut<'a, T> {
    assert!(dim == (self.dim, 1) || dim == (1, self.dim));
    if dim.1 == 1 {
      DeviceArray2dViewMut{
        buf:      self.buf,
        dim:      dim,
        stride:   (self.stride, self.stride * self.dim),
      }
    } else if dim.0 == 1 {
      DeviceArray2dViewMut{
        buf:      self.buf,
        dim:      dim,
        stride:   (1, self.stride),
      }
    } else {
      unreachable!();
    }
  }
}

impl<'a> DeviceArray1dViewMut<'a, u8> {
  pub fn set_constant(&'a mut self, c: u8, conn: DeviceConn) {
    if self.stride == 1 {
      self.buf.wait(&conn);
      unsafe { cuda_memset_async(self.buf.as_mut_ptr(), 0, self.buf.size_bytes(), &*conn.stream) }.unwrap();
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }
}

impl<'a> DeviceArray1dViewMut<'a, f32> {
  pub fn set_constant(&'a mut self, c: f32, conn: DeviceConn) {
    if self.stride == 1 {
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_set_scalar_f32(self.buf.as_mut_ptr(), self.dim(), c, conn.stream.ptr) };
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }
}

pub struct DeviceArray2d<T> where T: Copy {
  buf:      DeviceMem<T>,
  dim:      (usize, usize),
  stride:   (usize, usize),
}

impl<T> DeviceArray2d<T> where T: ZeroBits {
  pub fn zeros(dim: (usize, usize), conn: DeviceConn) -> DeviceArray2d<T> {
    let len = dim.flat_len();
    let mut buf = unsafe { DeviceMem::alloc(len, conn.clone()) };
    unsafe { cuda_memset_async(buf.dptr as *mut _, 0, buf.size_bytes(), &*conn.stream) }.unwrap();
    buf.tracker.borrow_mut().post(&conn);
    DeviceArray2d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<T> DeviceArray2d<T> where T: Copy {
  pub fn dim(&self) -> (usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }
}

impl<'a, T> AsView<'a, DeviceArray2dView<'a, T>> for DeviceArray2d<T> where T: Copy {
  fn as_view(&'a self) -> DeviceArray2dView<'a, T> {
    DeviceArray2dView{
      buf:      self.buf.as_ref(),
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T> AsViewMut<'a, DeviceArray2dViewMut<'a, T>> for DeviceArray2d<T> where T: Copy {
  fn as_view_mut(&'a mut self) -> DeviceArray2dViewMut<'a, T> {
    DeviceArray2dViewMut{
      buf:      self.buf.as_mut(),
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

#[derive(Clone)]
pub struct DeviceArray2dView<'a, T> where T: 'a + Copy {
  buf:      DeviceMemRef<'a, T>,
  dim:      (usize, usize),
  stride:   (usize, usize),
}

impl<'a, T> DeviceArray2dView<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }

  pub fn as_ptr(&self) -> *const T {
    self.buf.as_ptr()
  }

  pub fn post(&self, conn: &DeviceConn) {
    self.buf.post(conn);
  }

  pub fn wait(&self, conn: &DeviceConn) {
    self.buf.wait(conn);
  }

  pub fn store_sync(&mut self, mut output: Array2dViewMut<'a, T>, conn: DeviceConn) {
    assert_eq!(self.dim(), output.dim());
    if self.stride() == self.dim().least_stride() && self.stride() == output.stride() {
      self.buf.wait(&conn);
      let status = unsafe { cuda_memcpy_async(
          output.as_mut_ptr(),
          self.as_ptr(),
          self.dim().flat_len(),
          CudaMemcpyKind::DeviceToHost,
          &conn.stream,
      ) };
      self.buf.post(&conn);
      self.buf.wait(&conn);
      conn.sync();
    } else {
      unimplemented!();
    }
  }
}

impl<'a, T> View<'a, (usize, usize), DeviceArray2dView<'a, T>> for DeviceArray2dView<'a, T> where T: 'a + Copy {
  fn view(self, lo: (usize, usize), hi: (usize, usize)) -> DeviceArray2dView<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    DeviceArray2dView{
      buf:      self.buf.slice(new_offset, new_offset_end),
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

pub struct DeviceArray2dViewMut<'a, T> where T: 'a + Copy {
  buf:      DeviceMemRefMut<'a, T>,
  dim:      (usize, usize),
  stride:   (usize, usize),
}

impl<'a, T> DeviceArray2dViewMut<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }

  pub fn as_mut_ptr(&self) -> *mut T {
    self.buf.as_mut_ptr()
  }

  pub fn load_sync(&mut self, input: Array2dView<'a, T>, conn: DeviceConn) {
    assert_eq!(self.dim(), input.dim());
    if self.stride() == self.dim().least_stride() && self.stride() == input.stride() {
      self.buf.wait(&conn);
      let status = unsafe { cuda_memcpy_async(
          self.as_mut_ptr(),
          input.as_ptr(),
          self.dim().flat_len(),
          CudaMemcpyKind::HostToDevice,
          &conn.stream,
      ) };
      self.buf.post(&conn);
      self.buf.wait(&conn);
      conn.sync();
    } else {
      unimplemented!();
    }
  }
}

impl<'a, T> ViewMut<'a, (usize, usize), DeviceArray2dViewMut<'a, T>> for DeviceArray2dViewMut<'a, T> where T: 'a + Copy {
  fn view_mut(self, lo: (usize, usize), hi: (usize, usize)) -> DeviceArray2dViewMut<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    DeviceArray2dViewMut{
      buf:      self.buf.slice_mut(new_offset, new_offset_end),
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

impl<'a> DeviceArray2dViewMut<'a, f32> {
  pub fn set_constant(&'a mut self, c: f32, conn: DeviceConn) {
    if self.stride == self.dim.least_stride() {
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_set_scalar_f32(self.buf.as_mut_ptr(), self.dim.flat_len(), c, conn.stream.ptr) };
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }
}

/*impl<'a, T> ViewMut<'a, (usize, usize), Array2dViewMut<'a, T>> for Array2dViewMut<'a, T> where T: 'a + Copy {
  fn view_mut(&'a mut self, lo: (usize, usize), hi: (usize, usize)) -> Array2dViewMut<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    Array2dViewMut{
      buf:      &mut self.buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}*/

pub struct DeviceArray4d<T> where T: Copy {
  buf:      DeviceMem<T>,
  dim:      (usize, usize, usize, usize),
  stride:   (usize, usize, usize, usize),
}

impl<T> DeviceArray4d<T> where T: ZeroBits {
  pub fn zeros(dim: (usize, usize, usize, usize), conn: DeviceConn) -> DeviceArray4d<T> {
    let len = dim.flat_len();
    let mut buf = unsafe { DeviceMem::alloc(len, conn.clone()) };
    unsafe { cuda_memset_async(buf.dptr as *mut _, 0, buf.size_bytes(), &*conn.stream) }.unwrap();
    buf.tracker.borrow_mut().post(&conn);
    DeviceArray4d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<T> DeviceArray4d<T> where T: Copy {
  pub fn dim(&self) -> (usize, usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize, usize) {
    self.stride
  }
}

impl<'a, T> AsView<'a, DeviceArray4dView<'a, T>> for DeviceArray4d<T> where T: Copy {
  fn as_view(&'a self) -> DeviceArray4dView<'a, T> {
    DeviceArray4dView{
      buf:      self.buf.as_ref(),
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T> AsViewMut<'a, DeviceArray4dViewMut<'a, T>> for DeviceArray4d<T> where T: Copy {
  fn as_view_mut(&'a mut self) -> DeviceArray4dViewMut<'a, T> {
    DeviceArray4dViewMut{
      buf:      self.buf.as_mut(),
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

pub struct DeviceArray4dView<'a, T> where T: 'a + Copy {
  buf:      DeviceMemRef<'a, T>,
  dim:      (usize, usize, usize, usize),
  stride:   (usize, usize, usize, usize),
}

impl<'a, T> DeviceArray4dView<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize, usize) {
    self.stride
  }

  pub fn as_ptr(&self) -> *const T {
    self.buf.as_ptr()
  }
}

pub struct DeviceArray4dViewMut<'a, T> where T: 'a + Copy {
  buf:      DeviceMemRefMut<'a, T>,
  dim:      (usize, usize, usize, usize),
  stride:   (usize, usize, usize, usize),
}

impl<'a, T> DeviceArray4dViewMut<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize, usize) {
    self.stride
  }

  pub fn as_mut_ptr(&self) -> *mut T {
    self.buf.as_mut_ptr()
  }
}

impl<'a> DeviceArray4dViewMut<'a, f32> {
  pub fn set_constant(&'a mut self, c: f32, conn: DeviceConn) {
    if self.stride == self.dim.least_stride() {
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_set_scalar_f32(self.buf.as_mut_ptr(), self.dim.flat_len(), c, conn.stream.ptr) };
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }
}

/*
pub struct Array3d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      (usize, usize, usize),
  stride:   (usize, usize, usize),
  _marker:  PhantomData<T>,
}

impl<T, S> Array3d<T, S> where T: Copy, S: Deref<Target=[T]> {
  pub fn dim(&self) -> (usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize) {
    self.stride
  }

  pub fn as_slice(&self) -> &[T] {
    &*self.buf
  }
}

impl<T, S> Array3d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut *self.buf
  }
}

pub struct Array4d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      (usize, usize, usize, usize),
  stride:   (usize, usize, usize, usize),
  _marker:  PhantomData<T>,
}

impl<T, S> Array4d<T, S> where T: Copy, S: Deref<Target=[T]> {
  pub fn dim(&self) -> (usize, usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize, usize) {
    self.stride
  }

  pub fn as_slice(&self) -> &[T] {
    &*self.buf
  }
}

impl<T, S> Array4d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut *self.buf
  }
}

impl<'a, T, S> AsView<'a, Array4dView<'a, T>> for Array4d<T, S> where T: Copy, S: Deref<Target=[T]> {
  fn as_view(&'a self) -> Array4dView<'a, T> {
    Array4dView{
      buf:      &*self.buf,
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T, S> AsViewMut<'a, Array4dViewMut<'a, T>> for Array4d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  fn as_view_mut(&'a mut self) -> Array4dViewMut<'a, T> {
    Array4dViewMut{
      buf:      &mut *self.buf,
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

pub struct Array4dView<'a, T> where T: 'a + Copy {
  buf:      &'a [T],
  dim:      (usize, usize, usize, usize),
  stride:   (usize, usize, usize, usize),
}

impl<'a, T> Array4dView<'a, T> where T: 'a + Copy {
  pub fn as_ptr(&self) -> *const T {
    self.buf.as_ptr()
  }
}

pub struct Array4dViewMut<'a, T> where T: 'a + Copy {
  buf:      &'a mut [T],
  dim:      (usize, usize, usize, usize),
  stride:   (usize, usize, usize, usize),
}

impl<'a, T> Array4dViewMut<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize, usize) {
    self.stride
  }

  pub fn as_mut_ptr(&mut self) -> *mut T {
    self.buf.as_mut_ptr()
  }
}

impl<'a> Array4dViewMut<'a, f32> {
  pub fn set_constant(&'a mut self, c: f32) {
    if self.stride == self.dim.least_stride() {
      for i in 0 .. self.dim.flat_len() {
        self.buf[i] = c;
      }
    } else {
      unimplemented!();
    }
  }
}
*/
