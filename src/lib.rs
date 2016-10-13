#![feature(optin_builtin_traits)]
#![feature(rc_counts)]
//#![feature(zero_one)]

extern crate cuda;
extern crate cuda_blas;
extern crate densearray;

extern crate libc;

use kernels::*;

use cuda::runtime::*;
use cuda_blas::*;
use densearray::{ArrayIndex, AsView, AsViewMut, Reshape, ReshapeMut};

use std::cell::{RefCell};
use std::marker::{PhantomData};
use std::mem::{size_of};
//use std::num::{Zero};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc};
use std::sync::{Arc, Mutex};

pub mod kernels;
pub mod linalg;

/*impl<'a, T> Reshape<'a, usize, Array1dView<'a, T>> for [T] where T: Copy {
  fn reshape(&'a self, dim: usize) -> Array1dView<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim);
    Array1dView{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, usize, Array1dViewMut<'a, T>> for [T] where T: Copy {
  fn reshape_mut(&'a mut self, dim: usize) -> Array1dViewMut<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim);
    Array1dViewMut{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize), Array2dView<'a, T>> for [T] where T: Copy {
  fn reshape(&'a self, dim: (usize, usize)) -> Array2dView<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim.flat_len());
    Array2dView{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize), Array2dViewMut<'a, T>> for [T] where T: Copy {
  fn reshape_mut(&'a mut self, dim: (usize, usize)) -> Array2dViewMut<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim.flat_len());
    Array2dViewMut{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize), Array2dView<'a, T>> for Array1dView<'a, T> where T: Copy {
  fn reshape(&'a self, dim: (usize, usize)) -> Array2dView<'a, T> {
    assert!(dim == (self.dim, 1) || dim == (1, self.dim));
    if dim.1 == 1 {
      Array2dView{
        buf:      self.buf,
        dim:      dim,
        stride:   (self.stride, self.stride * self.dim),
      }
    } else if dim.0 == 1 {
      Array2dView{
        buf:      self.buf,
        dim:      dim,
        stride:   (1, self.stride),
      }
    } else {
      unreachable!();
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize), Array2dViewMut<'a, T>> for Array1dViewMut<'a, T> where T: Copy {
  fn reshape_mut(&'a mut self, dim: (usize, usize)) -> Array2dViewMut<'a, T> {
    assert!(dim == (self.dim, 1) || dim == (1, self.dim));
    if dim.1 == 1 {
      Array2dViewMut{
        buf:      self.buf,
        dim:      dim,
        stride:   (self.stride, self.stride * self.dim),
      }
    } else if dim.0 == 1 {
      Array2dViewMut{
        buf:      self.buf,
        dim:      dim,
        stride:   (1, self.stride),
      }
    } else {
      unreachable!();
    }
  }
}

impl<'a, T> Reshape<'a, usize, Array1dView<'a, T>> for Array2dView<'a, T> where T: Copy {
  fn reshape(&'a self, dim: usize) -> Array1dView<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dView{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> ReshapeMut<'a, usize, Array1dViewMut<'a, T>> for Array2dViewMut<'a, T> where T: Copy {
  fn reshape_mut(&'a mut self, dim: usize) -> Array1dViewMut<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dViewMut{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> Reshape<'a, usize, Array1dView<'a, T>> for Array4dView<'a, T> where T: Copy {
  fn reshape(&'a self, dim: usize) -> Array1dView<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dView{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> ReshapeMut<'a, usize, Array1dViewMut<'a, T>> for Array4dViewMut<'a, T> where T: Copy {
  fn reshape_mut(&'a mut self, dim: usize) -> Array1dViewMut<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dViewMut{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize), Array2dView<'a, T>> for Array4dView<'a, T> where T: Copy {
  fn reshape(&'a self, dim: (usize, usize)) -> Array2dView<'a, T> {
    // FIXME(20161008): should do a stricter check, but this is barely sufficient.
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim.flat_len());
    Array2dView{
      buf:      self.buf,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize), Array2dViewMut<'a, T>> for Array4dViewMut<'a, T> where T: Copy {
  fn reshape_mut(&'a mut self, dim: (usize, usize)) -> Array2dViewMut<'a, T> {
    // FIXME(20161008): should do a stricter check, but this is barely sufficient.
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim.flat_len());
    Array2dViewMut{
      buf:      self.buf,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}*/

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
}

impl DeviceConn {
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

pub struct DeviceMem<T> where T: Copy {
  dev_idx:  usize,
  dptr:     *mut T,
  len:      usize,
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
}

pub struct DeviceArray1d<T> where T: Copy {
  buf:      DeviceMem<T>,
  dim:      usize,
  stride:   usize,
  //_marker:  PhantomData<T>,
}

impl DeviceArray1d<u8> {
  pub fn zeros(dim: usize, conn: DeviceConn) -> DeviceArray1d<u8> {
    let mut buf = unsafe { DeviceMem::alloc(dim, conn.clone()) };
    unsafe { cuda_memset_async(buf.dptr, 0, buf.size_bytes(), &*conn.stream) }.unwrap();
    DeviceArray1d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
      //_marker:  PhantomData,
    }
  }
}

impl DeviceArray1d<f32> {
  pub fn zeros(dim: usize, conn: DeviceConn) -> DeviceArray1d<f32> {
    let mut buf = unsafe { DeviceMem::alloc(dim, conn.clone()) };
    unsafe { cuda_memset_async(buf.dptr as *mut _, 0, buf.size_bytes(), &*conn.stream) }.unwrap();
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
}

impl<'a> DeviceArray1dView<'a, f32> {
}

/*impl<'a, T> View<'a, usize, DeviceArray1dView<'a, T>> for DeviceArray1dView<'a, T> where T: 'a + Copy {
  fn view(&'a self, lo: usize, hi: usize) -> DeviceArray1dView<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    DeviceArray1dView{
      buf:      &self.buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}*/

pub struct DeviceArray1dViewMut<'a, T> where T: 'a + Copy {
  buf:      DeviceMemRefMut<'a, T>,
  dim:      usize,
  stride:   usize,
}

/*impl<'a, T> ViewMut<'a, usize, DeviceArray1dViewMut<'a, T>> for DeviceArray1dViewMut<'a, T> where T: 'a + Copy {
  fn view_mut(&'a mut self, lo: usize, hi: usize) -> DeviceArray1dViewMut<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    DeviceArray1dViewMut{
      buf:      &mut self.buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}*/

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
}

impl<'a> DeviceArray1dViewMut<'a, u8> {
  pub fn set_constant(&'a mut self, c: u8, conn: DeviceConn) {
    if self.stride == 1 {
      unsafe { cuda_memset_async(self.buf.as_mut_ptr(), 0, self.buf.size_bytes(), &*conn.stream) }.unwrap();
    } else {
      unimplemented!();
    }
  }
}

impl<'a> DeviceArray1dViewMut<'a, f32> {
  pub fn set_constant(&'a mut self, c: f32, conn: DeviceConn) {
    if self.stride == 1 {
      unsafe { devicemem_cuda_vector_set_scalar_f32(self.buf.as_mut_ptr(), self.dim(), c, conn.stream.ptr) };
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

impl DeviceArray2d<f32> {
  pub fn zeros(dim: (usize, usize), conn: DeviceConn) -> DeviceArray2d<f32> {
    let len = dim.flat_len();
    let mut buf = unsafe { DeviceMem::alloc(len, conn.clone()) };
    unsafe { cuda_memset_async(buf.dptr as *mut _, 0, buf.size_bytes(), &*conn.stream) }.unwrap();
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
}

/*impl<'a, T> View<'a, (usize, usize), Array2dView<'a, T>> for Array2dView<'a, T> where T: 'a + Copy {
  fn view(&'a self, lo: (usize, usize), hi: (usize, usize)) -> Array2dView<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    Array2dView{
      buf:      &self.buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}*/

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
}

impl<'a> DeviceArray2dViewMut<'a, f32> {
  pub fn set_constant(&'a mut self, c: f32, conn: DeviceConn) {
    if self.stride == self.dim.least_stride() {
      unsafe { devicemem_cuda_vector_set_scalar_f32(self.buf.as_mut_ptr(), self.dim.flat_len(), c, conn.stream.ptr) };
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
