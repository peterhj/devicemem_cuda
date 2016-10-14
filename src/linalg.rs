use super::*;
use kernels::*;

use cuda::runtime::*;
use cuda_blas::*;
use cuda_blas::ffi::*;
use densearray::linalg::{Transpose};

impl<'a> DeviceArray1dView<'a, f32> {
  pub fn l2_norm(&self, conn: DeviceConn) -> f32 {
    let cublas = conn.cublas();
    cublas.set_pointer_mode(CublasPointerMode::Host).unwrap();
    self.buf.wait(&conn);
    let mut y = 0.0;
    let status = unsafe { cublasSnrm2_v2(
        cublas.ptr,
        self.dim() as _,
        self.as_ptr(),
        self.stride() as _,
        &mut y as *mut _,
    ) };
    assert!(status.is_ok());
    self.buf.post(&conn);
    y
  }

  pub fn inner_prod(&self, x: DeviceArray1dView<'a, f32>, conn: DeviceConn) -> f32 {
    assert_eq!(self.dim(), x.dim());
    let cublas = conn.cublas();
    cublas.set_pointer_mode(CublasPointerMode::Host).unwrap();
    x.buf.wait(&conn);
    self.buf.wait(&conn);
    let mut y = 0.0;
    let status = unsafe { cublasSdot_v2(
        cublas.ptr,
        self.dim() as _,
        x.as_ptr(),
        x.stride() as _,
        self.as_ptr(),
        self.stride() as _,
        &mut y as *mut _,
    ) };
    assert!(status.is_ok());
    x.buf.post(&conn);
    self.buf.post(&conn);
    y
  }
}

impl<'a> DeviceArray1dViewMut<'a, f32> {
  pub fn set_scalar(&mut self, c: f32, conn: DeviceConn) {
    if self.stride == 1 {
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_set_scalar_f32(self.as_mut_ptr(), self.dim(), c, conn.stream.ptr) };
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }

  pub fn add_scalar(&mut self, c: f32, conn: DeviceConn) {
    if self.stride == 1 {
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_add_scalar_f32(self.as_mut_ptr(), self.dim(), c, conn.stream.ptr) };
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }

  pub fn scale(&mut self, alpha: f32, conn: DeviceConn) {
    if self.stride == 1 {
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_scale_f32(self.as_mut_ptr(), self.dim(), alpha, conn.stream.ptr) };
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }

  pub fn square(&mut self, conn: DeviceConn) {
    if self.stride == 1 {
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_square_f32(self.as_mut_ptr(), self.dim(), conn.stream.ptr) };
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }

  pub fn reciprocal(&mut self, conn: DeviceConn) {
    if self.stride == 1 {
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_reciprocal_f32(self.as_mut_ptr(), self.dim(), conn.stream.ptr) };
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }

  pub fn add(&mut self, alpha: f32, x: DeviceArray1dView<'a, f32>, beta: f32, conn: DeviceConn) {
    assert_eq!(self.dim(), x.dim());
    if self.stride == 1 {
      x.buf.wait(&conn);
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_add_f32(x.as_ptr(), self.dim(), alpha, beta, self.as_mut_ptr(), conn.stream.ptr) };
      x.buf.post(&conn);
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }

  pub fn average(&mut self, alpha: f32, x: DeviceArray1dView<'a, f32>, conn: DeviceConn) {
    assert_eq!(self.dim(), x.dim());
    if self.stride == 1 {
      x.buf.wait(&conn);
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_average_f32(x.as_ptr(), self.dim(), alpha, self.as_mut_ptr(), conn.stream.ptr) };
      x.buf.post(&conn);
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }

  pub fn elem_mult(&mut self, x: DeviceArray1dView<'a, f32>, conn: DeviceConn) {
    assert_eq!(self.dim(), x.dim());
    if self.stride == 1 {
      x.buf.wait(&conn);
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_elemwise_mult_f32(self.as_mut_ptr(), self.dim(), x.as_ptr(), conn.stream.ptr) };
      x.buf.post(&conn);
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }
}

impl<'a> DeviceArray2dViewMut<'a, f32> {
  pub fn matrix_prod(&mut self, alpha: f32, a: DeviceArray2dView<'a, f32>, a_trans: Transpose, b: DeviceArray2dView<'a, f32>, b_trans: Transpose, beta: f32, conn: DeviceConn) {
    let cublas = conn.cublas();
    cublas.set_pointer_mode(CublasPointerMode::Host).unwrap();
    let (a_m, a_n) = a.dim();
    let (b_m, b_n) = b.dim();
    let (c_m, c_n) = self.dim();
    let (at_m, at_n) = match a_trans {
      Transpose::N => (a_m, a_n),
      Transpose::T => (a_n, a_m),
    };
    let (bt_m, bt_n) = match b_trans {
      Transpose::N => (b_m, b_n),
      Transpose::T => (b_n, b_m),
    };
    assert_eq!(c_m, at_m);
    assert_eq!(c_n, bt_n);
    assert_eq!(at_n, bt_m);
    let k = at_n;
    let (a_inc, lda) = a.stride();
    let (b_inc, ldb) = b.stride();
    let (c_inc, ldc) = self.stride();
    assert_eq!(1, a_inc);
    assert_eq!(1, b_inc);
    assert_eq!(1, c_inc);
    a.buf.wait(&conn);
    b.buf.wait(&conn);
    self.buf.wait(&conn);
    unsafe { cublasSgemm_v2(
        cublas.ptr,
        match a_trans {
          Transpose::N => cublasOperation_t::CUBLAS_OP_N,
          Transpose::T => cublasOperation_t::CUBLAS_OP_T,
        },
        match b_trans {
          Transpose::N => cublasOperation_t::CUBLAS_OP_N,
          Transpose::T => cublasOperation_t::CUBLAS_OP_T,
        },
        c_m as _, c_n as _, k as _,
        &alpha as *const _,
        a.as_ptr(), lda as _,
        b.as_ptr(), ldb as _,
        &beta as *const _,
        self.as_mut_ptr(), ldc as _,
    ) };
    a.buf.post(&conn);
    b.buf.post(&conn);
    self.buf.post(&conn);
  }
}
