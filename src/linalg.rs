/*
Copyright 2016-2017 the devicemem_cuda authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use super::*;
use ffi::*;

use cuda::runtime::*;
use cuda_blas::*;
use cuda_blas::ffi::*;
use densearray::linalg::{Transpose};

impl<'a> DeviceMemRef<'a, f32> {
  pub fn unsafe_copy(&self, src: DeviceMemRef<'a, f32>, conn: DeviceConn) {
    // TODO
    unimplemented!();
  }

  pub fn unsafe_send(&self, dst: DeviceMemRef<'a, f32>, conn: DeviceConn) {
    // TODO
    unimplemented!();
  }

  pub fn unsafe_add(&self, x: DeviceMemRef<'a, f32>, conn: DeviceConn) {
    // TODO
    unimplemented!();
  }
}

impl<'a> DeviceMemRefMut<'a, f32> {
  pub fn inner_prod(&mut self, x: DeviceArray1dView<'a, f32>, y: DeviceArray1dView<'a, f32>, conn: DeviceConn) {
    assert_eq!(x.dim(), y.dim());
    let cublas = conn.cublas();
    cublas.set_pointer_mode(CublasPointerMode::Device).unwrap();
    // FIXME(20170325): `track` has been disabled (for now?) due to unrelated changes.
    /*let _ = x.buf.track(&conn);
    let _ = y.buf.track(&conn);
    let _ = self.track(&conn);*/
    x.wait(&conn);
    y.wait(&conn);
    self.wait(&conn);
    let status = unsafe { cublasSdot_v2(
        cublas.ptr,
        x.dim() as _,
        x.as_ptr(),
        x.stride() as _,
        y.as_ptr(),
        y.stride() as _,
        self.as_mut_ptr(),
    ) };
    assert!(status.is_ok());
    x.post(&conn);
    y.post(&conn);
    self.post(&conn);
  }

  pub fn self_inner_prod(&mut self, x: DeviceArray1dView<'a, f32>, conn: DeviceConn) {
    let cublas = conn.cublas();
    cublas.set_pointer_mode(CublasPointerMode::Device).unwrap();
    // FIXME(20170325): `track` has been disabled (for now?) due to unrelated changes.
    /*let _ = x.buf.track(&conn);
    let _ = y.buf.track(&conn);
    let _ = self.track(&conn);*/
    x.wait(&conn);
    self.wait(&conn);
    let status = unsafe { cublasSdot_v2(
        cublas.ptr,
        x.dim() as _,
        x.as_ptr(),
        x.stride() as _,
        x.as_ptr(),
        x.stride() as _,
        self.as_mut_ptr(),
    ) };
    assert!(status.is_ok());
    x.post(&conn);
    self.post(&conn);
  }

  pub fn reduce_sum(&mut self, x: DeviceArray1dView<'a, f32>, conn: DeviceConn) {
    assert_eq!(1, self.len());
    x.wait(&conn);
    self.wait(&conn);
    // TODO: do a deterministic reduce.
    unsafe { devicemem_cuda_kernel_reduce_sum_atomic_f32(
        x.dim(),
        x.as_ptr(),
        self.as_mut_ptr(),
        conn.raw_stream().as_ptr(),
    ) };
    x.post(&conn);
    self.post(&conn);
  }
}

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

  pub fn add_constant(&mut self, c: f32, conn: DeviceConn) {
    if self.stride == 1 {
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_add_constant_f32(self.as_mut_ptr(), self.dim(), c, conn.stream.ptr) };
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }

  pub fn add_scalar(&mut self, c: DeviceMemRef<'a, f32>, conn: DeviceConn) {
    if self.stride == 1 {
      c.wait(&conn);
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_add_scalar_f32(self.dim(), c.as_ptr(), self.as_mut_ptr(), conn.stream.ptr) };
      c.post(&conn);
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

  pub fn div_scalar(&mut self, alpha: f32, conn: DeviceConn) {
    if self.stride == 1 {
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_div_scalar_f32(self.as_mut_ptr(), self.dim(), alpha, conn.stream.ptr) };
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

  pub fn exp(&mut self, conn: DeviceConn) {
    if self.stride == 1 {
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_exp_f32(self.as_mut_ptr(), self.dim(), conn.stream.ptr) };
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }

  pub fn add(&mut self, alpha: f32, x: DeviceArray1dView<'a, f32>, conn: DeviceConn) {
    assert_eq!(self.dim(), x.dim());
    if self.stride == 1 {
      x.buf.wait(&conn);
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_vector_add_f32(x.as_ptr(), self.dim(), alpha, self.as_mut_ptr(), conn.stream.ptr) };
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

  pub fn elem_div(&mut self, x: DeviceArray1dView<'a, f32>, conn: DeviceConn) {
    assert_eq!(self.dim(), x.dim());
    if self.stride == 1 {
      x.buf.wait(&conn);
      self.buf.wait(&conn);
      /*unsafe { devicemem_cuda_vector_elemwise_div_f32(self.as_mut_ptr(), self.dim(), x.as_ptr(), conn.stream.ptr) };*/
      unsafe { devicemem_cuda_kernel_elem_div_f32(self.dim(), x.as_ptr(), self.as_mut_ptr(), conn.raw_stream().as_ptr()) };
      x.buf.post(&conn);
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }

  pub fn elem_ldiv(mut self, x: DeviceArray1dView<'a, f32>, conn: DeviceConn) {
    assert_eq!(self.dim(), x.dim());
    if self.stride == 1 {
      x.buf.wait(&conn);
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_kernel_elem_ldiv_f32(self.dim(), x.as_ptr(), self.as_mut_ptr(), conn.raw_stream().as_ptr()) };
      x.buf.post(&conn);
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }

  pub fn sqrt(mut self, conn: DeviceConn) {
    if self.stride == 1 {
      self.buf.wait(&conn);
      unsafe { devicemem_cuda_kernel_sqrt_f32(self.dim(), self.as_mut_ptr(), conn.raw_stream().as_ptr()) };
      self.buf.post(&conn);
    } else {
      unimplemented!();
    }
  }
}

impl<'a> DeviceArray2dViewMut<'a, f32> {
  pub fn matrix_add(&mut self, alpha: f32, x: DeviceArray2dView<'a, f32>, conn: DeviceConn) {
    let (x_m, x_n) = x.dim();
    let (y_m, y_n) = self.dim();
    assert_eq!(x_m, y_m);
    assert_eq!(x_n, y_n);
    let (incx, ldx) = x.stride();
    let (incy, ldy) = self.stride();
    if x_n == 1 {
      x.buf.wait(&conn);
      self.buf.wait(&conn);
      let cublas = conn.cublas();
      cublas.set_pointer_mode(CublasPointerMode::Host).unwrap();
      unsafe { cublasSaxpy_v2(
          cublas.ptr,
          x_m as _,
          &alpha as *const _,
          x.as_ptr(),
          incx as _,
          self.as_mut_ptr(),
          incy as _,
      ) };
      x.buf.post(&conn);
      self.buf.post(&conn);
    } else if x_m == 1 {
      unimplemented!();
    } else {
      unimplemented!();
    }
  }

  pub fn matrix_prod(&mut self, alpha: f32, a: DeviceArray2dView<'a, f32>, a_trans: Transpose, b: DeviceArray2dView<'a, f32>, b_trans: Transpose, beta: f32, conn: DeviceConn) {
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
    let cublas = conn.cublas();
    cublas.set_pointer_mode(CublasPointerMode::Host).unwrap();
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
