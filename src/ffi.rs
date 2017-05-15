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

use cuda::ffi::runtime::{cudaStream_t};
//use float::stub::{f16_stub};
use libc::*;

#[link(name = "devicemem_cuda_kernels", kind = "static")]
extern "C" {
  pub fn devicemem_cuda_cast_u8_to_f32(
      x: *const u8,
      dim: size_t,
      y: *mut f32,
      stream: cudaStream_t);

  pub fn devicemem_cuda_vector_set_scalar_f32(dst: *mut f32, dim: size_t, c: f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_add_constant_f32(dst: *mut f32, dim: size_t, c: f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_add_scalar_f32(dim: size_t, c: *const f32, y: *mut f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_scale_f32(dst: *mut f32, dim: size_t, alpha: f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_div_scalar_f32(dst: *mut f32, dim: size_t, c: f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_square_f32(dst: *mut f32, dim: size_t, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_reciprocal_f32(dst: *mut f32, dim: size_t, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_exp_f32(xs: *mut f32, dim: size_t, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_set_f32(src: *const f32, dim: size_t, alpha: f32, dst: *mut f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_add_f32(src: *const f32, dim: size_t, alpha: f32, dst: *mut f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_average_f32(src: *const f32, dim: size_t, alpha: f32, dst: *mut f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_elemwise_mult_f32(dst: *mut f32, len: size_t, xs: *const f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_elemwise_div_f32(dst: *mut f32, len: size_t, xs: *const f32, stream: cudaStream_t);

  pub fn devicemem_cuda_kernel_elem_div_f32(dim: usize, divisor: *const f32, x: *mut f32, stream: cudaStream_t);
  pub fn devicemem_cuda_kernel_elem_ldiv_f32(dim: usize, ldivisor: *const f32, x: *mut f32, stream: cudaStream_t);

  pub fn devicemem_cuda_kernel_sqrt_f32(dim: usize, x: *mut f32, stream: cudaStream_t);

  pub fn devicemem_cuda_kernel_reduce_sum_atomic_f32(
      dim: usize,
      x: *const f32,
      sum: *mut f32,
      stream: cudaStream_t);
  pub fn devicemem_cuda_kernel_elem_prereduce_moments2_f32(
      dim: usize,
      x: *const f32,
      mean: *mut f32,
      uvar: *mut f32,
      stream: cudaStream_t);
  pub fn devicemem_cuda_kernel_elem_increduce_moments2_f32(
      dim: usize,
      count: usize,
      x: *const f32,
      mean: *mut f32,
      uvar: *mut f32,
      stream: cudaStream_t);
  pub fn devicemem_cuda_kernel_elem_blockreduce_moments2_f32(
      dim: usize,
      src_count: usize,
      dst_count: usize,
      src_mean: *const f32,
      src_uvar: *const f32,
      mean: *mut f32,
      uvar: *mut f32,
      stream: cudaStream_t);
  pub fn devicemem_cuda_kernel_elem_postreduce_var_f32(
      dim: usize,
      count: usize,
      uvar: *mut f32,
      stream: cudaStream_t);

  /*pub fn devicemem_cuda_kernel_plane_image_catmullrom_resize_f32(
      in_pixels: *const f32,
      in_width: usize,
      in_height: usize,
      channels: usize,
      out_pixels: *mut f32,
      out_width: usize,
      out_height: usize,
      stream: cudaStream_t);*/
}
