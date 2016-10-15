use cuda::ffi::runtime::{cudaStream_t};
//use float::stub::{f16_stub};
use libc::*;

#[link(name = "devicemem_cuda_kernels", kind = "static")]
extern "C" {
  pub fn devicemem_cuda_vector_set_scalar_f32(dst: *mut f32, dim: size_t, c: f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_add_scalar_f32(dst: *mut f32, dim: size_t, c: f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_scale_f32(dst: *mut f32, dim: size_t, alpha: f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_square_f32(dst: *mut f32, dim: size_t, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_reciprocal_f32(dst: *mut f32, dim: size_t, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_exp_f32(xs: *mut f32, dim: size_t, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_set_f32(src: *const f32, dim: size_t, alpha: f32, dst: *mut f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_add_f32(src: *const f32, dim: size_t, alpha: f32, beta: f32, dst: *mut f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_average_f32(src: *const f32, dim: size_t, alpha: f32, dst: *mut f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_elemwise_mult_f32(dst: *mut f32, len: size_t, xs: *const f32, stream: cudaStream_t);
  pub fn devicemem_cuda_vector_elemwise_div_f32(dst: *mut f32, len: size_t, xs: *const f32, stream: cudaStream_t);
}
