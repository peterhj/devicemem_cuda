#include <cuda_runtime_api.h>
// FIXME(20160123): commentng out for cuda 7.0.
//#include <cuda_fp16.h>

#include <stdint.h>

__global__ void vector_set_scalar_f32_kernel(
    float *dst,
    int dim,
    float c)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    dst[idx] = c;
  }
}

extern "C" void devicemem_cuda_vector_set_scalar_f32(
    float *dst,
    size_t dim,
    float c,
    cudaStream_t stream)
{
  vector_set_scalar_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dst, dim, c);
}

__global__ void vector_add_scalar_f32_kernel(
    float *dst,
    int dim,
    float c)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float y = dst[idx] + c;
    dst[idx] = y;
  }
}

extern "C" void devicemem_cuda_vector_add_scalar_f32(
    float *dst,
    size_t dim,
    float c,
    cudaStream_t stream)
{
  vector_add_scalar_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dst, dim, c);
}

__global__ void vector_scale_f32_kernel(
    float *dst,
    int dim,
    float alpha)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float y = alpha * dst[idx];
    dst[idx] = y;
  }
}

extern "C" void devicemem_cuda_vector_scale_f32(
    float *dst,
    size_t dim,
    float alpha,
    cudaStream_t stream)
{
  vector_scale_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dst, dim, alpha);
}

__global__ void vector_exp_f32_kernel(
    float *xs,
    int dim)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = expf(xs[idx]);
    xs[idx] = x;
  }
}

extern "C" void devicemem_cuda_vector_exp_f32(
    float *xs,
    size_t dim,
    cudaStream_t stream)
{
  vector_exp_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      xs, dim);
}

__global__ void vector_square_f32_kernel(
    float *dst,
    int dim)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x = dst[idx];
    dst[idx] = x * x;
  }
}

extern "C" void devicemem_cuda_vector_square_f32(
    float *dst,
    size_t dim,
    cudaStream_t stream)
{
  vector_square_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dst, dim);
}

__global__ void vector_reciprocal_f32_kernel(
    float *dst,
    int dim)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float y = 1.0f / dst[idx];
    dst[idx] = y;
  }
}

extern "C" void devicemem_cuda_vector_reciprocal_f32(
    float *dst,
    size_t dim,
    cudaStream_t stream)
{
  vector_reciprocal_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dst, dim);
}

__global__ void vector_set_f32_kernel(
    const float *src,
    int dim,
    float alpha,
    float *dst)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float y = alpha * src[idx];
    dst[idx] = y;
  }
}

extern "C" void devicemem_cuda_vector_set_f32(
    const float *src,
    size_t dim,
    float alpha,
    float *dst,
    cudaStream_t stream)
{
  vector_set_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      src, dim, alpha, dst);
}

__global__ void vector_add_f32_kernel(
    const float *src,
    int dim,
    float alpha,
    float *dst)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float y = alpha * src[idx] + dst[idx];
    dst[idx] = y;
  }
}

extern "C" void devicemem_cuda_vector_add_f32(
    const float *src,
    size_t dim,
    float alpha,
    float *dst,
    cudaStream_t stream)
{
  vector_add_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      src, dim, alpha, dst);
}

__global__ void vector_average_f32_kernel(
    const float *src,
    int dim,
    float alpha,
    float *dst)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float y = dst[idx];
    y = y + alpha * (src[idx] - y);
    dst[idx] = y;
  }
}

extern "C" void devicemem_cuda_vector_average_f32(
    const float *src,
    size_t dim,
    float alpha,
    float *dst,
    cudaStream_t stream)
{
  vector_average_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      src, dim, alpha, dst);
}

__global__ void vector_elemwise_mult_f32_kernel(
    float *ys,
    int dim,
    const float *xs)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float y = xs[idx] * ys[idx];
    ys[idx] = y;
  }
}

extern "C" void devicemem_cuda_vector_elemwise_mult_f32(
    float *dst,
    size_t dim,
    const float *xs,
    cudaStream_t stream)
{
  vector_elemwise_mult_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dst, dim, xs);
}

__global__ void vector_elemwise_div_f32_kernel(
    float *ys,
    int dim,
    const float *xs)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float y = ys[idx] / xs[idx];
    ys[idx] = y;
  }
}

extern "C" void devicemem_cuda_vector_elemwise_div_f32(
    float *dst,
    size_t dim,
    const float *xs,
    cudaStream_t stream)
{
  vector_elemwise_div_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dst, dim, xs);
}
