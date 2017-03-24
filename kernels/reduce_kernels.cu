#include <cuda_runtime_api.h>

#include <stdint.h>

__global__ void elem_increduce_mean_f32_kernel(
    uint32_t dim,
    uint32_t count,
    const float *x,
    float *mean)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  float inv_n = 1.0f / ((float)(count + 1));
  if (idx < dim) {
    mean[idx] += inv_n * (x[idx] - mean[idx]);
  }
}

__global__ void elem_prereduce_moments2_f32_kernel(
    uint32_t dim,
    const float *x,
    float *mean,
    float *uvar)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    mean[idx] = x[idx];
    uvar[idx] = 0.0f;
  }
}

__global__ void elem_increduce_moments2_f32_kernel(
    uint32_t dim,
    uint32_t count,
    const float *x,
    float *mean,
    float *uvar)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  float inv_n = 1.0f / ((float)(count + 1));
  if (idx < dim) {
    float x_i = x[idx];
    float prev_mean_i = mean[idx];
    mean[idx] += inv_n * (x_i - prev_mean_i);
    uvar[idx] += (x_i - prev_mean_i) * (x_i - mean[idx]);
  }
}

__global__ void elem_blockreduce_moments2_f32_kernel(
    uint32_t dim,
    uint32_t src_count,
    uint32_t dst_count,
    const float *src_mean,
    const float *src_uvar,
    float *mean,
    float *uvar)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  float scale1 = ((float)(src_count)) / ((float)(src_count + dst_count));
  float scale2 = ((float)(src_count) * (float)(dst_count)) / ((float)(src_count + dst_count));
  if (idx < dim) {
    float delta = src_mean[idx] - mean[idx];
    mean[idx] += delta * scale1;
    uvar[idx] += src_uvar[idx] + delta * delta * scale2;
  }
}

__global__ void elem_postreduce_var_f32_kernel(
    uint32_t dim,
    uint32_t count,
    float *uvar)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  float scale = 1.0 / ((float)(count - 1));
  if (idx < dim) {
    uvar[idx] *= scale;
  }
}

extern "C" void devicemem_cuda_kernel_elem_prereduce_moments2_f32(
    size_t dim,
    const float *x,
    float *mean,
    float *uvar,
    cudaStream_t stream)
{
  elem_prereduce_moments2_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, x, mean, uvar);
}

extern "C" void devicemem_cuda_kernel_elem_increduce_moments2_f32(
    size_t dim,
    size_t count,
    const float *x,
    float *mean,
    float *uvar,
    cudaStream_t stream)
{
  elem_increduce_moments2_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, count, x, mean, uvar);
}

extern "C" void devicemem_cuda_kernel_elem_blockreduce_moments2_f32(
    size_t dim,
    size_t src_count,
    size_t dst_count,
    const float *src_mean,
    const float *src_uvar,
    float *mean,
    float *uvar,
    cudaStream_t stream)
{
  elem_blockreduce_moments2_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, src_count, dst_count, src_mean, src_uvar, mean, uvar);
}

extern "C" void devicemem_cuda_kernel_elem_postreduce_var_f32(
    size_t dim,
    size_t count,
    float *uvar,
    cudaStream_t stream)
{
  elem_postreduce_var_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, count, uvar);
}
