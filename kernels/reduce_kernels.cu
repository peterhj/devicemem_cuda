/*
Copyright 2017 the arraydiff authors

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

#include "common.cuh"
#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void reduce_sum_atomic_f32_kernel(
    uint32_t dim,
    const float *x,
    float *sum)
{
  __shared__ float cache[1024];
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    cache[threadIdx.x] = x[idx];
  } else {
    cache[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (0 == threadIdx.x) {
    atomicAdd(sum, cache[0]);
  }
}

extern "C" void devicemem_cuda_kernel_reduce_sum_atomic_f32(
    size_t dim,
    const float *x,
    float *sum,
    cudaStream_t stream)
{
  reduce_sum_atomic_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, x, sum);
}

__global__ void block_reduce_sum_f32_kernel(
    uint32_t dim,
    const float *x,
    float *sum)
{
  __shared__ float cache[1024];
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    cache[threadIdx.x] = x[idx];
  } else {
    cache[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (idx < dim) {
    if (0 == threadIdx.x) {
      sum[blockIdx.x] = cache[0];
    }
  }
}

extern "C" void devicemem_cuda_kernel_reduce_sum_scratch_f32(
    size_t dim,
    const float *x,
    float *sum,
    float *scratch,
    cudaStream_t stream)
{
  size_t level = 0;
  size_t width = dim;
  do {
    size_t next_width = (width + 1024 - 1) / 1024;
    const float *src = NULL;
    float *dst = NULL;
    // TODO
    block_reduce_sum_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
        dim, src, dst);
    level += 1;
    width = next_width;
  } while (width > 1);
}

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
