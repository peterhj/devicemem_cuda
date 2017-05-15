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

#include "common.cuh"
#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void sqrt_f32_kernel(
    uint32_t dim,
    float *x)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    x[idx] = sqrtf(x[idx]);
  }
}

extern "C" void devicemem_cuda_kernel_sqrt_f32(
    size_t dim,
    float *x,
    cudaStream_t stream)
{
  sqrt_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, x);
}
