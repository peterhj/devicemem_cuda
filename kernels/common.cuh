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

#ifndef __ARRAYDIFF_CUDA_KERNELS_COMMON_H__
#define __ARRAYDIFF_CUDA_KERNELS_COMMON_H__

#include <stdint.h>

__forceinline__ __device__ void threadblock1024_reduce_sum_f32(
    float *cache)
{
  for (uint32_t s = 512; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      cache[threadIdx.x] += cache[threadIdx.x + s];
    }
    __syncthreads();
  }
}

__forceinline__ __device__ void threadblock1024_reduce_max_f32(
    float *cache)
{
  for (uint32_t s = 512; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      cache[threadIdx.x] = max(cache[threadIdx.x], cache[threadIdx.x + s]);
    }
    __syncthreads();
  }
}

#endif
