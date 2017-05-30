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

__global__ void async_fence_init1_kernel(
    unsigned long long *fencebuf1)
{
  if (threadIdx.x == 0) {
    unsigned long long INIT_EPOCH = 1ULL;
    fencebuf1[0] = INIT_EPOCH;
  }
}

__global__ void async_fence_post1_kernel(
    unsigned long long prev_epoch1,
    unsigned long long *fencebuf1)
{
  if (threadIdx.x == 0) {
    unsigned long long next_epoch1 = prev_epoch1 + 1ULL;
    for (;;) {
      if (prev_epoch1 == atomicCAS(fencebuf1, prev_epoch1, next_epoch1)) {
        break;
      }
    }
  }
}

__global__ void async_fence_wait1_kernel(
    unsigned long long next_epoch1,
    unsigned long long *fencebuf1)
{
  if (threadIdx.x == 0) {
    for (;;) {
      if (next_epoch1 == atomicCAS(fencebuf1, next_epoch1, next_epoch1)) {
        break;
      }
    }
  }
}

extern "C" void devicemem_cuda_async_fence_init1(
    uint64_t *fencebuf1,
    cudaStream_t stream)
{
}

extern "C" void devicemem_cuda_async_fence_post1(
    uint64_t prev_epoch1,
    uint64_t *fencebuf1,
    cudaStream_t stream)
{
}

extern "C" void devicemem_cuda_async_fence_wait1(
    uint64_t next_epoch1,
    uint64_t *fencebuf1,
    cudaStream_t stream)
{
}
