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

#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void cast_u8_to_f32_kernel(
    const uint8_t *x,
    uint32_t dim,
    float *y)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float y_i = (float)(x[idx]);
    y[idx] = y_i;
  }
}

extern "C" void devicemem_cuda_cast_u8_to_f32(
    const uint8_t *x,
    size_t dim,
    float *y,
    cudaStream_t stream)
{
  cast_u8_to_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      x, dim, y);
}
