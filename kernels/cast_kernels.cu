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
