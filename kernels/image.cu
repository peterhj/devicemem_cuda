#include <cuda_runtime_api.h>
#include <stdint.h>

__device__ float catrom_w0_f32(float a) {
    //return -0.5f*a + a*a - 0.5f*a*a*a;
    return a*(-0.5f + a*(1.0f - 0.5f*a));
}

__device__ float catrom_w1_f32(float a) {
    //return 1.0f - 2.5f*a*a + 1.5f*a*a*a;
    return 1.0f + a*a*(-2.5f + 1.5f*a);
}

__device__ float catrom_w2_f32(float a) {
    //return 0.5f*a + 2.0f*a*a - 1.5f*a*a*a;
    return a*(0.5f + a*(2.0f - 1.5f*a));
}

__device__ float catrom_w3_f32(float a) {
    //return -0.5f*a*a + 0.5f*a*a*a;
    return a*a*(-0.5f + 0.5f*a);
}

__device__ float image_tex2d_clamp_f32(const float *pixels, int width, int height, int u, int v, int c) {
  int clamp_u = min(max(0, u), width-1);
  int clamp_v = min(max(0, v), height-1);
  return pixels[clamp_u + clamp_v * width + c * width * height];
}

__device__ float image_catmullrom_filter_f32(
    float x,
    float a0,
    float a1,
    float a2,
    float a3)
{
  float r = a0 * catrom_w0_f32(x);
  r += a1 * catrom_w1_f32(x);
  r += a2 * catrom_w2_f32(x);
  r += a3 * catrom_w3_f32(x);
  return r;
}

__device__ float image_catmullrom_interpolate_f32(
    const float *pixels,
    int width,
    int height,
    float u,
    float v,
    int c)
{
  u -= 0.5f;
  v -= 0.5f;
  float px = floorf(u);
  float py = floorf(v);
  float fx = u - px;
  float fy = v - py;
  int ipx = (int)px;
  int ipy = (int)py;
  return image_catmullrom_filter_f32(fy,
      image_catmullrom_filter_f32(fx,
          image_tex2d_clamp_f32(pixels, width, height, ipx-1, ipy-1, c),
          image_tex2d_clamp_f32(pixels, width, height, ipx,   ipy-1, c),
          image_tex2d_clamp_f32(pixels, width, height, ipx+1, ipy-1, c),
          image_tex2d_clamp_f32(pixels, width, height, ipx+2, ipy-1, c)),
      image_catmullrom_filter_f32(fx,
          image_tex2d_clamp_f32(pixels, width, height, ipx-1, ipy,   c),
          image_tex2d_clamp_f32(pixels, width, height, ipx,   ipy,   c),
          image_tex2d_clamp_f32(pixels, width, height, ipx+1, ipy,   c),
          image_tex2d_clamp_f32(pixels, width, height, ipx+2, ipy,   c)),
      image_catmullrom_filter_f32(fx,
          image_tex2d_clamp_f32(pixels, width, height, ipx-1, ipy+1, c),
          image_tex2d_clamp_f32(pixels, width, height, ipx,   ipy+1, c),
          image_tex2d_clamp_f32(pixels, width, height, ipx+1, ipy+1, c),
          image_tex2d_clamp_f32(pixels, width, height, ipx+2, ipy+1, c)),
      image_catmullrom_filter_f32(fx,
          image_tex2d_clamp_f32(pixels, width, height, ipx-1, ipy+2, c),
          image_tex2d_clamp_f32(pixels, width, height, ipx,   ipy+2, c),
          image_tex2d_clamp_f32(pixels, width, height, ipx+1, ipy+2, c),
          image_tex2d_clamp_f32(pixels, width, height, ipx+2, ipy+2, c)));
}

__global__ void image_catmullrom_resize_f32_kernel(
    const float *in_pixels,
    int in_width,
    int in_height,
    int channels,
    float *out_pixels,
    int out_width,
    int out_height)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int x = idx % out_width;
  int y = (idx / out_width) % out_height;
  int c = idx / (out_width * out_height);

  if ((x < out_width) && (y < out_height) && (c < channels)) {
    float u = ((float)x) / ((float)out_width) * ((float)in_width);
    float v = ((float)y) / ((float)out_height) * ((float)in_height);

    float interp_value = image_catmullrom_interpolate_f32(in_pixels, in_width, in_height, u, v, c);

    out_pixels[x + y * out_width + c * out_width * out_height] = interp_value;
  }
}

extern "C" void devicemem_cuda_kernel_plane_image_catmullrom_resize_f32(
    const float *in_pixels,
    size_t in_width,
    size_t in_height,
    size_t channels,
    float *out_pixels,
    size_t out_width,
    size_t out_height,
    cudaStream_t stream)
{
  int n = out_width * out_height * channels;
  image_catmullrom_resize_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      in_pixels,
      in_width,
      in_height,
      channels,
      out_pixels,
      out_width,
      out_height);
}
