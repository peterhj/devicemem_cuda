extern crate gcc;

use std::env;

fn main() {
  gcc::Config::new()
    //.compiler("/usr/local/cuda/bin/nvcc")
    .compiler("/usr/local/cuda-8.0/bin/nvcc")
    .opt_level(3)
    // FIXME(20151207): for working w/ K80.
    .flag("-arch=compute_37")
    .flag("-code=sm_37,sm_52")
    //.flag("-arch=sm_52")
    .flag("-prec-div=true")
    .flag("-prec-sqrt=true")
    .flag("-Xcompiler")
    .flag("\'-fno-strict-aliasing\'")
    .flag("-Xcompiler")
    .flag("\'-Werror\'")
    .pic(true)
    .include("cuda_kernels")
    .include("/usr/local/cuda/include")
    .file("kernels/cast_kernels.cu")
    .file("kernels/fence.cu")
    .file("kernels/reduce_kernels.cu")
    //.file("kernels/stats_kernels.cu")
    .file("kernels/special.cu")
    .file("kernels/vector_kernels.cu")
    .compile("libdevicemem_cuda_kernels.a");

  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64");
  let out_dir = env::var("OUT_DIR").unwrap();
  println!("cargo:rustc-link-search=native={}", out_dir);
}
