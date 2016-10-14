extern crate gcc;

use std::env;

fn main() {
  gcc::Config::new()
    .compiler("/usr/local/cuda/bin/nvcc")
    .opt_level(3)
    // FIXME(20151207): for working w/ K80.
    //.flag("-arch=sm_37")
    .flag("-arch=sm_52")
    .flag("-Xcompiler")
    .flag("\'-fPIC\'")
    .include("/usr/local/cuda/include")
    .file("kernels/vector_kernels.cu")
    .compile("libdevicemem_cuda_kernels.a");

  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64");
  let out_dir = env::var("OUT_DIR").unwrap();
  println!("cargo:rustc-link-search=native={}", out_dir);
}
