extern crate densearray;
extern crate devicemem_cuda;

use densearray::*;
use devicemem_cuda::{Device, DeviceArray1d};
//use devicemem_cuda::linalg::*;

fn main() {
  let device = Device::new(0);
  let conn = device.conn();
  let mut x: DeviceArray1d<f32> = DeviceArray1d::<f32>::zeros(1000, conn.clone());
  let mut y: DeviceArray1d<f32> = DeviceArray1d::<f32>::zeros(1000, conn.clone());
  x.as_view_mut().set_constant(1.0, conn.clone());
  y.as_view_mut().set_constant(1.0, conn.clone());
  let x_dot_y = x.as_view().inner_prod(y.as_view(), conn.clone());
  println!("DEBUG: dot: {:e}", x_dot_y);
}
