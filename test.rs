extern crate densearray;
extern crate devicemem_cuda;

use densearray::*;
use devicemem_cuda::{DeviceStream, DeviceArray1d};
//use devicemem_cuda::linalg::*;

fn main() {
  let stream = DeviceStream::new(0);
  let mut x: DeviceArray1d<f32> = DeviceArray1d::<f32>::zeros(1000, stream.conn());
  let mut y: DeviceArray1d<f32> = DeviceArray1d::<f32>::zeros(1000, stream.conn());
  x.as_view_mut().set_constant(1.0, stream.conn());
  y.as_view_mut().set_constant(1.0, stream.conn());
  let x_dot_y = x.as_view().inner_prod(y.as_view(), stream.conn());
  println!("DEBUG: dot: {:e}", x_dot_y);
}
