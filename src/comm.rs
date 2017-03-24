use super::*;
use kernels::*;
use linalg::*;

use cuda::runtime::*;
use densearray::prelude::*;
use nvsmi::*;
use sharedmem::sync::{SpinBarrier};
use stopwatch::{Stopwatch};

use std::cmp::{min};
use std::fs::{File};
use std::io::{Write, BufWriter};
use std::path::{PathBuf};
use std::sync::{Arc, Barrier, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};

#[derive(Clone, Copy, Default, Debug)]
pub struct TimingInfo {
  pub elapsed:  f64,
}

pub struct GPUAllreduceIo<T> where T: Copy {
  worker:       Option<GPURingAllreduceWorker<T>>,
  reduce_buf:   Option<DeviceMem<T>>,
  log_file:     Option<BufWriter<File>>,
  counter:      usize,
}

impl<T> GPUAllreduceIo<T> where T: ZeroBits {
  pub fn empty(worker_rank: usize) -> Self {
    let log_file = if worker_rank == 0 {
      let mut log_file = File::create(&PathBuf::from("comm_trace.csv")).unwrap();
      writeln!(&mut log_file, "iter,elapsed").unwrap();
      Some(BufWriter::new(log_file))
    } else {
      None
    };
    GPUAllreduceIo{
      worker:       None,
      reduce_buf:   None,
      log_file:     log_file,
      counter:      0,
    }
  }

  pub fn is_empty(&self) -> bool {
    self.worker.is_none()
  }

  pub fn attach(&mut self, worker: GPURingAllreduceWorker<T>, conn: DeviceConn) {
    let reduce_buf = DeviceMem::zeros(worker.buffer_size(), conn);
    self.worker = Some(worker);
    self.reduce_buf = Some(reduce_buf);
  }

  /*pub fn new(worker: GPURingAllreduceWorker<T>, conn: DeviceConn) -> Self {
    let reduce_buf = DeviceMem::zeros(worker.buffer_size(), conn);
    let log_file = if worker.worker_rank == 0 {
      let mut log_file = File::create(&PathBuf::from("comm_trace.csv")).unwrap();
      writeln!(&mut log_file, "iter,elapsed").unwrap();
      Some(BufWriter::new(log_file))
    } else {
      None
    };
    GPUAllreduceIo{
      worker:       worker,
      reduce_buf:   reduce_buf,
      log_file:     log_file,
      counter:      0,
    }
  }*/
}

impl<T> GPUAllreduceIo<T> where T: Copy {
  pub fn buffer(&self) -> &DeviceMem<T> {
    self.reduce_buf.as_ref().unwrap()
  }

  pub fn buffer_mut(&mut self) -> &mut DeviceMem<T> {
    self.reduce_buf.as_mut().unwrap()
  }

  pub fn as_ref(&self) -> DeviceMemRef<T> {
    self.reduce_buf.as_ref().unwrap().as_ref()
  }

  pub fn as_mut(&mut self) -> DeviceMemRefMut<T> {
    self.reduce_buf.as_mut().unwrap().as_mut()
  }
}

impl GPUAllreduceIo<f32> {
  pub fn write_allreduce_sum<'a, A>(&mut self, src_buf: A, stream: &DeviceStream) -> TimingInfo where A: FlatView<'a, DeviceArray1dView<'a, f32>> {
    let timing_info = self.worker.as_mut().unwrap().allreduce_sum(src_buf, self.reduce_buf.as_mut().unwrap(), stream);
    if let Some(ref mut log_file) = self.log_file {
      writeln!(log_file, "{},{:.9}", self.counter, timing_info.elapsed).unwrap();
      self.counter += 1;
    }
    timing_info
  }
}

pub struct GPUMomentsIo<T> where T: Copy {
  worker:       Option<GPURingAllreduceWorker<T>>,
  reduce_buf:   Option<DeviceMem<T>>,
  /*log_file:     Option<BufWriter<File>>,
  counter:      usize,*/
}

#[derive(Clone, Copy)]
pub enum GPUAllreduceChannelWorkerState {
  Empty,
  Full,
}

#[derive(Clone, Copy)]
pub enum GPUAllreduceChannelReq {
  BeginProduce,
  DoneProduce,
  BeginConsume,
  DoneConsume,
}

pub struct GPUAllreduceChannelBuilder<T> where T: Copy {
  builder:      GPURingAllreduceBuilder<T>,
}

impl<T> GPUAllreduceChannelBuilder<T> where T: Copy {
  pub fn new(num_workers: usize) -> Self {
    // TODO
    unimplemented!();
  }

  pub fn into_channel(self, worker_rank: usize, buf_sz: usize, stream: Arc<DeviceStream>) -> (GPUAllreduceChannelProducer<T>, GPUAllreduceChannelConsumer<T>) {
    // TODO
    unimplemented!();
  }
}

pub struct GPUAllreduceChannelWorker<T> where T: Copy {
  state:        GPUAllreduceChannelWorkerState,
  producer_rx:  Receiver<GPUAllreduceChannelReq>,
  consumer_rx:  Receiver<GPUAllreduceChannelReq>,
  stream:       Arc<DeviceStream>,
  worker:       GPURingAllreduceWorker<T>,
  src_buf:      Arc<Mutex<DeviceMem<T>>>,
  reduce_buf:   Arc<Mutex<DeviceMem<T>>>,
}

pub struct GPUAllreduceChannelProducer<T> where T: Copy {
  req_tx:       SyncSender<GPUAllreduceChannelReq>,
  stream:       Arc<DeviceStream>,
  src_buf:      Arc<Mutex<DeviceMem<T>>>,
}

impl GPUAllreduceChannelProducer<f32> {
  pub fn write_allreduce_sum<'a, A>(&mut self, ext_src_buf: A) where A: FlatView<'a, DeviceArray1dView<'a, f32>> {
    // TODO

    match self.req_tx.send(GPUAllreduceChannelReq::BeginProduce) {
      Err(_) => panic!(),
      Ok(_) => {}
    }

    {
      let mut src_buf = self.src_buf.lock().unwrap();
      src_buf.as_mut().flatten_mut().copy(ext_src_buf.flatten(), self.stream.conn());
    }

    match self.req_tx.send(GPUAllreduceChannelReq::DoneProduce) {
      Err(_) => panic!(),
      Ok(_) => {}
    }
  }
}

pub struct GPUAllreduceChannelConsumer<T> where T: Copy {
  req_tx:       SyncSender<GPUAllreduceChannelReq>,
  stream:       Arc<DeviceStream>,
  reduce_buf:   Arc<Mutex<DeviceMem<T>>>,
}

impl GPUAllreduceChannelConsumer<f32> {
  pub fn read<'a>(&mut self, dst_buf: &mut DeviceMemRefMut<'a, f32>) {
    // TODO

    match self.req_tx.send(GPUAllreduceChannelReq::BeginConsume) {
      Err(_) => panic!(),
      Ok(_) => {}
    }

    {
      let reduce_buf = self.reduce_buf.lock().unwrap();
      // TODO: lifetime acrobatics.
      //dst_buf.copy(reduce_buf.as_ref(), self.stream.conn());
      reduce_buf.as_ref().send(dst_buf, self.stream.conn());
    }

    match self.req_tx.send(GPUAllreduceChannelReq::DoneConsume) {
      Err(_) => panic!(),
      Ok(_) => {}
    }
  }
}

#[derive(Clone)]
pub struct GPURingAllreduceState<T> where T: Copy {
  barrier:  Arc<SpinBarrier>,
  buf_sz:   Arc<AtomicUsize>,
  parts:    Vec<Vec<Arc<Mutex<Option<(usize, DeviceMem<T>, DeviceMem<T>)>>>>>,
}

#[derive(Clone)]
pub struct GPURingAllreduceBuilder<T> where T: Copy {
  num_workers:  usize,
  state:    GPURingAllreduceState<T>,
}

impl<T> GPURingAllreduceBuilder<T> where T: Copy {
  pub fn new(num_workers: usize) -> Self {
    assert!(num_workers >= 1);
    let mut parts = Vec::with_capacity(num_workers);
    for _ in 0 .. num_workers {
      let mut rank_parts = Vec::with_capacity(num_workers);
      for _ in 0 .. num_workers {
        rank_parts.push(Arc::new(Mutex::new(None)));
      }
      parts.push(rank_parts);
    }
    GPURingAllreduceBuilder{
      num_workers:  num_workers,
      state:        GPURingAllreduceState{
        barrier:    Arc::new(SpinBarrier::new(num_workers)),
        buf_sz:     Arc::new(AtomicUsize::new(0)),
        parts:      parts,
      },
    }
  }
}

impl<T> GPURingAllreduceBuilder<T> where T: Copy + ZeroBits {
  pub fn into_worker(self, worker_rank: usize, buf_sz: usize, stream: &DeviceStream) -> GPURingAllreduceWorker<T> {
    assert!(worker_rank < self.num_workers);
    let prev_buf_sz = self.state.buf_sz.compare_and_swap(0, buf_sz, Ordering::SeqCst);
    assert!(prev_buf_sz == 0 || prev_buf_sz == buf_sz);
    let max_part_sz = (buf_sz + self.num_workers - 1) / self.num_workers;
    let mut part_offset = 0;
    for p in 0 .. self.num_workers {
      let part_sz = min(max_part_sz, buf_sz - p * max_part_sz);
      let part_buf = DeviceMem::zeros(part_sz, stream.conn());
      let part_recv_buf = DeviceMem::zeros(part_sz, stream.conn());
      let mut part = self.state.parts[worker_rank][p].lock().unwrap();
      *part = Some((part_offset, part_buf, part_recv_buf));
      part_offset += part_sz;
    }
    stream.sync();
    self.state.barrier.wait();
    GPURingAllreduceWorker{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      buf_sz:       buf_sz,
      state:        self.state,
      stopwatch:    Stopwatch::new(),
      //_marker:      PhantomData,
    }
  }
}

pub struct GPURingAllreduceWorker<T> where T: Copy {
  worker_rank:  usize,
  num_workers:  usize,
  buf_sz:       usize,
  state:        GPURingAllreduceState<T>,
  stopwatch:    Stopwatch,
}

impl<T> GPURingAllreduceWorker<T> where T: Copy {
  pub fn buffer_size(&self) -> usize {
    self.buf_sz
  }
}

impl GPURingAllreduceWorker<f32> {
  pub fn broadcast<'a, A>(&mut self, root: usize, in_buf: A, out_buf: &mut DeviceMem<f32>, stream: &DeviceStream) where A: FlatView<'a, DeviceArray1dView<'a, f32>> {
    // TODO
    unimplemented!();
  }

  //pub fn allreduce_sum(&self, in_buf: &DeviceMem<f32>, out_buf: &mut DeviceMem<f32>, stream: &DeviceStream) {
  pub fn allreduce_sum<'a, A>(&mut self, in_buf: A, out_buf: &mut DeviceMem<f32>, stream: &DeviceStream) -> TimingInfo where A: FlatView<'a, DeviceArray1dView<'a, f32>> {
    let in_arr = in_buf.flatten();
    /*assert_eq!(self.buf_sz, in_arr.dim());
    assert_eq!(self.buf_sz, out_buf.len());*/
    assert!(in_arr.dim() <= self.buf_sz);
    assert!(out_buf.len() <= self.buf_sz);
    assert!(in_arr.dim() <= out_buf.len());

    let mut timing_info = TimingInfo::default();

    if self.num_workers == 1 {
      out_buf.as_mut().flatten_mut().copy(in_arr, stream.conn());
      return timing_info;
    }

    for p in 0 .. self.num_workers {
      let mut part = self.state.parts[self.worker_rank][p].lock().unwrap();
      assert!(part.is_some());
      let &mut (part_offset, ref mut part_buf, ref mut part_recv_buf) = &mut *part.as_mut().unwrap();
      let part_sz = part_buf.len();
      //part_buf.as_mut().copy(in_buf.as_ref().slice(part_offset, part_offset + part_sz), stream.conn());
      let start_pos = min(in_arr.dim(), part_offset);
      let end_pos = min(in_arr.dim(), part_offset + part_sz);
      part_buf.as_mut().flatten_mut().view_mut(0, end_pos - start_pos)
        .copy(in_arr.clone().view(start_pos, end_pos), stream.conn());
    }
    stream.sync();
    self.state.barrier.wait();

    // TODO: start timing here.
    self.stopwatch.lap();

    for p in 0 .. self.num_workers - 1 {
      let src_rank = (self.worker_rank + p + 1) % self.num_workers;
      let src_part = self.state.parts[src_rank][self.worker_rank].lock().unwrap();
      let mut dst_part = self.state.parts[self.worker_rank][self.worker_rank].lock().unwrap();
      let &(_, ref src_buf, _) = &*src_part.as_ref().unwrap();
      let &mut (_, ref mut dst_buf, ref mut dst_recv_buf) = &mut *dst_part.as_mut().unwrap();
      dst_recv_buf.as_mut().copy(src_buf.as_ref(), stream.conn());
      dst_buf.as_mut().flatten_mut().add(1.0, dst_recv_buf.as_ref().flatten(), stream.conn());
    }
    stream.sync();
    self.state.barrier.wait();

    for p in 0 .. self.num_workers - 1 {
      let dst_rank = (self.worker_rank + p + 1) % self.num_workers;
      let src_part = self.state.parts[self.worker_rank][self.worker_rank].lock().unwrap();
      let mut dst_part = self.state.parts[dst_rank][self.worker_rank].lock().unwrap();
      let &(_, ref src_buf, _) = &*src_part.as_ref().unwrap();
      let &mut (_, ref mut dst_buf, _) = &mut *dst_part.as_mut().unwrap();
      dst_buf.as_mut().copy(src_buf.as_ref(), stream.conn());
    }
    stream.sync();
    self.state.barrier.wait();

    // TODO: stop timing here.
    timing_info.elapsed = self.stopwatch.lap().elapsed();

    for p in 0 .. self.num_workers {
      let mut part = self.state.parts[self.worker_rank][p].lock().unwrap();
      let &(part_offset, ref part_buf, _) = &*part.as_ref().unwrap();
      let part_sz = part_buf.len();
      let start_pos = min(in_arr.dim(), part_offset);
      let end_pos = min(in_arr.dim(), part_offset + part_sz);
      out_buf.as_mut().slice_mut(start_pos, end_pos)
        .copy(part_buf.as_ref().slice(0, end_pos - start_pos), stream.conn());
    }

    timing_info
  }
}

#[derive(Clone)]
pub struct GPURingMoments2AllreduceState<T> where T: Copy {
  barrier:  Arc<SpinBarrier>,
  buf_sz:   Arc<AtomicUsize>,
  src_pts:  Vec<Vec<Arc<Mutex<Option<(usize, DeviceMem<T>, DeviceMem<T>)>>>>>,
  mean_pts: Vec<Vec<Arc<Mutex<Option<(usize, DeviceMem<T>, DeviceMem<T>)>>>>>,
  var_pts:  Vec<Vec<Arc<Mutex<Option<(usize, DeviceMem<T>, DeviceMem<T>)>>>>>,
}

#[derive(Clone)]
pub struct GPURingMoments2AllreduceBuilder<T> where T: Copy {
  num_workers:  usize,
  state:    GPURingMoments2AllreduceState<T>,
}

impl<T> GPURingMoments2AllreduceBuilder<T> where T: Copy {
  pub fn new(num_workers: usize) -> Self {
    assert!(num_workers >= 1);
    let mut src_pts = Vec::with_capacity(num_workers);
    let mut mean_pts = Vec::with_capacity(num_workers);
    let mut var_pts = Vec::with_capacity(num_workers);
    for _ in 0 .. num_workers {
      let mut rank_parts = Vec::with_capacity(num_workers);
      for _ in 0 .. num_workers {
        rank_parts.push(Arc::new(Mutex::new(None)));
      }
      src_pts.push(rank_parts);
      let mut rank_parts = Vec::with_capacity(num_workers);
      for _ in 0 .. num_workers {
        rank_parts.push(Arc::new(Mutex::new(None)));
      }
      mean_pts.push(rank_parts);
      let mut rank_parts = Vec::with_capacity(num_workers);
      for _ in 0 .. num_workers {
        rank_parts.push(Arc::new(Mutex::new(None)));
      }
      var_pts.push(rank_parts);
    }
    GPURingMoments2AllreduceBuilder{
      num_workers:  num_workers,
      state:        GPURingMoments2AllreduceState{
        barrier:    Arc::new(SpinBarrier::new(num_workers)),
        buf_sz:     Arc::new(AtomicUsize::new(0)),
        src_pts:    src_pts,
        mean_pts:   mean_pts,
        var_pts:    var_pts,
      },
    }
  }
}

impl<T> GPURingMoments2AllreduceBuilder<T> where T: Copy + ZeroBits {
  pub fn into_worker(self, worker_rank: usize, buf_sz: usize, stream: &DeviceStream) -> GPURingMoments2AllreduceWorker<T> {
    assert!(worker_rank < self.num_workers);
    let prev_buf_sz = self.state.buf_sz.compare_and_swap(0, buf_sz, Ordering::SeqCst);
    assert!(prev_buf_sz == 0 || prev_buf_sz == buf_sz);
    let max_part_sz = (buf_sz + self.num_workers - 1) / self.num_workers;
    let mut part_offset = 0;
    for p in 0 .. self.num_workers {
      let part_sz = min(max_part_sz, buf_sz - p * max_part_sz);
      let part_buf = DeviceMem::zeros(part_sz, stream.conn());
      let part_recv_buf = DeviceMem::zeros(part_sz, stream.conn());
      let mut src_part = self.state.src_pts[worker_rank][p].lock().unwrap();
      *src_part = Some((part_offset, part_buf, part_recv_buf));
      let part_sz = min(max_part_sz, buf_sz - p * max_part_sz);
      let part_buf = DeviceMem::zeros(part_sz, stream.conn());
      let part_recv_buf = DeviceMem::zeros(part_sz, stream.conn());
      let mut mean_part = self.state.mean_pts[worker_rank][p].lock().unwrap();
      *mean_part = Some((part_offset, part_buf, part_recv_buf));
      let part_buf = DeviceMem::zeros(part_sz, stream.conn());
      let part_recv_buf = DeviceMem::zeros(part_sz, stream.conn());
      let mut var_part = self.state.var_pts[worker_rank][p].lock().unwrap();
      *var_part = Some((part_offset, part_buf, part_recv_buf));
      part_offset += part_sz;
    }
    stream.sync();
    self.state.barrier.wait();
    GPURingMoments2AllreduceWorker{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      buf_sz:       buf_sz,
      state:        self.state,
      stopwatch:    Stopwatch::new(),
    }
  }
}

pub struct GPURingMoments2AllreduceWorker<T> where T: Copy {
  worker_rank:  usize,
  num_workers:  usize,
  buf_sz:       usize,
  state:        GPURingMoments2AllreduceState<T>,
  stopwatch:    Stopwatch,
}

impl<T> GPURingMoments2AllreduceWorker<T> where T: Copy {
  pub fn buffer_size(&self) -> usize {
    self.buf_sz
  }
}

impl GPURingMoments2AllreduceWorker<f32> {
  pub fn allreduce_moments2<'a, A>(&mut self, in_buf: A, out_mean: &mut DeviceMem<f32>, out_var: &mut DeviceMem<f32>, stream: &DeviceStream) -> TimingInfo where A: FlatView<'a, DeviceArray1dView<'a, f32>> {
    let in_arr = in_buf.flatten();
    assert!(in_arr.dim() <= self.buf_sz);
    assert_eq!(in_arr.dim(), out_mean.len());
    assert_eq!(in_arr.dim(), out_var.len());

    let mut timing_info = TimingInfo::default();

    for p in 0 .. self.num_workers {
      let mut src_part = self.state.src_pts[self.worker_rank][p].lock().unwrap();
      let &mut (part_offset, ref mut part_buf, _) = &mut *src_part.as_mut().unwrap();
      let part_sz = part_buf.len();
      let start_pos = min(in_arr.dim(), part_offset);
      let end_pos = min(in_arr.dim(), part_offset + part_sz);
      part_buf.as_mut().flatten_mut().view_mut(0, end_pos - start_pos)
        .copy(in_arr.clone().view(start_pos, end_pos), stream.conn());

      let mut mean_part = self.state.mean_pts[self.worker_rank][p].lock().unwrap();
      let &mut (_, ref mut mean_part_buf, _) = &mut *mean_part.as_mut().unwrap();
      let mut var_part = self.state.var_pts[self.worker_rank][p].lock().unwrap();
      let &mut (_, ref mut var_part_buf, _) = &mut *var_part.as_mut().unwrap();
      // TODO: do prereduce or the initial increduce here.
      unsafe { devicemem_cuda_kernel_elem_increduce_moments2_f32(
          part_sz,
          0,
          part_buf.as_ref().as_ptr(),
          mean_part_buf.as_mut().as_mut_ptr(),
          var_part_buf.as_mut().as_mut_ptr(),
          stream.conn().raw_stream().as_ptr(),
      ) };
    }
    stream.sync();
    self.state.barrier.wait();

    self.stopwatch.lap();

    for p in 0 .. self.num_workers - 1 {
      let src_rank = (self.worker_rank + p + 1) % self.num_workers;

      let src_part = self.state.src_pts[src_rank][self.worker_rank].lock().unwrap();
      let mut dst_part = self.state.src_pts[self.worker_rank][self.worker_rank].lock().unwrap();
      let &(part_offset, ref src_buf, _) = &*src_part.as_ref().unwrap();
      let part_sz = src_buf.len();
      let &mut (_, _, ref mut dst_recv_buf) = &mut *dst_part.as_mut().unwrap();
      dst_recv_buf.as_mut().copy(src_buf.as_ref(), stream.conn());

      let mut dst_mean_part = self.state.mean_pts[self.worker_rank][self.worker_rank].lock().unwrap();
      let &mut (_, ref mut dst_mean_buf, _) = &mut *dst_mean_part.as_mut().unwrap();

      let mut dst_var_part = self.state.var_pts[self.worker_rank][self.worker_rank].lock().unwrap();
      let &mut (_, ref mut dst_var_buf, _) = &mut *dst_var_part.as_mut().unwrap();

      // TODO: do increduce here.
      unsafe { devicemem_cuda_kernel_elem_increduce_moments2_f32(
          part_sz,
          p + 1,
          dst_recv_buf.as_ref().as_ptr(),
          dst_mean_buf.as_mut().as_mut_ptr(),
          dst_var_buf.as_mut().as_mut_ptr(),
          stream.conn().raw_stream().as_ptr(),
      ) };

      // TODO: do postreduce here.
      if self.num_workers > 1 && p == self.num_workers - 2 {
        unsafe { devicemem_cuda_kernel_elem_postreduce_var_f32(
            part_sz,
            self.num_workers,
            dst_var_buf.as_mut().as_mut_ptr(),
            stream.conn().raw_stream().as_ptr(),
        ) };
      }
    }
    if self.num_workers > 1 {
      stream.sync();
      self.state.barrier.wait();
    }

    for p in 0 .. self.num_workers - 1 {
      let dst_rank = (self.worker_rank + p + 1) % self.num_workers;
      {
        let src_part = self.state.mean_pts[self.worker_rank][self.worker_rank].lock().unwrap();
        let mut dst_part = self.state.mean_pts[dst_rank][self.worker_rank].lock().unwrap();
        let &(_, ref src_buf, _) = &*src_part.as_ref().unwrap();
        let &mut (_, ref mut dst_buf, _) = &mut *dst_part.as_mut().unwrap();
        dst_buf.as_mut().copy(src_buf.as_ref(), stream.conn());
      }
      {
        let src_part = self.state.var_pts[self.worker_rank][self.worker_rank].lock().unwrap();
        let mut dst_part = self.state.var_pts[dst_rank][self.worker_rank].lock().unwrap();
        let &(_, ref src_buf, _) = &*src_part.as_ref().unwrap();
        let &mut (_, ref mut dst_buf, _) = &mut *dst_part.as_mut().unwrap();
        dst_buf.as_mut().copy(src_buf.as_ref(), stream.conn());
      }
    }
    if self.num_workers > 1 {
      stream.sync();
      self.state.barrier.wait();
    }

    timing_info.elapsed = self.stopwatch.lap().elapsed();

    for p in 0 .. self.num_workers {
      let mut mean_part = self.state.mean_pts[self.worker_rank][p].lock().unwrap();
      let mut var_part = self.state.var_pts[self.worker_rank][p].lock().unwrap();
      let &(part_offset, ref mean_part_buf, _) = &*mean_part.as_ref().unwrap();
      let &(_, ref var_part_buf, _) = &*var_part.as_ref().unwrap();
      let part_sz = mean_part_buf.len();
      let start_pos = min(in_arr.dim(), part_offset);
      let end_pos = min(in_arr.dim(), part_offset + part_sz);
      out_mean.as_mut().slice_mut(start_pos, end_pos)
        .copy(mean_part_buf.as_ref().slice(0, end_pos - start_pos), stream.conn());
      out_var.as_mut().slice_mut(start_pos, end_pos)
        .copy(var_part_buf.as_ref().slice(0, end_pos - start_pos), stream.conn());
    }

    timing_info
  }
}

#[derive(Clone)]
pub struct GPUPeerRingAllreduceState<T> where T: Copy {
  barrier:  Arc<SpinBarrier>,
  buf_sz:   Arc<AtomicUsize>,
  parts:    Vec<Vec<Arc<Mutex<Option<(usize, DeviceMem<T>)>>>>>,
}

#[derive(Clone)]
pub struct GPUPeerRingAllreduceBuilder<T> where T: Copy {
  num_workers:  usize,
  state:    GPUPeerRingAllreduceState<T>,
}

impl<T> GPUPeerRingAllreduceBuilder<T> where T: Copy {
  pub fn new(num_workers: usize) -> Self {
    assert!(num_workers >= 1);
    let mut parts = Vec::with_capacity(num_workers);
    for _ in 0 .. num_workers {
      let mut rank_parts = Vec::with_capacity(num_workers);
      for _ in 0 .. num_workers {
        rank_parts.push(Arc::new(Mutex::new(None)));
      }
      parts.push(rank_parts);
    }
    GPUPeerRingAllreduceBuilder{
      num_workers:  num_workers,
      state:        GPUPeerRingAllreduceState{
        barrier:    Arc::new(SpinBarrier::new(num_workers)),
        buf_sz:     Arc::new(AtomicUsize::new(0)),
        parts:      parts,
      },
    }
  }
}

impl<T> GPUPeerRingAllreduceBuilder<T> where T: Copy + ZeroBits {
  pub fn into_worker(self, worker_rank: usize, buf_sz: usize, stream: &DeviceStream) -> GPUPeerRingAllreduceWorker<T> {
    assert!(worker_rank < self.num_workers);
    let prev_buf_sz = self.state.buf_sz.compare_and_swap(0, buf_sz, Ordering::SeqCst);
    assert!(prev_buf_sz == 0 || prev_buf_sz == buf_sz);
    let max_part_sz = (buf_sz + self.num_workers - 1) / self.num_workers;
    let mut part_offset = 0;
    for p in 0 .. self.num_workers {
      let part_sz = min(max_part_sz, buf_sz - p * max_part_sz);
      let part_buf = DeviceMem::zeros(part_sz, stream.conn());
      let mut part = self.state.parts[worker_rank][p].lock().unwrap();
      *part = Some((part_offset, part_buf));
      part_offset += part_sz;
    }
    stream.sync();
    self.state.barrier.wait();
    GPUPeerRingAllreduceWorker{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      buf_sz:       buf_sz,
      state:        self.state,
      //_marker:      PhantomData,
    }
  }
}

pub struct GPUPeerRingAllreduceWorker<T> where T: Copy {
  worker_rank:  usize,
  num_workers:  usize,
  buf_sz:       usize,
  state:    GPUPeerRingAllreduceState<T>,
}

impl GPUPeerRingAllreduceWorker<f32> {
  pub fn allreduce_sum(&self, in_buf: &DeviceMem<f32>, out_buf: &mut DeviceMem<f32>, stream: &DeviceStream) {
    assert_eq!(self.buf_sz, in_buf.len());
    assert_eq!(self.buf_sz, out_buf.len());

    if self.num_workers == 1 {
      out_buf.as_mut().copy(in_buf.as_ref(), stream.conn());
      return;
    }

    for p in 0 .. self.num_workers {
      let mut part = self.state.parts[self.worker_rank][p].lock().unwrap();
      assert!(part.is_some());
      let &mut (part_offset, ref mut part_buf) = &mut *part.as_mut().unwrap();
      let part_sz = part_buf.len();
      part_buf.as_mut().copy(in_buf.as_ref().slice(part_offset, part_offset + part_sz), stream.conn());
    }
    stream.sync();
    self.state.barrier.wait();

    for p in 0 .. self.num_workers - 1 {
      let dst_rank = (self.worker_rank + p + 1) % self.num_workers;
      let src_part = self.state.parts[self.worker_rank][dst_rank].lock().unwrap();
      let mut dst_part = self.state.parts[dst_rank][dst_rank].lock().unwrap();
      let &(_, ref src_buf) = &*src_part.as_ref().unwrap();
      let &mut (_, ref mut dst_buf) = &mut *dst_part.as_mut().unwrap();
      dst_buf.as_mut().flatten_mut().add(1.0, src_buf.as_ref().flatten(), stream.conn());
    }
    stream.sync();
    self.state.barrier.wait();

    for p in 0 .. self.num_workers - 1 {
      let dst_rank = (self.worker_rank + p + 1) % self.num_workers;
      let src_part = self.state.parts[self.worker_rank][self.worker_rank].lock().unwrap();
      let mut dst_part = self.state.parts[dst_rank][self.worker_rank].lock().unwrap();
      let &(_, ref src_buf) = &*src_part.as_ref().unwrap();
      let &mut (_, ref mut dst_buf) = &mut *dst_part.as_mut().unwrap();
      dst_buf.as_mut().copy(src_buf.as_ref(), stream.conn());
    }
    stream.sync();
    self.state.barrier.wait();

    for p in 0 .. self.num_workers {
      let mut part = self.state.parts[self.worker_rank][p].lock().unwrap();
      let &(part_offset, ref part_buf) = &*part.as_ref().unwrap();
      let part_sz = part_buf.len();
      out_buf.as_mut().slice_mut(part_offset, part_offset + part_sz).copy(part_buf.as_ref(), stream.conn());
    }
  }
}

/*#[derive(Clone)]
pub struct GPUTwoLevelRingAllreduceState<T> where T: Copy {
  //num_groups:   Arc<AtomicUsize>,
  gpu_topo: NvsmiTopology,
  buf_sz:   Arc<AtomicUsize>,
  barrier:  Arc<SpinBarrier>,
  parts:    Vec<Vec<Arc<Mutex<Option<(usize, DeviceMem<T>, DeviceMem<T>)>>>>>,
}

#[derive(Clone)]
pub struct GPUTwoLevelRingAllreduceBuilder<T> where T: Copy {
  num_workers:  usize,
  state:    GPUTwoLevelRingAllreduceState<T>,
}

impl<T> GPUTwoLevelRingAllreduceBuilder<T> where T: Copy {
  pub fn new(num_workers: usize) -> Self {
    assert!(num_workers >= 1);
    let mut parts = Vec::with_capacity(num_workers);
    for _ in 0 .. num_workers {
      let mut rank_parts = Vec::with_capacity(num_workers);
      for _ in 0 .. num_workers {
        rank_parts.push(Arc::new(Mutex::new(None)));
      }
      parts.push(rank_parts);
    }
    let gpu_topo = NvsmiTopology::query_default();
    GPUTwoLevelRingAllreduceBuilder{
      num_workers:  num_workers,
      state:        GPUTwoLevelRingAllreduceState{
        //num_groups: Arc::new(AtomicUsize::new(0)),
        gpu_topo:   gpu_topo,
        buf_sz:     Arc::new(AtomicUsize::new(0)),
        barrier:    Arc::new(SpinBarrier::new(num_workers)),
        parts:      parts,
      },
    }
  }
}

impl<T> GPUTwoLevelRingAllreduceBuilder<T> where T: Copy + ZeroBits {
  pub fn into_worker(self, worker_rank: usize, buf_sz: usize, stream: &DeviceStream) -> GPUTwoLevelRingAllreduceWorker<T> {
    assert!(worker_rank < self.num_workers);
    let prev_buf_sz = self.state.buf_sz.compare_and_swap(0, buf_sz, Ordering::SeqCst);
    assert!(prev_buf_sz == 0 || prev_buf_sz == buf_sz);
    let max_part_sz = (buf_sz + self.num_workers - 1) / self.num_workers;
    let mut part_offset = 0;
    for p in 0 .. self.num_workers {
      let part_sz = min(max_part_sz, buf_sz - p * max_part_sz);
      let part_buf = DeviceMem::zeros(part_sz, stream.conn());
      let part_recv_buf = DeviceMem::zeros(part_sz, stream.conn());
      let mut part = self.state.parts[worker_rank][p].lock().unwrap();
      *part = Some((part_offset, part_buf, part_recv_buf));
      part_offset += part_sz;
    }
    stream.sync();
    self.state.barrier.wait();
    GPUTwoLevelRingAllreduceWorker{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      buf_sz:       buf_sz,
      state:        self.state,
      //_marker:      PhantomData,
    }
  }
}

pub struct GPUTwoLevelRingAllreduceWorker<T> where T: Copy {
  worker_rank:  usize,
  num_workers:  usize,
  //num_groups:   usize,
  buf_sz:       usize,
  state:    GPUTwoLevelRingAllreduceState<T>,
}

impl GPUTwoLevelRingAllreduceWorker<f32> {
  pub fn allreduce_sum(&self, in_buf: &DeviceMem<f32>, out_buf: &mut DeviceMem<f32>, stream: &DeviceStream) {
    assert_eq!(self.buf_sz, in_buf.len());
    assert_eq!(self.buf_sz, out_buf.len());

    let num_groups = self.state.gpu_topo.num_groups();

    if self.num_workers == 1 {
      out_buf.as_mut().copy(in_buf.as_ref(), stream.conn());
      return;
    }

    for p in 0 .. self.num_workers {
      let mut part = self.state.parts[self.worker_rank][p].lock().unwrap();
      assert!(part.is_some());
      let &mut (part_offset, ref mut part_buf, ref mut part_recv_buf) = &mut *part.as_mut().unwrap();
      let part_sz = part_buf.len();
      part_buf.as_mut().copy(in_buf.as_ref().slice(part_offset, part_offset + part_sz), stream.conn());
    }
    stream.sync();
    self.state.barrier.wait();

    // Switch-level reducer.
    for p in 0 .. self.num_workers - 1 {
      // TODO: project the rank into the switch group.
      let src_rank = (self.worker_rank + p + 1) % self.num_workers;
      let src_part = self.state.parts[src_rank][self.worker_rank].lock().unwrap();
      let mut dst_part = self.state.parts[self.worker_rank][self.worker_rank].lock().unwrap();
      let &(_, ref src_buf, _) = &*src_part.as_ref().unwrap();
      let &mut (_, ref mut dst_buf, ref mut dst_recv_buf) = &mut *dst_part.as_mut().unwrap();
      dst_recv_buf.as_mut().copy(src_buf.as_ref(), stream.conn());
      dst_buf.as_mut().flatten_mut().add(1.0, dst_recv_buf.as_ref().flatten(), stream.conn());
    }
    stream.sync();
    self.state.barrier.wait();

    for p in 0 .. self.num_workers - 1 {
      let dst_rank = (self.worker_rank + p + 1) % self.num_workers;
      let src_part = self.state.parts[self.worker_rank][self.worker_rank].lock().unwrap();
      let mut dst_part = self.state.parts[dst_rank][self.worker_rank].lock().unwrap();
      let &(_, ref src_buf, _) = &*src_part.as_ref().unwrap();
      let &mut (_, ref mut dst_buf, _) = &mut *dst_part.as_mut().unwrap();
      dst_buf.as_mut().copy(src_buf.as_ref(), stream.conn());
    }
    stream.sync();
    self.state.barrier.wait();

    // TODO: Bridge-level reducer.
    if self.state.gpu_topo.switch_groups.contains_key(&self.num_workers) {
      let group_rank = 0; // TODO
      for g in 0 .. num_groups - 1 {
        for p in 0 .. self.num_workers {
          // TODO
          let src_rank = (group_rank + g + 1) % num_groups;
          let dst_rank = group_rank;
          let src_part = self.state.parts[src_rank][p].lock().unwrap();
          let mut dst_part = self.state.parts[dst_rank][p].lock().unwrap();
          let &(_, ref src_buf, _) = &*src_part.as_ref().unwrap();
          let &mut (_, ref mut dst_buf, ref mut dst_recv_buf) = &mut *dst_part.as_mut().unwrap();
          dst_recv_buf.as_mut().copy(src_buf.as_ref(), stream.conn());
          dst_buf.as_mut().flatten_mut().add(1.0, dst_recv_buf.as_ref().flatten(), stream.conn());
        }
      }
    }

    for p in 0 .. self.num_workers {
      let mut part = self.state.parts[self.worker_rank][p].lock().unwrap();
      let &(part_offset, ref part_buf, _) = &*part.as_ref().unwrap();
      let part_sz = part_buf.len();
      out_buf.as_mut().slice_mut(part_offset, part_offset + part_sz).copy(part_buf.as_ref(), stream.conn());
    }
  }
}*/
