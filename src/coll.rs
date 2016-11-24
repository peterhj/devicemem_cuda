use super::*;

use nccl::*;
//use sharedmem::sync::{SpinBarrier};

//use std::cmp::{min};
//use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct DeviceNcclBuilder {
  num_workers:  usize,
  comm_id:      NcclUniqueId,
}

impl DeviceNcclBuilder {
  pub fn new(num_workers: usize) -> Self {
    let comm_id = NcclUniqueId::create().unwrap();
    DeviceNcclBuilder{
      num_workers:  num_workers,
      comm_id:      comm_id,
    }
  }

  pub fn into_worker(self, worker_rank: usize) -> DeviceNcclWorker {
    let comm = NcclComm::create(worker_rank, self.num_workers, self.comm_id.clone()).unwrap();
    DeviceNcclWorker{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      comm_id:      self.comm_id,
      comm:         comm,
    }
  }
}

pub struct DeviceNcclWorker {
  worker_rank:  usize,
  num_workers:  usize,
  comm_id:      NcclUniqueId,
  comm:         NcclComm,
}

impl DeviceNcclWorker {
  pub fn allreduce_sum<T>(&self, mut buf: DeviceMemRefMut<T>, conn: DeviceConn) where T: NcclDataType + Copy {
    let res = unsafe { self.comm.allreduce(buf.as_ptr(), buf.as_mut_ptr(), buf.len(), NcclSumOp, conn.raw_stream().ptr) };
    assert!(res.is_ok());
  }
}

/*#[derive(Clone)]
pub struct DeviceBufRingAllreduceBuilder<T> where T: Copy {
  num_workers:  usize,
  buf_sz:       usize,
  part_sizes:   Vec<usize>,
  part_offsets: Vec<usize>,
  barrier:      Arc<SpinBarrier>,
  work_bufs:    Arc<Mutex<Vec<Option<Arc<SharedDeviceMem<T>>>>>>,
  tmp_bufs:     Arc<Mutex<Vec<Option<Arc<SharedDeviceMem<T>>>>>>,
}

impl<T> DeviceBufRingAllreduceBuilder<T> where T: ZeroBits {
  pub fn new(num_workers: usize, buf_sz: usize) -> Self {
    let mut part_sizes = Vec::with_capacity(num_workers);
    let mut part_offsets = Vec::with_capacity(num_workers);
    // FIXME(20161122): split the buffer into more even parts.
    let part_sz = (buf_sz + num_workers - 1) / num_workers;
    let mut offset = 0;
    for r in 0 .. num_workers {
      let actual_part_sz = min(buf_sz, (r+1) * part_sz) - r * part_sz;
      part_sizes.push(actual_part_sz);
      part_offsets.push(offset);
      offset += actual_part_sz;
    }
    let barrier = Arc::new(SpinBarrier::new(num_workers));
    let mut work_bufs = Vec::with_capacity(num_workers);
    let mut tmp_bufs = Vec::with_capacity(num_workers);
    for _ in 0 .. num_workers {
      work_bufs.push(None);
      tmp_bufs.push(None);
    }
    DeviceBufRingAllreduceBuilder{
      num_workers:  num_workers,
      buf_sz:       buf_sz,
      part_sizes:   part_sizes,
      part_offsets: part_offsets,
      barrier:      barrier,
      work_bufs:    Arc::new(Mutex::new(work_bufs)),
      tmp_bufs:     Arc::new(Mutex::new(tmp_bufs)),
    }
  }

  pub fn into_worker(self, worker_rank: usize, stream: DeviceStream) -> DeviceBufRingAllreduceWorker<T> {
    {
      let mut work_bufs = self.work_bufs.lock().unwrap();
      let mut tmp_bufs = self.work_bufs.lock().unwrap();
      work_bufs[worker_rank] = Some(Arc::new(SharedDeviceMem::zeros(self.buf_sz, stream.conn())));
      tmp_bufs[worker_rank] = Some(Arc::new(SharedDeviceMem::zeros(self.buf_sz, stream.conn())));
    }
    self.barrier.wait();
    let mut local_work_bufs = Vec::with_capacity(self.num_workers);
    let mut local_tmp_bufs = Vec::with_capacity(self.num_workers);
    {
      let work_bufs = self.work_bufs.lock().unwrap();
      let tmp_bufs = self.tmp_bufs.lock().unwrap();
      for r in 0 .. self.num_workers {
        local_work_bufs.push(work_bufs[r].as_ref().unwrap().clone());
        local_tmp_bufs.push(tmp_bufs[r].as_ref().unwrap().clone());
      }
    }
    DeviceBufRingAllreduceWorker{
      num_workers:  self.num_workers,
      buf_sz:       self.buf_sz,
      part_sizes:   self.part_sizes,
      part_offsets: self.part_offsets,
      barrier:      self.barrier,
      worker_rank:  worker_rank,
      work_bufs:    local_work_bufs,
      tmp_bufs:     local_tmp_bufs,
    }
  }
}

pub struct DeviceBufRingAllreduceWorker<T> where T: Copy {
  num_workers:  usize,
  buf_sz:       usize,
  part_sizes:   Vec<usize>,
  part_offsets: Vec<usize>,
  barrier:      Arc<SpinBarrier>,
  worker_rank:  usize,
  work_bufs:    Vec<Arc<SharedDeviceMem<T>>>,
  tmp_bufs:     Vec<Arc<SharedDeviceMem<T>>>,
}

impl DeviceBufRingAllreduceWorker<f32> {
  pub fn allreduce(&self, mut buf: DeviceMemRefMut<f32>, stream: DeviceStream) {
    let rank = self.worker_rank;

    self.work_bufs[rank].as_mut().copy(buf.clone().as_ref(), stream.conn());

    self.work_bufs[rank].as_ref().wait(&stream.conn());
    stream.conn().sync();
    self.barrier.wait();

    for r in 0 .. self.num_workers - 1 {
      let part_rank = (self.num_workers + rank - r - 2) % self.num_workers;
      let src_rank = (self.num_workers + rank - 1) % self.num_workers;
      let part_size = self.part_sizes[part_rank];
      let part_offset = self.part_offsets[part_rank];
      self.tmp_bufs[rank].as_mut().slice_mut(part_offset, part_offset + part_size)
        .copy((*self.work_bufs[src_rank]).as_ref().slice(part_offset, part_offset + part_size), stream.conn());
      self.work_bufs[rank].as_mut().slice_mut(part_offset, part_offset + part_size)
        .add((*self.tmp_bufs[rank]).as_ref().slice(part_offset, part_offset + part_size), stream.conn());
      self.work_bufs[rank].as_ref().wait(&stream.conn());
      stream.conn().sync();
      self.barrier.wait();
    }

    let part_size = self.part_sizes[rank];
    let part_offset = self.part_offsets[rank];
    for r in 0 .. self.num_workers - 1 {
      let dst_rank = (rank + r + 1) % self.num_workers;
      self.work_bufs[dst_rank].as_ref_unsync().slice(part_offset, part_offset + part_size)
        .copy_unsync(self.work_bufs[rank].as_ref_unsync().slice(part_offset, part_offset + part_size), stream.conn());
    }
    self.work_bufs[rank].as_ref().post(&stream.conn());

    buf.as_mut().copy(self.work_bufs[rank].as_ref(), stream.conn());
  }
}*/
