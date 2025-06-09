use dataformat::PackedSample;
use rand::seq::SliceRandom;
use std::{
    fs::File, io::{self, Read, Seek}, mem, sync::mpsc, thread::{self, JoinHandle}
};

use crate::batch::Batch;

pub const BUFFER_SIZE: usize = 4194304;

#[derive(Debug)]
pub struct BatchLoader {
    batch_receiver: mpsc::Receiver<Batch>,
    _worker: JoinHandle<()>,
}

impl BatchLoader {
    pub fn from_file(file: File, batch_size: usize) -> Self {
        let (batch_sender, batch_receiver) = mpsc::sync_channel(32);
        Self {
            batch_receiver,
            _worker: thread::spawn(move || loader_thread(file, batch_size, batch_sender))
        }
    }

    pub fn load(&mut self) -> Batch {
        self.batch_receiver.recv().expect("batch loading thread has disconnected")
    }
}

fn loader_thread(file: File, batch_size: usize, batch_sender: mpsc::SyncSender<Batch>) {
    let mut batch_loader = BufferedLoader::from_file(file);
    loop {
        let mut batch = Batch::new(batch_size);
        batch_loader.load_into(&mut batch);
        if batch_sender.send(batch).is_err() {
            return;
        }
    }
}

#[derive(Debug)]
struct BufferedLoader {
    file: File,
    buffer: Vec<PackedSample>,
}

impl BufferedLoader {
    pub fn from_file(file: File) -> Self {
        Self {
            file,
            buffer: Vec::with_capacity(BUFFER_SIZE),
        }
    }

    pub fn load_into(&mut self, batch: &mut Batch) {
        batch.clear();
        for _ in 0..batch.capacity {
            if let Some(sample) = self.next() {
                let sample = match sample.unpack() {
                    Ok(sample) => sample,
                    Err(err) => {
                        eprintln!("error: failed to unpack sample: {}", err);
                        continue;
                    }
                };
                batch.add(&sample);
            } else {
                break;
            }
        }
    }

    fn next(&mut self) -> Option<PackedSample> {
        if self.buffer.is_empty() {
            self.fill_buffer().expect("failed to read from dataset file");
        }
        self.buffer.pop()
    }

    fn fill_buffer(&mut self) -> io::Result<()> {
        unsafe { self.buffer.set_len(BUFFER_SIZE) };
        let mut buf_size = self.file.read(bytemuck::cast_slice_mut(&mut self.buffer))?;
        if buf_size == 0 {
            self.file.rewind()?;
            buf_size = self.file.read(bytemuck::cast_slice_mut(&mut self.buffer))?;
        }
        self.buffer.resize(buf_size / mem::size_of::<PackedSample>(), Default::default());
        self.buffer.shuffle(&mut rand::rng());
        Ok(())
    }
}

/*
pub const BUFFER_SIZE: usize = 4194304;

#[derive(Debug)]
pub struct BatchLoader {
    file: File,
    buffer: Vec<PackedSample>,
}

impl BatchLoader {
    pub fn from_file(file: File) -> Self {
        Self {
            file,
            buffer: Vec::with_capacity(BUFFER_SIZE),
        }
    }

    pub fn load(&mut self, batch: &mut Batch) {
        batch.clear();
        for _ in 0..batch.capacity {
            if let Some(sample) = self.next() {
                let sample = match sample.unpack() {
                    Ok(sample) => sample,
                    Err(err) => {
                        eprintln!("error: failed to unpack sample: {}", err);
                        continue;
                    }
                };
                batch.add(&sample);
            } else {
                break;
            }
        }
    }

    fn next(&mut self) -> Option<PackedSample> {
        if self.buffer.is_empty() {
            self.fill_buffer().expect("failed to read from dataset file");
        }
        self.buffer.pop()
    }

    fn fill_buffer(&mut self) -> io::Result<()> {
        unsafe { self.buffer.set_len(BUFFER_SIZE) };
        let mut buf_size = self.file.read(bytemuck::cast_slice_mut(&mut self.buffer))?;
        if buf_size == 0 {
            self.file.rewind()?;
            buf_size = self.file.read(bytemuck::cast_slice_mut(&mut self.buffer))?;
        }
        self.buffer.resize(buf_size / mem::size_of::<PackedSample>(), Default::default());
        self.buffer.shuffle(&mut rand::rng());
        Ok(())
    }
}
*/
