use dataformat::PackedSample;
use rand::seq::SliceRandom;
use std::{
    fs::File,
    io::{BufReader, Read},
    mem,
};

use crate::batch::Batch;

pub const BUFFER_SAMPLES: usize = 2097152;

#[derive(Debug)]
pub struct BatchLoader {
    reader: BufReader<File>,
    buffer: Vec<PackedSample>,
}

impl BatchLoader {
    pub fn from_file(file: File) -> Self {
        Self {
            reader: BufReader::new(file),
            buffer: vec![],
        }
    }

    pub fn load(&mut self, batch: &mut Batch) -> bool {
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
        batch.entries > 0
    }

    fn next(&mut self) -> Option<PackedSample> {
        if self.buffer.is_empty() && !self.fill_buffer() {
            return None;
        }
        self.buffer.pop()
    }

    fn fill_buffer(&mut self) -> bool {
        self.buffer.clear();
        for _ in 0..BUFFER_SAMPLES {
            let mut buffer = [0; mem::size_of::<PackedSample>()];
            if self.reader.read_exact(&mut buffer).is_ok() {
                self.buffer.push(*bytemuck::from_bytes(&buffer));
            } else {
                break;
            }
        }
        self.buffer.shuffle(&mut rand::rng());
        !self.buffer.is_empty()
    }
}
