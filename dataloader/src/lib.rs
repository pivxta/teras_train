use batch::Batch;
use core::ptr;
use loader::BatchLoader;
use std::{
    ffi::{CStr, c_char},
    fs::File,
};

pub mod batch;
pub mod feature;
pub mod loader;

#[unsafe(no_mangle)]
unsafe extern "C" fn open_loader(path: *const c_char, batch_size: u32) -> *mut BatchLoader {
    let path = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(path) => path,
        Err(_) => return ptr::null_mut(),
    };
    let file = match File::open(path) {
        Ok(file) => file,
        Err(_) => return ptr::null_mut(),
    };
    Box::into_raw(Box::new(BatchLoader::from_file(file, batch_size as usize)))
}

#[unsafe(no_mangle)]
unsafe extern "C" fn close_loader(loader: *mut BatchLoader) {
    drop(unsafe { Box::from_raw(loader) })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn load_batch(loader: *mut BatchLoader) -> *mut Batch {
    unsafe { Box::into_raw(Box::new(loader.as_mut().unwrap().load())) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn drop_batch(batch: *mut Batch) {
    drop(unsafe { Box::from_raw(batch) })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn batch_capacity(batch: *const Batch) -> u32 {
    unsafe { batch.as_ref().unwrap().capacity as u32 }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn batch_size(batch: *const Batch) -> u32 {
    unsafe { batch.as_ref().unwrap().entries as u32 }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn batch_total_features(batch: *const Batch) -> u32 {
    unsafe { batch.as_ref().unwrap().total_features as u32 }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn batch_stm_features(batch: *const Batch) -> *const u32 {
    unsafe { batch.as_ref().unwrap().stm_features.as_ptr() }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn batch_non_stm_features(batch: *const Batch) -> *const u32 {
    unsafe { batch.as_ref().unwrap().non_stm_features.as_ptr() }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn batch_evals(batch: *const Batch) -> *const f32 {
    unsafe { batch.as_ref().unwrap().eval_centipawns.as_ptr() }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn batch_outcomes(batch: *const Batch) -> *const f32 {
    unsafe { batch.as_ref().unwrap().outcomes.as_ptr() }
}

