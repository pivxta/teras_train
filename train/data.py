import ctypes
import numpy as np
import os
import torch
from dataclasses import dataclass

FEATURE_COUNT = 768

def load_data_lib():
    lib = ctypes.cdll.LoadLibrary(
        "./target/release/libdataloader.so" if os.name != "nt" else
        "./target/release/dataloader.dll"
    )
    lib.create_batch.restype = ctypes.c_void_p
    lib.batch_capacity.restype = ctypes.c_uint32
    lib.batch_size.restype = ctypes.c_uint32
    lib.batch_total_features.restype = ctypes.c_uint32
    lib.batch_stm_features.restype = ctypes.POINTER(ctypes.c_uint32)
    lib.batch_non_stm_features.restype = ctypes.POINTER(ctypes.c_uint32)
    lib.batch_evals.restype = ctypes.POINTER(ctypes.c_float)
    lib.batch_outcomes.restype = ctypes.POINTER(ctypes.c_float)
    lib.open_loader.restype = ctypes.c_void_p
    lib.load_batch.restype = ctypes.c_bool
    return lib

lib = load_data_lib()

@dataclass 
class Batch:
    size: int
    stm_features: torch.Tensor
    non_stm_features: torch.Tensor
    evals: torch.Tensor
    outcomes: torch.Tensor

class _Batch:
    def __init__(self, capacity: int):
        self._ptr = ctypes.c_void_p(lib.create_batch(ctypes.c_uint32(capacity)))

    def __del__(self):
        self.drop()

    def drop(self):
        if self._ptr.value is not None:
            lib.drop_batch(self._ptr)
            self._ptr.value = None

    def capacity(self) -> int:
        return ctypes.c_uint32(lib.batch_capacity(self._ptr)).value

    def size(self) -> int:
        return ctypes.c_uint32(lib.batch_size(self._ptr)).value

    def total_features(self) -> int:
        return ctypes.c_uint32(lib.batch_total_features(self._ptr)).value

    def stm_features(self):
        return lib.batch_stm_features(self._ptr)

    def non_stm_features(self):
        return lib.batch_non_stm_features(self._ptr)

    def evals(self):
        return lib.batch_evals(self._ptr)

    def outcomes(self):
        return lib.batch_outcomes(self._ptr)

    def to_torch(self) -> Batch:
        size = self.size()
        evals = torch.from_numpy(np.ctypeslib.as_array(self.evals(), shape=(size, 1)))
        outcomes = torch.from_numpy(np.ctypeslib.as_array(self.outcomes(), shape=(size, 1)))
        
        active_features = self.total_features()
        stm_indices = torch.transpose(
            torch.from_numpy(
                np.ctypeslib.as_array(self.stm_features(), shape=(active_features, 2))
            ), 0, 1
        ).long()
        non_stm_indices = torch.transpose(
            torch.from_numpy(
                np.ctypeslib.as_array(self.non_stm_features(), shape=(active_features, 2))
            ), 0, 1
        ).long()

        stm_values = torch.ones(active_features)
        non_stm_values = torch.ones(active_features)

        stm_features = torch.sparse_coo_tensor(stm_indices, stm_values, (size, FEATURE_COUNT))
        non_stm_features = torch.sparse_coo_tensor(non_stm_indices, non_stm_values, (size, FEATURE_COUNT))

        return Batch(
            size=size,
            evals=evals,
            outcomes=outcomes,
            stm_features=stm_features,
            non_stm_features=non_stm_features,
        )

class _BatchLoader:
    def __init__(self, path: str):
        self._ptr = ctypes.c_void_p(lib.open_loader(ctypes.create_string_buffer(bytes(path, "ascii"))))
        if self._ptr.value is None:
            raise Exception(f"failed to load data from file '{path}'")

    def close(self):
        if self._ptr.value is not None:
            lib.close_loader(self._ptr)
            self._ptr.value = None

    def load(self, batch: _Batch) -> bool:
        return ctypes.c_bool(lib.load_batch(self._ptr, batch._ptr)).value

class DataLoader:
    def __init__(self, path: str, batch_size: int):
        self._path = path
        self._loader = _BatchLoader(path)
        self._batch = _Batch(batch_size)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        same_epoch, batch = self.load_batch()
        if not same_epoch:
            raise StopIteration
        return batch

    def load_batch(self) -> tuple[bool, Batch]:
        if not self._loader.load(self._batch):
            self._loader.close()
            self._loader = _BatchLoader(self._path)
            return False, self._batch.to_torch()

        return True, self._batch.to_torch()

    def close(self):
        self._loader.close()
        self._batch.drop()
