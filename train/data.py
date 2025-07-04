import ctypes
import numpy as np
import os
import torch
from feature import FEATURE_COUNT
from dataclasses import dataclass

@dataclass 
class Batch:
    size: int
    stm_features: torch.Tensor
    non_stm_features: torch.Tensor
    evals: torch.Tensor
    outcomes: torch.Tensor

def load_data_lib():
    lib = ctypes.cdll.LoadLibrary(
        "./target/release/libdataloader.so" if os.name != "nt" else
        "./target/release/dataloader.dll"
    )
    lib.load_batch.restype = ctypes.c_void_p
    lib.batch_capacity.restype = ctypes.c_uint32
    lib.batch_size.restype = ctypes.c_uint32
    lib.batch_total_features.restype = ctypes.c_uint32
    lib.batch_stm_features.restype = ctypes.POINTER(ctypes.c_uint32)
    lib.batch_non_stm_features.restype = ctypes.POINTER(ctypes.c_uint32)
    lib.batch_evals.restype = ctypes.POINTER(ctypes.c_float)
    lib.batch_outcomes.restype = ctypes.POINTER(ctypes.c_float)
    lib.open_loader.restype = ctypes.c_void_p
    return lib

lib = load_data_lib()

class _Batch:
    def __init__(self, ptr):
        self._ptr = ptr

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
    def __init__(self, path: str, batch_size: int):
        self._ptr = ctypes.c_void_p(lib.open_loader(
            ctypes.create_string_buffer(bytes(path, "ascii")), 
            ctypes.c_uint32(batch_size)
            ))
        if self._ptr.value is None:
            raise Exception(f"failed to load data from file '{path}'")

    def close(self):
        if self._ptr.value is not None:
            lib.close_loader(self._ptr)
            self._ptr.value = None

    def load(self) -> _Batch:
        return _Batch(ctypes.c_void_p(lib.load_batch(self._ptr)))

class NnueDataset(torch.utils.data.IterableDataset):
    def __init__(self, path: str, batch_size: int, epoch_size: int):
        self._last_batch = None
        self._loader = _BatchLoader(path, batch_size)
        self.batches = (epoch_size + batch_size - 1) // batch_size

    def __del__(self):
        self._loader.close()

    def __len__(self):
        return self.batches

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        if self._last_batch is not None:
            self._last_batch.drop()
        self._last_batch = self._loader.load()
        tensor_batch = self._last_batch.to_torch()
        return tensor_batch
