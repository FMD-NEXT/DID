# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import os
import random
import struct
import threading
import numpy as np

import torch
from torch.utils.data import IterableDataset, get_worker_info

dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def code(dtype):
    for k in dtypes:
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    def __init__(
        self, filenames, n_chunks, block_size, seed=12345, shuffle=False, wrap=False, num_processes=1, process_rank=0, rank_split=True
    ):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank
        self.rank_split = rank_split

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        num_files_per_shard = len(self._filenames) // num_shards
        max_num_files = num_files_per_shard * num_shards
        # NOTE: use all filenames only for eval or debug, times 4 as num_workers = 4
        filenames = self._filenames if num_files_per_shard < self._n_chunks * 4 or not self.rank_split else self._filenames[shard_id:max_num_files:num_shards]

        return PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
        )


class PackedDatasetBuilder(object):
    def __init__(self, outdir, prefix, chunk_size, sep_token, dtype="auto", vocab_size=None):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames = []

    def _write_chunk(self):
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)

        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        self._filenames.append(filename)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr):
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_reminder(self):
        self._write_chunk()


class PackedDatasetIterator:
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None

        self._wrap = wrap

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None

        self._mmaps = []
        self._buffers = []

        self._block_idxs = []
        self._curr_idx = 0

        self._empty_files = []
        assert self._n_chunks <= len(self._filenames), f"Chunk size {self._n_chunks} should lower than Num files {len(self._filenames)}"
        self._load_n_chunks()

    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _iter_filenames(self):
        filename = self._filenames[self._file_idx]
        self._file_idx += 1
        if self._file_idx >= len(self._filenames):
            self._file_idx = 0
        return filename

    def _load_n_chunks(self):
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        loaded_files = 0
        while loaded_files < self._n_chunks:
            filename = self._iter_filenames()
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(filename)
                self._n_blocks = self._chunk_size // self._block_size
            # TODO: check header matches with previous files
            try:
                mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            except:
                self._empty_files.append(filename)
                self._empty_files = list(set(self._empty_files))
                print(f"{'#' * 20}")
                print(self._empty_files)
                print(f"{'#' * 20}")
                continue
            # print(f"read filename: {filename}")
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))
            loaded_files += 1

        n_all_blocks = self._n_chunks * self._n_blocks
        # self._block_idxs = self._rng.permutation(n_all_blocks) if self._shuffle else range(n_all_blocks)
        
        self._block_idxs = range(n_all_blocks)
        # print(f"check block idxs: {self._block_idxs}")
        self._curr_idx = 0

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
        self._curr_idx += 1
        return torch.from_numpy(arr.astype(np.int64))


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)

#########################################################

class NewPackedDatasetIterator:
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None

        self._wrap = wrap
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks
        self._block_size = block_size
        self._n_blocks = None
        self._dtype = None

        self._curr_mmaps = []
        self._next_mmaps = []
        self._curr_buffers = []
        self._next_buffers = []
        self._curr_block_idxs = []
        self._next_block_idxs = []
        self._curr_idx = 0

        self._preload_thread = None
        self._load_state()
        self._start_preload_next()
    
    def __iter__(self):
        return self
    
    def _close_mmaps(self, close_next=False):
        if close_next:
            for mmap in self._next_mmaps:
                mmap._mmap.close()
        else:
            for mmap in self._curr_mmaps:
                mmap._mmap.close()
    
    def _clear_data(self, clear_next=False):
        self._close_mmaps(clear_next)
        if clear_next:
            del self._next_mmaps
            del self._next_buffers
            del self._next_block_idxs
            self._next_mmaps = []
            self._next_buffers = []
            self._next_block_idxs = []
        else:
            del self._curr_mmaps
            del self._curr_buffers
            del self._curr_block_idxs
            self._curr_mmaps = []
            self._curr_buffers = []
            self._curr_block_idxs = []
    
    def __del__(self):
        self._clear_data()
        self._clear_data(True)

    def _read_header(self, path):
        try:
            with open(path, "rb") as f:
                magic = f.read(len(HDR_MAGIC))
                assert magic == HDR_MAGIC, "File doesn't match expected format."
                version = struct.unpack("<Q", f.read(8))
                assert version == (1,)
                (dtype_code,) = struct.unpack("<B", f.read(1))
                dtype = dtypes[dtype_code]
                (chunk_size,) = struct.unpack("<Q", f.read(8))
            return dtype, chunk_size
        except:
            return None, None
    
    def _iter_filenames(self):
        filename = self._filenames[self._file_idx]
        self._file_idx += 1
        if self._file_idx >= len(self._filenames):
            self._file_idx = 0
        return filename

    def _load_state(self, load_next=False):
        self._clear_data(load_next)
        _mmaps = []
        _buffers = []
        loaded_files = 0

        while loaded_files < self._n_chunks:
            filename = self._iter_filenames()
            _dtype, _chunk_size = self._read_header(filename)
            if _dtype is None:
                continue
            _n_blocks = _chunk_size // self._block_size

            if self._dtype is None:
                self._dtype, self._chunk_size, self._n_blocks = _dtype, _chunk_size, _n_blocks
            elif self._dtype != _dtype or self._chunk_size != _chunk_size:
                print(f"{'#' * 20}\nFile header do not match with previous file\n{'#' * 20}")

            try:
                mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            except:
                continue

            _mmaps.append(mmap)
            _buffers.append(memoryview(mmap))
            loaded_files += 1        

        n_all_blocks = self._n_chunks * self._n_blocks
        _block_idxs = self._rng.permutation(n_all_blocks) if self._shuffle else range(n_all_blocks)
        if load_next:
            self._next_mmaps = _mmaps
            self._next_buffers = _buffers
            self._next_block_idxs = _block_idxs
        else:
            self._curr_mmaps = _mmaps
            self._curr_buffers = _buffers
            self._curr_block_idxs = _block_idxs

    def _start_preload_next(self):
        def preload_task():
            self._load_state(load_next=True)

        self._preload_thread = threading.Thread(target=preload_task)
        self._preload_thread.daemon = True
        self._preload_thread.start()

    def _wait_for_preload(self):
        if self._preload_thread is not None:
            self._preload_thread.join()
            self._preload_thread = None
    
    def _check_curr_data(self):
        if self._curr_idx >= len(self._curr_block_idxs):
            self._curr_idx = 0
            self._wait_for_preload()
            self._curr_mmaps, self._next_mmaps = self._next_mmaps, self._curr_mmaps
            self._curr_buffers, self._next_buffers = self._next_buffers, self._curr_buffers
            self._curr_block_idxs, self._next_block_idxs = self._next_block_idxs, self._curr_block_idxs
            self._start_preload_next()

    def __next__(self):
        self._check_curr_data()
        block_idx = self._curr_block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._curr_buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
        self._curr_idx += 1
        self._check_curr_data()
        return torch.from_numpy(arr.astype(np.int64))
