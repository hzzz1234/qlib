# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pickle
from multiprocessing import shared_memory
from typing import Any, Dict, Tuple
import pandas as pd

from ..log import get_module_logger


class CSFeatureSharedMemCache:
    """
    Shared memory cache for cross-section features.
    
    This cache stores precomputed cross-section feature results in shared memory,
    allowing worker processes to access them without recomputation.
    
    Design:
    - Main process computes CS features for all instruments
    - Results are stored in shared memory blocks
    - Worker processes read from shared memory when needed
    """
    
    def __init__(self):
        """Initialize the CS feature cache."""
        self.logger = get_module_logger("CSFeatureSharedMemCache")
        
        # Dict mapping (feature_name, instrument) -> (shm_name, offset, size)
        self._metadata = {}
        
        # Dict of opened shared memory blocks: {block_name: block}
        self._shm_blocks = {}
        
        # Current block for writing
        self._current_block = None
        self._current_offset = 0
        
        # Default block size: 200MB
        self.BLOCK_SIZE = 200 * 1024 * 1024
        
        # Track if this is the creator process
        self._is_creator = True
    
    def _ensure_block(self, required_size: int):
        """Ensure there's a block with enough space."""
        if self._current_block is None or self._current_offset + required_size > self.BLOCK_SIZE:
            # Create new block
            new_block = shared_memory.SharedMemory(create=True, size=self.BLOCK_SIZE)
            self._shm_blocks[new_block.name] = new_block
            self._current_block = new_block
            self._current_offset = 0
            self.logger.info(f"Created new shared memory block: {new_block.name}")
    
    def set(self, feature_name: str, instrument: str, value: pd.Series):
        """
        Store a feature value for an instrument.
        
        :param feature_name: Name of the cross-section feature (e.g., "CSRank($close)")
        :param instrument: Instrument code
        :param value: Feature value (typically a pd.Series)
        """
        if not self._is_creator:
            raise RuntimeError("Only creator process can write to CSFeatureSharedMemCache")
        
        import pandas as pd
        import numpy as np
        
        # Serialize values
        values_bytes = value.values.tobytes()
        n_elements = len(value)
        
        # Optimize index storage based on type
        index = value.index
        if isinstance(index, pd.RangeIndex):
            # RangeIndex: store start, stop, step as 3 int32
            index_type = 0
            index_bytes = b''
            # Header: n_elements(4) + start(4) + stop(4) + step(4) = 16 bytes
            header_size = 16
        elif hasattr(index, 'values') and np.issubdtype(index.dtype, np.integer):
            # Integer index: store as numpy array
            index_type = 1
            index_bytes = index.values.tobytes()
            # Header: n_elements(4) = 4 bytes
            header_size = 4
        else:
            # Complex index (DatetimeIndex, etc.): use pickle
            index_type = 2
            index_bytes = pickle.dumps(index)
            # Header: n_elements(4) = 4 bytes
            header_size = 4
        
        total_size = header_size + len(index_bytes) + len(values_bytes)
        
        # Ensure we have space
        self._ensure_block(total_size)
        
        # Write to current block
        block_name = self._current_block.name
        offset = self._current_offset
        
        buf = self._current_block.buf
        pos = offset
        
        # Write header (no index_type, it's in metadata)
        buf[pos:pos+4] = n_elements.to_bytes(4, 'little')
        pos += 4
        
        # For RangeIndex, write start, stop, step as int32
        if index_type == 0:
            buf[pos:pos+4] = index.start.to_bytes(4, 'little', signed=True)
            pos += 4
            buf[pos:pos+4] = index.stop.to_bytes(4, 'little', signed=True)
            pos += 4
            buf[pos:pos+4] = index.step.to_bytes(4, 'little', signed=True)
            pos += 4
        
        # Write index bytes (for type 1 and 2)
        if index_bytes:
            buf[pos:pos+len(index_bytes)] = index_bytes
            pos += len(index_bytes)
        
        # Write values
        buf[pos:pos+len(values_bytes)] = values_bytes
        
        # Update metadata: (block_name, offset, size, index_type, values_dtype, index_dtype)
        if instrument not in self._metadata:
            self._metadata[instrument] = {}
        self._metadata[instrument][feature_name] = (block_name, offset, total_size, index_type, value.dtype, index.dtype)
        
        # Update offset
        self._current_offset += total_size
    
    def get(self, feature_name: str, instrument: str) -> Any:
        """
        Retrieve a feature value for an instrument.
        
        :param feature_name: Name of the cross-section feature (also used as Series.name)
        :param instrument: Instrument code
        :return: Feature value or None if not found
        """
        if instrument not in self._metadata or feature_name not in self._metadata[instrument]:
            return None
        
        meta = self._metadata[instrument][feature_name]
        
        # New format: (block_name, offset, size, index_type, values_dtype, index_dtype)
        if len(meta) == 6:
            block_name, offset, size, index_type, values_dtype, index_dtype = meta
        else:
            # Old formats not supported in new version
            raise ValueError(f"Unsupported metadata format: {len(meta)} fields")
        
        # Open the shared memory block if needed
        shm = self._shm_blocks.get(block_name)
        
        # no shm no data
        if shm is None:
            return None
        
        # Read from shared memory
        import pandas as pd
        import numpy as np
        
        buf = shm.buf
        pos = offset
        
        # Read header (index_type is in metadata now)
        n_elements = int.from_bytes(bytes(buf[pos:pos+4]), 'little')
        pos += 4
        
        # Reconstruct index based on type (from metadata)
        if index_type == 0:
            # RangeIndex - read 3 int32 values
            start = int.from_bytes(bytes(buf[pos:pos+4]), 'little', signed=True)
            pos += 4
            stop = int.from_bytes(bytes(buf[pos:pos+4]), 'little', signed=True)
            pos += 4
            step = int.from_bytes(bytes(buf[pos:pos+4]), 'little', signed=True)
            pos += 4
            index = pd.RangeIndex(start, stop, step)
        elif index_type == 1:
            # Integer array - calculate length from n_elements and dtype
            index_len = n_elements * index_dtype.itemsize
            index_values = np.frombuffer(buf[pos:pos+index_len], dtype=index_dtype).copy()
            index = pd.Index(index_values)
            pos += index_len
        else:
            # Pickled index - calculate length from total size
            values_len = n_elements * values_dtype.itemsize
            remaining = size - (pos - offset) - values_len
            index = pickle.loads(bytes(buf[pos:pos+remaining]))
            pos += remaining
        
        # Reconstruct numpy array from bytes
        values_len = n_elements * values_dtype.itemsize
        values = np.frombuffer(buf[pos:pos+values_len], dtype=values_dtype).copy()
        
        # Create Series with feature_name as name
        series = pd.Series(values, index=index, name=feature_name)
        return series
    
    def cleanup(self):
        """Cleanup all shared memory blocks."""
        for shm in self._shm_blocks.values():
            try:
                shm.close()
                if self._is_creator:
                    self.logger.info(f"Unlinking shared memory block: {shm.name}")
                    shm.unlink()
            except:
                pass
        self._shm_blocks.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            for shm in self._shm_blocks.values():
                shm.close()
        except:
            pass
