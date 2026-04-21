"""
Sliding-window frame buffer for accumulating video frames before inference.
Thread-safe, configurable window size and stride.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class FrameBuffer:
    """
    Accumulates frames and yields batches for inference.
    
    Args:
        window_size: Number of frames per inference window (e.g. 16 or 32)
        stride: How many new frames before triggering next inference.
                If stride < window_size, windows overlap.
        max_queue_size: Max frames to hold in memory (backpressure).
    """
    window_size: int = 16
    stride: int = 8
    max_queue_size: int = 128
    
    _frames: deque = field(default_factory=deque, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _frames_since_last_inference: int = field(default=0, init=False, repr=False)
    _total_frames: int = field(default=0, init=False, repr=False)

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Add a frame to the buffer.
        Returns True if enough frames are ready for inference.
        """
        with self._lock:
            if len(self._frames) >= self.max_queue_size:
                # Drop oldest frame (backpressure)
                self._frames.popleft()
            
            self._frames.append(frame)
            self._frames_since_last_inference += 1
            self._total_frames += 1
            
            return self.is_ready()

    def is_ready(self) -> bool:
        """Check if we have enough frames for an inference window."""
        return (len(self._frames) >= self.window_size and 
                self._frames_since_last_inference >= self.stride)

    def get_window(self) -> Optional[List[np.ndarray]]:
        """
        Get the latest window of frames for inference.
        Returns None if not enough frames.
        """
        with self._lock:
            if not self.is_ready():
                return None
            
            # Take the last `window_size` frames
            window = list(self._frames)[-self.window_size:]
            self._frames_since_last_inference = 0
            return window

    def clear(self):
        """Reset the buffer."""
        with self._lock:
            self._frames.clear()
            self._frames_since_last_inference = 0
            self._total_frames = 0

    @property
    def frame_count(self) -> int:
        return self._total_frames

    @property
    def buffered_count(self) -> int:
        return len(self._frames)
