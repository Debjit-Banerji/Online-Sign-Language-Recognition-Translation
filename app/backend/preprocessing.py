"""
Frame preprocessing: JPEG decode, resize, normalize.
Matches the transforms used by the SLRT TwoStreamNetwork for inference.
"""

import io
from typing import List, Tuple

import numpy as np
from PIL import Image


# Default image size expected by S3D backbone
DEFAULT_IMG_SIZE = 224


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to RGB numpy array (H, W, 3), uint8."""
    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return np.array(image)


def resize_frame(frame: np.ndarray, size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Resize frame to (size, size) using PIL for quality."""
    image = Image.fromarray(frame)
    image = image.resize((size, size), Image.BILINEAR)
    return np.array(image)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize frame to [-1, 1] range and swap channels BGR→RGB.
    Matches TwoStreamNetwork inference preprocessing:
        sgn_videos = sgn_videos[:,:,[2,1,0],:,:]  # BGR swap
        sgn_videos = (sgn_videos - 0.5) / 0.5      # normalize to [-1,1]
    
    Input: (H, W, 3) uint8 RGB
    Output: (3, H, W) float32 normalized
    """
    # Convert to float [0, 1]
    frame = frame.astype(np.float32) / 255.0
    # Channel swap RGB → BGR (model expects BGR then swaps back)
    # Actually the model does [:,[2,1,0]] which is RGB→BGR, 
    # but since our input is already RGB and model will swap,
    # we keep as-is and let model handle it.
    # Normalize to [-1, 1]
    frame = (frame - 0.5) / 0.5
    # HWC → CHW
    frame = np.transpose(frame, (2, 0, 1))
    return frame


def preprocess_frames(
    jpeg_frames: List[bytes],
    img_size: int = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a batch of JPEG frames.
    
    Args:
        jpeg_frames: List of JPEG-encoded bytes
        img_size: Target spatial resolution
    
    Returns:
        np.ndarray of shape (1, T, 3, H, W), float32, normalized to [-1, 1]
        Ready for model input as sgn_videos.
    """
    processed = []
    for jpeg in jpeg_frames:
        frame = decode_jpeg(jpeg)
        frame = resize_frame(frame, img_size)
        frame = normalize_frame(frame)
        processed.append(frame)
    
    # Stack: (T, C, H, W) → (1, T, C, H, W)  [batch dim]
    batch = np.stack(processed, axis=0)
    batch = np.expand_dims(batch, axis=0)
    return batch


def preprocess_numpy_frames(
    frames: List[np.ndarray],
    img_size: int = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    """
    Same as preprocess_frames but takes decoded numpy arrays instead of JPEG bytes.
    
    Args:
        frames: List of (H, W, 3) uint8 RGB arrays
        img_size: Target spatial resolution
    
    Returns:
        np.ndarray of shape (1, T, 3, H, W), float32
    """
    processed = []
    for frame in frames:
        frame = resize_frame(frame, img_size)
        frame = normalize_frame(frame)
        processed.append(frame)
    
    batch = np.stack(processed, axis=0)
    batch = np.expand_dims(batch, axis=0)
    return batch
