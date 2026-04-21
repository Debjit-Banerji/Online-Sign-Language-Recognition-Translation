"""
End-to-end test: Load real SLRT model and run inference with random video frames.
Tests the full pipeline: random frames → S3D → visual head → VLMapper → mBART → German text → English.
"""
import sys, os
import numpy as np
import time

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_loader import load_model
from preprocessing import preprocess_numpy_frames

print("=" * 60)
print("End-to-end SLRT pipeline test")
print("=" * 60)

# Step 1: Load real model
print("\n[1/3] Loading SLRT model...")
t0 = time.time()
model = load_model(mode="slrt")
print(f"  Model loaded in {time.time() - t0:.1f}s")

# Step 2: Create fake video frames (16 random frames, 224x224, RGB)
print("\n[2/3] Creating fake video frames...")
NUM_FRAMES = 16
fake_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(NUM_FRAMES)]
preprocessed = preprocess_numpy_frames(fake_frames, img_size=224)
print(f"  Preprocessed shape: {preprocessed.shape}")  # expect (1, 16, 3, 224, 224)
print(f"  dtype: {preprocessed.dtype}, range: [{preprocessed.min():.2f}, {preprocessed.max():.2f}]")

# Step 3: Run inference
print("\n[3/3] Running inference...")
t0 = time.time()
result = model.predict(preprocessed)
inference_time = time.time() - t0

print(f"\n{'=' * 60}")
print(f"RESULTS (inference took {inference_time:.2f}s):")
print(f"  Glosses:      {result['glosses']}")
print(f"  German text:  {result.get('text_de', 'N/A')}")
print(f"  English text: {result['text']}")
print(f"{'=' * 60}")
print("\n✅ Pipeline test complete!")
