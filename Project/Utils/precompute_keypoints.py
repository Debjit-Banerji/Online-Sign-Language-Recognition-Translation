# precompute_keypoints.py

import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

from pose_extractor import PoseKeypointExtractor

MAX_FRAMES = 300

def process_split(root_dir, split):
    extractor = PoseKeypointExtractor()

    frames_dir = os.path.join(root_dir, "features", "fullFrame-210x260px", split)
    save_dir   = os.path.join("/home/abdullahm/jaleel/CV_project/Project/keypoints", split)
    os.makedirs(save_dir, exist_ok=True)

    video_folders = sorted(os.listdir(frames_dir))

    for vid in tqdm(video_folders, desc=f"{split}"):
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, vid, "*.png")))[:MAX_FRAMES]

        kps = []
        for p in frame_paths:
            img = cv2.imread(p)
            if img is None:
                kp = np.zeros((105, 2), dtype=np.float32)
            else:
                try:
                    kp = extractor.extract_keypoints(img)  # (105, 2)
                except:
                    kp = np.zeros((105, 2), dtype=np.float32)

            # 🔴 enforce shape safety (VERY IMPORTANT)
            if kp.shape != (105, 2):
                kp = np.zeros((105, 2), dtype=np.float32)

            kps.append(kp)

        kps = np.stack(kps)  # (T, 105, 2)

        np.save(os.path.join(save_dir, f"{vid}.npy"), kps)


if __name__ == "__main__":
    root = "/home/abdullahm/jaleel/CV_project/CLIP-SLA/Data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T"

    for split in ["train", "dev", "test"]:
        process_split(root, split)