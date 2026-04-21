import cv2
import mediapipe as mp
import numpy as np

class PoseKeypointExtractor:
    def __init__(self, face_indices=None):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.face_indices = face_indices or [
            10, 152, 234, 454, 33, 263, 61, 291, 13, 14,
            78, 308, 82, 312, 87, 317, 95, 324, 107, 336,
            55, 285, 52, 282, 65, 295, 70, 300, 63, 293
        ]

    def extract_keypoints(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.holistic.process(image)

        def get_pts(lm_list, count):
            if not lm_list:
                return [[0.0, 0.0]] * count
            return [[lm.x, lm.y] for lm in lm_list.landmark]

        pose_pts = get_pts(res.pose_landmarks, 33)
        left_pts = get_pts(res.left_hand_landmarks, 21)
        right_pts = get_pts(res.right_hand_landmarks, 21)

        face_all = get_pts(res.face_landmarks, 468)
        face_pts = [face_all[i] for i in self.face_indices] if len(face_all) >= 468 else [[0.0, 0.0]] * len(self.face_indices)

        keypoints = pose_pts + left_pts + right_pts + face_pts

        if len(keypoints) != 105:
            print(f"[WARN] keypoint length = {len(keypoints)}", flush=True)

        return np.array(keypoints, dtype=np.float32)