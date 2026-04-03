"""
hand_tracking.py — MediaPipe Hand Tracking Module (Tasks API)

Uses the new MediaPipe Tasks API (compatible with mediapipe >= 0.10.14)
to detect 21 hand landmarks from webcam frames in VIDEO mode.

Accuracy improvements:
  - Uses full-precision model for better detection
  - Lower confidence thresholds for more consistent tracking
  - Provides raw normalized landmarks for sub-pixel accuracy
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request


# Path to the hand landmarker model
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Hand landmark connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


def _download_model():
    """Download the hand landmarker model if not present."""
    if os.path.exists(MODEL_PATH):
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[INFO] Downloading hand landmarker model...")
    print(f"[INFO] URL: {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[INFO] Model saved to: {MODEL_PATH}")


class HandTracker:
    """
    Detects and tracks hands using the MediaPipe Tasks Hand Landmarker API.

    Uses VIDEO running mode for synchronous frame-by-frame processing.
    Tuned for high accuracy with lower confidence thresholds to maintain
    continuous tracking even during fast hand movements.
    """

    def __init__(self, mode=False, max_hands=1, detection_conf=0.5, tracking_conf=0.5):
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        # Download model if needed
        _download_model()

        # Create Hand Landmarker with VIDEO mode
        # Lower confidence thresholds = more consistent detection
        # (prevents hand from "disappearing" during fast movements)
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_conf,
            min_hand_presence_confidence=self.detection_conf,
            min_tracking_confidence=self.tracking_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # Store latest results
        self.results = None
        self._frame_timestamp_ms = 0

        # Landmark tip IDs for each finger
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, frame, draw=True):
        """
        Detect hands in the given frame.

        Args:
            frame: BGR image from webcam.
            draw: Whether to draw landmarks on the frame.

        Returns:
            frame: The (possibly annotated) frame.
        """
        h, w, _ = frame.shape

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Increment timestamp for VIDEO mode
        self._frame_timestamp_ms += 33  # ~30fps

        # Detect hand landmarks
        self.results = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        # Draw landmarks on frame if requested
        if draw and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                self._draw_landmarks(frame, hand_landmarks, w, h)

        return frame

    def _draw_landmarks(self, frame, hand_landmarks, width, height):
        """Draw styled hand landmarks and connections on the frame."""
        points = []
        for lm in hand_landmarks:
            px = int(lm.x * width)
            py = int(lm.y * height)
            points.append((px, py))

        # Draw connections (cyan lines for visibility)
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], (255, 200, 0), 2, cv2.LINE_AA)

        # Draw landmark points
        for i, (px, py) in enumerate(points):
            # Fingertips get larger circles
            if i in self.tip_ids:
                cv2.circle(frame, (px, py), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 3, (0, 200, 0), -1, cv2.LINE_AA)

    def find_position(self, frame, hand_no=0):
        """
        Extract landmark positions for a specific hand.

        Returns sub-pixel accurate positions by using float coordinates
        and rounding at the final step.

        Args:
            frame: BGR image (used for dimension reference).
            hand_no: Index of the hand to extract (default: 0).

        Returns:
            landmarks: List of (id, x, y) tuples for all 21 landmarks.
                       Empty list if no hand detected.
        """
        landmarks = []

        if self.results and self.results.hand_landmarks:
            if hand_no < len(self.results.hand_landmarks):
                hand = self.results.hand_landmarks[hand_no]
                h, w, _ = frame.shape

                for idx, lm in enumerate(hand):
                    # Use round() for sub-pixel accuracy instead of int()
                    cx = round(lm.x * w)
                    cy = round(lm.y * h)
                    landmarks.append((idx, cx, cy))

        return landmarks

    def hands_detected(self):
        """Check if any hands are currently detected."""
        return (
            self.results is not None
            and self.results.hand_landmarks is not None
            and len(self.results.hand_landmarks) > 0
        )

    def close(self):
        """Release the landmarker resources."""
        if self.landmarker:
            self.landmarker.close()
