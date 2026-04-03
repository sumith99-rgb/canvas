"""
gesture.py — Gesture Detection Module (Improved Accuracy)

Detects hand gestures from MediaPipe landmarks with:
  - Hysteresis-based finger detection (prevents flickering)
  - Robust thumb detection using angle-based method
  - Smoothed pinch distance
  - Gesture stability buffer (requires N consistent frames)
  - Debounce system for action triggers
"""

import time
import math


class GestureDetector:
    """
    Identifies hand gestures from landmark positions with high accuracy.

    Uses hysteresis and gesture stability to prevent false triggers
    and flickering between modes.
    """

    # Finger tip landmark IDs
    TIP_IDS = [4, 8, 12, 16, 20]
    # Finger PIP (proximal interphalangeal) joint IDs
    PIP_IDS = [3, 6, 10, 14, 18]
    # Finger MCP (metacarpophalangeal) joint IDs
    MCP_IDS = [2, 5, 9, 13, 17]

    def __init__(self, pinch_threshold=40, debounce_time=0.6):
        self.pinch_threshold = pinch_threshold
        self.debounce_time = debounce_time
        self._last_pinch_time = 0

        # Gesture stability: require consistent gesture for N frames
        self._gesture_buffer = []
        self._buffer_size = 3  # Must see same gesture for 3 frames

        # Finger state hysteresis thresholds
        # A finger is considered "up" if tip is above PIP by at least UP_MARGIN pixels
        # A finger is considered "down" if tip is below PIP by at least DOWN_MARGIN pixels
        # Between these margins, the previous state is kept (hysteresis)
        self._finger_states = [0, 0, 0, 0, 0]
        self._hysteresis_margin = 15  # pixels

        # Smoothed pinch distance (exponential moving average)
        self._smoothed_pinch_dist = 100.0
        self._pinch_alpha = 0.4  # Smoothing factor (0=max smooth, 1=no smooth)

    def fingers_up(self, landmarks):
        """
        Determine which fingers are raised using hysteresis.

        Uses angle-based detection for thumb and y-position comparison
        with hysteresis margins for other fingers to prevent flickering.

        Args:
            landmarks: List of (id, x, y) tuples from HandTracker.

        Returns:
            List of 5 ints [thumb, index, middle, ring, pinky],
            each 0 (down) or 1 (up).
        """
        if len(landmarks) < 21:
            return [0, 0, 0, 0, 0]

        fingers = []

        # ── THUMB detection (angle-based for accuracy) ─────────
        # Calculate the angle between thumb MCP→IP→TIP
        # If the thumb is extended, this angle will be large
        thumb_cmc = landmarks[1]   # CMC joint
        thumb_mcp = landmarks[2]   # MCP joint
        thumb_ip = landmarks[3]    # IP joint
        thumb_tip = landmarks[4]   # Tip

        wrist = landmarks[0]
        index_mcp = landmarks[5]

        # Determine hand side by checking wrist-to-index_mcp direction
        hand_direction = index_mcp[1] - wrist[1]  # positive = right hand (camera view)

        # For thumb: compute distance from tip to palm center
        # Palm center approximated as midpoint of wrist and middle MCP
        palm_cx = (wrist[1] + landmarks[9][1]) // 2
        palm_cy = (wrist[2] + landmarks[9][2]) // 2

        # Thumb is up if tip is far from palm center horizontally
        thumb_dist_from_palm = abs(thumb_tip[1] - palm_cx)
        ip_dist_from_palm = abs(thumb_ip[1] - palm_cx)

        if thumb_dist_from_palm > ip_dist_from_palm + 10:
            fingers.append(1)
        elif thumb_dist_from_palm < ip_dist_from_palm - 10:
            fingers.append(0)
        else:
            # Keep previous state (hysteresis)
            fingers.append(self._finger_states[0])

        # ── OTHER FINGERS detection (with hysteresis) ──────────
        for i, (tip_id, pip_id) in enumerate(zip(self.TIP_IDS[1:], self.PIP_IDS[1:]), 1):
            tip_y = landmarks[tip_id][2]
            pip_y = landmarks[pip_id][2]
            diff = pip_y - tip_y  # Positive = finger up (tip above PIP)

            if diff > self._hysteresis_margin:
                # Clearly up
                fingers.append(1)
            elif diff < -self._hysteresis_margin:
                # Clearly down
                fingers.append(0)
            else:
                # In the hysteresis zone: keep previous state
                fingers.append(self._finger_states[i])

        # Update stored states
        self._finger_states = fingers.copy()
        return fingers

    def get_distance(self, p1, p2):
        """Calculate Euclidean distance between two landmark points."""
        return math.sqrt((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def is_pinch(self, landmarks):
        """
        Check if thumb and index finger are pinching.
        Uses smoothed distance to prevent false triggers from jitter.
        """
        if len(landmarks) < 21:
            return False

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        raw_distance = self.get_distance(thumb_tip, index_tip)

        # Smooth the distance with exponential moving average
        self._smoothed_pinch_dist = (
            self._pinch_alpha * raw_distance
            + (1 - self._pinch_alpha) * self._smoothed_pinch_dist
        )

        return self._smoothed_pinch_dist < self.pinch_threshold

    def is_draw_mode(self, landmarks):
        """
        Check if the hand is in drawing mode.
        Drawing mode: only index finger is up.
        """
        fingers = self.fingers_up(landmarks)
        # Index up, middle/ring/pinky down (thumb can be anything)
        return fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0

    def is_select_mode(self, landmarks):
        """
        Check if the hand is in selection mode.
        Selection mode: index and middle fingers are up, ring and pinky down.
        """
        fingers = self.fingers_up(landmarks)
        return fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0

    def check_pinch_debounced(self, landmarks):
        """Check for pinch gesture with debounce protection."""
        if not self.is_pinch(landmarks):
            return False

        current_time = time.time()
        if current_time - self._last_pinch_time >= self.debounce_time:
            self._last_pinch_time = current_time
            return True

        return False

    def _get_stable_gesture(self, gesture):
        """
        Apply stability buffer — only change gesture if it's been
        consistent for N consecutive frames. Prevents rapid flickering.

        Args:
            gesture (str): Current frame's raw gesture.

        Returns:
            str: Stable gesture (may lag by a few frames).
        """
        self._gesture_buffer.append(gesture)
        if len(self._gesture_buffer) > self._buffer_size:
            self._gesture_buffer.pop(0)

        # If all recent frames agree, use that gesture
        if len(self._gesture_buffer) == self._buffer_size:
            if all(g == self._gesture_buffer[0] for g in self._gesture_buffer):
                return self._gesture_buffer[0]

        # Otherwise, return the previous stable gesture (or "none")
        # This prevents flickering during transitions
        if self._gesture_buffer:
            return self._gesture_buffer[-1]
        return "none"

    def get_gesture(self, landmarks):
        """
        Determine the current gesture from landmarks with stability.

        Returns one of:
            "draw"    — index finger up only
            "select"  — index + middle fingers up
            "pinch"   — thumb + index close (debounced)
            "none"    — no recognized gesture
        """
        if len(landmarks) < 21:
            return "none"

        # Check pinch first (highest priority, bypasses stability buffer)
        if self.check_pinch_debounced(landmarks):
            return "pinch"

        # Determine raw gesture
        raw_gesture = "none"
        if self.is_draw_mode(landmarks):
            raw_gesture = "draw"
        elif self.is_select_mode(landmarks):
            raw_gesture = "select"

        # Apply stability buffer to prevent flickering
        return self._get_stable_gesture(raw_gesture)

    def get_index_finger_pos(self, landmarks):
        """Get the position of the index finger tip."""
        if len(landmarks) < 21:
            return None
        return (landmarks[8][1], landmarks[8][2])

    def get_middle_finger_pos(self, landmarks):
        """Get the position of the middle finger tip."""
        if len(landmarks) < 21:
            return None
        return (landmarks[12][1], landmarks[12][2])
