"""
utils.py — Helper Utility Functions

Provides general-purpose helper functions used across the application:
  - FPS calculation
  - Frame resizing
  - Text overlay rendering
  - Prediction display
  - Visual feedback indicators
"""

import cv2
import time
import numpy as np


class FPSCounter:
    """
    Tracks and calculates real-time frames per second.
    Uses a rolling average for smooth display.
    """

    def __init__(self, avg_count=30):
        self._prev_time = time.time()
        self._fps_values = []
        self._avg_count = avg_count

    def update(self):
        """
        Call once per frame to update the FPS counter.

        Returns:
            float: Current smoothed FPS value.
        """
        current_time = time.time()
        delta = current_time - self._prev_time
        self._prev_time = current_time

        if delta > 0:
            fps = 1.0 / delta
            self._fps_values.append(fps)
            if len(self._fps_values) > self._avg_count:
                self._fps_values.pop(0)

        return self.get_fps()

    def get_fps(self):
        """
        Get the current smoothed FPS.

        Returns:
            float: Average FPS over the last N frames.
        """
        if not self._fps_values:
            return 0.0
        return sum(self._fps_values) / len(self._fps_values)


def overlay_fps(frame, fps):
    """
    Draw the FPS counter on the frame.

    Args:
        frame: BGR image to draw on.
        fps (float): Current FPS value.

    Returns:
        frame: The frame with FPS displayed.
    """
    h, w = frame.shape[:2]
    text = f"FPS: {int(fps)}"

    # Background rectangle for readability
    cv2.rectangle(frame, (w - 140, h - 40), (w - 5, h - 5), (0, 0, 0), -1)
    cv2.rectangle(frame, (w - 140, h - 40), (w - 5, h - 5), (80, 80, 80), 1)

    # FPS text
    color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(
        frame, text, (w - 130, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )

    return frame


def overlay_mode_indicator(frame, mode, eraser_active=False):
    """
    Draw the current mode indicator on the frame.

    Args:
        frame: BGR image to draw on.
        mode (str): Current mode ("draw", "select", "none").
        eraser_active (bool): Whether eraser is active.

    Returns:
        frame: The frame with mode indicator displayed.
    """
    h, w = frame.shape[:2]

    if eraser_active:
        text = "MODE: ERASER"
        color = (0, 100, 255)
    elif mode == "draw":
        text = "MODE: DRAW"
        color = (0, 255, 0)
    elif mode == "select":
        text = "MODE: SELECT"
        color = (255, 255, 0)
    else:
        text = "MODE: IDLE"
        color = (150, 150, 150)

    # Background
    cv2.rectangle(frame, (5, h - 40), (200, h - 5), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, h - 40), (200, h - 5), (80, 80, 80), 1)

    cv2.putText(
        frame, text, (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )

    return frame


def overlay_prediction(frame, digit, confidence, duration=3.0, start_time=None):
    """
    Display the prediction result prominently on the frame.

    Args:
        frame: BGR image to draw on.
        digit: Predicted digit (int or None).
        confidence (float): Prediction confidence (0-1).
        duration (float): How long to display the result (seconds).
        start_time (float): When the prediction was made (time.time()).

    Returns:
        tuple: (frame, still_showing) — the frame and whether to keep displaying.
    """
    if digit is None:
        return frame, False

    if start_time is not None:
        elapsed = time.time() - start_time
        if elapsed > duration:
            return frame, False

    h, w = frame.shape[:2]

    # Large prediction display box (centered)
    box_w, box_h = 300, 150
    box_x = (w - box_w) // 2
    box_y = (h - box_h) // 2

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Border
    border_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), border_color, 3)

    # Title
    cv2.putText(
        frame, "Prediction", (box_x + 85, box_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1,
    )

    # Predicted digit (large)
    cv2.putText(
        frame, str(digit), (box_x + 120, box_y + 100),
        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4,
    )

    # Confidence
    conf_text = f"Confidence: {confidence:.1%}"
    cv2.putText(
        frame, conf_text, (box_x + 60, box_y + 135),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2,
    )

    return frame, True


def overlay_message(frame, message, duration=2.0, start_time=None, color=(255, 255, 255)):
    """
    Display a temporary message on the frame.

    Args:
        frame: BGR image to draw on.
        message (str): Message to display.
        duration (float): How long to display (seconds).
        start_time (float): When the message was triggered.
        color (tuple): BGR text color.

    Returns:
        tuple: (frame, still_showing).
    """
    if start_time is not None:
        elapsed = time.time() - start_time
        if elapsed > duration:
            return frame, False

    h, w = frame.shape[:2]

    # Background bar
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    bar_w = text_size[0] + 40
    bar_x = (w - bar_w) // 2
    bar_y = h - 80

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 40), (0, 0, 0), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 40), (100, 100, 100), 1)

    text_x = bar_x + 20
    text_y = bar_y + 28
    cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame, True


def draw_cursor(frame, position, color, size, eraser_mode=False):
    """
    Draw a visual cursor at the finger position.

    Args:
        frame: BGR image to draw on.
        position: Tuple (x, y) of cursor position.
        color (tuple): Current brush color (BGR).
        size (int): Current brush size.
        eraser_mode (bool): Whether eraser is active.

    Returns:
        frame: The frame with cursor drawn.
    """
    if position is None:
        return frame

    x, y = position

    if eraser_mode:
        # Eraser cursor: circle with X
        radius = size * 2
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)
        cv2.line(frame, (x - 8, y - 8), (x + 8, y + 8), (0, 0, 255), 2)
        cv2.line(frame, (x + 8, y - 8), (x - 8, y + 8), (0, 0, 255), 2)
    else:
        # Draw cursor: filled circle with outline
        cv2.circle(frame, (x, y), size // 2 + 2, color, -1)
        cv2.circle(frame, (x, y), size // 2 + 4, (255, 255, 255), 1)

    return frame


def resize_frame(frame, width=1280):
    """
    Resize a frame to a target width while maintaining aspect ratio.

    Args:
        frame: BGR image to resize.
        width (int): Target width in pixels.

    Returns:
        numpy.ndarray: Resized frame.
    """
    h, w = frame.shape[:2]
    ratio = width / w
    new_h = int(h * ratio)
    return cv2.resize(frame, (width, new_h))


def draw_help_text(frame):
    """
    Draw keyboard shortcut help text on the frame.

    Args:
        frame: BGR image to draw on.

    Returns:
        frame: The frame with help text.
    """
    h, w = frame.shape[:2]
    help_lines = [
        "Keys: Q=Quit  S=Save  C=Clear  U=Undo  R=Recognize",
    ]

    y_pos = h - 55
    for line in help_lines:
        cv2.putText(
            frame, line, (220, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1,
        )
        y_pos += 20

    return frame
