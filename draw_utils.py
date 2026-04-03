"""
draw_utils.py — Drawing Canvas Engine

Manages a NumPy-based drawing canvas that overlays on the webcam feed.
Supports multi-color drawing, eraser, undo, brush sizes, and save.
"""

import cv2
import numpy as np
import os
from datetime import datetime


class DrawingCanvas:
    """
    A transparent drawing canvas backed by a NumPy array.

    The canvas uses black (0,0,0) as the transparent/empty color.
    Drawing is done by setting pixel values to the brush color.
    The canvas is merged with the webcam frame using masking.

    Attributes:
        width (int): Canvas width in pixels.
        height (int): Canvas height in pixels.
        brush_color (tuple): Current BGR color for drawing.
        brush_size (int): Current brush thickness in pixels.
        eraser_mode (bool): Whether eraser is active.
    """

    # Default drawing color (purple-blue)
    DEFAULT_COLOR = (255, 0, 128)

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Brush settings
        self.brush_color = self.DEFAULT_COLOR
        self.brush_size = 8
        self.eraser_mode = False
        self._saved_color = self.DEFAULT_COLOR

        # Stroke history for undo (stores canvas snapshots)
        self._stroke_history = []
        self._max_history = 20  # Limit memory usage

        # Track whether we're mid-stroke (for saving history at stroke boundaries)
        self._is_drawing = False

    def start_stroke(self):
        """
        Mark the beginning of a new stroke.
        Saves the current canvas state for undo support.
        """
        if not self._is_drawing:
            self._is_drawing = True
            # Save a snapshot before the stroke begins
            snapshot = self.canvas.copy()
            self._stroke_history.append(snapshot)
            # Trim history if too long
            if len(self._stroke_history) > self._max_history:
                self._stroke_history.pop(0)

    def end_stroke(self):
        """Mark the end of the current stroke."""
        self._is_drawing = False

    def draw_line(self, prev_point, curr_point):
        """
        Draw a smooth anti-aliased line from prev_point to curr_point.

        Uses cv2.LINE_AA for anti-aliasing and adds circular end caps
        for smooth, professional-looking strokes.
        """
        if prev_point is None or curr_point is None:
            return

        color = (0, 0, 0) if self.eraser_mode else self.brush_color
        thickness = self.brush_size * 3 if self.eraser_mode else self.brush_size

        # Anti-aliased line between points
        cv2.line(self.canvas, prev_point, curr_point, color, thickness, cv2.LINE_AA)

        # Add circular caps at both ends for smooth joints
        radius = thickness // 2
        cv2.circle(self.canvas, prev_point, radius, color, -1, cv2.LINE_AA)
        cv2.circle(self.canvas, curr_point, radius, color, -1, cv2.LINE_AA)

    def draw_point(self, point):
        """
        Draw a single anti-aliased point (circle) on the canvas.
        """
        if point is None:
            return

        color = (0, 0, 0) if self.eraser_mode else self.brush_color
        thickness = self.brush_size * 3 if self.eraser_mode else self.brush_size

        cv2.circle(self.canvas, point, thickness // 2, color, -1, cv2.LINE_AA)

    def set_brush_size(self, size):
        """
        Set the brush thickness.

        Args:
            size (int): Brush size in pixels (e.g., 5, 10, 20).
        """
        self.brush_size = max(1, min(size, 50))

    def set_color(self, color):
        """
        Set the brush color.

        Args:
            color (tuple): BGR color tuple, e.g., (0, 0, 255) for red.
        """
        self.brush_color = color
        self._saved_color = color
        if self.eraser_mode:
            self.eraser_mode = False

    def toggle_eraser(self):
        """Toggle eraser mode on/off."""
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self._saved_color = self.brush_color
        else:
            self.brush_color = self._saved_color

    def clear(self):
        """
        Clear the entire canvas to black (transparent).
        Saves current state to history first.
        """
        self._stroke_history.append(self.canvas.copy())
        if len(self._stroke_history) > self._max_history:
            self._stroke_history.pop(0)
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def undo(self):
        """
        Undo the last stroke by restoring the previous canvas snapshot.

        Returns:
            bool: True if undo was performed, False if no history.
        """
        if self._stroke_history:
            self.canvas = self._stroke_history.pop()
            return True
        return False

    def get_canvas(self):
        """
        Get the current canvas array.

        Returns:
            numpy.ndarray: The canvas (height x width x 3, BGR).
        """
        return self.canvas

    def get_drawing_region(self):
        """
        Extract the bounding box region that contains drawn content.
        Useful for handwriting recognition preprocessing.

        Returns:
            numpy.ndarray or None: Cropped region containing the drawing,
                                    or None if canvas is empty.
        """
        # Convert to grayscale to find non-zero (drawn) pixels
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)

        # Find bounding box of all non-zero pixels
        coords = cv2.findNonZero(gray)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)

        # Add padding
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(self.width - x, w + 2 * pad)
        h = min(self.height - y, h + 2 * pad)

        return self.canvas[y : y + h, x : x + w]

    def merge_with_frame(self, frame):
        """
        Overlay the canvas onto the webcam frame.

        Uses masking: wherever the canvas has non-black pixels,
        those pixels replace the frame pixels.

        Args:
            frame: BGR webcam frame (same dimensions as canvas).

        Returns:
            numpy.ndarray: Merged frame.
        """
        # Ensure dimensions match
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        # Create a mask where canvas has drawn content (non-black pixels)
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Black-out the drawn regions in the frame
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        # Extract drawn regions from canvas
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

        # Combine
        merged = cv2.add(frame_bg, canvas_fg)
        return merged

    def save_image(self, directory="saved_drawings"):
        """
        Save the current canvas as a PNG image.

        Args:
            directory (str): Directory to save images in.

        Returns:
            str: Path to the saved image.
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drawing_{timestamp}.png"
        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, self.canvas)
        return filepath

    def is_empty(self):
        """
        Check if the canvas has any drawn content.

        Returns:
            bool: True if canvas is completely empty (all black).
        """
        return not np.any(self.canvas)
