"""
ui_toolbar.py — On-Screen Toolbar UI

Renders a toolbar at the top of the webcam frame with:
  - Color palette (6 colors)
  - Brush size selectors (Small, Medium, Large)
  - Eraser toggle
  - Clear button
  - Recognize button
  - Save button
  - Undo button

All rendering uses OpenCV drawing primitives for maximum performance.
"""

import cv2
import numpy as np


class ToolbarButton:
    """
    Represents a single clickable button on the toolbar.
    """

    def __init__(self, x, y, w, h, label, action, color=(60, 60, 60), text_color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.action = action  # Action string returned on click
        self.color = color
        self.text_color = text_color
        self.active = False  # Whether this button is currently selected

    def contains(self, px, py):
        """Check if point (px, py) is inside this button."""
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def draw(self, frame):
        """Draw this button on the frame."""
        # Button background
        bg_color = self.color
        border_color = (255, 255, 255) if self.active else (100, 100, 100)
        border_thickness = 3 if self.active else 1

        # Draw filled rectangle
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.w, self.y + self.h),
            bg_color,
            -1,
        )

        # Draw border
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.w, self.y + self.h),
            border_color,
            border_thickness,
        )

        # Draw label text (centered)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        text_size = cv2.getTextSize(self.label, font, font_scale, thickness)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        cv2.putText(frame, self.label, (text_x, text_y), font, font_scale, self.text_color, thickness)


class Toolbar:
    """
    On-screen toolbar rendered at the top of the webcam frame.

    Provides color palette, brush sizes, eraser, clear, recognize, save, and undo buttons.
    """

    # Toolbar dimensions
    TOOLBAR_HEIGHT = 70
    BUTTON_MARGIN = 6
    BUTTON_HEIGHT = 50

    # Color palette (BGR format)
    COLORS = {
        "Red": (0, 0, 255),
        "Blue": (255, 50, 0),
        "Green": (0, 200, 0),
        "Yellow": (0, 255, 255),
        "White": (255, 255, 255),
        "Purple": (255, 0, 128),
    }

    # Brush sizes
    SIZES = {
        "S": 5,
        "M": 10,
        "L": 20,
    }

    def __init__(self, frame_width=1280):
        self.frame_width = frame_width
        self.buttons = []
        self.active_color = "Purple"
        self.active_size = "M"
        self.eraser_active = False
        self._build_buttons()

    def _build_buttons(self):
        """Create all toolbar buttons with their positions."""
        self.buttons = []
        x = self.BUTTON_MARGIN
        y = self.BUTTON_MARGIN + 5
        h = self.BUTTON_HEIGHT

        # --- Color palette buttons ---
        color_w = 55
        for name, bgr in self.COLORS.items():
            # Use contrasting text color
            brightness = sum(bgr) / 3
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
            btn = ToolbarButton(x, y, color_w, h, name, f"color_{name.lower()}", bgr, text_color)
            if name == self.active_color:
                btn.active = True
            self.buttons.append(btn)
            x += color_w + self.BUTTON_MARGIN

        # --- Separator ---
        x += 10

        # --- Brush size buttons ---
        size_w = 40
        for label, size in self.SIZES.items():
            btn = ToolbarButton(x, y, size_w, h, label, f"size_{label.lower()}", (80, 80, 80))
            if label == self.active_size:
                btn.active = True
            self.buttons.append(btn)
            x += size_w + self.BUTTON_MARGIN

        # --- Separator ---
        x += 10

        # --- Action buttons ---
        action_w = 70

        # Eraser
        eraser_btn = ToolbarButton(x, y, action_w, h, "Eraser", "eraser", (50, 50, 120))
        self.buttons.append(eraser_btn)
        x += action_w + self.BUTTON_MARGIN

        # Clear
        clear_btn = ToolbarButton(x, y, action_w, h, "Clear", "clear", (30, 30, 150))
        self.buttons.append(clear_btn)
        x += action_w + self.BUTTON_MARGIN

        # Undo
        undo_btn = ToolbarButton(x, y, action_w, h, "Undo", "undo", (80, 80, 40))
        self.buttons.append(undo_btn)
        x += action_w + self.BUTTON_MARGIN

        # Save
        save_btn = ToolbarButton(x, y, action_w, h, "Save", "save", (40, 100, 40))
        self.buttons.append(save_btn)
        x += action_w + self.BUTTON_MARGIN

        # Recognize
        recog_btn = ToolbarButton(x, y, 90, h, "Recognize", "recognize", (120, 60, 20))
        self.buttons.append(recog_btn)

    def draw(self, frame):
        """
        Render the toolbar on the frame.

        Args:
            frame: BGR webcam frame to draw on.

        Returns:
            frame: The frame with toolbar drawn.
        """
        # Semi-transparent toolbar background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.TOOLBAR_HEIGHT), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Draw all buttons
        for btn in self.buttons:
            btn.draw(frame)

        return frame

    def check_click(self, x, y):
        """
        Check if a point (x, y) falls inside any toolbar button.

        Args:
            x (int): X coordinate of the click/finger position.
            y (int): Y coordinate of the click/finger position.

        Returns:
            str or None: The action string of the clicked button, or None.
        """
        for btn in self.buttons:
            if btn.contains(x, y):
                return btn.action
        return None

    def update_state(self, action):
        """
        Update toolbar state based on an action.

        Args:
            action (str): The action triggered (e.g., "color_red", "size_m").

        Returns:
            dict: State changes to apply. Keys may include:
                  "color", "size", "eraser", "clear", "undo",
                  "save", "recognize".
        """
        result = {}

        if action is None:
            return result

        # Color selection
        if action.startswith("color_"):
            color_name = action.split("_")[1].capitalize()
            if color_name in self.COLORS:
                self.active_color = color_name
                self.eraser_active = False
                result["color"] = self.COLORS[color_name]
                # Update button active states
                for btn in self.buttons:
                    if btn.action.startswith("color_"):
                        btn.active = (btn.action == action)
                    if btn.action == "eraser":
                        btn.active = False

        # Size selection
        elif action.startswith("size_"):
            size_label = action.split("_")[1].upper()
            if size_label in self.SIZES:
                self.active_size = size_label
                result["size"] = self.SIZES[size_label]
                for btn in self.buttons:
                    if btn.action.startswith("size_"):
                        btn.active = (btn.action == action)

        # Eraser toggle
        elif action == "eraser":
            self.eraser_active = not self.eraser_active
            result["eraser"] = self.eraser_active
            for btn in self.buttons:
                if btn.action == "eraser":
                    btn.active = self.eraser_active
                # Deactivate color buttons when eraser is on
                if btn.action.startswith("color_") and self.eraser_active:
                    btn.active = False

        # Clear canvas
        elif action == "clear":
            result["clear"] = True

        # Undo
        elif action == "undo":
            result["undo"] = True

        # Save
        elif action == "save":
            result["save"] = True

        # Recognize
        elif action == "recognize":
            result["recognize"] = True

        return result

    def is_in_toolbar(self, y):
        """
        Check if a y-coordinate is within the toolbar region.

        Args:
            y (int): Y coordinate to check.

        Returns:
            bool: True if within toolbar region.
        """
        return y < self.TOOLBAR_HEIGHT

    def get_active_color(self):
        """Get the currently active color as BGR tuple."""
        return self.COLORS.get(self.active_color, self.COLORS["Purple"])

    def get_active_size(self):
        """Get the currently active brush size."""
        return self.SIZES.get(self.active_size, 10)
