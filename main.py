"""
main.py — AI Virtual Paint App with Hand Gesture Control

Main application loop that orchestrates:
  - Webcam capture
  - Hand tracking (MediaPipe)
  - Gesture recognition
  - Drawing engine
  - Toolbar UI
  - Handwriting recognition (MNIST CNN)

Controls:
  - Index finger up → Drawing mode
  - Index + middle fingers up → Selection mode (interact with toolbar)
  - Pinch (thumb + index) → Toggle eraser

Keyboard shortcuts:
  Q → Quit
  S → Save drawing
  C → Clear canvas
  U → Undo last stroke
  R → Recognize handwriting
"""

import cv2
import time
import sys
import os

from hand_tracking import HandTracker
from gesture import GestureDetector
from draw_utils import DrawingCanvas
from ui_toolbar import Toolbar
from model import load_trained_model, predict_digit
from utils import (
    FPSCounter,
    overlay_fps,
    overlay_mode_indicator,
    overlay_prediction,
    overlay_message,
    draw_cursor,
    draw_help_text,
)


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "AI Virtual Paint - Hand Gesture Control"


def main():
    """Main application entry point."""

    print("=" * 60)
    print("  AI Virtual Paint App")
    print("  Hand Gesture Control + Handwriting Recognition")
    print("=" * 60)
    print()

    # ── Initialize webcam ──────────────────────────────────────
    print("[INFO] Opening webcam...")
    # Use DirectShow backend on Windows for reliable camera access
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        # Fallback to default backend
        print("[INFO] DirectShow failed, trying default backend...")
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check your camera connection.")
        sys.exit(1)

    # Try to set resolution (camera may not support it — that's OK)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Read one frame to get actual dimensions
    ret, test_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read from webcam.")
        sys.exit(1)

    actual_h, actual_w = test_frame.shape[:2]
    print(f"[INFO] Camera resolution: {actual_w}x{actual_h}")

    # If camera resolution is lower than target, we'll resize frames up
    # so the canvas and toolbar have enough space
    use_resize = (actual_w < FRAME_WIDTH)
    if use_resize:
        print(f"[INFO] Will resize frames to {FRAME_WIDTH}x{FRAME_HEIGHT} for better UI")
        canvas_w, canvas_h = FRAME_WIDTH, FRAME_HEIGHT
    else:
        canvas_w, canvas_h = actual_w, actual_h

    # ── Initialize modules ──────────────────────────────────────
    print("[INFO] Initializing hand tracker...")
    tracker = HandTracker(max_hands=1, detection_conf=0.5, tracking_conf=0.5)

    print("[INFO] Initializing gesture detector...")
    gesture = GestureDetector(pinch_threshold=40, debounce_time=0.6)

    print("[INFO] Initializing drawing canvas...")
    canvas = DrawingCanvas(width=canvas_w, height=canvas_h)

    print("[INFO] Initializing toolbar...")
    toolbar = Toolbar(frame_width=canvas_w)

    # Set initial brush from toolbar defaults
    canvas.set_color(toolbar.get_active_color())
    canvas.set_brush_size(toolbar.get_active_size())

    # ── Load ML model ──────────────────────────────────────────
    print("[INFO] Loading handwriting recognition model...")
    model = load_trained_model()
    if model is None:
        print("[WARNING] Model not loaded. Recognition will be disabled.")
        print("[INFO] Run 'python train_model.py' to train the model first.")
    print()

    # ── State variables ────────────────────────────────────────
    fps_counter = FPSCounter()
    prev_point = None  # Previous drawing position for line interpolation

    # Position smoothing (exponential moving average)
    # Dramatically reduces finger jitter for smooth drawing
    smooth_x, smooth_y = 0.0, 0.0
    smooth_alpha = 0.45  # 0 = max smoothing, 1 = no smoothing
    current_mode = "none"

    # Prediction display state
    prediction_digit = None
    prediction_confidence = 0.0
    prediction_time = None
    show_prediction = False

    # Message display state
    message_text = None
    message_time = None
    show_message = False

    # Toolbar interaction cooldown
    toolbar_cooldown = 0
    TOOLBAR_COOLDOWN_FRAMES = 10

    print("[INFO] Starting main loop... Press 'Q' to quit.")
    print()

    # ── Main loop ──────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Resize frame if camera resolution is lower than target
        if use_resize:
            frame = cv2.resize(frame, (canvas_w, canvas_h))

        # Update FPS
        fps = fps_counter.update()

        # ── Hand tracking ──────────────────────────────────────
        frame = tracker.find_hands(frame, draw=True)
        landmarks = tracker.find_position(frame)

        # Decrease toolbar cooldown
        if toolbar_cooldown > 0:
            toolbar_cooldown -= 1

        # ── Gesture processing ─────────────────────────────────
        if landmarks:
            raw_index_pos = gesture.get_index_finger_pos(landmarks)
            current_gesture = gesture.get_gesture(landmarks)

            # Apply position smoothing to index finger
            if raw_index_pos:
                if prev_point is None:
                    # First point: snap immediately
                    smooth_x, smooth_y = float(raw_index_pos[0]), float(raw_index_pos[1])
                else:
                    # Smooth subsequent points with EMA
                    smooth_x = smooth_alpha * raw_index_pos[0] + (1 - smooth_alpha) * smooth_x
                    smooth_y = smooth_alpha * raw_index_pos[1] + (1 - smooth_alpha) * smooth_y
                index_pos = (int(round(smooth_x)), int(round(smooth_y)))
            else:
                index_pos = None

            # ── DRAWING MODE ───────────────────────────────────
            if current_gesture == "draw":
                current_mode = "draw"

                if index_pos:
                    # Don't draw in the toolbar region
                    if not toolbar.is_in_toolbar(index_pos[1]):
                        canvas.start_stroke()
                        if prev_point is not None:
                            canvas.draw_line(prev_point, index_pos)
                        else:
                            canvas.draw_point(index_pos)
                        prev_point = index_pos
                    else:
                        prev_point = None
                        canvas.end_stroke()

            # ── SELECTION MODE ─────────────────────────────────
            elif current_gesture == "select":
                current_mode = "select"
                prev_point = None
                canvas.end_stroke()

                # Check toolbar interaction
                if index_pos and toolbar.is_in_toolbar(index_pos[1]):
                    if toolbar_cooldown <= 0:
                        action = toolbar.check_click(index_pos[0], index_pos[1])
                        if action:
                            state_changes = toolbar.update_state(action)
                            toolbar_cooldown = TOOLBAR_COOLDOWN_FRAMES

                            # Apply state changes
                            if "color" in state_changes:
                                canvas.set_color(state_changes["color"])

                            if "size" in state_changes:
                                canvas.set_brush_size(state_changes["size"])

                            if "eraser" in state_changes:
                                if state_changes["eraser"]:
                                    canvas.eraser_mode = True
                                else:
                                    canvas.eraser_mode = False

                            if "clear" in state_changes:
                                canvas.clear()
                                message_text = "Canvas Cleared"
                                message_time = time.time()
                                show_message = True

                            if "undo" in state_changes:
                                if canvas.undo():
                                    message_text = "Undo"
                                    message_time = time.time()
                                    show_message = True
                                else:
                                    message_text = "Nothing to undo"
                                    message_time = time.time()
                                    show_message = True

                            if "save" in state_changes:
                                path = canvas.save_image()
                                message_text = f"Saved: {os.path.basename(path)}"
                                message_time = time.time()
                                show_message = True

                            if "recognize" in state_changes:
                                _do_recognition(
                                    model, canvas, frame
                                )
                                # Set prediction display state
                                if model and not canvas.is_empty():
                                    drawing = canvas.get_canvas()
                                    digit, conf = predict_digit(model, drawing)
                                    if digit is not None:
                                        prediction_digit = digit
                                        prediction_confidence = conf
                                        prediction_time = time.time()
                                        show_prediction = True
                                    else:
                                        message_text = "Could not recognize. Draw a digit."
                                        message_time = time.time()
                                        show_message = True
                                elif model is None:
                                    message_text = "Model not loaded! Run train_model.py"
                                    message_time = time.time()
                                    show_message = True
                                else:
                                    message_text = "Canvas is empty!"
                                    message_time = time.time()
                                    show_message = True

            # ── PINCH GESTURE ──────────────────────────────────
            elif current_gesture == "pinch":
                current_mode = "none"
                prev_point = None
                canvas.end_stroke()

                # Toggle eraser on pinch
                canvas.toggle_eraser()
                toolbar.eraser_active = canvas.eraser_mode
                # Update eraser button visual
                for btn in toolbar.buttons:
                    if btn.action == "eraser":
                        btn.active = canvas.eraser_mode

                mode_name = "ERASER ON" if canvas.eraser_mode else "ERASER OFF"
                message_text = mode_name
                message_time = time.time()
                show_message = True

            # ── NO GESTURE ─────────────────────────────────────
            else:
                current_mode = "none"
                prev_point = None
                canvas.end_stroke()

        else:
            # No hand detected
            current_mode = "none"
            prev_point = None
            canvas.end_stroke()

        # ── Merge canvas with frame ────────────────────────────
        frame = canvas.merge_with_frame(frame)

        # ── Draw toolbar ───────────────────────────────────────
        frame = toolbar.draw(frame)

        # ── Draw cursor ────────────────────────────────────────
        if landmarks and current_mode in ("draw", "select"):
            cursor_pos = gesture.get_index_finger_pos(landmarks)
            frame = draw_cursor(
                frame, cursor_pos,
                canvas.brush_color, canvas.brush_size,
                canvas.eraser_mode,
            )

        # ── Overlay UI elements ────────────────────────────────
        frame = overlay_fps(frame, fps)
        frame = overlay_mode_indicator(frame, current_mode, canvas.eraser_mode)
        frame = draw_help_text(frame)

        # Show prediction result
        if show_prediction:
            frame, show_prediction = overlay_prediction(
                frame, prediction_digit, prediction_confidence,
                duration=4.0, start_time=prediction_time,
            )

        # Show temporary message
        if show_message:
            frame, show_message = overlay_message(
                frame, message_text,
                duration=2.0, start_time=message_time,
            )

        # ── Display frame ──────────────────────────────────────
        cv2.imshow(WINDOW_NAME, frame)

        # ── Keyboard input ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            print("[INFO] Quitting...")
            break

        elif key == ord("s") or key == ord("S"):
            path = canvas.save_image()
            message_text = f"Saved: {os.path.basename(path)}"
            message_time = time.time()
            show_message = True
            print(f"[INFO] Drawing saved to: {path}")

        elif key == ord("c") or key == ord("C"):
            canvas.clear()
            message_text = "Canvas Cleared"
            message_time = time.time()
            show_message = True
            print("[INFO] Canvas cleared.")

        elif key == ord("u") or key == ord("U"):
            if canvas.undo():
                message_text = "Undo"
                message_time = time.time()
                show_message = True
                print("[INFO] Undo performed.")
            else:
                message_text = "Nothing to undo"
                message_time = time.time()
                show_message = True

        elif key == ord("r") or key == ord("R"):
            if model and not canvas.is_empty():
                drawing = canvas.get_canvas()
                digit, conf = predict_digit(model, drawing)
                if digit is not None:
                    prediction_digit = digit
                    prediction_confidence = conf
                    prediction_time = time.time()
                    show_prediction = True
                    print(f"[INFO] Predicted: {digit} (confidence: {conf:.1%})")
                else:
                    message_text = "Could not recognize. Try drawing a digit."
                    message_time = time.time()
                    show_message = True
            elif model is None:
                message_text = "Model not loaded! Run train_model.py"
                message_time = time.time()
                show_message = True
            else:
                message_text = "Canvas is empty!"
                message_time = time.time()
                show_message = True

    # ── Cleanup ────────────────────────────────────────────────
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed.")


def _do_recognition(model, canvas, frame):
    """
    Helper to avoid code duplication for recognition.
    The actual prediction logic is handled inline
    since we need to update the main loop's state variables.
    This is a placeholder for any pre-recognition processing.
    """
    pass


if __name__ == "__main__":
    main()
