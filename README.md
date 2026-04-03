# 🎨 AI Virtual Paint App — Hand Gesture Control + Handwriting Recognition

An advanced, AI-powered virtual paint application that lets you draw on your screen using hand gestures captured through your webcam. It features highly responsive drawing, real-time gesture recognition, and a custom Convolutional Neural Network (CNN) for recognizing handwritten digits.

---

## ✨ Features

- **Hand Gesture Drawing** — Draw continuously with your index finger, tracked via MediaPipe Tasks API.
- **Gesture Controls** — Switch modes, toggle the eraser, and interact with the toolbar using intuitive hand gestures.
- **Color Palette & Brushes** — 6 vibrant colors (Red, Blue, Green, Yellow, White, Purple) and 3 brush sizes (Small, Medium, Large).
- **Smooth Anti-Aliased Lines** — Position smoothing and OpenCV anti-aliasing ensures professional, jitter-free lines.
- **Undo & Clear Functionality** — Easily undo your last stroke or clear the entire canvas.
- **Save Drawings** — Export your artwork as PNG images.
- **Handwriting Recognition** — Built-in TensorFlow/Keras CNN model recognizes drawn digits (0-9) in real-time.
- **Real-time FPS & Robust Backend** — Optimized for smooth performance with a reliable DirectShow camera backend for Windows.

---

## 📁 Project Structure

```text
hd/
├── main.py             # Main application loop, camera handling, and integration
├── hand_tracking.py    # MediaPipe HandLandmarker Tasks API module
├── gesture.py          # Gesture detection logic (hysteresis, debouncing, pinch calculations)
├── draw_utils.py       # Drawing canvas engine (anti-aliasing, strokes, rendering)
├── ui_toolbar.py       # On-screen toolbar UI rendering and interaction logic
├── model.py            # CNN model definition + prediction logic
├── train_model.py      # Standalone training script for the digit recognizer
├── utils.py            # Helper utility functions
├── test_camera.py      # Camera diagnostic tool for debugging video feeds
├── requirements.txt    # Python dependencies
├── models/             # Directory for trained ML models
│   └── mnist_cnn.keras # Saved MNIST CNN model (Auto-generated after training)
├── saved_drawings/     # Saved user drawing images (.png)
└── README.md           # Project documentation
```

---

## 🚀 Setup & Installation

### 1. Install Dependencies

Ensure you have Python 3.8+ installed. Run the following command to install the required libraries:

```bash
pip install opencv-python mediapipe numpy tensorflow
```

### 2. Train the Handwriting Recognition Model

Before using the recognition feature, you must train the CNN model:

```bash
python train_model.py
```

This will automatically:
- Download the MNIST digits dataset.
- Train a CNN for 10 epochs.
- Save the trained model to `models/mnist_cnn.keras` (achieves ~99% accuracy).

### 3. Run the Application

```bash
python main.py
```

*Note: On its very first run, the app will automatically download a ~10MB `hand_landmarker.task` file required for MediaPipe.*

---

## 🎮 How to Use (Controls & Gestures)

### ✋ Hand Gestures

| Gesture | How to execute | Action |
|---|---|---|
| **Draw** | ☝️ Raise **only your index finger**. Keep other fingers closed. | Draw on the canvas. |
| **Select** | ✌️ Raise **index + middle fingers**. Keep others closed. | Move your cursor to hover over toolbar buttons. |
| **Pinch** | 🤏 Touch your **thumb tip to your index tip**. | Toggle the eraser on/off. |

### ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `S` | Save current drawing as a PNG image in the `saved_drawings/` folder |
| `C` | Clear the entire canvas |
| `U` | Undo the last drawn stroke |
| `R` | Run handwriting recognition on the canvas |

### 🎨 Toolbar Buttons (Use "Select" mode ✌️ to click)

Hover over the buttons at the top of the window while in Select Mode to activate them:
- **Colors**: Red, Blue, Green, Yellow, White, Purple
- **Brush Sizes**: S (Small), M (Medium), L (Large)
- **Tools**: Eraser, Clear Canvas, Undo Stroke, Save Image, Recognize Digit

---

## 🧠 Under the Hood: Accuracy & Tech Details

This application implements several advanced techniques to ensure a premium user experience:

1. **Jitter-Free Drawing:** Exponential Moving Average (EMA) smoothing is applied to the index finger coordinates, completely eliminating camera jitter and resulting in beautifully smooth curves.
2. **Hysteresis Finger Tracking:** To prevent the app from rapidly fluttering between "Drawing" and "Selection" modes, a 15-pixel hysteresis margin is applied to finger joint tracking. 
3. **Gesture Stability Buffer:** Gestures require 3 consecutive frames of validation before the app registers a mode change.
4. **Anti-Aliasing:** The canvas uses `cv2.LINE_AA` and circular end caps, making brush strokes look professional and eliminating jagged pixels.
5. **DirectShow Camera Backend:** Prioritizes `cv2.CAP_DSHOW` on Windows machines mapping to better camera frame-rates, falling back to MSMF/default if needed.

### Handwriting Recognition Model Architecture
```text
Conv2D(32) → BatchNorm → MaxPool → Conv2D(64) → BatchNorm → MaxPool → Conv2D(128) → BatchNorm → MaxPool → Flatten → Dense(128) → Dropout(0.5) → Dense(10, softmax)
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| **Gray screen / Lock Icon** | Windows camera privacy issue. Go to Windows Settings → Privacy & Security → Camera and enable "Let desktop apps access your camera". Make sure no other apps (Zoom/Teams) are using it. |
| **Model Recognition Fails** | Run `python train_model.py` to generate the `.keras` model file. |
| **Lag / Low FPS** | Ensure you are in a well-lit room. MediaPipe requires good lighting to process frames efficiently. |
| **Webcam won't open** | Run `python test_camera.py` to diagnose camera indices and backend compatibility. |

---

## 📝 License
This project was developed for educational purposes. Feel free to fork, modify, and expand upon it!
