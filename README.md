# 🎨 AI Virtual Paint App — Hand Gesture Control + Handwriting Recognition

An AI-powered virtual paint application that lets you draw on screen using hand gestures captured through your webcam. Includes a trained CNN model for recognizing handwritten digits.

---

## ✨ Features

- **Hand Gesture Drawing** — Draw with your index finger tracked via MediaPipe
- **Gesture Controls** — Switch modes, toggle eraser, and interact with the toolbar using hand gestures
- **Color Palette** — 6 vibrant colors (Red, Blue, Green, Yellow, White, Purple)
- **Brush Sizes** — Small, Medium, Large brush options
- **Eraser Mode** — Toggle eraser with pinch gesture or toolbar button
- **Undo Support** — Undo last stroke with keyboard or toolbar
- **Save Drawings** — Export your artwork as PNG images
- **Handwriting Recognition** — CNN model recognizes drawn digits (0-9) in real-time
- **Real-time FPS** — Optimized for smooth performance

---

## 📁 Project Structure

```
hd/
├── main.py             # Main application loop
├── hand_tracking.py    # MediaPipe hand tracking module
├── gesture.py          # Gesture detection logic
├── draw_utils.py       # Drawing canvas engine
├── ui_toolbar.py       # On-screen toolbar UI
├── model.py            # CNN model definition + prediction
├── train_model.py      # Standalone model training script
├── utils.py            # Helper utility functions
├── requirements.txt    # Python dependencies
├── models/             # Trained model storage
│   └── mnist_cnn.keras # Trained MNIST CNN model
├── saved_drawings/     # Saved drawing images
└── README.md           # This file
```

---

## 🚀 Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Handwriting Recognition Model

```bash
python train_model.py
```

This will:
- Download the MNIST dataset (auto-handled by TensorFlow)
- Train a CNN for 10 epochs
- Save the model to `models/mnist_cnn.keras`
- Typically achieves ~99% accuracy

### 3. Run the Application

```bash
python main.py
```

---

## 🎮 Controls

### Hand Gestures

| Gesture | Action |
|---|---|
| ☝️ Index finger up (only) | **Drawing mode** — draw on canvas |
| ✌️ Index + middle fingers up | **Selection mode** — interact with toolbar |
| 🤏 Pinch (thumb + index close) | **Toggle eraser** on/off |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `S` | Save current drawing as PNG |
| `C` | Clear canvas |
| `U` | Undo last stroke |
| `R` | Recognize handwritten digit |

### Toolbar Buttons

The toolbar at the top of the screen provides:
- **Color buttons** — Red, Blue, Green, Yellow, White, Purple
- **Size buttons** — S (Small), M (Medium), L (Large)
- **Eraser** — Toggle eraser mode
- **Clear** — Clear entire canvas
- **Undo** — Undo last stroke
- **Save** — Save drawing to file
- **Recognize** — Run handwriting recognition on current drawing

---

## 🧠 Handwriting Recognition

The app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits (0-9).

### How to Use:
1. Draw a digit on the canvas
2. Press `R` or click the "Recognize" button on the toolbar
3. The predicted digit and confidence score will appear on screen

### Model Architecture:
```
Conv2D(32) → BatchNorm → MaxPool
Conv2D(64) → BatchNorm → MaxPool
Conv2D(128) → BatchNorm → MaxPool
Flatten → Dense(128) → Dropout(0.5) → Dense(10, softmax)
```

---

## 📋 Requirements

- Python 3.8+
- Webcam / built-in camera
- Windows / macOS / Linux

### Python Packages:
- `opencv-python >= 4.8.0`
- `mediapipe >= 0.10.0`
- `numpy >= 1.24.0`
- `tensorflow >= 2.15.0`

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| Webcam not detected | Check camera connection. Try changing `CAMERA_INDEX` in `main.py` |
| Low FPS | Close other camera-using apps. Reduce resolution in `main.py` |
| Model not found | Run `python train_model.py` first |
| MediaPipe errors | Ensure `mediapipe >= 0.10.0` is installed |
| Hand not detected | Ensure good lighting and keep hand within camera view |

---

## 📝 License

This project is for educational purposes. Feel free to modify and extend it!
