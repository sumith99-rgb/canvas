"""
model.py — Handwriting Recognition Model

Provides functions to:
  - Build a CNN model for MNIST digit recognition
  - Train the model on the MNIST dataset
  - Load a pre-trained model
  - Preprocess canvas drawings for prediction
  - Predict digits from drawings
"""

import os
import numpy as np
import cv2

# Suppress TensorFlow info/warning logs for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Path to the saved model
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_cnn.keras")


def build_model():
    """
    Build a CNN model for MNIST digit recognition.

    Architecture:
        Conv2D(32, 3x3) → ReLU → MaxPool(2x2)
        Conv2D(64, 3x3) → ReLU → MaxPool(2x2)
        Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(10, softmax)

    Returns:
        keras.Model: Compiled CNN model.
    """
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # First convolutional block
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Dense classification head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(epochs=10, batch_size=128):
    """
    Train the CNN on the MNIST dataset and save the model.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.

    Returns:
        keras.Model: The trained model.
    """
    print("[INFO] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess: normalize to [0, 1] and reshape to (N, 28, 28, 1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"[INFO] Training data: {x_train.shape}")
    print(f"[INFO] Test data: {x_test.shape}")

    # Build and train
    model = build_model()
    model.summary()

    print(f"\n[INFO] Training for {epochs} epochs...")
    model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    # Evaluate
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[INFO] Test accuracy: {accuracy:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[INFO] Model saved to: {MODEL_PATH}")

    return model


def load_trained_model():
    """
    Load the pre-trained MNIST model from disk.

    Returns:
        keras.Model or None: The loaded model, or None if not found.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model not found at {MODEL_PATH}")
        print("[INFO] Please run 'python train_model.py' first to train the model.")
        return None

    print(f"[INFO] Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
    return model


def preprocess_for_prediction(canvas_image):
    """
    Preprocess a canvas drawing for MNIST prediction.

    Steps:
        1. Convert to grayscale
        2. Find the bounding box of drawn content
        3. Crop to the drawn region with padding
        4. Make the crop square (preserve aspect ratio)
        5. Resize to 28x28
        6. Normalize pixel values to [0, 1]
        7. Reshape to (1, 28, 28, 1)

    Args:
        canvas_image: BGR image from the drawing canvas.

    Returns:
        numpy.ndarray or None: Preprocessed image ready for prediction,
                                or None if the image is empty.
    """
    if canvas_image is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)

    # Check if image is empty
    if not np.any(gray):
        return None

    # Find bounding box of non-zero pixels
    coords = cv2.findNonZero(gray)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    # Add padding around the digit
    pad = max(w, h) // 4
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(gray.shape[1] - x, w + 2 * pad)
    h = min(gray.shape[0] - y, h + 2 * pad)

    # Crop to bounding box
    cropped = gray[y : y + h, x : x + w]

    # Make square by padding the shorter dimension
    max_dim = max(w, h)
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    x_offset = (max_dim - w) // 2
    y_offset = (max_dim - h) // 2
    square[y_offset : y_offset + h, x_offset : x_offset + w] = cropped

    # Resize to 28x28 (MNIST standard size)
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    normalized = resized.astype("float32") / 255.0

    # Reshape for model input: (1, 28, 28, 1)
    processed = normalized.reshape(1, 28, 28, 1)

    return processed


def predict_digit(model, canvas_image):
    """
    Predict the digit drawn on the canvas.

    Args:
        model: Trained Keras model.
        canvas_image: BGR image from the drawing canvas (full or cropped).

    Returns:
        tuple: (predicted_digit, confidence) or (None, 0.0) if prediction fails.
    """
    if model is None:
        return None, 0.0

    processed = preprocess_for_prediction(canvas_image)
    if processed is None:
        return None, 0.0

    # Run prediction
    predictions = model.predict(processed, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])

    return int(predicted_class), confidence
