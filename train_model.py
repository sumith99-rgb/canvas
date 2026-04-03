"""
train_model.py — Standalone MNIST Model Training Script

Run this script to train the CNN model on the MNIST dataset
and save it for use by the virtual paint application.

Usage:
    python train_model.py

The trained model will be saved to: models/mnist_cnn.keras
"""

from model import train_model


def main():
    print("=" * 60)
    print("  MNIST CNN Model Training")
    print("=" * 60)
    print()

    # Train the model (default: 10 epochs)
    model = train_model(epochs=10, batch_size=128)

    print()
    print("=" * 60)
    print("  Training complete!")
    print("  Model saved to: models/mnist_cnn.keras")
    print("  You can now run: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
