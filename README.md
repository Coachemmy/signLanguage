## Sign Language Detection Using Computer Vision (OpenCV & cvzone)
## Overview

This project implements a real-time hand gesture recognition system for sign language detection using computer vision techniques. Leveraging OpenCV for image processing and cvzone for hand tracking, the system captures hand gestures from a webcam, preprocesses them, and classifies them using a pretrained Keras model.

Designed for educational and research purposes, this project demonstrates how hand detection, image normalization, and machine learning can work together to interpret hand signs corresponding to alphabet letters. The modular design allows for easy dataset expansion, model retraining, and integration with applications such as sign-to-text translation or assistive technology.

## Features

- Real-time hand tracking: Detects a single hand using cvzone.HandDetector.

- Dynamic cropping & resizing: Extracts hand regions and normalizes them to a fixed input size (300x300 px) while preserving aspect ratio.

- Dataset collection: Easily capture gesture images for training via the train.py script.

- Pretrained classifier integration: Supports Keras .h5 models for gesture recognition.

- Visual feedback: Displays cropped and normalized hand images during both training and prediction.


## Installation

- Clone the repository:

git clone https://github.com/yourusername/SignLangDetector.git
cd SignLangDetector

- Install required dependencies (preferably in a virtual environment):

pip install opencv-python numpy cvzone tensorflow

- Verify your webcam is working, as both training and testing scripts use live feed.


## Usage
1. Capturing Training Data

- Run train.py to collect hand gesture images:

- python train.py

- Press s to save the current normalized hand image (imgWhite) to the dataset folder.

Images are standardized to 300x300 pixels, with aspect ratio preserved.

2. Real-time Gesture Prediction

- Run dataTest.py to perform live gesture classification:

python dataTest.py

Make gestures in front of the camera.

The system detects the hand, crops, normalizes, and classifies the gesture.

Predicted labels appear on the video feed in real time.

## Technical Details
Hand Detection

- Uses cvzone’s HandDetector to locate keypoints and bounding boxes.

- Tracks one hand per frame for optimal performance and accuracy.

Image Preprocessing

- Offset padding: Adds margin around the detected hand for robust capture.

- Aspect ratio preservation: Resizes images to a fixed square while maintaining proportions.

- White background padding: Ensures consistent input for the classifier regardless of hand size or position.

## Classification

- Uses a Keras .h5 model for gesture recognition.

- Labels are loaded from labels.txt.

- Predictions return both probabilities and class index for real-time display.

## Example Output

Training Output:
Displays cropped hand images (imgCrop) and normalized white-background images (imgWhite) ready to save for training.

Prediction Output:
Live webcam feed with hand labeled by predicted class:

Prediction: F

## References

cvzone Documentation

OpenCV Python Tutorials

TensorFlow Keras Guide
