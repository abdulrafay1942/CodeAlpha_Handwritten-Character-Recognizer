

# Handwritten Digit Recognition using CNN (MNIST)

## Overview

This project focuses on recognizing handwritten digits (0–9) using a Convolutional Neural Network (CNN).
The model is trained on the MNIST dataset, which contains grayscale images of handwritten numbers.

The main goal of this project was to understand how CNNs learn visual patterns and how they can be applied to image classification tasks.

---

## What This Project Does

* Loads handwritten digit images from the MNIST dataset
* Preprocesses images by normalizing and reshaping them
* Trains a CNN to learn digit patterns such as edges and curves
* Evaluates the model on unseen test data
* Verifies predictions on sample images

---

## Dataset

* **Name:** MNIST
* **Type:** Handwritten digits (0–9)
* **Image size:** 28 × 28 pixels
* **Total images:** 70,000

  * 60,000 training images
  * 10,000 testing images

The dataset is loaded locally from an `mnist.npz` file.

---

## Model Architecture

The CNN used in this project consists of:

* Convolutional layers to extract visual features
* Max pooling layers to reduce spatial dimensions
* Fully connected (dense) layers for classification
* Softmax output layer to predict digit probabilities

This architecture is simple but effective for handwritten digit recognition.

---

## Training Details

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Evaluation Metric:** Accuracy
* **Early Stopping:** Used to stop training when validation performance stops improving

---

## Results

The model achieves high accuracy on the test dataset (approximately **97–99%**).
Sample predictions were manually verified to confirm that the model outputs match the correct labels.

---

## Technologies Used

* Python
* NumPy
* TensorFlow / Keras

---

## How to Run

1. Ensure Python and required libraries are installed
2. Place `mnist.npz` in the project’s working directory
3. Run the training script
4. Observe training progress, accuracy, and sample predictions

---

## Future Improvements

* Extend the model to recognize handwritten letters using the EMNIST dataset
* Apply sequence models (e.g., CRNN) for word or sentence recognition
* Experiment with deeper or more complex CNN architectures

---

## Conclusion

This project demonstrates how convolutional neural networks can be used to recognize handwritten digits with high accuracy.
It serves as a strong foundation for exploring more advanced handwriting recognition and computer vision tasks.

