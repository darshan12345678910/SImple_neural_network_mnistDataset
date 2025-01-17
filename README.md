# Simple Neural Network - MNIST Dataset

This project demonstrates how to build and compare two neural network models for classifying digits from the MNIST dataset. The project involves:

- Loading and preprocessing the MNIST dataset.
- Building two models: a simple Convolutional Neural Network (CNN) and a custom fully connected neural network (with dense layers).
- Training both models for 5 epochs.
- Visualizing training and validation accuracy over epochs.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup](#setup)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Models](#training-the-models)
- [Plotting Results](#plotting-results)
- [License](#license)

## Project Overview

This project compares two models for image classification using the MNIST dataset:
1. **Keras CNN Model**: A simple Convolutional Neural Network with two convolutional layers and fully connected layers.
2. **Custom Fully Connected Model**: A custom-defined model with three dense layers.

Both models are trained for 5 epochs, and their training and validation accuracies are plotted to compare their performance.

## Setup

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/simple_neural_network_mnistDataset.git
2. cd simple_neural_network_mnistDataset
   ```bash
   cd simple_neural_network_mnistDataset
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Dataset
The dataset used in this project is the MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0-9). It is widely used for benchmarking machine learning models in classification tasks.

The MNIST dataset is automatically downloaded and loaded using keras.datasets.mnist in the code.
## Preprocessing
The dataset is normalized to have pixel values between 0 and 1.
The images are reshaped to fit the input shape expected by the models.

## Model-Architecture
# Custom Neural Network (from scratch)
In the custom model:
- It uses a simple architecture with one hidden layer and applies forward and backward passes to update weights using gradient descent.
This section outlines the key concepts and parameters involved in the custom neural network used for digit classification on the MNIST dataset.

## Activation Functions

1. **Sigmoid**:
   - Used in the hidden layer.
   - Provides non-linear transformations, allowing the model to learn complex patterns.
   
2. **Softmax**:
   - Used in the output layer.
   - Converts the raw output into probabilities, ensuring that the sum of probabilities equals 1.
   - Suitable for multi-class classification tasks like MNIST digit classification.

## Loss Function

- **Cross-Entropy Loss**:
  - Measures the difference between the true labels and predicted probabilities.
  - Lower values of cross-entropy loss indicate better performance, meaning the model's predictions are close to the actual labels.

## Custom Neural Network

1. **Weights and Biases**:
   - Weights and biases are initialized randomly with small values (multiplied by `0.01`).
   - The model starts with random weights, which are adjusted during training to minimize the loss function.

2. **Forward Pass**:
   - The forward pass calculates activations for each layer, starting from the input layer and progressing to the output layer.
   - The output of the hidden layer is passed through a **sigmoid** activation function, while the output layer uses a **softmax** activation.

3. **Backward Pass**:
   - In the backward pass, gradients of weights and biases are calculated using the **chain rule** of differentiation.
   - The gradients are used to update the weights and biases via **gradient descent**, minimizing the loss function.

## Training

- The model is trained for **1000 epochs**.
- The loss is printed every **100 epochs** to track the training progress and to monitor overfitting.

## Prediction and Evaluation

1. **Prediction**:
   - After training, the model makes predictions on the test images using the learned weights and biases.
   
2. **Evaluation**:
   - The predicted classes are compared to the true labels to compute the **accuracy** of the model.

## Key Parameters

- **Input Size**: 784 (28x28 image flattened into a 1D vector).
- **Hidden Size**: 64 neurons in the hidden layer.
- **Output Size**: 10 classes (digits 0-9).
- **Epochs**: 1000 iterations of training.
- **Learning Rate**: 0.1 for weight updates.
- **Subset for Training**: Only the first 10,000 training samples are used to speed up the training process.

By using this custom neural network, we aim to train a model that effectively classifies the MNIST digits using the principles of forward propagation, backpropagation, and gradient descent.

# Keras Neural Network (using TensorFlow/Keras)

This section outlines the process of building, training, and evaluating a simple neural network model using Keras, and comparing it with a custom gradient descent implementation.

## Keras Model Overview

- The Keras model consists of a simple feedforward neural network with two hidden layers.
- The input images are flattened into a 784-dimensional vector (28x28 pixels) and passed through dense layers.
- The model uses the following configurations:
  - **Optimizer**: Adam optimizer
  - **Loss Function**: Categorical Crossentropy
  - **Metric**: Accuracy
- The model is trained for **5 epochs**.

## Data Preprocessing

1. **Normalization**:
   - The pixel values of the images are scaled to the range [0, 1] by dividing each pixel value by 255.

2. **One-Hot Encoding**:
   - The labels are converted to one-hot encoded vectors using the `to_categorical` function from Keras. This prepares the labels for categorical crossentropy loss.

3. **Train-Test Split**:
   - The training data is split into training and validation sets using `train_test_split` from the `sklearn.model_selection` module.

## Model Training

1. **Custom Model**:
   - The custom model is trained using a custom gradient descent implementation with manual backpropagation.

2. **Keras Model**:
   - The Keras model is trained using the **Adam optimizer** for 5 epochs.

## Model Evaluation

- After training, both models (custom and Keras) are evaluated on the test set.
- The **test loss** and **test accuracy** are displayed to gauge the model’s performance.

## Model Saving

- After training, the Keras model is saved in **HDF5 format** (`.h5`) for later use.
  - The model will be saved as `mnist_simple_nn.h5`.

## Expected Output

During training, you will see the following output for both models:
