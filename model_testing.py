import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ActivationFunctions import ReLU, Sigmoid, Tanh, Softmax
from CostFunctions import mean_squared_error, cross_entropy_loss
from NeuralNetwork import FullyConnected, Model

# Load MNIST Dataset
def load_mnist():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data / 255.0  # Normalize pixel values to range [0, 1]
    Y = mnist.target.astype(int)  # Labels as integers
    return X, Y

# Preprocess MNIST dataset
def preprocess_mnist(X, Y):
    print("Preprocessing MNIST dataset...")
    # Flatten images (already flattened by fetch_openml)
    X = X.T  # Transpose to shape (features, examples)
    
    # Convert Y to NumPy array
    Y = np.array(Y)

    # One-hot encode labels
    one_hot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
    Y = one_hot_encoder.fit_transform(Y.reshape(-1, 1)).T  # Shape (classes, examples)
    return X, Y

# Split dataset
def split_dataset(X, Y):
    print("Splitting dataset...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X.T, Y.T, test_size=0.2, random_state=42)
    return X_train.T, X_test.T, Y_train.T, Y_test.T

# Test the model on the test set
def test_model(model, X_test, Y_test):
    print("Testing the model...")
    predictions = model.forward(X_test)
    accuracy = np.mean(np.argmax(predictions, axis=0) == np.argmax(Y_test, axis=0))
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Main script
if __name__ == "__main__":
    # Load and preprocess MNIST dataset
    X, Y = load_mnist()
    X, Y = preprocess_mnist(X, Y)
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)
    
    # Define the model
    model = Model([
        FullyConnected(784, 128, "xavier"),  # Fully connected layer from 784 inputs to 128 neurons
        Tanh(),
        FullyConnected(128, 64, "xavier"),  # Fully connected layer from 128 neurons to 64 neurons
        Tanh(),
        FullyConnected(64, 10, "xavier"),   # Fully connected layer from 64 neurons to 10 outputs (classes)
        Softmax()
    ])

    # Train the model
    print("Training the model...")
    model.train(X_train, Y_train, learning_rate=0.1, epochs=20, cost_function=cross_entropy_loss)

    # Test the model
    test_model(model, X_test, Y_test)