import numpy as np

from ActivationFunctions import ReLU, Sigmoid, Tanh
from CostFunctions import mean_squared_error, cross_entropy_loss

class FullyConnected:        
    def __init__(self, input_size, output_size, initialization="he"):
        if initialization == "he":
            self.W = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        elif initialization == "xavier":
            self.W = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)
        self.b = np.zeros((output_size, 1))
        self.b = np.zeros((output_size, 1))

    def forward_pass(self, X):
        self.X = X
        self.Z = np.dot(self.W, X) + self.b
        return self.Z
    
    def backward_pass(self, dZ, learning_rate):
        m = self.X.shape[1]
        dW = (1 / m) * np.dot(dZ, self.X.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dA_prev    

class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def train(self, X, Y, learning_rate, epochs, cost_function):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            cost = cost_function(output, Y)
            print(f"Epoch {epoch + 1}, Cost: {cost}")

            # Backward pass
            dA = output - Y
            for layer in reversed(self.layers):
                dA = layer.backward_pass(dA, learning_rate)


