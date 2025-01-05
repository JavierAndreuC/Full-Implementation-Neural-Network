import numpy as np

class ReLU:
    def forward_pass(self, Z):
        self.Z = Z
        self.A = np.maximum(0, Z)
        return self.A
    
    def backward_pass(self, dA, lr):
        return dA * (self.Z > 0)  # Gradient of ReLU
        

class Sigmoid:
    def forward_pass(self, Z):
        self.Z = Z
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
    
    def backward_pass(self, dA, lr):
        return dA * (self.A * (1 - self.A))  # Gradient of Sigmoid


class Tanh:
    def forward_pass(self, Z):
        self.Z = Z
        self.A = np.tanh(Z)
        return self.A
    
    def backward_pass(self, dA, lr):
        return dA * (1 - self.A ** 2)  # Gradient of Tanh
    
class Softmax:
    def forward_pass(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Avoid overflow
        self.A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return self.A
    
    def backward_pass(self, dA, lr):
        # Gradient calculation for Softmax is done combined with cross-entropy loss.
        return dA