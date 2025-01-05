import numpy as np

def mean_squared_error(output, target):
    """
    Calculate the Mean Squared Error (MSE) cost.
    :param output: Predicted values, NumPy array of shape (m, 1).
    :param target: True values, NumPy array of shape (m, 1).
    :return: Scalar MSE value.
    """
    m = len(output)
    result = (1 / m) * np.sum((output - target) ** 2)
    return result

def cross_entropy_loss(output, target):
    m = target.shape[1]  # Number of examples
    cost = -np.sum(target * np.log(output + 1e-8)) / m
    return cost