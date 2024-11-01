import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def generate_hf_2d_data(num_points: int, noise_std: float = 1):
    # Generate random x values
    x_values = np.linspace(0.3, 1, num_points)
    print(x_values[0:min(num_points,30)])
    inv_x_values = 1 / x_values
    y_values = np.sin(inv_x_values) * np.exp(inv_x_values) + np.random.normal(0, noise_std, num_points)
    return x_values, y_values

def generate_noised_data(f, num_points: int, start: float, stop: float, noise_std: float = 1):
    x_values, y_values = generate(f, num_points, start, stop)
    y_values += np.random.normal(0, noise_std, num_points)
    return x_values, y_values

def generate(f, num_points: int, start: float, stop: float):
    x_values = np.linspace(start, stop, num_points)
    print(x_values[0:min(num_points,30)])
    y_values = f(x_values)
    return x_values, y_values
