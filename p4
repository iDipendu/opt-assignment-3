import numpy as np
import matplotlib.pyplot as plt

# Replace this with your own dataset
x = np.array([5.1, 4.8, 5.0, 5.3, 5.2, 5.5, 4.9])  # Example data

# Negative Log-Likelihood
def f(mu, sigma):
    return np.sum(np.log(sigma) + ((x - mu)**2) / (2 * sigma**2))

# Gradient
def grad_f(mu, sigma):
    n = len(x)
    d_mu = np.sum((mu - x) / (sigma**2))
    d_sigma = np.sum(1 / sigma - ((x - mu)**2) / (sigma**3))
    return np.array([d_mu, d_sigma])

# Parameters
mu, sigma = 0.0, 1.0  # Initial values
eta = 0.01
epsilon = 1e-5
max_iter = 10000

loss_history = []
trajectory = [(mu, sigma]()
