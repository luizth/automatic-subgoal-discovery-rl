import os
import numpy as np
import torch


def save_linear_model(w, filename):
    """Save the linear model weights to a file."""
    np.save("./models/" + filename, w)


def load_linear_model(filename):
    """Load the linear model weights from a file."""
    return np.load("./models/" + filename)


def load_linear_model_from_path(path, filename):
    """Load the linear model weights from a file."""
    return np.load(path + filename)


def save_nparray(dir, filename, nparray):
    """Save the option model to a file."""
    path = f"./models/{dir}/"
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + filename, nparray)


def save_option(dir, filename, option):
    """Save the option model to a file."""
    path = f"./models/{dir}/"
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + filename, {"w": option.w, "theta": option.theta, "rw": option.rw, "W": option.W})


def load_option(filename):
    """Load the option model from a file."""
    data = np.load("./models/" + filename + ".npy", allow_pickle=True).item()
    return data.get("w", None), data.get("theta", None), data.get("rw", None), data.get("W", None)


def value_function(state_features, w):
    """Linear value function approximation"""
    return np.dot(state_features, w)


def delta_function(c, z, v, v_, B, gamma=0.99):
    """Temporal Difference error (TD Error)"""
    return c + B * z + gamma * (1 - B) * v_ - v


def UWT(w, e, gradient, alpha_delta, rho, gamma_lambda):
    """UpdateWeights&Traces procedure"""
    e = rho * (e + gradient)
    w = w + alpha_delta * e
    e = gamma_lambda * e
    return w, e


def one_hot(x, num_classes):
    """One-hot encoding"""
    return np.eye(num_classes)[x]


def softmax(env, state, theta):
    """Softmax policy"""
    action_preferences = np.zeros(env.observation_space.n)
    for a in range(env.action_space.n):
        features = env.state_action_to_features(state, a)
        action_preferences[a] = np.dot(theta, features)  # state-action value
    # Numerical stability
    max_value = np.max(action_preferences)
    action_preferences -= max_value
    # Softmax
    exp_values = np.exp(action_preferences)
    probs = exp_values / np.sum(exp_values)
    return probs


def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error between predicted and actual values

    Parameters:
    y_true (array-like): Ground truth (correct) target values
    y_pred (array-like): Estimated target values

    Returns:
    float: Root Mean Squared Error
    """
    # Convert inputs to numpy arrays for consistent handling
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate squared error for each prediction
    squared_errors = (y_true - y_pred) ** 2

    # Calculate mean of squared errors
    mean_squared_error = np.mean(squared_errors)

    # Take square root to get RMSE
    root_mean_squared_error = np.sqrt(mean_squared_error)

    return root_mean_squared_error


def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs
