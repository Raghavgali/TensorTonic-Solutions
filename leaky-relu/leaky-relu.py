import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x = np.array(x, dtype=float)

    v_func = np.vectorize(lambda x: x if x >= 0 else alpha * x)

    return v_func(x)