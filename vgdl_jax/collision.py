import jax.numpy as jnp


def detect_eos(positions, alive, height, width):
    """Returns [max_n] bool — which alive sprites are out of bounds."""
    oob = ((positions[:, 0] < 0) | (positions[:, 0] >= height) |
           (positions[:, 1] < 0) | (positions[:, 1] >= width))
    return oob & alive
