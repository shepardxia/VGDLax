import jax.numpy as jnp


_EOS_EPS = 0.01  # absorb float32 rounding; min movement step is 0.1


def detect_eos(positions, alive, height, width):
    """Returns [max_n] bool — which alive sprites are out of bounds."""
    eps = _EOS_EPS
    oob = ((positions[:, 0] < -eps) | (positions[:, 0] > height - 1 + eps) |
           (positions[:, 1] < -eps) | (positions[:, 1] > width - 1 + eps))
    return oob & alive
