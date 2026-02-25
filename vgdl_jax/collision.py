import jax.numpy as jnp


_EOS_EPS = 0.01  # absorb float32 rounding; min movement step is 0.1
AABB_EPS = 1e-3  # tolerance for float drift at cell boundaries


def in_bounds(ipos, height, width):
    """Check which sprites have positions within the grid. Returns [max_n] bool."""
    return (
        (ipos[:, 0] >= 0) & (ipos[:, 0] < height) &
        (ipos[:, 1] >= 0) & (ipos[:, 1] < width)
    )


def detect_eos(positions, alive, height, width):
    """Returns [max_n] bool — which alive sprites are out of bounds."""
    eps = _EOS_EPS
    oob = ((positions[:, 0] < -eps) | (positions[:, 0] > height - 1 + eps) |
           (positions[:, 1] < -eps) | (positions[:, 1] > width - 1 + eps))
    return oob & alive
