import jax.numpy as jnp


def detect_collisions(pos_a, alive_a, pos_b, alive_b):
    """Pairwise collision detection between two groups.
    Returns [len_a, len_b] bool collision matrix."""
    same_pos = jnp.all(pos_a[:, None] == pos_b[None, :], axis=-1)
    both_alive = alive_a[:, None] & alive_b[None, :]
    return same_pos & both_alive


def detect_eos(positions, alive, height, width):
    """Returns [max_n] bool — which alive sprites are out of bounds."""
    oob = ((positions[:, 0] < 0) | (positions[:, 0] >= height) |
           (positions[:, 1] < 0) | (positions[:, 1] >= width))
    return oob & alive
