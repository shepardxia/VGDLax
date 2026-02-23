import jax.numpy as jnp
from vgdl_jax.collision import detect_eos


def test_detect_eos():
    positions = jnp.array([[0, 0], [-1, 3], [5, 5], [2, 10]])
    alive = jnp.array([True, True, True, True])
    oob = detect_eos(positions, alive, height=6, width=6)
    assert oob[0] == False  # in bounds
    assert oob[1] == True   # y < 0
    assert oob[2] == False  # y=5 < 6, x=5 < 6
    assert oob[3] == True   # x=10 >= 6


def test_detect_eos_dead_sprite():
    positions = jnp.array([[-1, -1]])
    alive = jnp.array([False])
    oob = detect_eos(positions, alive, height=5, width=5)
    assert oob[0] == False  # dead, so not OOB
