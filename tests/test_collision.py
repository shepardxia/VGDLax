import jax.numpy as jnp
from vgdl_jax.collision import detect_collisions, detect_eos


def test_detect_collisions_basic():
    pos_a = jnp.array([[2, 3], [4, 5], [0, 0]])
    alive_a = jnp.array([True, True, False])
    pos_b = jnp.array([[2, 3], [1, 1]])
    alive_b = jnp.array([True, True])
    coll = detect_collisions(pos_a, alive_a, pos_b, alive_b)
    assert coll.shape == (3, 2)
    assert coll[0, 0] == True   # same position, both alive
    assert coll[1, 0] == False  # different position
    assert coll[2, 0] == False  # sprite a is dead


def test_detect_collisions_self_overlap():
    pos_a = jnp.array([[2, 3], [2, 3]])
    alive_a = jnp.array([True, True])
    pos_b = pos_a
    alive_b = alive_a
    coll = detect_collisions(pos_a, alive_a, pos_b, alive_b)
    assert coll[0, 0] == True
    assert coll[0, 1] == True
    assert coll[1, 0] == True
    assert coll[1, 1] == True


def test_detect_collisions_no_alive():
    pos_a = jnp.array([[2, 3]])
    alive_a = jnp.array([False])
    pos_b = jnp.array([[2, 3]])
    alive_b = jnp.array([True])
    coll = detect_collisions(pos_a, alive_a, pos_b, alive_b)
    assert coll[0, 0] == False


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
