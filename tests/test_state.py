import jax
import jax.numpy as jnp
from vgdl_jax.state import GameState, create_initial_state


def test_game_state_creation():
    state = create_initial_state(n_types=3, max_n=5, height=10, width=10)
    assert state.positions.shape == (3, 5, 2)
    assert state.alive.shape == (3, 5)
    assert state.orientations.shape == (3, 5, 2)
    assert state.done == False
    assert state.score == 0


def test_game_state_is_pytree():
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5)
    leaves = jax.tree.leaves(state)
    assert all(isinstance(l, jnp.ndarray) for l in leaves)


def test_game_state_replace():
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5)
    new_state = state.replace(score=jnp.int32(10))
    assert new_state.score == 10
    assert state.score == 0  # original unchanged


def test_game_state_sprite_ops():
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5)
    # Place a sprite at (2, 3) for type 0, slot 0
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2, 3])),
        alive=state.alive.at[0, 0].set(True),
    )
    assert state.alive[0, 0] == True
    assert state.positions[0, 0, 0] == 2
    assert state.positions[0, 0, 1] == 3


def test_game_state_vmappable():
    """GameState should work with vmap."""
    def make_state(rng):
        return create_initial_state(n_types=2, max_n=3, height=5, width=5, rng_key=rng)
    rngs = jax.random.split(jax.random.PRNGKey(0), 4)
    batch = jax.vmap(make_state)(rngs)
    assert batch.positions.shape == (4, 2, 3, 2)
    assert batch.alive.shape == (4, 2, 3)
