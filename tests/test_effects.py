import jax.numpy as jnp
from vgdl_jax.state import create_initial_state
from vgdl_jax.effects import (
    apply_kill_sprite, apply_kill_both, apply_step_back,
    apply_transform_to, apply_turn_around, apply_reverse_direction,
)


def _make_state_with_sprites():
    """Two types, max 3 each, on a 5x5 grid."""
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5)
    # Type 0, slot 0: alive at (2, 3)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2, 3])),
        alive=state.alive.at[0, 0].set(True),
        orientations=state.orientations.at[0, 0].set(jnp.array([0., 1.])),
    )
    # Type 1, slot 0: alive at (2, 3)
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2, 3])),
        alive=state.alive.at[1, 0].set(True),
    )
    return state


def test_kill_sprite():
    state = _make_state_with_sprites()
    state = apply_kill_sprite(state, 0, 0, score_change=2)
    assert state.alive[0, 0] == False
    assert state.score == 2


def test_kill_sprite_no_score():
    state = _make_state_with_sprites()
    state = apply_kill_sprite(state, 0, 0, score_change=0)
    assert state.alive[0, 0] == False
    assert state.score == 0


def test_kill_both():
    state = _make_state_with_sprites()
    state = apply_kill_both(state, 0, 0, 1, 0, score_change=-1)
    assert state.alive[0, 0] == False
    assert state.alive[1, 0] == False
    assert state.score == -1


def test_step_back():
    state = _make_state_with_sprites()
    prev_positions = state.positions  # save at (2,3)
    # Move type 0 slot 0 to (3, 3)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([3, 3])))
    state = apply_step_back(state, prev_positions, 0, 0)
    assert jnp.array_equal(state.positions[0, 0], jnp.array([2, 3]))


def test_transform_to():
    state = _make_state_with_sprites()
    # Transform type 0 slot 0 into type 1
    state = apply_transform_to(state, 0, 0, new_type_idx=1)
    assert state.alive[0, 0] == False  # old sprite dead
    # New sprite created in type 1 — slot 1 (slot 0 already taken)
    assert state.alive[1, 1] == True
    assert jnp.array_equal(state.positions[1, 1], jnp.array([2, 3]))
    assert jnp.array_equal(state.orientations[1, 1], jnp.array([0., 1.]))


def test_transform_to_empty_type():
    """Transform into a type that has no existing sprites."""
    state = create_initial_state(n_types=3, max_n=3, height=5, width=5)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([1, 1])),
        alive=state.alive.at[0, 0].set(True),
        orientations=state.orientations.at[0, 0].set(jnp.array([1., 0.])),
    )
    state = apply_transform_to(state, 0, 0, new_type_idx=2)
    assert state.alive[0, 0] == False
    assert state.alive[2, 0] == True
    assert jnp.array_equal(state.positions[2, 0], jnp.array([1, 1]))


def test_turn_around():
    state = _make_state_with_sprites()
    state = apply_turn_around(state, 0, 0)
    assert jnp.array_equal(state.orientations[0, 0], jnp.array([0., -1.]))


def test_reverse_direction():
    state = _make_state_with_sprites()
    state = apply_reverse_direction(state, 0, 0)
    assert jnp.array_equal(state.orientations[0, 0], jnp.array([0., -1.]))
