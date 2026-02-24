import jax.numpy as jnp
from vgdl_jax.state import create_initial_state
from vgdl_jax.terminations import (
    check_sprite_counter, check_multi_sprite_counter,
    check_timeout, check_resource_counter, check_all_terminations,
)


def test_sprite_counter_triggers():
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5)
    ended, win = check_sprite_counter(state, type_indices=[0], limit=0, win=False)
    assert ended == True
    assert win == False


def test_sprite_counter_not_triggered():
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5)
    state = state.replace(alive=state.alive.at[0, 0].set(True))
    ended, win = check_sprite_counter(state, type_indices=[0], limit=0, win=True)
    assert ended == False


def test_sprite_counter_multiple_types():
    """stype resolves to multiple concrete types."""
    state = create_initial_state(n_types=3, max_n=3, height=5, width=5)
    state = state.replace(alive=state.alive.at[1, 0].set(True))
    # Types 1 and 2: 1 alive total, limit=0 → not triggered
    ended, win = check_sprite_counter(state, type_indices=[1, 2], limit=0, win=True)
    assert ended == False


def test_multi_sprite_counter():
    state = create_initial_state(n_types=3, max_n=3, height=5, width=5)
    # Types 1 and 2 both have 0 alive sprites → sum=0, limit=0
    ended, win = check_multi_sprite_counter(
        state, type_indices_list=[[1], [2]], limit=0, win=True)
    assert ended == True


def test_multi_sprite_counter_not_triggered():
    state = create_initial_state(n_types=3, max_n=3, height=5, width=5)
    state = state.replace(alive=state.alive.at[1, 0].set(True))
    ended, win = check_multi_sprite_counter(
        state, type_indices_list=[[1], [2]], limit=0, win=True)
    assert ended == False


def test_timeout():
    state = create_initial_state(n_types=1, max_n=1, height=5, width=5)
    state = state.replace(step_count=jnp.int32(100))
    ended, win = check_timeout(state, limit=100, win=False)
    assert ended == True


def test_timeout_not_reached():
    state = create_initial_state(n_types=1, max_n=1, height=5, width=5)
    state = state.replace(step_count=jnp.int32(50))
    ended, win = check_timeout(state, limit=100, win=False)
    assert ended == False


def test_resource_counter_triggers():
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5, n_resource_types=1)
    # Set avatar (type 0, slot 0) resource[0] = 5
    state = state.replace(resources=state.resources.at[0, 0, 0].set(5))
    ended, win = check_resource_counter(state, avatar_type_idx=0, resource_idx=0, limit=5, win=True)
    assert ended == True
    assert win == True


def test_resource_counter_not_triggered():
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5, n_resource_types=1)
    state = state.replace(resources=state.resources.at[0, 0, 0].set(3))
    ended, win = check_resource_counter(state, avatar_type_idx=0, resource_idx=0, limit=5, win=True)
    assert ended == False


def test_check_all_first_wins():
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5)
    # Both type 0 and type 1 empty — two terminations both trigger
    terms = [
        (lambda s: check_sprite_counter(s, [0], 0, False), 0),
        (lambda s: check_sprite_counter(s, [1], 0, True), 0),
    ]
    state, done, win = check_all_terminations(state, terms)
    assert done == True
    assert win == False  # first one wins


def test_check_all_score():
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5)
    terms = [
        (lambda s: check_sprite_counter(s, [0], 0, True), 5),
    ]
    state, done, win = check_all_terminations(state, terms)
    assert done == True
    assert state.score == 5
