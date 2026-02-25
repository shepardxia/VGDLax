import os
import jax
from vgdl_jax.parser import parse_vgdl
from vgdl_jax.compiler import compile_game
from conftest import GAMES_DIR


def test_compile_chase():
    gd = parse_vgdl(
        os.path.join(GAMES_DIR, 'chase.txt'),
        os.path.join(GAMES_DIR, 'chase_lvl0.txt'))
    compiled = compile_game(gd)
    assert compiled.init_state is not None
    assert compiled.step_fn is not None
    assert compiled.n_actions > 0

    # Verify initial state has correct sprites
    state = compiled.init_state
    n_alive = state.alive.sum()
    assert n_alive > 0  # Should have avatar + goats + walls


def test_compile_chase_step():
    gd = parse_vgdl(
        os.path.join(GAMES_DIR, 'chase.txt'),
        os.path.join(GAMES_DIR, 'chase_lvl0.txt'))
    compiled = compile_game(gd)
    state = compiled.init_state

    # Take a NOOP action
    noop = compiled.noop_action
    new_state = compiled.step_fn(state, noop)
    assert new_state.step_count == 1
    assert new_state.done == False


def test_compile_zelda():
    gd = parse_vgdl(
        os.path.join(GAMES_DIR, 'zelda.txt'),
        os.path.join(GAMES_DIR, 'zelda_lvl0.txt'))
    compiled = compile_game(gd)
    state = compiled.init_state
    assert state.alive.sum() > 0

    noop = compiled.noop_action
    new_state = compiled.step_fn(state, noop)
    assert new_state.step_count == 1


def test_compile_aliens():
    gd = parse_vgdl(
        os.path.join(GAMES_DIR, 'aliens.txt'),
        os.path.join(GAMES_DIR, 'aliens_lvl0.txt'))
    compiled = compile_game(gd)
    state = compiled.init_state
    assert state.alive.sum() > 0

    noop = compiled.noop_action
    new_state = compiled.step_fn(state, noop)
    assert new_state.step_count == 1
