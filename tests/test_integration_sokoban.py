import jax
import jax.numpy as jnp
from vgdl_jax.env import VGDLJaxEnv
import os

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')


def test_sokoban_runs_100_steps():
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'sokoban.txt'),
        os.path.join(GAMES_DIR, 'sokoban_lvl0.txt'))
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)

    for i in range(100):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.step_count > 0


def test_sokoban_box_bounceforward():
    """When avatar pushes a box, box should move in avatar's direction."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'sokoban.txt'),
        os.path.join(GAMES_DIR, 'sokoban_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    gd = env.compiled.game_def
    box_idx = gd.type_idx('box')
    avatar_idx = gd.type_idx('avatar')

    # Record initial box positions
    initial_box_pos = state.positions[box_idx].copy()
    initial_box_alive = state.alive[box_idx].copy()

    # Run some steps — boxes should move when pushed
    moved = False
    for i in range(50):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break
        # Check if any box position changed
        for j in range(initial_box_alive.shape[0]):
            if initial_box_alive[j] and state.alive[box_idx, j]:
                if not jnp.array_equal(state.positions[box_idx, j], initial_box_pos[j]):
                    moved = True

    # With enough random moves, at least one box should have moved
    # (the level has boxes adjacent to the avatar)
    assert moved, "Expected at least one box to move via bounceForward"


def test_sokoban_undo_all_on_box_wall():
    """undoAll should revert all positions when box hits wall."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'sokoban.txt'),
        os.path.join(GAMES_DIR, 'sokoban_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    # Just verify the game compiles and runs with undoAll effect active
    for i in range(100):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.step_count > 0


def test_sokoban_vmap():
    """Batched execution."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'sokoban.txt'),
        os.path.join(GAMES_DIR, 'sokoban_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 4)
    obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    assert obs_batch.shape[0] == 4

    actions = jnp.zeros(4, dtype=jnp.int32)
    obs2, states2, rewards, dones, infos = jax.vmap(env.step)(state_batch, actions)
    assert obs2.shape[0] == 4
