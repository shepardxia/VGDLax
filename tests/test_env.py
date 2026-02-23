import jax
import jax.numpy as jnp
from vgdl_jax.env import VGDLJaxEnv
import os

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')


def test_env_reset():
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'chase.txt'),
        os.path.join(GAMES_DIR, 'chase_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    assert obs.shape == env.obs_shape
    assert state.done == False


def test_env_step():
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'chase.txt'),
        os.path.join(GAMES_DIR, 'chase_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    obs2, state2, reward, done, info = env.step(state, 0)
    assert obs2.shape == obs.shape
    assert state2.step_count == 1


def test_env_vmap():
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'chase.txt'),
        os.path.join(GAMES_DIR, 'chase_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 4)
    obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    assert obs_batch.shape[0] == 4

    actions = jnp.zeros(4, dtype=jnp.int32)
    obs2, states2, rewards, dones, infos = jax.vmap(env.step)(state_batch, actions)
    assert obs2.shape[0] == 4


def test_env_multiple_steps():
    """Run 20 steps without crashing."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'chase.txt'),
        os.path.join(GAMES_DIR, 'chase_lvl0.txt'))
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)
    for i in range(20):
        action = i % env.n_actions
        obs, state, reward, done, info = env.step(state, action)
    assert state.step_count == 20


def test_env_render_jax():
    """env.render() returns an RGB image with correct shape."""
    game_file = os.path.join(GAMES_DIR, 'chase.txt')
    level_file = os.path.join(GAMES_DIR, 'chase_lvl0.txt')
    env = VGDLJaxEnv(game_file, level_file)

    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    img = env.render(state, block_size=10)
    H, W = 11, 24  # chase_lvl0 dimensions
    assert img.shape == (H * 10, W * 10, 3)
    assert img.dtype == jnp.uint8

    # Walls should be gray (90, 90, 90), not all white
    assert not jnp.all(img == 255)
