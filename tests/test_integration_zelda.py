import jax
import jax.numpy as jnp
from vgdl_jax.env import VGDLJaxEnv
import os

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')


def test_zelda_runs_100_steps():
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'zelda.txt'),
        os.path.join(GAMES_DIR, 'zelda_lvl0.txt'))
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)

    for i in range(100):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.step_count > 0


def test_zelda_shoot_creates_sword():
    """ShootAvatar's shoot action should create a sword sprite."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'zelda.txt'),
        os.path.join(GAMES_DIR, 'zelda_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    gd = env.compiled.game_def
    sword_idx = gd.type_idx('sword')

    # Initially no sword alive
    assert state.alive[sword_idx].sum() == 0

    # Shoot action (action index = n_move = 4 for ShootAvatar)
    shoot_action = 4
    obs, state, _, _, _ = env.step(state, shoot_action)

    # Sword should be spawned
    assert state.alive[sword_idx].sum() == 1


def test_zelda_game_ends():
    """Game should eventually end with random play."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'zelda.txt'),
        os.path.join(GAMES_DIR, 'zelda_lvl0.txt'))
    rng = jax.random.PRNGKey(123)
    obs, state = env.reset(rng)

    for i in range(1000):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.done == True


def test_zelda_vmap():
    """Batched execution with vmap."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'zelda.txt'),
        os.path.join(GAMES_DIR, 'zelda_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 8)
    obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    assert obs_batch.shape[0] == 8

    actions = jnp.zeros(8, dtype=jnp.int32)
    obs2, states2, rewards, dones, infos = jax.vmap(env.step)(state_batch, actions)
    assert obs2.shape[0] == 8
