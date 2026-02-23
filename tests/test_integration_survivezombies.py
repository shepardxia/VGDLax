import jax
import jax.numpy as jnp
from vgdl_jax.env import VGDLJaxEnv
import os

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')


def test_survivezombies_runs_100_steps():
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'survivezombies.txt'),
        os.path.join(GAMES_DIR, 'survivezombies_lvl0.txt'))
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)

    for i in range(100):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.step_count > 0


def test_survivezombies_game_ends():
    """Game should eventually end (avatar dies or timeout)."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'survivezombies.txt'),
        os.path.join(GAMES_DIR, 'survivezombies_lvl0.txt'))
    rng = jax.random.PRNGKey(7)
    obs, state = env.reset(rng)

    for i in range(3000):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.done == True


def test_survivezombies_collect_honey():
    """Collecting honey should increase score."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'survivezombies.txt'),
        os.path.join(GAMES_DIR, 'survivezombies_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    gd = env.compiled.game_def
    honey_idx = gd.type_idx('honey')
    initial_honey = int(state.alive[honey_idx].sum())
    assert initial_honey > 0

    max_score = 0
    for i in range(500):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        max_score = max(max_score, int(state.score))
        if done:
            break

    assert max_score > 0, "Expected to collect at least one honey"


def test_survivezombies_vmap():
    """Batched execution."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'survivezombies.txt'),
        os.path.join(GAMES_DIR, 'survivezombies_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 4)
    obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    assert obs_batch.shape[0] == 4

    actions = jnp.zeros(4, dtype=jnp.int32)
    obs2, states2, rewards, dones, infos = jax.vmap(env.step)(state_batch, actions)
    assert obs2.shape[0] == 4
