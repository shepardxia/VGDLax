import jax
import jax.numpy as jnp
from vgdl_jax.env import VGDLJaxEnv
import os

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')


def test_boulderdash_runs_100_steps():
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'boulderdash.txt'),
        os.path.join(GAMES_DIR, 'boulderdash_lvl0.txt'))
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)

    for i in range(100):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.step_count > 0


def test_boulderdash_game_ends():
    """Game should eventually end (avatar dies)."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'boulderdash.txt'),
        os.path.join(GAMES_DIR, 'boulderdash_lvl0.txt'))
    rng = jax.random.PRNGKey(7)
    obs, state = env.reset(rng)

    for i in range(2000):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.done == True


def test_boulderdash_collect_diamonds():
    """Collecting diamonds should increase score and resources."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'boulderdash.txt'),
        os.path.join(GAMES_DIR, 'boulderdash_lvl0.txt'))

    gd = env.compiled.game_def
    diamond_idx = gd.type_idx('diamond')

    # Try multiple seeds — random play may kill avatar before reaching diamond
    collected = False
    for seed in range(10):
        rng = jax.random.PRNGKey(seed)
        obs, state = env.reset(rng)
        initial_diamonds = int(state.alive[diamond_idx].sum())
        assert initial_diamonds > 0, "Level should have diamonds"
        for i in range(500):
            rng, key = jax.random.split(rng)
            action = jax.random.randint(key, (), 0, env.n_actions)
            obs, state, reward, done, info = env.step(state, action)
            if int(state.score) > 0:
                collected = True
                break
            if done:
                break
        if collected:
            break

    assert collected, "Expected to collect at least one diamond across 10 seeds"


def test_boulderdash_vmap():
    """Batched execution."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'boulderdash.txt'),
        os.path.join(GAMES_DIR, 'boulderdash_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 4)
    obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    assert obs_batch.shape[0] == 4

    actions = jnp.zeros(4, dtype=jnp.int32)
    obs2, states2, rewards, dones, infos = jax.vmap(env.step)(state_batch, actions)
    assert obs2.shape[0] == 4
