"""Parameterized integration tests for all supported games."""
import jax
import jax.numpy as jnp
import pytest
from conftest import make_env


ALL_GAMES = ['chase', 'zelda', 'aliens', 'sokoban', 'portals',
             'boulderdash', 'survivezombies', 'frogs']

# Max random-play steps before asserting game ends (per game)
GAME_END_MAX_STEPS = {
    'chase': 5000,
    'zelda': 3000,
    'aliens': 2000,
    'portals': 2000,
    'boulderdash': 2000,
    'survivezombies': 3000,
    'frogs': 1000,
}


@pytest.mark.parametrize('game', ALL_GAMES)
def test_runs_100_steps(game):
    env = make_env(game)
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)

    for i in range(100):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.step_count > 0


@pytest.mark.parametrize('game', list(GAME_END_MAX_STEPS.keys()))
def test_game_ends(game):
    env = make_env(game)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    max_steps = GAME_END_MAX_STEPS[game]

    for i in range(max_steps):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.done == True


@pytest.mark.parametrize('game', ALL_GAMES)
def test_vmap(game):
    env = make_env(game)
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 4)
    obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    assert obs_batch.shape[0] == 4

    actions = jnp.zeros(4, dtype=jnp.int32)
    obs2, states2, rewards, dones, infos = jax.vmap(env.step)(state_batch, actions)
    assert obs2.shape[0] == 4
