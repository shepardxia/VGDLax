import jax
import jax.numpy as jnp
from vgdl_jax.env import VGDLJaxEnv
import os

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')


def test_frogs_runs_100_steps():
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'frogs.txt'),
        os.path.join(GAMES_DIR, 'frogs_lvl0.txt'))
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)

    for i in range(100):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.step_count > 0


def test_frogs_game_ends():
    """Game should eventually end (avatar dies on water or wins)."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'frogs.txt'),
        os.path.join(GAMES_DIR, 'frogs_lvl0.txt'))
    rng = jax.random.PRNGKey(7)
    obs, state = env.reset(rng)

    for i in range(1000):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.done == True


def test_frogs_trucks_wrap():
    """Trucks should wrap around when hitting EOS."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'frogs.txt'),
        os.path.join(GAMES_DIR, 'frogs_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    gd = env.compiled.game_def
    # Pick any truck type
    truck_types = []
    for sd in gd.sprites:
        if 'truck' in sd.key.lower():
            truck_types.append(sd.type_idx)

    assert len(truck_types) > 0, "Should have truck types"

    # Run enough steps for trucks to hit edges
    for i in range(100):
        rng, key = jax.random.split(rng)
        action = env.n_actions - 1  # NOOP
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    # After 100 steps, trucks should still be alive (wrapping, not dying)
    total_trucks = sum(int(state.alive[t].sum()) for t in truck_types)
    assert total_trucks > 0, "Trucks should survive via wrapAround"


def test_frogs_vmap():
    """Batched execution."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'frogs.txt'),
        os.path.join(GAMES_DIR, 'frogs_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 4)
    obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    assert obs_batch.shape[0] == 4

    actions = jnp.zeros(4, dtype=jnp.int32)
    obs2, states2, rewards, dones, infos = jax.vmap(env.step)(state_batch, actions)
    assert obs2.shape[0] == 4
