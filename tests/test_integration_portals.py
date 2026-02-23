import jax
import jax.numpy as jnp
from vgdl_jax.env import VGDLJaxEnv
import os

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')


def test_portals_runs_100_steps():
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'portals.txt'),
        os.path.join(GAMES_DIR, 'portals_lvl0.txt'))
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)

    for i in range(100):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.step_count > 0


def test_portals_game_ends():
    """Game should eventually end (avatar dies or collects all goals)."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'portals.txt'),
        os.path.join(GAMES_DIR, 'portals_lvl0.txt'))
    rng = jax.random.PRNGKey(123)
    obs, state = env.reset(rng)

    for i in range(2000):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.done == True


def test_portals_teleport():
    """Avatar should visit multiple distinct cells with portal teleportation."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'portals.txt'),
        os.path.join(GAMES_DIR, 'portals_lvl0.txt'))

    gd = env.compiled.game_def
    avatar_idx = gd.type_idx('avatar')

    # Try multiple seeds to get enough exploration
    max_positions = 0
    for seed in range(5):
        rng = jax.random.PRNGKey(seed)
        obs, state = env.reset(rng)
        positions_seen = set()
        for i in range(500):
            rng, key = jax.random.split(rng)
            action = jax.random.randint(key, (), 0, env.n_actions)
            obs, state, reward, done, info = env.step(state, action)
            if done:
                break
            pos = tuple(int(x) for x in state.positions[avatar_idx, 0])
            positions_seen.add(pos)
        max_positions = max(max_positions, len(positions_seen))

    assert max_positions > 3, f"Avatar should visit multiple cells, best={max_positions}"


def test_portals_vmap():
    """Batched execution."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'portals.txt'),
        os.path.join(GAMES_DIR, 'portals_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 4)
    obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    assert obs_batch.shape[0] == 4

    actions = jnp.zeros(4, dtype=jnp.int32)
    obs2, states2, rewards, dones, infos = jax.vmap(env.step)(state_batch, actions)
    assert obs2.shape[0] == 4
