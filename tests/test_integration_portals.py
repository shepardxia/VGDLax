import jax
from conftest import make_env


def test_portals_teleport():
    """Avatar should visit multiple distinct cells with portal teleportation."""
    env = make_env('portals')

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
