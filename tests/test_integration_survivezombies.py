import jax
from conftest import make_env


def test_survivezombies_collect_honey():
    """Collecting honey should increase score."""
    env = make_env('survivezombies')
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
