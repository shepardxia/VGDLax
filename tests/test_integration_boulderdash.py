import jax
from conftest import make_env


def test_boulderdash_collect_diamonds():
    """Collecting diamonds should increase score and resources."""
    env = make_env('boulderdash')

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
