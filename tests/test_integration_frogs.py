import jax
from conftest import make_env


def test_frogs_trucks_wrap():
    """Trucks should wrap around when hitting EOS."""
    env = make_env('frogs')
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
        action = env.noop_action  # NOOP
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    # After 100 steps, trucks should still be alive (wrapping, not dying)
    total_trucks = sum(int(state.alive[t].sum()) for t in truck_types)
    assert total_trucks > 0, "Trucks should survive via wrapAround"
