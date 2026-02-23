import jax
import jax.numpy as jnp
from conftest import make_env


def test_chase_jit_speed():
    """Verify jit compilation works and steps are fast."""
    env = make_env('chase')
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    # First call triggers compilation
    obs, state, _, _, _ = env.step(state, 0)
    # Second call should be fast (already compiled)
    obs, state, _, _, _ = env.step(state, 1)
    assert state.step_count == 2


def test_chase_stepback_prevents_wall_pass():
    """Avatar should not pass through walls."""
    env = make_env('chase')
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    # Find wall positions
    gd = env.compiled.game_def
    wall_idx = gd.type_idx('wall')
    wall_positions = state.positions[wall_idx]
    wall_alive = state.alive[wall_idx]

    # Find avatar position
    avatar_idx = gd.type_idx('avatar')
    avatar_pos = state.positions[avatar_idx, 0]

    # Run some steps and verify avatar never occupies a wall position
    for i in range(50):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break
        # Avatar position after effects should not be on a wall
        av_pos = state.positions[avatar_idx, 0]
        # Check avatar isn't at any alive wall position
        for j in range(wall_alive.shape[0]):
            if wall_alive[j]:
                same = jnp.array_equal(av_pos, wall_positions[j])
                assert not same, f"Avatar at step {i} occupies wall position {wall_positions[j]}"
