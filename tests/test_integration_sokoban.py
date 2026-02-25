import jax
import jax.numpy as jnp
from conftest import make_env


def test_sokoban_box_bounceforward():
    """When avatar pushes a box, box should move in avatar's direction."""
    env = make_env('sokoban')
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    gd = env.compiled.game_def
    box_idx = gd.type_idx('box')

    # Record initial box positions
    initial_box_pos = state.positions[box_idx].copy()
    initial_box_alive = state.alive[box_idx].copy()

    # Run some steps — boxes should move when pushed
    moved = False
    for i in range(50):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break
        # Check if any box position changed
        for j in range(initial_box_alive.shape[0]):
            if initial_box_alive[j] and state.alive[box_idx, j]:
                if not jnp.array_equal(state.positions[box_idx, j], initial_box_pos[j]):
                    moved = True

    # With enough random moves, at least one box should have moved
    # (the level has boxes adjacent to the avatar)
    assert moved, "Expected at least one box to move via bounceForward"
