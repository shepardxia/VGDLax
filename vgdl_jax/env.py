"""
Gymnax-style JAX environment wrapper for VGDL games.
Supports jit, vmap for batched RL training.
"""
from functools import partial

import jax
import jax.numpy as jnp

from vgdl_jax.parser import parse_vgdl
from vgdl_jax.compiler import compile_game
from vgdl_jax.render import render_rgb


class VGDLJaxEnv:
    """
    A VGDL game environment compatible with JAX transformations.

    Usage:
        env = VGDLJaxEnv('game.txt', 'game_lvl0.txt')
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)
        obs, state, reward, done, info = env.step(state, action)

        # Batched:
        obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    """

    def __init__(self, game_file, level_file, max_sprites_per_type=None):
        game_def = parse_vgdl(game_file, level_file)
        self.compiled = compile_game(game_def, max_sprites_per_type)
        self.n_actions = self.compiled.n_actions
        n_types = len(game_def.sprites)
        self._height = game_def.level.height
        self._width = game_def.level.width
        self._n_types = n_types
        self.obs_shape = (n_types, self._height, self._width)

        # Build color table from sprite definitions
        self._colors = jnp.array(
            [sd.color for sd in game_def.sprites], dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng):
        """Reset the environment and return (obs, state)."""
        state = self.compiled.init_state.replace(rng=rng)
        return self._get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        """Take a step and return (obs, state, reward, done, info)."""
        prev_score = state.score
        state = self.compiled.step_fn(state, action)
        obs = self._get_obs(state)
        reward = state.score - prev_score
        return obs, state, reward, state.done, {}

    def _get_obs(self, state):
        """Render state as a [n_types, height, width] binary grid."""
        grid = jnp.zeros((self._n_types, self._height, self._width),
                         dtype=jnp.bool_)
        for t in range(self._n_types):
            pos = state.positions[t].astype(jnp.int32)
            alive = state.alive[t]
            row = jnp.clip(pos[:, 0], 0, self._height - 1)
            col = jnp.clip(pos[:, 1], 0, self._width - 1)
            grid = grid.at[t, row, col].set(grid[t, row, col] | alive)
        return grid

    def render(self, state, block_size=10):
        """Render game state to RGB image.

        Args:
            state: GameState
            block_size: pixels per grid cell

        Returns:
            [H*block_size, W*block_size, 3] uint8 RGB image
        """
        obs = self._get_obs(state)
        return render_rgb(obs, self._colors, block_size)
