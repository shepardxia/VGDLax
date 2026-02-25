"""Validation subpackage for cross-engine comparison between py-vgdl and vgdl-jax."""
from .constants import ALL_GAMES, DETERMINISTIC_GAMES, STOCHASTIC_GAMES, GAMES_DIR, BLOCK_SIZE
from .harness import (setup_jax_game, setup_pyvgdl_game, run_comparison, run_jax_trajectory,
                      run_pyvgdl_trajectory, compare_states, StepComparison, TrajectoryResult,
                      get_sprite_configs, get_effects,
                      validate_pyvgdl_loads, validate_pyvgdl_state_extraction,
                      validate_pyvgdl_trajectory)
from .state_extractor import extract_pyvgdl_state, extract_jax_state
from .rng_replay import RNGRecorder, ReplayRandomGenerator, patch_chaser_directions
