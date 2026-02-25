"""Shared test fixtures and constants for vgdl-jax tests."""
import os
import sys
from vgdl_jax.validate.constants import GAMES_DIR, ALL_GAMES
from vgdl_jax.env import VGDLJaxEnv

# py-vgdl on sys.path for test_cross_engine.py (module-level import)
PYVGDL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl')
if PYVGDL_DIR not in sys.path:
    sys.path.insert(0, PYVGDL_DIR)


def make_env(game_name):
    """Create a VGDLJaxEnv for the given game name."""
    return VGDLJaxEnv(
        os.path.join(GAMES_DIR, f'{game_name}.txt'),
        os.path.join(GAMES_DIR, f'{game_name}_lvl0.txt'))
