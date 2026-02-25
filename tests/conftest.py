"""Shared test fixtures and constants for vgdl-jax tests."""
import os
import sys
from vgdl_jax.env import VGDLJaxEnv


GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')

# Centralized py-vgdl path setup (used by validate_harness, state_extractor, etc.)
PYVGDL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl')
if PYVGDL_DIR not in sys.path:
    sys.path.insert(0, PYVGDL_DIR)

ALL_GAMES = [
    'chase', 'zelda', 'aliens', 'missilecommand', 'sokoban',
    'portals', 'boulderdash', 'survivezombies', 'frogs',
]


def make_env(game_name):
    """Create a VGDLJaxEnv for the given game name."""
    return VGDLJaxEnv(
        os.path.join(GAMES_DIR, f'{game_name}.txt'),
        os.path.join(GAMES_DIR, f'{game_name}_lvl0.txt'))
