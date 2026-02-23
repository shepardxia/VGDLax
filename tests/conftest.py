"""Shared test fixtures and constants for vgdl-jax tests."""
import os
import pytest
from vgdl_jax.env import VGDLJaxEnv


GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')


def make_env(game_name):
    """Create a VGDLJaxEnv for the given game name."""
    return VGDLJaxEnv(
        os.path.join(GAMES_DIR, f'{game_name}.txt'),
        os.path.join(GAMES_DIR, f'{game_name}_lvl0.txt'))
