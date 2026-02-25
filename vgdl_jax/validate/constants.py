"""Single source of truth for validation constants."""
import os

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'py-vgdl', 'vgdl', 'games')
BLOCK_SIZE = 10

ALL_GAMES = [
    'chase', 'zelda', 'aliens', 'missilecommand', 'sokoban',
    'portals', 'boulderdash', 'survivezombies', 'frogs',
]

DETERMINISTIC_GAMES = ['sokoban']
STOCHASTIC_GAMES = [g for g in ALL_GAMES if g not in DETERMINISTIC_GAMES]
