import os
from vgdl_jax.parser import parse_vgdl
from vgdl_jax.data_model import SpriteClass, EffectType

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')


def test_parse_chase():
    game_file = os.path.join(GAMES_DIR, 'chase.txt')
    level_file = os.path.join(GAMES_DIR, 'chase_lvl0.txt')
    gd = parse_vgdl(game_file, level_file)

    keys = [s.key for s in gd.sprites]
    assert 'avatar' in keys
    assert 'angry' in keys
    assert 'scared' in keys
    assert 'wall' in keys

    avatar_def = next(s for s in gd.sprites if s.key == 'avatar')
    assert avatar_def.sprite_class == SpriteClass.MOVING_AVATAR

    angry_def = next(s for s in gd.sprites if s.key == 'angry')
    assert angry_def.sprite_class == SpriteClass.CHASER

    # Check effects
    assert any(e.effect_type == EffectType.STEP_BACK for e in gd.effects)
    assert any(e.effect_type == EffectType.TRANSFORM_TO for e in gd.effects)

    # Check level
    assert gd.level is not None
    assert gd.level.height > 0
    assert gd.level.width > 0
    assert len(gd.level.initial_sprites) > 0

    # Check stype resolution
    assert 'goat' in gd.stype_to_indices
    goat_indices = gd.stype_to_indices['goat']
    assert len(goat_indices) == 2  # angry and scared


def test_parse_zelda():
    game_file = os.path.join(GAMES_DIR, 'zelda.txt')
    level_file = os.path.join(GAMES_DIR, 'zelda_lvl0.txt')
    gd = parse_vgdl(game_file, level_file)

    keys = [s.key for s in gd.sprites]
    assert 'nokey' in keys
    assert 'withkey' in keys
    assert 'sword' in keys
    assert 'monsterQuick' in keys

    sword_def = next(s for s in gd.sprites if s.key == 'sword')
    assert sword_def.sprite_class == SpriteClass.ORIENTED_FLICKER
    assert sword_def.flicker_limit == 5

    # ShootAvatar
    nokey_def = next(s for s in gd.sprites if s.key == 'nokey')
    assert nokey_def.sprite_class == SpriteClass.SHOOT_AVATAR

    # stype hierarchy: movable includes avatar subtypes, enemy subtypes, wall
    assert 'movable' in gd.stype_to_indices
    assert 'enemy' in gd.stype_to_indices


def test_parse_aliens():
    game_file = os.path.join(GAMES_DIR, 'aliens.txt')
    level_file = os.path.join(GAMES_DIR, 'aliens_lvl0.txt')
    gd = parse_vgdl(game_file, level_file)

    keys = [s.key for s in gd.sprites]
    assert 'sam' in keys
    assert 'bomb' in keys
    assert 'alienGreen' in keys

    sam_def = next(s for s in gd.sprites if s.key == 'sam')
    assert sam_def.sprite_class == SpriteClass.MISSILE
    assert sam_def.singleton == True

    portal_def = next(s for s in gd.sprites if s.key == 'portalSlow')
    assert portal_def.sprite_class == SpriteClass.SPAWN_POINT

    # FlakAvatar
    avatar_def = next(s for s in gd.sprites if s.key == 'avatar')
    assert avatar_def.sprite_class == SpriteClass.FLAK_AVATAR

    # EOS effects
    eos_effects = [e for e in gd.effects if e.actee_stype == 'EOS']
    assert len(eos_effects) >= 3  # avatar, alien, missile all have EOS effects

    # MultiSpriteCounter termination
    terms = gd.terminations
    assert any(t.term_type == 1 for t in terms)  # MULTI_SPRITE_COUNTER


def test_parse_level_dimensions():
    """Level dimensions match the actual text file."""
    game_file = os.path.join(GAMES_DIR, 'chase.txt')
    level_file = os.path.join(GAMES_DIR, 'chase_lvl0.txt')
    gd = parse_vgdl(game_file, level_file)
    # chase_lvl0: 11 rows, 24 cols
    assert gd.level.height == 11
    assert gd.level.width == 24


def test_parse_orientation():
    """Orientation kwarg parsed correctly for missiles."""
    game_file = os.path.join(GAMES_DIR, 'aliens.txt')
    level_file = os.path.join(GAMES_DIR, 'aliens_lvl0.txt')
    gd = parse_vgdl(game_file, level_file)

    sam_def = next(s for s in gd.sprites if s.key == 'sam')
    assert sam_def.orientation == (-1.0, 0.0)  # UP

    bomb_def = next(s for s in gd.sprites if s.key == 'bomb')
    assert bomb_def.orientation == (1.0, 0.0)  # DOWN


def test_parse_default_colors():
    """Sprite classes get correct default colors when no color= override."""
    game_file = os.path.join(GAMES_DIR, 'chase.txt')
    level_file = os.path.join(GAMES_DIR, 'chase_lvl0.txt')
    gd = parse_vgdl(game_file, level_file)

    wall_def = next(s for s in gd.sprites if s.key == 'wall')
    assert wall_def.color == (90, 90, 90)  # Immovable → GRAY

    avatar_def = next(s for s in gd.sprites if s.key == 'avatar')
    assert avatar_def.color == (250, 250, 250)  # MovingAvatar → WHITE


def test_parse_color_override():
    """Colors specified in game files override class defaults."""
    game_file = os.path.join(GAMES_DIR, 'aliens.txt')
    level_file = os.path.join(GAMES_DIR, 'aliens_lvl0.txt')
    gd = parse_vgdl(game_file, level_file)

    # FlakAvatar default is GREEN (no color= override in aliens.txt)
    avatar_def = next(s for s in gd.sprites if s.key == 'avatar')
    assert avatar_def.color == (0, 200, 0)  # FlakAvatar → GREEN

    # sam has explicit color=BLUE override (parent class Missile default is WHITE)
    sam_def = next(s for s in gd.sprites if s.key == 'sam')
    assert sam_def.color == (0, 0, 200)  # BLUE override

    # bomb has explicit color=RED override
    bomb_def = next(s for s in gd.sprites if s.key == 'bomb')
    assert bomb_def.color == (200, 0, 0)  # RED override

    # base has explicit color=WHITE override (Immovable default is GRAY)
    base_def = next(s for s in gd.sprites if s.key == 'base')
    assert base_def.color == (250, 250, 250)  # WHITE override


def test_parse_img_and_shrinkfactor():
    """Sprite image paths and shrinkfactors are preserved."""
    game_file = os.path.join(GAMES_DIR, 'zelda.txt')
    level_file = os.path.join(GAMES_DIR, 'zelda_lvl0.txt')
    gd = parse_vgdl(game_file, level_file)

    # sword has img=oryx/slash1
    sword_def = next(s for s in gd.sprites if s.key == 'sword')
    assert sword_def.img == 'oryx/slash1'

    # nokey avatar has img and default shrinkfactor 0.0
    # (py-vgdl Avatar mixin sets 0.15, but VGDLSprite.shrinkfactor=0.0 wins via MRO)
    nokey_def = next(s for s in gd.sprites if s.key == 'nokey')
    assert nokey_def.img == 'oryx/swordman1_0'
    assert nokey_def.shrinkfactor == 0.0

    # wall has img=oryx/wall3 and default shrinkfactor
    wall_def = next(s for s in gd.sprites if s.key == 'wall')
    assert wall_def.img == 'oryx/wall3'
    assert wall_def.shrinkfactor == 0.0
