import os
import pytest
from vgdl_jax.parser import parse_vgdl
from vgdl_jax.data_model import SpriteClass
from conftest import GAMES_DIR


# ── Module-scoped fixtures (each game parsed once) ──────────────────────

@pytest.fixture(scope='module')
def chase_gd():
    return parse_vgdl(
        os.path.join(GAMES_DIR, 'chase.txt'),
        os.path.join(GAMES_DIR, 'chase_lvl0.txt'))


@pytest.fixture(scope='module')
def aliens_gd():
    return parse_vgdl(
        os.path.join(GAMES_DIR, 'aliens.txt'),
        os.path.join(GAMES_DIR, 'aliens_lvl0.txt'))


@pytest.fixture(scope='module')
def zelda_gd():
    return parse_vgdl(
        os.path.join(GAMES_DIR, 'zelda.txt'),
        os.path.join(GAMES_DIR, 'zelda_lvl0.txt'))


# ── Tests ───────────────────────────────────────────────────────────────

def test_parse_chase(chase_gd):
    gd = chase_gd
    keys = [s.key for s in gd.sprites]
    assert 'avatar' in keys
    assert 'angry' in keys
    assert 'scared' in keys
    assert 'wall' in keys

    avatar_def = next(s for s in gd.sprites if s.key == 'avatar')
    assert avatar_def.sprite_class == SpriteClass.MOVING_AVATAR

    angry_def = next(s for s in gd.sprites if s.key == 'angry')
    assert angry_def.sprite_class == SpriteClass.CHASER

    # Effects
    assert any(e.effect_type == 'step_back' for e in gd.effects)
    assert any(e.effect_type == 'transform_to' for e in gd.effects)

    # Level dimensions (absorbs test_parse_level_dimensions)
    assert gd.level is not None
    assert gd.level.height == 11
    assert gd.level.width == 24
    assert len(gd.level.initial_sprites) > 0

    # stype resolution
    assert 'goat' in gd.stype_to_indices
    goat_indices = gd.stype_to_indices['goat']
    assert len(goat_indices) == 2  # angry and scared

    # Default colors (absorbs test_parse_default_colors)
    wall_def = next(s for s in gd.sprites if s.key == 'wall')
    assert wall_def.color == (90, 90, 90)  # Immovable → GRAY
    assert avatar_def.color == (250, 250, 250)  # MovingAvatar → WHITE


def test_parse_aliens(aliens_gd):
    gd = aliens_gd
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
    assert len(eos_effects) >= 3

    # MultiSpriteCounter termination
    terms = gd.terminations
    assert any(t.term_type == 1 for t in terms)  # MULTI_SPRITE_COUNTER

    # Orientation (absorbs test_parse_orientation)
    assert sam_def.orientation == (-1.0, 0.0)  # UP
    bomb_def = next(s for s in gd.sprites if s.key == 'bomb')
    assert bomb_def.orientation == (1.0, 0.0)  # DOWN

    # Color overrides (absorbs test_parse_color_override)
    assert avatar_def.color == (0, 200, 0)  # FlakAvatar → GREEN
    assert sam_def.color == (0, 0, 200)  # BLUE override
    assert bomb_def.color == (200, 0, 0)  # RED override
    base_def = next(s for s in gd.sprites if s.key == 'base')
    assert base_def.color == (250, 250, 250)  # WHITE override


def test_parse_zelda(zelda_gd):
    gd = zelda_gd
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

    # stype hierarchy
    assert 'movable' in gd.stype_to_indices
    assert 'enemy' in gd.stype_to_indices

    # img and shrinkfactor (absorbs test_parse_img_and_shrinkfactor)
    assert sword_def.img == 'oryx/slash1'
    assert nokey_def.img == 'oryx/swordman1_0'
    assert nokey_def.shrinkfactor == 0.0
    wall_def = next(s for s in gd.sprites if s.key == 'wall')
    assert wall_def.img == 'oryx/wall3'
    assert wall_def.shrinkfactor == 0.0
