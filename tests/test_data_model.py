from vgdl_jax.data_model import (
    GameDef, SpriteDef, EffectDef, TerminationDef, LevelDef,
    SpriteClass, EffectType, TerminationType,
)


def test_sprite_class_enum():
    assert SpriteClass.IMMOVABLE == 0
    assert SpriteClass.MISSILE == 1
    assert SpriteClass.BOMBER == 7


def test_sprite_def():
    sd = SpriteDef(
        key="sam", type_idx=0, sprite_class=SpriteClass.MISSILE,
        stypes=["missile", "sam"], speed=1.0, orientation=(0, -1),
        cooldown=1, is_static=False, singleton=True,
        flicker_limit=0, spawner_stype=None, spawner_prob=0.0,
        spawner_total=0, color=(250, 250, 250),
        img=None, shrinkfactor=0.0,
    )
    assert sd.key == "sam"
    assert sd.singleton is True


def test_sprite_def_has_color():
    sd = SpriteDef(
        key='wall', type_idx=0, sprite_class=SpriteClass.IMMOVABLE,
        stypes=['wall'], speed=0.0, orientation=(0.0, 1.0),
        cooldown=0, is_static=True, singleton=False,
        flicker_limit=0, spawner_stype=None,
        spawner_prob=1.0, spawner_total=0,
        color=(90, 90, 90),
        img=None, shrinkfactor=0.0,
    )
    assert sd.color == (90, 90, 90)


def test_effect_def():
    ed = EffectDef(
        effect_type=EffectType.KILL_SPRITE,
        actor_stype="alien", actee_stype="sam",
        score_change=2, kwargs={},
    )
    assert ed.effect_type == EffectType.KILL_SPRITE


def test_game_def_type_index():
    sprites = [
        SpriteDef(key="wall", type_idx=0, sprite_class=SpriteClass.IMMOVABLE,
                  stypes=["wall"], speed=0, orientation=(0, 0), cooldown=0,
                  is_static=True, singleton=False, flicker_limit=0,
                  spawner_stype=None, spawner_prob=0, spawner_total=0,
                  color=(250, 250, 250), img=None, shrinkfactor=0.0),
        SpriteDef(key="avatar", type_idx=1, sprite_class=SpriteClass.MOVING_AVATAR,
                  stypes=["avatar"], speed=1, orientation=(0, 0), cooldown=1,
                  is_static=False, singleton=False, flicker_limit=0,
                  spawner_stype=None, spawner_prob=0, spawner_total=0,
                  color=(250, 250, 250), img=None, shrinkfactor=0.0),
    ]
    gd = GameDef(sprites=sprites, effects=[], terminations=[],
                 level=None, char_mapping={}, sprite_order=["wall", "avatar"],
                 stype_to_indices={"wall": [0], "avatar": [1]})
    assert gd.type_idx("wall") == 0
    assert gd.type_idx("avatar") == 1
    assert gd.resolve_stype("wall") == [0]
