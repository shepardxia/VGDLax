from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


class SpriteClass:
    IMMOVABLE = 0
    MISSILE = 1
    RANDOM_NPC = 2
    CHASER = 3
    FLEEING = 4
    FLICKER = 5
    SPAWN_POINT = 6
    BOMBER = 7
    WALKER = 8
    ORIENTED_FLICKER = 9
    # Avatar classes (handled specially but tracked here)
    MOVING_AVATAR = 10
    FLAK_AVATAR = 11
    SHOOT_AVATAR = 12
    HORIZONTAL_AVATAR = 13
    ORIENTED_AVATAR = 14
    # New sprite classes
    RESOURCE = 15
    PASSIVE = 16
    PORTAL = 17
    INERTIAL_AVATAR = 18
    MARIO_AVATAR = 19
    VERTICAL_AVATAR = 20
    CONVEYOR = 21
    ERRATIC_MISSILE = 22
    RANDOM_INERTIAL = 23
    RANDOM_MISSILE = 24
    ROTATING_AVATAR = 25
    ROTATING_FLIPPING_AVATAR = 26
    NOISY_ROTATING_FLIPPING_AVATAR = 27
    SHOOT_EVERYWHERE_AVATAR = 28
    AIMED_AVATAR = 29
    AIMED_FLAK_AVATAR = 30
    SPREADER = 31
    WALK_JUMPER = 32


class EffectType:
    KILL_SPRITE = 0
    KILL_BOTH = 1
    STEP_BACK = 2
    TRANSFORM_TO = 3
    TURN_AROUND = 4
    REVERSE_DIRECTION = 5
    NULL = 6
    # New effect types
    CHANGE_RESOURCE = 7
    COLLECT_RESOURCE = 8
    KILL_IF_HAS_LESS = 9
    KILL_IF_HAS_MORE = 10
    KILL_IF_OTHER_HAS_MORE = 11
    KILL_IF_OTHER_HAS_LESS = 12
    KILL_IF_FROM_ABOVE = 13
    WRAP_AROUND = 14
    BOUNCE_FORWARD = 15
    UNDO_ALL = 16
    TELEPORT_TO_EXIT = 17
    PULL_WITH_IT = 18
    WALL_STOP = 19
    WALL_BOUNCE = 20
    BOUNCE_DIRECTION = 21
    FLIP_DIRECTION = 22
    KILL_IF_ALIVE = 23
    KILL_IF_SLOW = 24
    CONVEY_SPRITE = 25
    CLONE_SPRITE = 26
    SPAWN_IF_HAS_MORE = 27
    WIND_GUST = 28
    SLIP_FORWARD = 29
    ATTRACT_GAZE = 30
    SPEND_RESOURCE = 31
    SPEND_AVATAR_RESOURCE = 32
    KILL_OTHERS = 33
    KILL_IF_AVATAR_WITHOUT_RESOURCE = 34
    AVATAR_COLLECT_RESOURCE = 35
    TRANSFORM_OTHERS_TO = 36


class TerminationType:
    SPRITE_COUNTER = 0
    MULTI_SPRITE_COUNTER = 1
    TIMEOUT = 2
    RESOURCE_COUNTER = 3


# Sprite classes that never move and should be skipped in NPC update
STATIC_CLASSES = {
    SpriteClass.IMMOVABLE, SpriteClass.PASSIVE,
    SpriteClass.RESOURCE, SpriteClass.PORTAL,
    SpriteClass.CONVEYOR,
}

# Avatar classes handled by _update_avatar (not NPC update)
AVATAR_CLASSES = {
    SpriteClass.MOVING_AVATAR, SpriteClass.FLAK_AVATAR,
    SpriteClass.SHOOT_AVATAR, SpriteClass.HORIZONTAL_AVATAR,
    SpriteClass.ORIENTED_AVATAR,
    SpriteClass.INERTIAL_AVATAR, SpriteClass.MARIO_AVATAR,
    SpriteClass.VERTICAL_AVATAR,
    SpriteClass.ROTATING_AVATAR, SpriteClass.ROTATING_FLIPPING_AVATAR,
    SpriteClass.NOISY_ROTATING_FLIPPING_AVATAR,
    SpriteClass.SHOOT_EVERYWHERE_AVATAR,
    SpriteClass.AIMED_AVATAR, SpriteClass.AIMED_FLAK_AVATAR,
}


@dataclass
class SpriteDef:
    key: str
    type_idx: int
    sprite_class: int
    stypes: List[str]
    speed: float
    orientation: Tuple[float, float]
    cooldown: int
    is_static: bool
    singleton: bool
    flicker_limit: int
    spawner_stype: Optional[str]
    spawner_prob: float
    spawner_total: int
    color: Tuple[int, int, int]
    img: Optional[str]        # sprite image path, e.g. "oryx/alien1" (no .png extension)
    shrinkfactor: float       # 0.0 = full size, 0.15 = avatar default, up to ~0.6
    # Resource fields (for Resource sprite class)
    resource_name: Optional[str] = None   # resource type name (e.g. "diamond")
    resource_value: int = 1               # value when collected
    resource_limit: int = 1               # max amount of this resource
    # Portal fields
    portal_exit_stype: Optional[str] = None  # stype of the exit portal
    # Continuous/gravity physics fields
    physics_type: str = 'grid'           # 'grid', 'continuous', or 'gravity'
    mass: float = 1.0
    strength: float = 1.0               # force multiplier (VGDLSprite default)
    jump_strength: float = 10.0         # MarioAvatar upward impulse
    airsteering: bool = False            # MarioAvatar air control
    angle_diff: float = 0.05             # AimedAvatar rotation step (radians)


@dataclass
class EffectDef:
    effect_type: int
    actor_stype: str
    actee_stype: str
    score_change: int = 0
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminationDef:
    term_type: int
    win: bool
    score_change: int = 0
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LevelDef:
    height: int
    width: int
    # List of (type_idx, y, x) for each sprite to place
    initial_sprites: List[Tuple[int, int, int]]


@dataclass
class GameDef:
    sprites: List[SpriteDef]
    effects: List[EffectDef]
    terminations: List[TerminationDef]
    level: Optional[LevelDef]
    char_mapping: Dict[str, List[str]]
    sprite_order: List[str]
    stype_to_indices: Dict[str, List[int]]

    def type_idx(self, key: str) -> int:
        for s in self.sprites:
            if s.key == key:
                return s.type_idx
        raise KeyError(f"Unknown sprite key: {key}")

    def resolve_stype(self, stype: str) -> List[int]:
        return self.stype_to_indices.get(stype, [])
