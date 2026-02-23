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


class TerminationType:
    SPRITE_COUNTER = 0
    MULTI_SPRITE_COUNTER = 1
    TIMEOUT = 2


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
