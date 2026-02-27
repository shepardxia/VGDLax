import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# py-vgdl physics operates in pixel coordinates where 1 grid cell = block_size pixels.
# vgdl-jax positions are in grid-cell units. All physics constants (forces, velocities)
# from VGDL files are in pixel units and must be divided by this scale factor.
PHYSICS_SCALE = 24

# Named constants
AABB_THRESHOLD = 1.0 - 1e-3        # collision overlap threshold (1.0 - AABB_EPS)
N_DIRECTIONS = 4                     # cardinal directions (UP, DOWN, LEFT, RIGHT)
DEFAULT_RESOURCE_LIMIT = 100         # fallback resource capacity
NOISY_AVATAR_NOISE_LEVEL = 0.4      # NoisyRotatingFlippingAvatar noise probability
GRAVITY_ACCEL = 1.0 / PHYSICS_SCALE  # standard gravity in grid-cell units
SPRITE_HEADROOM = 10                 # extra slots per type beyond level count

# Effects that modify sprite positions (used to determine occupancy-grid cache
# safety in step.py and chaser-target stability in compiler.py).
POSITION_MODIFYING_EFFECTS = frozenset({
    'step_back', 'wall_stop', 'wall_bounce', 'bounce_direction',
    'bounce_forward', 'pull_with_it', 'wrap_around', 'teleport_to_exit',
    'convey_sprite', 'wind_gust', 'slip_forward', 'undo_all', 'turn_around',
})

# Effects that require per-pair partner identity (partner_idx) from collision.
# All other effects use only the boolean collision mask.
PARTNER_IDX_EFFECTS = frozenset({
    'kill_both', 'kill_if_from_above',
    'kill_if_other_has_more', 'kill_if_other_has_less',
    'collect_resource', 'convey_sprite', 'wind_gust', 'attract_gaze',
    'bounce_forward', 'pull_with_it',
})


class SpriteClass(enum.IntEnum):
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


class TerminationType(enum.IntEnum):
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
    effect_type: str   # internal key (e.g. 'kill_sprite'), see effects.EFFECT_REGISTRY
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


@dataclass
class CompiledEffect:
    type_a: int
    is_eos: bool
    effect_type: str
    score_change: int
    max_a: int
    static_a_grid_idx: Optional[int] = None
    static_b_grid_idx: Optional[int] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    # Non-EOS fields (ignored when is_eos=True)
    type_b: int = -1
    collision_mode: str = 'grid'
    max_speed_cells: int = 1
    max_b: int = 0


@dataclass
class AvatarConfig:
    avatar_type_idx: int
    n_move_actions: int
    cooldown: int
    can_shoot: bool
    shoot_action_idx: int = -1
    projectile_type_idx: int = -1
    projectile_orientation_from_avatar: bool = False
    projectile_default_orientation: Tuple[float, float] = (0.0, 0.0)
    projectile_speed: float = 0.0
    direction_offset: int = 0
    physics_type: str = 'grid'
    mass: float = 1.0
    strength: float = 1.0
    jump_strength: float = 1.0
    airsteering: bool = False
    gravity: float = 1.0
    is_rotating: bool = False
    is_flipping: bool = False
    noise_level: float = 0.0
    shoot_everywhere: bool = False
    is_aimed: bool = False
    can_move_aimed: bool = False
    angle_diff: float = 0.05


@dataclass
class SpriteConfig:
    sprite_class: int
    cooldown: int
    flicker_limit: int = 0
    target_type_idx: int = -1
    prob: float = 1.0
    total: int = 0
    spreadprob: float = 1.0
    mass: float = 1.0
    strength: float = 1.0
    gravity: float = 0.0
    spawn_cooldown: int = 1
    target_orientation: Tuple[float, float] = (0.0, 0.0)
    target_speed: float = 0.0
