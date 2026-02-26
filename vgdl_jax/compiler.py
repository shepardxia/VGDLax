"""
Compiler: converts a GameDef (parsed VGDL) into a jit-compiled step function
and an initial GameState.
"""
import math
import warnings
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable

import jax
import jax.numpy as jnp

from vgdl_jax.data_model import (
    GameDef, SpriteClass, TerminationType, PHYSICS_SCALE, STATIC_CLASSES,
)
from vgdl_jax.state import GameState, create_initial_state
from vgdl_jax.step import build_step_fn
from vgdl_jax.sprites import DIRECTION_DELTAS
from vgdl_jax.terminations import check_sprite_counter, check_multi_sprite_counter, check_timeout, check_resource_counter


@dataclass
class CompiledGame:
    init_state: GameState
    step_fn: Callable
    n_actions: int
    noop_action: int
    game_def: GameDef
    static_grid_map: dict  # type_idx → static_grid_idx (empty if no static types)


# Avatar type → (n_move_actions, can_shoot)
AVATAR_INFO = {
    SpriteClass.MOVING_AVATAR: (4, False),
    SpriteClass.ORIENTED_AVATAR: (4, False),
    SpriteClass.HORIZONTAL_AVATAR: (2, False),
    SpriteClass.FLAK_AVATAR: (2, True),
    SpriteClass.SHOOT_AVATAR: (4, True),
    SpriteClass.INERTIAL_AVATAR: (4, False),
    SpriteClass.MARIO_AVATAR: (5, False),  # LEFT, RIGHT, JUMP, J+L, J+R
    SpriteClass.VERTICAL_AVATAR: (2, False),
    SpriteClass.ROTATING_AVATAR: (4, False),
    SpriteClass.ROTATING_FLIPPING_AVATAR: (4, False),
    SpriteClass.NOISY_ROTATING_FLIPPING_AVATAR: (4, False),
    SpriteClass.SHOOT_EVERYWHERE_AVATAR: (4, True),
    SpriteClass.AIMED_AVATAR: (2, True),      # AIM_UP, AIM_DOWN + shoot
    SpriteClass.AIMED_FLAK_AVATAR: (4, True),  # LEFT, RIGHT, AIM_UP, AIM_DOWN + shoot
}

# Avatars whose action indices start at LEFT/RIGHT instead of UP/DOWN
_HORIZONTAL_AVATARS = {SpriteClass.HORIZONTAL_AVATAR, SpriteClass.FLAK_AVATAR}

def _resolve_first(game_def, stype, default=None):
    """Resolve stype to list of indices, return first or default."""
    if stype is None:
        return default
    indices = game_def.resolve_stype(stype)
    return indices[0] if indices else default


def compile_game(game_def: GameDef, max_sprites_per_type=None):
    """
    Compile a GameDef into a CompiledGame with jitted step function.
    """
    n_types = len(game_def.sprites)
    height = game_def.level.height
    width = game_def.level.width

    # ── 0. Find avatar early (needed for active type analysis) ─────────
    avatar_sd = None
    for sd in game_def.sprites:
        if sd.sprite_class in AVATAR_INFO:
            avatar_sd = sd
            break
    assert avatar_sd is not None, "No avatar found in game definition"

    # ── 0b. Build resource registry ────────────────────────────────────
    resource_name_to_idx = {}
    resource_limits = []
    # Scan Resource-class sprites for resource definitions
    for sd in game_def.sprites:
        if sd.sprite_class == SpriteClass.RESOURCE and sd.resource_name:
            if sd.resource_name not in resource_name_to_idx:
                resource_name_to_idx[sd.resource_name] = len(resource_limits)
                resource_limits.append(sd.resource_limit)
    # Also scan effects for resource= kwargs that may reference resources
    # not defined as Resource sprites (e.g. "safety" in frogs)
    for ed in game_def.effects:
        res_name = ed.kwargs.get('resource', None)
        if res_name and res_name not in resource_name_to_idx:
            # Find limit from any Resource sprite or effect kwargs
            limit = ed.kwargs.get('limit', 100)
            # Search for a Resource SpriteDef with this name
            for sd in game_def.sprites:
                if sd.resource_name == res_name:
                    limit = sd.resource_limit
                    break
            resource_name_to_idx[res_name] = len(resource_limits)
            resource_limits.append(limit)

    n_resource_types = len(resource_limits)

    # ── 0c. Identify static types ────────────────────────────────────────
    # A type is "static" if: sprite class is non-moving, speed == 0,
    # NOT a spawn/transform target, and NOT type_a in position-modifying effects.
    spawn_targets = set()
    for ed in game_def.effects:
        if ed.effect_type in ('transform_to', 'transform_others_to',
                               'spawn_if_has_more', 'clone_sprite'):
            stype = ed.kwargs.get('stype', '') or ed.kwargs.get('target', '')
            for idx in game_def.resolve_stype(stype):
                spawn_targets.add(idx)
    for sd in game_def.sprites:
        if sd.sprite_class in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER,
                                SpriteClass.SPREADER):
            if sd.spawner_stype:
                for idx in game_def.resolve_stype(sd.spawner_stype):
                    spawn_targets.add(idx)

    # Types that are type_a in force-move or state-modifying effects can't be static.
    # Revert-only effects (step_back, wall_stop, etc.) are safe for speed=0 types
    # because the sprite never moves → revert is a no-op.
    FORCE_MOVE_EFFECTS = {
        'bounce_forward', 'pull_with_it', 'convey_sprite',
        'wind_gust', 'slip_forward', 'teleport_to_exit', 'wrap_around',
    }
    MODIFY_TYPE_A_EFFECTS = {
        'transform_to', 'clone_sprite', 'change_resource',
    }
    DISQUALIFYING_EFFECTS = FORCE_MOVE_EFFECTS | MODIFY_TYPE_A_EFFECTS
    position_modified_types = set()
    for ed in game_def.effects:
        if ed.effect_type in DISQUALIFYING_EFFECTS:
            for idx in game_def.resolve_stype(ed.actor_stype):
                position_modified_types.add(idx)

    # Portals are in STATIC_CLASSES for NPC update skip, but teleport_to_exit
    # reads their positions from arrays, so exclude from static grid storage.
    STATIC_GRID_CLASSES = STATIC_CLASSES - {SpriteClass.PORTAL}

    # Also exclude types used as exit targets in teleport effects
    teleport_exit_types = set()
    for ed in game_def.effects:
        if ed.effect_type == 'teleport_to_exit':
            actee_indices = game_def.resolve_stype(ed.actee_stype)
            for aidx in actee_indices:
                portal_sd = game_def.sprites[aidx]
                if portal_sd.portal_exit_stype:
                    for eidx in game_def.resolve_stype(portal_sd.portal_exit_stype):
                        teleport_exit_types.add(eidx)

    static_type_indices = []  # ordered list of type_idx that are static
    static_type_set = set()
    for sd in game_def.sprites:
        if (sd.sprite_class in STATIC_GRID_CLASSES
                and sd.speed == 0
                and sd.type_idx not in spawn_targets
                and sd.type_idx not in position_modified_types
                and sd.type_idx not in teleport_exit_types):
            static_type_indices.append(sd.type_idx)
            static_type_set.add(sd.type_idx)

    static_grid_map = {ti: i for i, ti in enumerate(static_type_indices)}
    n_static = len(static_type_indices)

    # Auto-compute per-type max_n from level sprite counts + headroom
    if max_sprites_per_type is None:
        active_types = _find_active_types(game_def, avatar_sd)
        counts = defaultdict(int)
        for type_idx, _, _ in game_def.level.initial_sprites:
            counts[type_idx] += 1
        HEADROOM = 10
        type_max_n = []
        for idx in range(n_types):
            if idx in static_type_set:
                type_max_n.append(0)
            else:
                base = counts.get(idx, 0)
                type_max_n.append(max(base + HEADROOM, HEADROOM) if idx in active_types else 1)
        max_n = max(type_max_n)
        max_n = max(max_n, 10)
    else:
        max_n = max_sprites_per_type
        type_max_n = [max_n] * n_types

    # ── 1. Build initial state ─────────────────────────────────────────
    state = create_initial_state(n_types=n_types, max_n=max_n,
                                 height=height, width=width,
                                 n_resource_types=n_resource_types,
                                 n_static_types=n_static)

    # Place sprites from level: static types go into grids, others into arrays
    import numpy as np
    static_grid_data = np.zeros((max(n_static, 1), height, width), dtype=bool)
    slot_counts = defaultdict(int)
    for type_idx, row, col in game_def.level.initial_sprites:
        if type_idx in static_type_set:
            sg_idx = static_grid_map[type_idx]
            static_grid_data[sg_idx, row, col] = True
            continue
        slot = slot_counts[type_idx]
        if slot >= max_n:
            continue
        sd = game_def.sprites[type_idx]
        state = state.replace(
            positions=state.positions.at[type_idx, slot].set(
                jnp.array([row, col], dtype=jnp.float32)),
            alive=state.alive.at[type_idx, slot].set(True),
            orientations=state.orientations.at[type_idx, slot].set(
                jnp.array(sd.orientation, dtype=jnp.float32)),
            speeds=state.speeds.at[type_idx, slot].set(
                jnp.float32(sd.speed)),
            cooldown_timers=state.cooldown_timers.at[type_idx, slot].set(
                jnp.int32(0)),
        )
        slot_counts[type_idx] += 1
    state = state.replace(static_grids=jnp.array(static_grid_data))

    # Randomize orientations for RandomMissile sprites
    rng = jax.random.PRNGKey(0)
    for sd in game_def.sprites:
        if sd.sprite_class == SpriteClass.RANDOM_MISSILE:
            rng, key = jax.random.split(rng)
            n_placed = slot_counts.get(sd.type_idx, 0)
            if n_placed > 0:
                dir_indices = jax.random.randint(key, (n_placed,), 0, 4)
                random_oris = DIRECTION_DELTAS[dir_indices]
                state = state.replace(
                    orientations=state.orientations.at[
                        sd.type_idx, :n_placed].set(random_oris))

    # ── 2. Build avatar config ──────────────────────────────────────────
    n_move, can_shoot = AVATAR_INFO[avatar_sd.sprite_class]

    # Direction offset: horizontal avatars map actions to LEFT/RIGHT (index 2,3)
    direction_offset = 2 if avatar_sd.sprite_class in _HORIZONTAL_AVATARS else 0

    # Resolve projectile info for shooting avatars
    proj_type_idx = -1
    proj_ori_from_avatar = False
    proj_default_ori = [0., 0.]
    proj_speed = 0.0
    shoot_action_idx = -1

    if can_shoot and avatar_sd.spawner_stype:
        proj_type_idx = _resolve_first(game_def, avatar_sd.spawner_stype, -1)
        if proj_type_idx >= 0:
            proj_sd = game_def.sprites[proj_type_idx]
            proj_speed = proj_sd.speed
            if avatar_sd.sprite_class == SpriteClass.SHOOT_AVATAR:
                proj_ori_from_avatar = True
            else:
                proj_default_ori = list(proj_sd.orientation)
                proj_ori_from_avatar = False
        shoot_action_idx = n_move + 1  # NOOP is n_move, SHOOT is n_move+1

    n_actions = n_move + (1 if can_shoot else 0) + 1  # +1 for NOOP

    avatar_config = dict(
        avatar_type_idx=avatar_sd.type_idx,
        n_move_actions=n_move,
        cooldown=max(avatar_sd.cooldown, 1),
        can_shoot=can_shoot,
        shoot_action_idx=shoot_action_idx,
        projectile_type_idx=proj_type_idx,
        projectile_orientation_from_avatar=proj_ori_from_avatar,
        projectile_default_orientation=proj_default_ori,
        projectile_speed=proj_speed,
        direction_offset=direction_offset,
        # Continuous/gravity physics parameters (scaled from pixel to grid-cell units)
        physics_type=avatar_sd.physics_type,
        mass=avatar_sd.mass,
        strength=avatar_sd.strength / PHYSICS_SCALE,
        jump_strength=avatar_sd.jump_strength / PHYSICS_SCALE,
        airsteering=avatar_sd.airsteering,
        gravity=1.0 / PHYSICS_SCALE,  # py-vgdl GravityPhysics.gravity = 1 (pixel unit)
        # Rotating avatar parameters
        is_rotating=avatar_sd.sprite_class in (
            SpriteClass.ROTATING_AVATAR,
            SpriteClass.ROTATING_FLIPPING_AVATAR,
            SpriteClass.NOISY_ROTATING_FLIPPING_AVATAR),
        is_flipping=avatar_sd.sprite_class in (
            SpriteClass.ROTATING_FLIPPING_AVATAR,
            SpriteClass.NOISY_ROTATING_FLIPPING_AVATAR),
        noise_level=0.4 if avatar_sd.sprite_class == SpriteClass.NOISY_ROTATING_FLIPPING_AVATAR else 0.0,
        shoot_everywhere=(avatar_sd.sprite_class == SpriteClass.SHOOT_EVERYWHERE_AVATAR),
        is_aimed=avatar_sd.sprite_class in (SpriteClass.AIMED_AVATAR, SpriteClass.AIMED_FLAK_AVATAR),
        can_move_aimed=avatar_sd.sprite_class == SpriteClass.AIMED_FLAK_AVATAR,
        angle_diff=avatar_sd.angle_diff,
    )

    # ── 3. Build sprite configs ────────────────────────────────────────
    sprite_configs = []
    for sd in game_def.sprites:
        cfg = dict(
            sprite_class=sd.sprite_class,
            cooldown=max(sd.cooldown, 1),
            flicker_limit=sd.flicker_limit,
        )

        if sd.sprite_class in (SpriteClass.CHASER, SpriteClass.FLEEING):
            cfg['target_type_idx'] = _resolve_first(game_def, sd.spawner_stype, 0)

        elif sd.sprite_class == SpriteClass.SPREADER:
            cfg['spreadprob'] = sd.spawner_prob  # prob param doubles as spreadprob

        elif sd.sprite_class == SpriteClass.ERRATIC_MISSILE:
            cfg['prob'] = sd.spawner_prob  # probability of direction change per tick

        elif sd.sprite_class == SpriteClass.RANDOM_INERTIAL:
            cfg['mass'] = sd.mass
            cfg['strength'] = sd.strength / PHYSICS_SCALE

        elif sd.sprite_class == SpriteClass.WALK_JUMPER:
            cfg['mass'] = sd.mass
            cfg['strength'] = sd.strength / PHYSICS_SCALE
            cfg['prob'] = sd.spawner_prob  # jump probability threshold
            cfg['gravity'] = 1.0 / PHYSICS_SCALE

        elif sd.sprite_class in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER):
            target_idx = _resolve_first(game_def, sd.spawner_stype, -1)
            cfg['target_type_idx'] = target_idx if target_idx >= 0 else 0
            cfg['prob'] = sd.spawner_prob
            cfg['total'] = sd.spawner_total
            if target_idx >= 0:
                target_sd = game_def.sprites[target_idx]
                cfg['target_orientation'] = list(target_sd.orientation)
                cfg['target_speed'] = target_sd.speed
            else:
                cfg['target_orientation'] = [0., 0.]
                cfg['target_speed'] = 0.0

        sprite_configs.append(cfg)

    # ── 4. Build effects ───────────────────────────────────────────────
    # Determine which types use continuous physics (for per-pair AABB)
    continuous_types = {sd.type_idx for sd in game_def.sprites
                        if sd.physics_type in ('continuous', 'gravity')}
    # Types with fractional speed also need AABB collision
    fractional_speed_types = {sd.type_idx for sd in game_def.sprites
                               if sd.speed != 1.0 and sd.speed > 0}
    # Types with non-integer positions (continuous physics or fractional speed)
    frac_or_cont = continuous_types | fractional_speed_types
    # Build speed lookup for sweep flag computation
    _speed_by_idx = {sd.type_idx: sd.speed for sd in game_def.sprites}

    compiled_effects = []
    for ed in game_def.effects:
        is_eos = (ed.actee_stype == 'EOS')
        actor_indices = game_def.resolve_stype(ed.actor_stype)

        if is_eos:
            for ta_idx in actor_indices:
                # EOS effects never involve static types (they check bounds)
                compiled_effects.append(dict(
                    type_a=ta_idx,
                    is_eos=True,
                    effect_type=ed.effect_type,
                    score_change=ed.score_change,
                    max_a=type_max_n[ta_idx],
                    static_a_grid_idx=static_grid_map.get(ta_idx),
                    static_b_grid_idx=None,
                    kwargs=_compile_effect_kwargs(
                        ed, game_def, resource_name_to_idx, resource_limits,
                        concrete_actor_idx=ta_idx,
                        avatar_type_idx=avatar_sd.type_idx),
                ))
        else:
            actee_indices = game_def.resolve_stype(ed.actee_stype)
            for ta_idx in actor_indices:
                for tb_idx in actee_indices:
                    speed_a = _speed_by_idx.get(ta_idx, 1.0)
                    speed_b = _speed_by_idx.get(tb_idx, 1.0)
                    a_static = ta_idx in static_type_set
                    b_static = tb_idx in static_type_set
                    # Collision mode: static grid > sweep > expanded_grid > aabb > grid
                    a_frac = ta_idx in frac_or_cont
                    b_frac = tb_idx in frac_or_cont
                    if a_static and b_static:
                        # Both static — collision is compile-time constant, skip
                        collision_mode = 'static_both'
                    elif b_static:
                        # type_b is a static grid — collision is a direct grid lookup
                        if a_frac:
                            collision_mode = 'static_b_expanded'
                        else:
                            collision_mode = 'static_b_grid'
                    elif a_static:
                        # type_a is a static grid
                        collision_mode = 'static_a_grid'
                    elif speed_a > 1.0 or speed_b > 1.0:
                        collision_mode = 'sweep'
                    elif a_frac and not b_frac:
                        collision_mode = 'expanded_grid_a'
                    elif b_frac and not a_frac:
                        collision_mode = 'expanded_grid_b'
                    elif a_frac and b_frac:
                        collision_mode = 'aabb'
                    else:
                        collision_mode = 'grid'
                    compiled_effects.append(dict(
                        type_a=ta_idx,
                        type_b=tb_idx,
                        is_eos=False,
                        effect_type=ed.effect_type,
                        score_change=ed.score_change,
                        collision_mode=collision_mode,
                        max_speed_cells=max(1, math.ceil(max(speed_a, speed_b))),
                        max_a=type_max_n[ta_idx],
                        max_b=type_max_n[tb_idx],
                        static_a_grid_idx=static_grid_map.get(ta_idx),
                        static_b_grid_idx=static_grid_map.get(tb_idx),
                        kwargs=_compile_effect_kwargs(
                            ed, game_def, resource_name_to_idx, resource_limits,
                            concrete_actor_idx=ta_idx, concrete_actee_idx=tb_idx,
                            avatar_type_idx=avatar_sd.type_idx),
                    ))

    # ── 5. Build terminations ──────────────────────────────────────────
    compiled_terminations = []
    for td in game_def.terminations:
        if td.term_type == TerminationType.SPRITE_COUNTER:
            stype = td.kwargs.get('stype', '')
            indices = game_def.resolve_stype(stype)
            # Split into dynamic (alive array) and static (grid) indices
            dyn_idx = [i for i in indices if i not in static_type_set]
            sg_idx = [static_grid_map[i] for i in indices if i in static_type_set]
            limit = td.kwargs.get('limit', 0)
            win = td.win
            check_fn = lambda s, _di=dyn_idx, _si=sg_idx, _lim=limit, _win=win: \
                check_sprite_counter(s, _di, _lim, _win, _si)
            compiled_terminations.append((check_fn, td.score_change))

        elif td.term_type == TerminationType.MULTI_SPRITE_COUNTER:
            stypes_list = td.kwargs.get('stypes', [])
            indices_list = [game_def.resolve_stype(st) for st in stypes_list]
            # Split each stype group into dynamic and static
            dyn_indices_list = [[i for i in grp if i not in static_type_set]
                                for grp in indices_list]
            sg_indices_list = [[static_grid_map[i] for i in grp if i in static_type_set]
                               for grp in indices_list]
            limit = td.kwargs.get('limit', 0)
            win = td.win
            check_fn = lambda s, _di=dyn_indices_list, _si=sg_indices_list, _lim=limit, _win=win: \
                check_multi_sprite_counter(s, _di, _lim, _win, _si)
            compiled_terminations.append((check_fn, td.score_change))

        elif td.term_type == TerminationType.RESOURCE_COUNTER:
            res_name = td.kwargs.get('resource', '')
            r_idx = resource_name_to_idx.get(res_name, 0)
            limit = td.kwargs.get('limit', 0)
            win = td.win
            check_fn = lambda s, _ati=avatar_sd.type_idx, _ri=r_idx, _lim=limit, _win=win: \
                check_resource_counter(s, _ati, _ri, _lim, _win)
            compiled_terminations.append((check_fn, td.score_change))

        elif td.term_type == TerminationType.TIMEOUT:
            limit = td.kwargs.get('limit', 0)
            win = td.win
            check_fn = lambda s, _lim=limit, _win=win: \
                check_timeout(s, _lim, _win)
            compiled_terminations.append((check_fn, td.score_change))

    # ── 5b. Compile-time validation ─────────────────────────────────────
    if can_shoot and proj_type_idx < 0:
        warnings.warn(f"Avatar can_shoot=True but projectile type not resolved "
                       f"(spawner_stype={avatar_sd.spawner_stype!r})")
    for cfg in sprite_configs:
        sc = cfg['sprite_class']
        if sc in (SpriteClass.CHASER, SpriteClass.FLEEING):
            if cfg.get('target_type_idx', 0) == 0 and not game_def.sprites[0].sprite_class in AVATAR_INFO:
                warnings.warn(f"Chaser/Fleeing target defaulted to type 0")
        if sc in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER):
            if cfg.get('target_type_idx', 0) == 0:
                warnings.warn(f"SpawnPoint/Bomber target defaulted to type 0")

    # ── 6. Build step function ─────────────────────────────────────────
    params = dict(n_types=n_types, max_n=max_n, height=height, width=width,
                  n_resource_types=max(n_resource_types, 1),
                  resource_limits=resource_limits)
    step_fn = build_step_fn(compiled_effects, compiled_terminations,
                            sprite_configs, avatar_config, params)

    noop_action = n_move  # NOOP is right after movement actions

    return CompiledGame(
        init_state=state,
        step_fn=step_fn,
        n_actions=n_actions,
        noop_action=noop_action,
        game_def=game_def,
        static_grid_map=static_grid_map,
    )



def _resolve_collect_resource_kwargs(ed, game_def, concrete_actor_idx,
                                      resource_name_to_idx, resource_limits):
    """Resolve resource kwargs for COLLECT_RESOURCE / AVATAR_COLLECT_RESOURCE."""
    if concrete_actor_idx is not None:
        res_sd = game_def.sprites[concrete_actor_idx]
    else:
        actor_indices = game_def.resolve_stype(ed.actor_stype)
        res_sd = game_def.sprites[actor_indices[0]] if actor_indices else None
    kwargs = {}
    if res_sd is not None:
        res_name = res_sd.resource_name or res_sd.key
        kwargs['resource_idx'] = resource_name_to_idx.get(res_name, 0)
        kwargs['resource_value'] = res_sd.resource_value
        kwargs['limit'] = resource_limits[kwargs['resource_idx']] if resource_limits else 100
    return kwargs


def _compile_effect_kwargs(ed, game_def, resource_name_to_idx, resource_limits,
                           concrete_actor_idx=None, concrete_actee_idx=None,
                           avatar_type_idx=None):
    """Compile effect kwargs, resolving stype references to type indices."""
    et = ed.effect_type  # string key (e.g. 'kill_sprite')
    kwargs = {}
    if et == 'transform_to':
        idx = _resolve_first(game_def, ed.kwargs.get('stype', ''))
        if idx is not None:
            kwargs['new_type_idx'] = idx

    elif et == 'change_resource':
        res_name = ed.kwargs.get('resource', '')
        kwargs['resource_idx'] = resource_name_to_idx.get(res_name, 0)
        kwargs['value'] = ed.kwargs.get('value', 0)
        kwargs['limit'] = resource_limits[kwargs['resource_idx']] if resource_limits else 100

    elif et == 'collect_resource':
        kwargs.update(_resolve_collect_resource_kwargs(
            ed, game_def, concrete_actor_idx, resource_name_to_idx, resource_limits))

    elif et in ('kill_if_has_less', 'kill_if_has_more',
                'kill_if_other_has_more', 'kill_if_other_has_less'):
        res_name = ed.kwargs.get('resource', '')
        kwargs['resource_idx'] = resource_name_to_idx.get(res_name, 0)
        kwargs['limit'] = ed.kwargs.get('limit', 0)

    elif et == 'kill_if_slow':
        kwargs['limitspeed'] = ed.kwargs.get('limitspeed', 0.0)

    elif et == 'convey_sprite':
        if concrete_actee_idx is not None:
            partner_sd = game_def.sprites[concrete_actee_idx]
            kwargs['strength'] = partner_sd.strength
        else:
            kwargs['strength'] = 1.0

    elif et == 'spawn_if_has_more':
        res_name = ed.kwargs.get('resource', '')
        kwargs['resource_idx'] = resource_name_to_idx.get(res_name, 0)
        kwargs['limit'] = ed.kwargs.get('limit', 0)
        idx = _resolve_first(game_def, ed.kwargs.get('stype', ''))
        if idx is not None:
            kwargs['spawn_type_idx'] = idx

    elif et == 'slip_forward':
        kwargs['prob'] = float(ed.kwargs.get('prob', 0.5))

    elif et == 'attract_gaze':
        kwargs['prob'] = float(ed.kwargs.get('prob', 0.5))

    elif et == 'spend_resource':
        res_name = ed.kwargs.get('resource', ed.kwargs.get('target', ''))
        kwargs['resource_idx'] = resource_name_to_idx.get(res_name, 0)
        kwargs['amount'] = int(ed.kwargs.get('amount', 1))

    elif et == 'spend_avatar_resource':
        res_name = ed.kwargs.get('resource', ed.kwargs.get('target', ''))
        kwargs['resource_idx'] = resource_name_to_idx.get(res_name, 0)
        kwargs['amount'] = int(ed.kwargs.get('amount', 1))
        kwargs['avatar_type_idx'] = avatar_type_idx

    elif et == 'kill_others':
        idx = _resolve_first(game_def, ed.kwargs.get('stype', ed.kwargs.get('target', '')))
        if idx is not None:
            kwargs['kill_type_idx'] = idx

    elif et == 'kill_if_avatar_without_resource':
        res_name = ed.kwargs.get('resource', ed.kwargs.get('target', ''))
        kwargs['resource_idx'] = resource_name_to_idx.get(res_name, 0)
        kwargs['avatar_type_idx'] = avatar_type_idx

    elif et == 'avatar_collect_resource':
        kwargs.update(_resolve_collect_resource_kwargs(
            ed, game_def, concrete_actor_idx, resource_name_to_idx, resource_limits))
        kwargs['avatar_type_idx'] = avatar_type_idx

    elif et == 'transform_others_to':
        idx = _resolve_first(game_def, ed.kwargs.get('target', ''))
        if idx is not None:
            kwargs['target_type_idx'] = idx
        idx = _resolve_first(game_def, ed.kwargs.get('stype', ''))
        if idx is not None:
            kwargs['new_type_idx'] = idx

    elif et in ('wall_stop', 'wall_bounce', 'bounce_direction'):
        if 'friction' in ed.kwargs:
            kwargs['friction'] = float(ed.kwargs['friction'])

    elif et == 'teleport_to_exit':
        if concrete_actee_idx is not None:
            portal_sd = game_def.sprites[concrete_actee_idx]
        else:
            actee_idx = _resolve_first(game_def, ed.actee_stype)
            portal_sd = game_def.sprites[actee_idx] if actee_idx is not None else None
        if portal_sd is not None and portal_sd.portal_exit_stype:
            exit_idx = _resolve_first(game_def, portal_sd.portal_exit_stype)
            if exit_idx is not None:
                kwargs['exit_type_idx'] = exit_idx

    return kwargs


def _find_active_types(game_def, avatar_sd):
    """Find type indices that participate in game logic.

    Inert types (e.g. hidden backgrounds) are excluded so they don't inflate max_n.
    Loops until stable to handle transitive spawner chains.
    """
    active = set()

    # Avatar and its projectile
    active.add(avatar_sd.type_idx)
    if avatar_sd.spawner_stype:
        for idx in game_def.resolve_stype(avatar_sd.spawner_stype):
            active.add(idx)

    # Types referenced in effects
    for ed in game_def.effects:
        for idx in game_def.resolve_stype(ed.actor_stype):
            active.add(idx)
        if ed.actee_stype != 'EOS':
            for idx in game_def.resolve_stype(ed.actee_stype):
                active.add(idx)

    # Types referenced in terminations
    for td in game_def.terminations:
        if td.term_type == TerminationType.SPRITE_COUNTER:
            for idx in game_def.resolve_stype(td.kwargs.get('stype', '')):
                active.add(idx)
        elif td.term_type == TerminationType.MULTI_SPRITE_COUNTER:
            for stype in td.kwargs.get('stypes', []):
                for idx in game_def.resolve_stype(stype):
                    active.add(idx)

    # Spawner targets of active types — loop until stable
    prev_size = 0
    while len(active) != prev_size:
        prev_size = len(active)
        for sd in game_def.sprites:
            if sd.type_idx in active and sd.spawner_stype:
                for idx in game_def.resolve_stype(sd.spawner_stype):
                    active.add(idx)

    return active
