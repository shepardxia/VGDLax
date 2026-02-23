"""
Compiler: converts a GameDef (parsed VGDL) into a jit-compiled step function
and an initial GameState.
"""
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable

import jax
import jax.numpy as jnp

from vgdl_jax.data_model import (
    GameDef, SpriteClass, EffectType, TerminationType,
)
from vgdl_jax.state import GameState, create_initial_state
from vgdl_jax.step import build_step_fn
from vgdl_jax.terminations import check_sprite_counter, check_multi_sprite_counter, check_timeout


@dataclass
class CompiledGame:
    init_state: GameState
    step_fn: Callable
    n_actions: int
    game_def: GameDef


# Avatar type → (n_move_actions, can_shoot)
AVATAR_INFO = {
    SpriteClass.MOVING_AVATAR: (4, False),
    SpriteClass.ORIENTED_AVATAR: (4, False),
    SpriteClass.HORIZONTAL_AVATAR: (2, False),
    SpriteClass.FLAK_AVATAR: (2, True),
    SpriteClass.SHOOT_AVATAR: (4, True),
    SpriteClass.INERTIAL_AVATAR: (4, False),
    SpriteClass.MARIO_AVATAR: (5, False),  # LEFT, RIGHT, JUMP, J+L, J+R
}

# Avatars whose action indices start at LEFT/RIGHT instead of UP/DOWN
_HORIZONTAL_AVATARS = {SpriteClass.HORIZONTAL_AVATAR, SpriteClass.FLAK_AVATAR}

# py-vgdl physics operates in pixel coordinates where 1 grid cell = block_size pixels.
# vgdl-jax positions are in grid-cell units. All physics constants (forces, velocities)
# from VGDL files are in pixel units and must be divided by this scale factor.
PHYSICS_SCALE = 24


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

    # Auto-compute max_n from level sprite counts + headroom
    if max_sprites_per_type is None:
        active_types = _find_active_types(game_def, avatar_sd)
        counts = defaultdict(int)
        for type_idx, _, _ in game_def.level.initial_sprites:
            counts[type_idx] += 1
        # Only count active types for max_n sizing
        active_counts = [counts.get(idx, 0) for idx in active_types
                         if idx in counts]
        max_n = max(active_counts, default=1) + 10
        max_n = max(max_n, 10)
    else:
        max_n = max_sprites_per_type

    # ── 1. Build initial state ─────────────────────────────────────────
    state = create_initial_state(n_types=n_types, max_n=max_n,
                                 height=height, width=width,
                                 n_resource_types=n_resource_types)

    # Place sprites from level
    slot_counts = defaultdict(int)
    for type_idx, row, col in game_def.level.initial_sprites:
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
                jnp.int32(sd.cooldown)),
        )
        slot_counts[type_idx] += 1

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
        proj_key = avatar_sd.spawner_stype
        proj_indices = game_def.resolve_stype(proj_key)
        if proj_indices:
            proj_type_idx = proj_indices[0]
            proj_sd = game_def.sprites[proj_type_idx]
            proj_speed = proj_sd.speed
            if avatar_sd.sprite_class == SpriteClass.SHOOT_AVATAR:
                proj_ori_from_avatar = True
            else:
                proj_default_ori = list(proj_sd.orientation)
                proj_ori_from_avatar = False
        shoot_action_idx = n_move

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
    )

    # ── 3. Build sprite configs ────────────────────────────────────────
    sprite_configs = []
    for sd in game_def.sprites:
        # Compute effective cooldown from speed
        base_cooldown = max(sd.cooldown, 1)
        if sd.speed > 0 and sd.speed != 1.0:
            effective_cooldown = max(1, round(base_cooldown / sd.speed))
        else:
            effective_cooldown = base_cooldown

        cfg = dict(
            sprite_class=sd.sprite_class,
            cooldown=effective_cooldown,
            flicker_limit=sd.flicker_limit,
        )

        if sd.sprite_class in (SpriteClass.CHASER, SpriteClass.FLEEING):
            target_key = sd.spawner_stype
            target_indices = game_def.resolve_stype(target_key) if target_key else []
            cfg['target_type_idx'] = target_indices[0] if target_indices else 0

        elif sd.sprite_class in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER):
            target_key = sd.spawner_stype
            target_indices = game_def.resolve_stype(target_key) if target_key else []
            cfg['target_type_idx'] = target_indices[0] if target_indices else 0
            cfg['prob'] = sd.spawner_prob
            cfg['total'] = sd.spawner_total
            if target_indices:
                target_sd = game_def.sprites[target_indices[0]]
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

    compiled_effects = []
    for ed in game_def.effects:
        is_eos = (ed.actee_stype == 'EOS')
        actor_indices = game_def.resolve_stype(ed.actor_stype)

        if is_eos:
            for ta_idx in actor_indices:
                compiled_effects.append(dict(
                    type_a=ta_idx,
                    is_eos=True,
                    effect_type=_effect_type_str(ed.effect_type),
                    score_change=ed.score_change,
                    kwargs=_compile_effect_kwargs(
                        ed, game_def, resource_name_to_idx, resource_limits,
                        concrete_actor_idx=ta_idx),
                ))
        else:
            actee_indices = game_def.resolve_stype(ed.actee_stype)
            for ta_idx in actor_indices:
                for tb_idx in actee_indices:
                    compiled_effects.append(dict(
                        type_a=ta_idx,
                        type_b=tb_idx,
                        is_eos=False,
                        effect_type=_effect_type_str(ed.effect_type),
                        score_change=ed.score_change,
                        use_aabb=(ta_idx in continuous_types or
                                  tb_idx in continuous_types),
                        kwargs=_compile_effect_kwargs(
                            ed, game_def, resource_name_to_idx, resource_limits,
                            concrete_actor_idx=ta_idx, concrete_actee_idx=tb_idx),
                    ))

    # ── 5. Build terminations ──────────────────────────────────────────
    compiled_terminations = []
    for td in game_def.terminations:
        if td.term_type == TerminationType.SPRITE_COUNTER:
            stype = td.kwargs.get('stype', '')
            indices = game_def.resolve_stype(stype)
            limit = td.kwargs.get('limit', 0)
            win = td.win
            check_fn = lambda s, _idx=indices, _lim=limit, _win=win: \
                check_sprite_counter(s, _idx, _lim, _win)
            compiled_terminations.append((check_fn, td.score_change))

        elif td.term_type == TerminationType.MULTI_SPRITE_COUNTER:
            stypes_list = td.kwargs.get('stypes', [])
            indices_list = [game_def.resolve_stype(st) for st in stypes_list]
            limit = td.kwargs.get('limit', 0)
            win = td.win
            check_fn = lambda s, _idxs=indices_list, _lim=limit, _win=win: \
                check_multi_sprite_counter(s, _idxs, _lim, _win)
            compiled_terminations.append((check_fn, td.score_change))

        elif td.term_type == TerminationType.TIMEOUT:
            limit = td.kwargs.get('limit', 0)
            win = td.win
            check_fn = lambda s, _lim=limit, _win=win: \
                check_timeout(s, _lim, _win)
            compiled_terminations.append((check_fn, td.score_change))

    # ── 6. Build step function ─────────────────────────────────────────
    params = dict(n_types=n_types, max_n=max_n, height=height, width=width,
                  n_resource_types=max(n_resource_types, 1),
                  resource_limits=resource_limits)
    step_fn = build_step_fn(compiled_effects, compiled_terminations,
                            sprite_configs, avatar_config, params)

    return CompiledGame(
        init_state=state,
        step_fn=step_fn,
        n_actions=n_actions,
        game_def=game_def,
    )


def _effect_type_str(effect_type_int):
    """Map EffectType enum int to the string expected by step.py."""
    mapping = {
        EffectType.KILL_SPRITE: 'kill_sprite',
        EffectType.KILL_BOTH: 'kill_both',
        EffectType.STEP_BACK: 'step_back',
        EffectType.TRANSFORM_TO: 'transform_to',
        EffectType.TURN_AROUND: 'turn_around',
        EffectType.REVERSE_DIRECTION: 'reverse_direction',
        EffectType.NULL: 'null',
        EffectType.CHANGE_RESOURCE: 'change_resource',
        EffectType.COLLECT_RESOURCE: 'collect_resource',
        EffectType.KILL_IF_HAS_LESS: 'kill_if_has_less',
        EffectType.KILL_IF_HAS_MORE: 'kill_if_has_more',
        EffectType.KILL_IF_OTHER_HAS_MORE: 'kill_if_other_has_more',
        EffectType.KILL_IF_OTHER_HAS_LESS: 'kill_if_other_has_less',
        EffectType.KILL_IF_FROM_ABOVE: 'kill_if_from_above',
        EffectType.WRAP_AROUND: 'wrap_around',
        EffectType.BOUNCE_FORWARD: 'bounce_forward',
        EffectType.UNDO_ALL: 'undo_all',
        EffectType.TELEPORT_TO_EXIT: 'teleport_to_exit',
        EffectType.PULL_WITH_IT: 'pull_with_it',
        EffectType.WALL_STOP: 'wall_stop',
        EffectType.WALL_BOUNCE: 'wall_bounce',
        EffectType.BOUNCE_DIRECTION: 'bounce_direction',
    }
    return mapping.get(effect_type_int, 'null')


def _compile_effect_kwargs(ed, game_def, resource_name_to_idx, resource_limits,
                           concrete_actor_idx=None, concrete_actee_idx=None):
    """Compile effect kwargs, resolving stype references to type indices."""
    kwargs = {}
    if ed.effect_type == EffectType.TRANSFORM_TO:
        stype = ed.kwargs.get('stype', '')
        indices = game_def.resolve_stype(stype)
        if indices:
            kwargs['new_type_idx'] = indices[0]

    elif ed.effect_type in (EffectType.CHANGE_RESOURCE,):
        res_name = ed.kwargs.get('resource', '')
        kwargs['resource_idx'] = resource_name_to_idx.get(res_name, 0)
        kwargs['value'] = ed.kwargs.get('value', 0)
        kwargs['limit'] = resource_limits[kwargs['resource_idx']] if resource_limits else 100

    elif ed.effect_type == EffectType.COLLECT_RESOURCE:
        # The actor is a Resource sprite — look up its resource info
        if concrete_actor_idx is not None:
            res_sd = game_def.sprites[concrete_actor_idx]
        else:
            actor_indices = game_def.resolve_stype(ed.actor_stype)
            res_sd = game_def.sprites[actor_indices[0]] if actor_indices else None
        if res_sd is not None:
            res_name = res_sd.resource_name or res_sd.key
            kwargs['resource_idx'] = resource_name_to_idx.get(res_name, 0)
            kwargs['resource_value'] = res_sd.resource_value
            kwargs['limit'] = resource_limits[kwargs['resource_idx']] if resource_limits else 100

    elif ed.effect_type in (EffectType.KILL_IF_HAS_LESS, EffectType.KILL_IF_HAS_MORE):
        res_name = ed.kwargs.get('resource', '')
        kwargs['resource_idx'] = resource_name_to_idx.get(res_name, 0)
        kwargs['limit'] = ed.kwargs.get('limit', 0)

    elif ed.effect_type in (EffectType.KILL_IF_OTHER_HAS_MORE,
                             EffectType.KILL_IF_OTHER_HAS_LESS):
        res_name = ed.kwargs.get('resource', '')
        kwargs['resource_idx'] = resource_name_to_idx.get(res_name, 0)
        kwargs['limit'] = ed.kwargs.get('limit', 0)

    elif ed.effect_type in (EffectType.WALL_STOP, EffectType.WALL_BOUNCE,
                             EffectType.BOUNCE_DIRECTION):
        if 'friction' in ed.kwargs:
            kwargs['friction'] = float(ed.kwargs['friction'])

    elif ed.effect_type == EffectType.TELEPORT_TO_EXIT:
        # Resolve exit type from the concrete portal's stype
        if concrete_actee_idx is not None:
            portal_sd = game_def.sprites[concrete_actee_idx]
        else:
            actee_indices = game_def.resolve_stype(ed.actee_stype)
            portal_sd = game_def.sprites[actee_indices[0]] if actee_indices else None
        if portal_sd is not None and portal_sd.portal_exit_stype:
            exit_indices = game_def.resolve_stype(portal_sd.portal_exit_stype)
            if exit_indices:
                kwargs['exit_type_idx'] = exit_indices[0]

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
