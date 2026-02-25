"""
The jittable step function: combines sprite updates, collision detection,
effects, and termination checks into a single state → state transition.

Collision detection uses occupancy grids [height, width] for O(max_n) checks
instead of O(max_n²) pairwise comparisons. Effects are applied via boolean
masks over the sprite arrays (see effects.py for all 37 effect handlers).
"""
import jax
import jax.numpy as jnp
from vgdl_jax.state import GameState
from vgdl_jax.collision import detect_eos, in_bounds, AABB_EPS
from vgdl_jax.effects import apply_masked_effect
from vgdl_jax.sprites import (
    DIRECTION_DELTAS, spawn_sprite,
    update_inertial_avatar, update_mario_avatar,
    update_missile, update_erratic_missile, update_random_npc,
    update_random_inertial, update_spreader, update_chaser,
    update_spawn_point, update_walk_jumper,
)
from vgdl_jax.terminations import check_all_terminations
from vgdl_jax.data_model import SpriteClass, STATIC_CLASSES, AVATAR_CLASSES, PHYSICS_SCALE


def build_step_fn(effects, terminations, sprite_configs, avatar_config, params):
    """
    Build a jit-compiled step function from compiled game configuration.

    Args:
        effects: list of dicts with keys:
            - type_a, type_b: int type indices
            - is_eos: bool (if True, type_b is ignored, check bounds instead)
            - effect_type: str
            - score_change: int
            - kwargs: dict
        terminations: list of (check_fn, score_change) tuples
        sprite_configs: list of dicts per type with keys:
            - sprite_class: int (SpriteClass enum)
            - cooldown: int
            - flicker_limit: int
            - (class-specific keys: target_type_idx, prob, total, etc.)
        avatar_config: dict with keys:
            - avatar_type_idx, n_move_actions, cooldown, can_shoot,
              shoot_action_idx, projectile_type_idx,
              projectile_orientation_from_avatar, projectile_default_orientation,
              projectile_speed, direction_offset, physics_type
        params: dict with keys:
            - n_types, max_n, height, width

    Returns:
        A function step(state, action) → state
    """
    n_types = params['n_types']
    max_n = params['max_n']
    height = params['height']
    width = params['width']

    def _step_inner(state: GameState, action: int) -> GameState:
        # Save previous positions for stepBack / undoAll / bounceForward
        prev_positions = state.positions

        # 1. Increment cooldown timers for alive sprites
        state = state.replace(
            cooldown_timers=jnp.where(
                state.alive, state.cooldown_timers + 1,
                state.cooldown_timers))

        # 2. Update avatar (dispatch based on compile-time physics_type)
        avatar_physics = avatar_config.get('physics_type', 'grid')
        if avatar_physics == 'continuous':
            state = update_inertial_avatar(
                state, action,
                avatar_type=avatar_config['avatar_type_idx'],
                n_move=avatar_config['n_move_actions'],
                mass=avatar_config['mass'],
                strength=avatar_config['strength'])
        elif avatar_physics == 'gravity':
            state = update_mario_avatar(
                state, action,
                avatar_type=avatar_config['avatar_type_idx'],
                mass=avatar_config['mass'],
                strength=avatar_config['strength'],
                jump_strength=avatar_config['jump_strength'],
                gravity=avatar_config['gravity'],
                airsteering=avatar_config['airsteering'])
        elif avatar_config.get('is_aimed', False):
            state = _update_aimed_avatar(state, action, avatar_config, height, width)
        elif avatar_config.get('is_rotating', False):
            state = _update_rotating_avatar(state, action, avatar_config, height, width)
        else:
            state = _update_avatar(state, action, avatar_config, height, width)

        # 3. Update NPC sprites
        for type_idx, cfg in enumerate(sprite_configs):
            sc = cfg['sprite_class']
            if sc in AVATAR_CLASSES:
                continue
            if sc in STATIC_CLASSES:
                continue
            state = _update_npc(state, type_idx, cfg, height, width)

        # 4. Age sprites and kill expired flickers
        new_ages = jnp.where(state.alive, state.ages + 1, state.ages)
        flicker_limits = jnp.array(
            [cfg.get('flicker_limit', 0) for cfg in sprite_configs],
            dtype=jnp.int32)[:, None]  # [n_types, 1]
        flicker_expired = (flicker_limits > 0) & (new_ages >= flicker_limits)
        state = state.replace(
            ages=new_ages,
            alive=state.alive & ~flicker_expired,
        )

        # 5. Apply effects via collision detection + masks
        state = _apply_all_effects(state, prev_positions, effects,
                                   height, width, max_n)

        # 6. Check terminations
        state, done, win = check_all_terminations(state, terminations)

        state = state.replace(
            done=done, win=win,
            step_count=state.step_count + 1,
        )
        return state

    def step(state: GameState, action: int) -> GameState:
        # Done-state guard: if already done, return state unchanged
        return jax.lax.cond(
            state.done,
            lambda s, a: s,
            _step_inner,
            state, action)

    return jax.jit(step)


# ── Grid-based collision detection ────────────────────────────────────


def _build_occupancy_grid(positions, alive, height, width):
    """Build a [height, width] boolean grid from sprite positions."""
    # Truncate float32 → int32 (floors toward zero). Correct for grid-physics
    # sprites with integer positions. Fractional-speed pairs use AABB collision
    # instead (see _collision_mask_aabb).
    ipos = positions.astype(jnp.int32)
    ib = in_bounds(ipos, height, width)
    effective = alive & ib
    grid = jnp.zeros((height, width), dtype=jnp.bool_)
    grid = grid.at[ipos[:, 0], ipos[:, 1]].max(effective)
    return grid


def _collision_mask(state, type_a, type_b, height, width):
    """Which type_a sprites overlap with any type_b sprite? Returns [max_n] bool."""
    pos_a = state.positions[type_a].astype(jnp.int32)
    alive_a = state.alive[type_a]
    in_bounds_a = in_bounds(pos_a, height, width)

    if type_a != type_b:
        grid_b = _build_occupancy_grid(
            state.positions[type_b], state.alive[type_b], height, width)
        r = jnp.clip(pos_a[:, 0], 0, height - 1)
        c = jnp.clip(pos_a[:, 1], 0, width - 1)
        mask = grid_b[r, c] & alive_a & in_bounds_a
    else:
        r = jnp.clip(pos_a[:, 0], 0, height - 1)
        c = jnp.clip(pos_a[:, 1], 0, width - 1)
        counts = jnp.zeros((height, width), dtype=jnp.int32)
        effective = alive_a & in_bounds_a
        counts = counts.at[r, c].add(effective.astype(jnp.int32))
        mask = (counts[r, c] > 1) & effective

    return mask


# ── AABB collision detection (for continuous physics) ─────────────────


def _collision_mask_aabb(state, type_a, type_b, height, width):
    """AABB overlap: two 1x1 sprites overlap when |pos_a - pos_b| < 1.0 on both axes."""
    pos_a = state.positions[type_a]    # [max_n, 2] float32
    alive_a = state.alive[type_a]      # [max_n]
    pos_b = state.positions[type_b]    # [max_n, 2] float32
    alive_b = state.alive[type_b]      # [max_n]

    # [max_n_a, max_n_b, 2]
    diff = jnp.abs(pos_a[:, None, :] - pos_b[None, :, :])
    overlap = jnp.all(diff < (1.0 - AABB_EPS), axis=-1) & alive_a[:, None] & alive_b[None, :]

    if type_a == type_b:
        max_n = alive_a.shape[0]
        overlap = overlap & ~jnp.eye(max_n, dtype=jnp.bool_)

    return jnp.any(overlap, axis=1) & alive_a


# ── Sweep collision detection (for speed > 1) ─────────────────────────


def _build_swept_occupancy_grid(positions, prev_positions, alive,
                                 height, width, max_speed_cells):
    """Build a [H, W] boolean grid marking all cells along each sprite's path."""
    grid = jnp.zeros((height, width), dtype=jnp.bool_)
    iprev = prev_positions.astype(jnp.int32)
    ipos = positions.astype(jnp.int32)
    ib = in_bounds(ipos, height, width)
    effective = alive & ib

    dir_row = jnp.sign(ipos[:, 0] - iprev[:, 0])
    dir_col = jnp.sign(ipos[:, 1] - iprev[:, 1])
    max_cells = jnp.maximum(
        jnp.abs(ipos[:, 0] - iprev[:, 0]),
        jnp.abs(ipos[:, 1] - iprev[:, 1]))

    def mark_step(step_i, g):
        r = jnp.clip(iprev[:, 0] + dir_row * step_i, 0, height - 1)
        c = jnp.clip(iprev[:, 1] + dir_col * step_i, 0, width - 1)
        valid = effective & (step_i <= max_cells)
        return g.at[r, c].max(valid)

    return jax.lax.fori_loop(0, max_speed_cells + 1, mark_step, grid)


def _collision_mask_sweep(state, prev_positions, type_a, type_b,
                           height, width, max_speed_cells):
    """Sweep collision: checks if type_a's path overlaps with type_b's path."""
    grid_b = _build_swept_occupancy_grid(
        state.positions[type_b], prev_positions[type_b],
        state.alive[type_b], height, width, max_speed_cells)

    pos_a = state.positions[type_a]
    prev_a = prev_positions[type_a]
    alive_a = state.alive[type_a]

    iprev_a = prev_a.astype(jnp.int32)
    ipos_a = pos_a.astype(jnp.int32)
    in_bounds_a = in_bounds(ipos_a, height, width)

    dir_row = jnp.sign(ipos_a[:, 0] - iprev_a[:, 0])
    dir_col = jnp.sign(ipos_a[:, 1] - iprev_a[:, 1])
    max_cells = jnp.maximum(
        jnp.abs(ipos_a[:, 0] - iprev_a[:, 0]),
        jnp.abs(ipos_a[:, 1] - iprev_a[:, 1]))

    def check_step(step_i, hit):
        r = jnp.clip(iprev_a[:, 0] + dir_row * step_i, 0, height - 1)
        c = jnp.clip(iprev_a[:, 1] + dir_col * step_i, 0, width - 1)
        valid = alive_a & in_bounds_a & (step_i <= max_cells)
        return hit | (valid & grid_b[r, c])

    init_hit = jnp.zeros_like(alive_a)
    return jax.lax.fori_loop(0, max_speed_cells + 1, check_step, init_hit)


# ── Effect application ────────────────────────────────────────────────


def _apply_all_effects(state, prev_positions, effects, height, width, max_n):
    """Apply all effects using collision detection and mask operations."""
    bounced = {}  # type_idx → [max_n] bool — once-per-step guard for wallBounce
    for eff in effects:
        type_a = eff['type_a']
        effect_type = eff['effect_type']
        score_change = eff.get('score_change', 0)
        kwargs = eff.get('kwargs', {})

        if eff['is_eos']:
            eos_mask = detect_eos(
                state.positions[type_a], state.alive[type_a], height, width)
            state = apply_masked_effect(
                state, prev_positions, type_a, -1, eos_mask,
                effect_type, score_change, kwargs, height, width, max_n)
        else:
            type_b = eff['type_b']
            use_aabb = eff.get('use_aabb', False)
            needs_sweep = eff.get('needs_sweep', False)
            max_speed_cells = eff.get('max_speed_cells', 1)
            if needs_sweep:
                coll_mask = _collision_mask_sweep(
                    state, prev_positions, type_a, type_b,
                    height, width, max_speed_cells)
            elif use_aabb:
                coll_mask = _collision_mask_aabb(state, type_a, type_b, height, width)
            else:
                coll_mask = _collision_mask(state, type_a, type_b, height, width)

            # wallBounce once-per-step: skip sprites already bounced this step
            if effect_type == 'wall_bounce':
                already = bounced.get(type_a, jnp.zeros(max_n, dtype=jnp.bool_))
                coll_mask = coll_mask & ~already
                bounced[type_a] = already | coll_mask

            state = apply_masked_effect(
                state, prev_positions, type_a, type_b, coll_mask,
                effect_type, score_change, kwargs, height, width, max_n)

    return state


# ── Avatar and NPC update ─────────────────────────────────────────────


def _update_avatar(state, action, cfg, height, width):
    """Update avatar position and optionally shoot."""
    avatar_type = cfg['avatar_type_idx']
    n_move = cfg['n_move_actions']
    cooldown = cfg['cooldown']
    direction_offset = cfg.get('direction_offset', 0)

    # Movement
    is_move = action < n_move
    # Apply direction_offset so HorizontalAvatar maps actions 0,1 to LEFT,RIGHT
    move_idx = jnp.clip(action + direction_offset, 0, 3)
    delta = jax.lax.cond(
        is_move,
        lambda: DIRECTION_DELTAS[move_idx],
        lambda: jnp.array([0.0, 0.0], dtype=jnp.float32))

    can_move = state.cooldown_timers[avatar_type, 0] >= cooldown
    should_move = is_move & can_move
    new_pos = state.positions[avatar_type, 0] + delta * should_move

    state = state.replace(
        positions=state.positions.at[avatar_type, 0].set(new_pos),
        cooldown_timers=jnp.where(
            should_move,
            state.cooldown_timers.at[avatar_type, 0].set(0),
            state.cooldown_timers),
    )

    # Update orientation only if actually moved (not just is_move)
    new_ori = jax.lax.cond(
        should_move,
        lambda: DIRECTION_DELTAS[move_idx].astype(jnp.float32),
        lambda: state.orientations[avatar_type, 0])
    state = state.replace(
        orientations=state.orientations.at[avatar_type, 0].set(new_ori))

    # Shoot
    if cfg['can_shoot']:
        is_shoot = (action == cfg['shoot_action_idx'])
        proj_type = cfg['projectile_type_idx']
        proj_speed = cfg['projectile_speed']

        if cfg.get('shoot_everywhere', False):
            # ShootEverywhereAvatar: fire in all 4 cardinal directions
            def _shoot_everywhere(s):
                for i in range(4):
                    s = spawn_sprite(s, avatar_type, 0, proj_type,
                                     DIRECTION_DELTAS[i], proj_speed)
                return s
            state = jax.lax.cond(
                is_shoot, _shoot_everywhere, lambda s: s, state)
        else:
            if cfg.get('projectile_orientation_from_avatar', False):
                proj_ori = state.orientations[avatar_type, 0]
            else:
                proj_ori = jnp.array(cfg['projectile_default_orientation'],
                                      dtype=jnp.float32)

            state = jax.lax.cond(
                is_shoot,
                lambda s: spawn_sprite(s, avatar_type, 0, proj_type,
                                        proj_ori, proj_speed),
                lambda s: s,
                state,
            )

    return state


def _maybe_shoot(state, action, cfg, avatar_type):
    """Conditionally spawn projectile using avatar's current orientation."""
    is_shoot = (action == cfg['shoot_action_idx'])
    proj_type = cfg['projectile_type_idx']
    proj_speed = cfg['projectile_speed']
    proj_ori = state.orientations[avatar_type, 0]
    return jax.lax.cond(
        is_shoot,
        lambda s: spawn_sprite(s, avatar_type, 0, proj_type, proj_ori, proj_speed),
        lambda s: s,
        state,
    )


def _update_aimed_avatar(state, action, cfg, height, width):
    """Update AimedAvatar / AimedFlakAvatar: continuous-angle aiming + optional horizontal movement.

    AimedAvatar: AIM_UP=0, AIM_DOWN=1, SHOOT=n_move, NOOP=n_move+1
    AimedFlakAvatar: LEFT=0, RIGHT=1, AIM_UP=2, AIM_DOWN=3, SHOOT=4, NOOP=5
    """
    avatar_type = cfg['avatar_type_idx']
    angle_diff = cfg.get('angle_diff', 0.05)
    can_move = cfg.get('can_move_aimed', False)
    n_move = cfg['n_move_actions']

    ori = state.orientations[avatar_type, 0]
    pos = state.positions[avatar_type, 0]

    if can_move:
        # AimedFlakAvatar: actions 0,1 = LEFT,RIGHT; 2,3 = AIM_UP,AIM_DOWN
        is_left = (action == 0)
        is_right = (action == 1)
        is_aim_up = (action == 2)
        is_aim_down = (action == 3)
        h_delta = jnp.where(is_left, -1.0, jnp.where(is_right, 1.0, 0.0))
        new_pos = pos.at[1].add(h_delta)
    else:
        # AimedAvatar: actions 0,1 = AIM_UP,AIM_DOWN
        is_aim_up = (action == 0)
        is_aim_down = (action == 1)
        new_pos = pos

    # Apply 2D rotation to orientation
    # AIM_UP: rotate by -angle_diff (CCW), AIM_DOWN: rotate by +angle_diff (CW)
    # Rotation matrix: [[cos, -sin], [sin, cos]]
    # Our orientation is (row, col) = (-sin(theta), cos(theta)) for rightward = (0, 1)
    theta = jnp.where(is_aim_up, -angle_diff,
                      jnp.where(is_aim_down, angle_diff, 0.0))
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    new_ori_r = cos_t * ori[0] - sin_t * ori[1]
    new_ori_c = sin_t * ori[0] + cos_t * ori[1]
    new_ori = jnp.array([new_ori_r, new_ori_c])

    state = state.replace(
        positions=state.positions.at[avatar_type, 0].set(new_pos),
        orientations=state.orientations.at[avatar_type, 0].set(new_ori),
    )

    if cfg['can_shoot']:
        state = _maybe_shoot(state, action, cfg, avatar_type)

    return state


def _update_rotating_avatar(state, action, cfg, height, width):
    """Update rotating avatar: ego-centric forward/backward + rotation.

    Actions:
        0: thrust forward (move 1 step in current orientation)
        1: thrust backward (RotatingAvatar) or flip 180 (Flipping variants)
        2: rotate CCW
        3: rotate CW
        4+ or NOOP: no-op
    """
    avatar_type = cfg['avatar_type_idx']
    is_flipping = cfg.get('is_flipping', False)
    noise_level = cfg.get('noise_level', 0.0)

    # Optionally apply noise
    if noise_level > 0:
        rng, key = jax.random.split(state.rng)
        # With probability noise_level, replace action with random
        noisy = jax.random.uniform(key) < noise_level
        rng, key2 = jax.random.split(rng)
        rand_action = jax.random.randint(key2, (), 0, 5)
        action = jnp.where(noisy, rand_action, action)
        state = state.replace(rng=rng)

    ori = state.orientations[avatar_type, 0]

    # Find current orientation index in DIRECTION_DELTAS
    # UP=0, DOWN=1, LEFT=2, RIGHT=3
    diffs = jnp.sum(jnp.abs(DIRECTION_DELTAS - ori), axis=-1)
    ori_idx = jnp.argmin(diffs)

    # Action 0: forward — move in current orientation
    is_forward = (action == 0)
    fwd_pos = state.positions[avatar_type, 0] + ori * is_forward

    # Action 1: backward (non-flipping) or flip (flipping)
    is_action1 = (action == 1)
    if is_flipping:
        # Flip: rotate 180 degrees, no movement
        # BASEDIRS cycle: UP(0), DOWN(1), LEFT(2), RIGHT(3)
        # 180 flip: UP↔DOWN, LEFT↔RIGHT → index XOR with specific mapping
        # Actually use: (ori_idx + 2) wouldn't work with our BASEDIRS order
        # UP=0→DOWN=1, DOWN=1→UP=0, LEFT=2→RIGHT=3, RIGHT=3→LEFT=2
        flipped_idx = jnp.array([1, 0, 3, 2])[ori_idx]
        new_ori_flip = DIRECTION_DELTAS[flipped_idx]
        new_ori = jnp.where(is_action1, new_ori_flip, ori)
        new_pos = jnp.where(is_forward, fwd_pos, state.positions[avatar_type, 0])
    else:
        # Backward: move opposite to current orientation
        bwd_pos = state.positions[avatar_type, 0] - ori * is_action1
        new_pos = jnp.where(is_forward, fwd_pos,
                           jnp.where(is_action1, bwd_pos,
                                    state.positions[avatar_type, 0]))
        new_ori = ori

    # Action 2: CCW rotation
    # In our BASEDIRS: UP=0,DOWN=1,LEFT=2,RIGHT=3
    # CCW: UP→LEFT→DOWN→RIGHT→UP
    ccw_map = jnp.array([2, 3, 1, 0])  # UP→LEFT, DOWN→RIGHT, LEFT→DOWN, RIGHT→UP
    is_ccw = (action == 2)
    ccw_idx = ccw_map[ori_idx]
    new_ori = jnp.where(is_ccw, DIRECTION_DELTAS[ccw_idx], new_ori)

    # Action 3: CW rotation
    # CW: UP→RIGHT→DOWN→LEFT→UP
    cw_map = jnp.array([3, 2, 0, 1])  # UP→RIGHT, DOWN→LEFT, LEFT→UP, RIGHT→DOWN
    is_cw = (action == 3)
    cw_idx = cw_map[ori_idx]
    new_ori = jnp.where(is_cw, DIRECTION_DELTAS[cw_idx], new_ori)

    state = state.replace(
        positions=state.positions.at[avatar_type, 0].set(new_pos),
        orientations=state.orientations.at[avatar_type, 0].set(new_ori),
    )

    if cfg['can_shoot']:
        state = _maybe_shoot(state, action, cfg, avatar_type)

    return state


def _update_npc(state, type_idx, cfg, height, width):
    """Update a single NPC type based on its sprite class."""
    sc = cfg['sprite_class']
    cooldown = cfg.get('cooldown', 1)

    if sc == SpriteClass.MISSILE:
        return update_missile(state, type_idx, cooldown)

    elif sc == SpriteClass.ERRATIC_MISSILE:
        return update_erratic_missile(state, type_idx, cooldown,
                                       prob=cfg.get('prob', 0.1))

    elif sc == SpriteClass.RANDOM_NPC:
        return update_random_npc(state, type_idx, cooldown)

    elif sc == SpriteClass.CHASER:
        return update_chaser(state, type_idx,
                             cfg['target_type_idx'], cooldown, fleeing=False,
                             height=height, width=width)

    elif sc == SpriteClass.FLEEING:
        return update_chaser(state, type_idx,
                             cfg['target_type_idx'], cooldown, fleeing=True,
                             height=height, width=width)

    elif sc in (SpriteClass.FLICKER, SpriteClass.ORIENTED_FLICKER):
        return state

    elif sc == SpriteClass.SPAWN_POINT:
        return update_spawn_point(
            state, type_idx, cooldown,
            prob=cfg.get('prob', 1.0),
            total=cfg.get('total', 0),
            target_type=cfg['target_type_idx'],
            target_orientation=jnp.array(cfg.get('target_orientation', [0., 0.]),
                                          dtype=jnp.float32),
            target_speed=cfg.get('target_speed', 1.0),
        )

    elif sc == SpriteClass.BOMBER:
        state = update_missile(state, type_idx, cooldown)
        state = update_spawn_point(
            state, type_idx,
            cooldown=cfg.get('spawn_cooldown', cooldown),
            prob=cfg.get('prob', 1.0),
            total=cfg.get('total', 0),
            target_type=cfg['target_type_idx'],
            target_orientation=jnp.array(cfg.get('target_orientation', [0., 0.]),
                                          dtype=jnp.float32),
            target_speed=cfg.get('target_speed', 1.0),
        )
        return state

    elif sc == SpriteClass.WALKER:
        return update_missile(state, type_idx, cooldown)

    elif sc == SpriteClass.SPREADER:
        return update_spreader(state, type_idx,
                                spreadprob=cfg.get('spreadprob', 1.0))

    elif sc == SpriteClass.RANDOM_INERTIAL:
        return update_random_inertial(state, type_idx,
                                       mass=cfg.get('mass', 1.0),
                                       strength=cfg.get('strength', 1.0))

    elif sc == SpriteClass.RANDOM_MISSILE:
        # After initialization (randomized orientation), behaves as a regular missile
        return update_missile(state, type_idx, cooldown)

    elif sc == SpriteClass.WALK_JUMPER:
        return update_walk_jumper(state, type_idx,
                                   prob=cfg.get('prob', 0.1),
                                   strength=cfg.get('strength', 10.0 / PHYSICS_SCALE),
                                   gravity=cfg.get('gravity', 1.0 / PHYSICS_SCALE),
                                   mass=cfg.get('mass', 1.0))

    return state
