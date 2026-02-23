"""
The jittable step function: combines sprite updates, collision detection,
effects, and termination checks into a single state → state transition.

Collision detection uses occupancy grids [height, width] for O(max_n) checks
instead of O(max_n²) pairwise comparisons. Effects are applied via boolean
masks over the sprite arrays.
"""
import jax
import jax.numpy as jnp
from vgdl_jax.state import GameState
from vgdl_jax.collision import detect_eos
from vgdl_jax.sprites import (
    DIRECTION_DELTAS, spawn_sprite,
    update_inertial_avatar, update_mario_avatar,
    update_missile, update_random_npc, update_chaser,
    update_spawn_point,
)
from vgdl_jax.terminations import check_all_terminations
from vgdl_jax.data_model import SpriteClass, STATIC_CLASSES, AVATAR_CLASSES


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
            - cooldown: int (effective cooldown, accounting for speed)
            - kwargs: dict
        avatar_config: dict with keys:
            - avatar_type_idx, n_move_actions, cooldown, can_shoot,
              shoot_action_idx, projectile_type_idx,
              projectile_orientation_from_avatar, projectile_default_orientation,
              projectile_speed, direction_offset
        params: dict with keys:
            - n_types, max_n, height, width, n_resource_types, resource_limits

    Returns:
        A function step(state, action) → state
    """
    n_types = params['n_types']
    max_n = params['max_n']
    height = params['height']
    width = params['width']
    n_resource_types = params.get('n_resource_types', 1)
    resource_limits_list = params.get('resource_limits', [])
    resource_limits = (jnp.array(resource_limits_list + [100], dtype=jnp.int32)
                       if resource_limits_list else jnp.array([100], dtype=jnp.int32))
    # use_aabb is now per-effect (stored in each effect dict), not global

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
                strength=avatar_config['strength'],
                height=height, width=width)
        elif avatar_physics == 'gravity':
            state = update_mario_avatar(
                state, action,
                avatar_type=avatar_config['avatar_type_idx'],
                mass=avatar_config['mass'],
                strength=avatar_config['strength'],
                jump_strength=avatar_config['jump_strength'],
                gravity=avatar_config['gravity'],
                airsteering=avatar_config['airsteering'],
                height=height, width=width)
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
                                   height, width, max_n,
                                   n_resource_types, resource_limits)

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


# ── Shared helpers ─────────────────────────────────────────────────────


def _in_bounds(ipos, height, width):
    """Check which sprites have positions within the grid. Returns [max_n] bool."""
    return (
        (ipos[:, 0] >= 0) & (ipos[:, 0] < height) &
        (ipos[:, 1] >= 0) & (ipos[:, 1] < width)
    )


def _get_partner_coords(state, type_b, height, width):
    """Get int32 positions, clipped coords, and in-bounds mask for type_b sprites.

    Returns (ipos_b, r_b, c_b, in_bounds_b).
    """
    ipos_b = state.positions[type_b].astype(jnp.int32)
    r_b = jnp.clip(ipos_b[:, 0], 0, height - 1)
    c_b = jnp.clip(ipos_b[:, 1], 0, width - 1)
    in_bounds_b = _in_bounds(ipos_b, height, width)
    return ipos_b, r_b, c_b, in_bounds_b


def _apply_partner_delta(state, prev_positions, type_a, type_b, mask,
                          height, width, score_delta):
    """Apply type_b's movement delta to type_a sprites (used by bounceForward/pullWithIt)."""
    if type_b >= 0:
        b_delta = (state.positions[type_b] - prev_positions[type_b]).astype(jnp.int32)
        ipos_b, r_b, c_b, _ = _get_partner_coords(state, type_b, height, width)
        delta_grid = jnp.zeros((height, width, 2), dtype=jnp.int32)
        alive_b_expanded = (state.alive[type_b])[:, None]
        delta_grid = delta_grid.at[r_b, c_b].set(
            jnp.where(alive_b_expanded, b_delta, 0))
        pos_a = state.positions[type_a]
        ipos_a = pos_a.astype(jnp.int32)
        r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
        c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
        partner_delta = delta_grid[r_a, c_a]
        new_pos = jnp.where(mask[:, None], pos_a + partner_delta, pos_a)
        new_pos = jnp.clip(new_pos,
                           jnp.array([0, 0]),
                           jnp.array([height - 1, width - 1]))
    else:
        new_pos = state.positions[type_a]
    return state.replace(
        positions=state.positions.at[type_a].set(new_pos),
        score=state.score + score_delta,
    )


# ── Grid-based collision detection ────────────────────────────────────


def _build_occupancy_grid(positions, alive, height, width):
    """Build a [height, width] boolean grid from sprite positions."""
    ipos = positions.astype(jnp.int32)
    in_bounds = _in_bounds(ipos, height, width)
    effective = alive & in_bounds
    grid = jnp.zeros((height, width), dtype=jnp.bool_)
    grid = grid.at[ipos[:, 0], ipos[:, 1]].max(effective)
    return grid


def _collision_mask(state, type_a, type_b, height, width):
    """Which type_a sprites overlap with any type_b sprite? Returns [max_n] bool."""
    pos_a = state.positions[type_a].astype(jnp.int32)
    alive_a = state.alive[type_a]
    in_bounds_a = _in_bounds(pos_a, height, width)

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


_AABB_EPS = 1e-3  # tolerance for float drift at cell boundaries


def _collision_mask_aabb(state, type_a, type_b, height, width):
    """AABB overlap: two 1x1 sprites overlap when |pos_a - pos_b| < 1.0 on both axes."""
    pos_a = state.positions[type_a]    # [max_n, 2] float32
    alive_a = state.alive[type_a]      # [max_n]
    pos_b = state.positions[type_b]    # [max_n, 2] float32
    alive_b = state.alive[type_b]      # [max_n]

    # [max_n_a, max_n_b, 2]
    diff = jnp.abs(pos_a[:, None, :] - pos_b[None, :, :])
    overlap = jnp.all(diff < (1.0 - _AABB_EPS), axis=-1) & alive_a[:, None] & alive_b[None, :]

    if type_a == type_b:
        max_n = alive_a.shape[0]
        overlap = overlap & ~jnp.eye(max_n, dtype=jnp.bool_)

    return jnp.any(overlap, axis=1) & alive_a


# ── Masked effect application ─────────────────────────────────────────


def _apply_all_effects(state, prev_positions, effects, height, width, max_n,
                       n_resource_types, resource_limits):
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
            state = _apply_masked_effect(
                state, prev_positions, type_a, -1, eos_mask,
                effect_type, score_change, kwargs, height, width, max_n,
                n_resource_types, resource_limits)
        else:
            type_b = eff['type_b']
            use_aabb = eff.get('use_aabb', False)
            if use_aabb:
                coll_mask = _collision_mask_aabb(state, type_a, type_b, height, width)
            else:
                coll_mask = _collision_mask(state, type_a, type_b, height, width)

            # wallBounce once-per-step: skip sprites already bounced this step
            if effect_type == 'wall_bounce':
                already = bounced.get(type_a, jnp.zeros(max_n, dtype=jnp.bool_))
                coll_mask = coll_mask & ~already
                bounced[type_a] = already | coll_mask

            state = _apply_masked_effect(
                state, prev_positions, type_a, type_b, coll_mask,
                effect_type, score_change, kwargs, height, width, max_n,
                n_resource_types, resource_limits)

    return state


def _apply_masked_effect(state, prev_positions, type_a, type_b, mask,
                          effect_type, score_change, kwargs,
                          height, width, max_n,
                          n_resource_types, resource_limits):
    """Apply an effect to all type_a sprites indicated by mask [max_n]."""
    n_affected = mask.sum()
    score_delta = n_affected * jnp.int32(score_change)

    if effect_type == 'kill_sprite':
        return state.replace(
            alive=state.alive.at[type_a].set(state.alive[type_a] & ~mask),
            score=state.score + score_delta,
        )

    elif effect_type == 'kill_both':
        state = state.replace(
            alive=state.alive.at[type_a].set(state.alive[type_a] & ~mask))
        if type_b >= 0:
            ipos_a = state.positions[type_a].astype(jnp.int32)
            r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
            c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
            grid_coll = jnp.zeros((height, width), dtype=jnp.bool_)
            grid_coll = grid_coll.at[r_a, c_a].max(mask)
            _, r_b, c_b, in_bounds_b = _get_partner_coords(state, type_b, height, width)
            mask_b = grid_coll[r_b, c_b] & state.alive[type_b] & in_bounds_b
            state = state.replace(
                alive=state.alive.at[type_b].set(state.alive[type_b] & ~mask_b))
        return state.replace(score=state.score + score_delta)

    elif effect_type == 'step_back':
        new_pos = jnp.where(
            mask[:, None], prev_positions[type_a], state.positions[type_a])
        return state.replace(
            positions=state.positions.at[type_a].set(new_pos),
            score=state.score + score_delta,
        )

    elif effect_type == 'reverse_direction':
        new_ori = jnp.where(
            mask[:, None],
            -state.orientations[type_a],
            state.orientations[type_a])
        return state.replace(
            orientations=state.orientations.at[type_a].set(new_ori),
            score=state.score + score_delta,
        )

    elif effect_type == 'turn_around':
        new_ori = jnp.where(
            mask[:, None],
            -state.orientations[type_a],
            state.orientations[type_a])
        new_pos = jnp.where(
            mask[:, None], prev_positions[type_a], state.positions[type_a])
        return state.replace(
            orientations=state.orientations.at[type_a].set(new_ori),
            positions=state.positions.at[type_a].set(new_pos),
            score=state.score + score_delta,
        )

    elif effect_type == 'transform_to':
        new_type = kwargs['new_type_idx']
        state = state.replace(
            alive=state.alive.at[type_a].set(state.alive[type_a] & ~mask),
            score=state.score + score_delta,
        )
        n_spawns = mask.sum()
        available = ~state.alive[new_type]
        slot_rank = jnp.cumsum(available)
        should_fill = available & (slot_rank <= n_spawns)
        source_order = jnp.argsort(~mask)
        src_idx = source_order[jnp.clip(slot_rank - 1, 0, max_n - 1)]
        src_pos = prev_positions[type_a][src_idx]
        src_ori = state.orientations[type_a][src_idx]
        state = state.replace(
            alive=state.alive.at[new_type].set(
                state.alive[new_type] | should_fill),
            positions=state.positions.at[new_type].set(
                jnp.where(should_fill[:, None], src_pos,
                          state.positions[new_type])),
            orientations=state.orientations.at[new_type].set(
                jnp.where(should_fill[:, None], src_ori,
                          state.orientations[new_type])),
            ages=state.ages.at[new_type].set(
                jnp.where(should_fill, 0, state.ages[new_type])),
        )
        return state

    # ── New effects ────────────────────────────────────────────────────

    elif effect_type == 'change_resource':
        r_idx = kwargs.get('resource_idx', 0)
        value = kwargs.get('value', 0)
        limit = kwargs.get('limit', 100)
        cur = state.resources[type_a, :, r_idx]
        new_val = jnp.where(mask, jnp.clip(cur + value, 0, limit), cur)
        return state.replace(
            resources=state.resources.at[type_a, :, r_idx].set(new_val),
            score=state.score + score_delta,
        )

    elif effect_type == 'collect_resource':
        # actor=Resource sprite, actee=collector (partner)
        # The collector (type_b) gains the resource value
        r_idx = kwargs.get('resource_idx', 0)
        r_value = kwargs.get('resource_value', 1)
        limit = kwargs.get('limit', 100)
        if type_b >= 0:
            # Build [H,W] grid marking cells where masked type_a sprites are
            ipos_a = state.positions[type_a].astype(jnp.int32)
            r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
            c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
            grid_coll = jnp.zeros((height, width), dtype=jnp.bool_)
            grid_coll = grid_coll.at[r_a, c_a].max(mask)
            # Find type_b sprites at those positions
            _, r_b, c_b, in_bounds_b = _get_partner_coords(state, type_b, height, width)
            b_mask = grid_coll[r_b, c_b] & state.alive[type_b] & in_bounds_b
            cur = state.resources[type_b, :, r_idx]
            new_val = jnp.where(b_mask, jnp.clip(cur + r_value, 0, limit), cur)
            state = state.replace(
                resources=state.resources.at[type_b, :, r_idx].set(new_val))
        return state.replace(score=state.score + score_delta)

    elif effect_type == 'kill_if_has_less':
        # Kill type_a if its resource value <= limit.
        # NOTE: py-vgdl's docstring says "less than" but the code uses <=,
        # so resource == limit also triggers the kill. We match that behavior.
        r_idx = kwargs.get('resource_idx', 0)
        limit = kwargs.get('limit', 0)
        cur = state.resources[type_a, :, r_idx]
        should_kill = mask & (cur <= limit)
        return state.replace(
            alive=state.alive.at[type_a].set(state.alive[type_a] & ~should_kill),
            score=state.score + should_kill.sum() * jnp.int32(score_change),
        )

    elif effect_type == 'kill_if_has_more':
        # Kill type_a if its resource value >= limit.
        # NOTE: py-vgdl's docstring says "more than" but the code uses >=,
        # so resource == limit also triggers the kill. We match that behavior.
        r_idx = kwargs.get('resource_idx', 0)
        limit = kwargs.get('limit', 0)
        cur = state.resources[type_a, :, r_idx]
        should_kill = mask & (cur >= limit)
        return state.replace(
            alive=state.alive.at[type_a].set(state.alive[type_a] & ~should_kill),
            score=state.score + should_kill.sum() * jnp.int32(score_change),
        )

    elif effect_type == 'kill_if_other_has_more':
        # Kill type_a if type_b (partner) has resource >= limit.
        # NOTE: py-vgdl uses >= despite docstring saying "more than".
        r_idx = kwargs.get('resource_idx', 0)
        limit = kwargs.get('limit', 0)
        if type_b >= 0:
            _, r_b, c_b, _ = _get_partner_coords(state, type_b, height, width)
            res_grid = jnp.zeros((height, width), dtype=jnp.int32)
            b_res = state.resources[type_b, :, r_idx]
            res_grid = res_grid.at[r_b, c_b].max(
                jnp.where(state.alive[type_b], b_res, 0))
            ipos_a = state.positions[type_a].astype(jnp.int32)
            r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
            c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
            partner_res = res_grid[r_a, c_a]
            should_kill = mask & (partner_res >= limit)
        else:
            should_kill = jnp.zeros_like(mask)
        return state.replace(
            alive=state.alive.at[type_a].set(state.alive[type_a] & ~should_kill),
            score=state.score + should_kill.sum() * jnp.int32(score_change),
        )

    elif effect_type == 'kill_if_other_has_less':
        # Kill type_a if type_b (partner) has resource <= limit.
        # NOTE: py-vgdl uses <= despite docstring saying "less than".
        r_idx = kwargs.get('resource_idx', 0)
        limit = kwargs.get('limit', 0)
        if type_b >= 0:
            _, r_b, c_b, _ = _get_partner_coords(state, type_b, height, width)
            # Use -1 as sentinel for "no partner here"
            res_grid = jnp.full((height, width), -1, dtype=jnp.int32)
            b_res = state.resources[type_b, :, r_idx]
            res_grid = res_grid.at[r_b, c_b].max(
                jnp.where(state.alive[type_b], b_res, -1))
            ipos_a = state.positions[type_a].astype(jnp.int32)
            r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
            c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
            partner_res = res_grid[r_a, c_a]
            should_kill = mask & (partner_res >= 0) & (partner_res <= limit)
        else:
            should_kill = jnp.zeros_like(mask)
        return state.replace(
            alive=state.alive.at[type_a].set(state.alive[type_a] & ~should_kill),
            score=state.score + should_kill.sum() * jnp.int32(score_change),
        )

    elif effect_type == 'kill_if_from_above':
        # Kill type_a if type_b's prev position was directly above
        if type_b >= 0:
            ipos_a = state.positions[type_a].astype(jnp.int32)
            r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
            c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
            # Build grid marking cells where partner was one row above
            iprev_b = prev_positions[type_b].astype(jnp.int32)
            icurr_b = state.positions[type_b].astype(jnp.int32)
            # Partner fell from above: prev row < curr row (moved down)
            fell_down = (icurr_b[:, 0] > iprev_b[:, 0]) & state.alive[type_b]
            r_prev_b = jnp.clip(iprev_b[:, 0], 0, height - 1)
            c_prev_b = jnp.clip(iprev_b[:, 1], 0, width - 1)
            from_above_grid = jnp.zeros((height, width), dtype=jnp.bool_)
            # The partner came from the cell above current type_a position
            # i.e., partner is now at type_a's position and moved down from above
            r_curr_b = jnp.clip(icurr_b[:, 0], 0, height - 1)
            c_curr_b = jnp.clip(icurr_b[:, 1], 0, width - 1)
            from_above_grid = from_above_grid.at[r_curr_b, c_curr_b].max(fell_down)
            should_kill = mask & from_above_grid[r_a, c_a]
        else:
            should_kill = jnp.zeros_like(mask)
        return state.replace(
            alive=state.alive.at[type_a].set(state.alive[type_a] & ~should_kill),
            score=state.score + should_kill.sum() * jnp.int32(score_change),
        )

    elif effect_type == 'wrap_around':
        # Teleport to opposite edge along orientation axis (EOS-triggered)
        ori = state.orientations[type_a]
        pos = state.positions[type_a]
        # Determine axis: row axis if ori[:,0] != 0, else col axis
        row_axis = ori[:, 0] != 0
        # For row axis: wrap row. For col axis: wrap col.
        # Moving up (ori[0]<0) → row=height-1, moving down (ori[0]>0) → row=0
        # Moving left (ori[1]<0) → col=width-1, moving right (ori[1]>0) → col=0
        new_row = jnp.where(
            mask & row_axis & (ori[:, 0] < 0), height - 1,
            jnp.where(mask & row_axis & (ori[:, 0] > 0), 0, pos[:, 0]))
        new_col = jnp.where(
            mask & ~row_axis & (ori[:, 1] < 0), width - 1,
            jnp.where(mask & ~row_axis & (ori[:, 1] > 0), 0, pos[:, 1]))
        new_pos = jnp.stack([new_row, new_col], axis=-1)
        return state.replace(
            positions=state.positions.at[type_a].set(new_pos),
            score=state.score + score_delta,
        )

    elif effect_type == 'bounce_forward':
        return _apply_partner_delta(
            state, prev_positions, type_a, type_b, mask,
            height, width, score_delta)

    elif effect_type == 'undo_all':
        # When ANY masked sprite triggers, revert ALL positions globally
        any_triggered = mask.any()
        new_positions = jnp.where(any_triggered, prev_positions, state.positions)
        return state.replace(
            positions=new_positions,
            score=state.score + score_delta,
        )

    elif effect_type == 'teleport_to_exit':
        # Teleport type_a to a random alive exit sprite
        exit_type = kwargs.get('exit_type_idx', -1)
        if exit_type >= 0:
            rng, key = jax.random.split(state.rng)
            exit_pos = state.positions[exit_type]
            exit_alive = state.alive[exit_type]
            n_exits = exit_alive.sum()
            # Pick random exit index among alive exits
            rand_idx = jax.random.randint(key, (), 0, jnp.maximum(n_exits, 1))
            exit_rank = jnp.cumsum(exit_alive)
            chosen = exit_alive & (exit_rank == rand_idx + 1)
            chosen_idx = jnp.argmax(chosen)
            target_pos = exit_pos[chosen_idx]
            # Move all masked type_a sprites to the exit position
            pos_a = state.positions[type_a]
            new_pos = jnp.where(mask[:, None] & (n_exits > 0),
                                target_pos, pos_a)
            state = state.replace(
                positions=state.positions.at[type_a].set(new_pos),
                score=state.score + score_delta,
                rng=rng,
            )
        else:
            state = state.replace(score=state.score + score_delta)
        return state

    elif effect_type == 'pull_with_it':
        return _apply_partner_delta(
            state, prev_positions, type_a, type_b, mask,
            height, width, score_delta)

    elif effect_type == 'wall_stop':
        # Axis-separated collision resolution for continuous physics.
        # Mimics py-vgdl's "move horizontal, check; move vertical, check" by
        # testing intermediate positions: (curr_row, prev_col) for vertical-only
        # movement, and (prev_row, curr_col) for horizontal-only movement.
        friction = kwargs.get('friction', 0.0)
        pos = state.positions[type_a]
        prev = prev_positions[type_a]
        vel = state.velocities[type_a]
        pf = state.passive_forces[type_a]

        if type_b >= 0:
            pos_b = state.positions[type_b]
            alive_b = state.alive[type_b]
            threshold = 1.0 - _AABB_EPS

            # Check vertical-only intermediate: (curr_row, prev_col)
            # If this overlaps a wall, vertical movement caused the collision
            v_rdiff = jnp.abs(pos[:, None, 0] - pos_b[None, :, 0])
            v_cdiff = jnp.abs(prev[:, None, 1] - pos_b[None, :, 1])
            check_v = (v_rdiff < threshold) & (v_cdiff < threshold) & alive_b[None, :]

            # Check horizontal-only intermediate: (prev_row, curr_col)
            # If this overlaps a wall, horizontal movement caused the collision
            h_rdiff = jnp.abs(prev[:, None, 0] - pos_b[None, :, 0])
            h_cdiff = jnp.abs(pos[:, None, 1] - pos_b[None, :, 1])
            check_h = (h_rdiff < threshold) & (h_cdiff < threshold) & alive_b[None, :]

            has_row_cross = jnp.any(check_v, axis=1)
            has_col_cross = jnp.any(check_h, axis=1)
        else:
            has_row_cross = jnp.zeros_like(mask)
            has_col_cross = jnp.zeros_like(mask)

        # Fallback: if neither axis detected (e.g. already overlapping),
        # use delta-magnitude heuristic
        neither = mask & ~has_row_cross & ~has_col_cross
        delta = pos - prev
        is_vert_fb = jnp.abs(delta[:, 0]) >= jnp.abs(delta[:, 1])
        has_row_cross = has_row_cross | (neither & is_vert_fb)
        has_col_cross = has_col_cross | (neither & ~is_vert_fb)

        # ── Vertical collision: flush row against nearest wall ──
        vert_mask = mask & has_row_cross
        if type_b >= 0:
            # Distance in row axis to each wall sprite (using vertical intermediate check)
            v_dist = jnp.where(check_v, v_rdiff, 1e10)
            nearest_v = jnp.argmin(v_dist, axis=1)  # [max_n]
            wall_row = pos_b[nearest_v, 0]
            # Moving down (+row) → place at wall_row - 1.0; up (-row) → wall_row + 1.0
            flush_row = jnp.where(vel[:, 0] > 0, wall_row - 1.0, wall_row + 1.0)
            new_pos_row = jnp.where(vert_mask, flush_row, pos[:, 0])
        else:
            new_pos_row = jnp.where(vert_mask, prev[:, 0], pos[:, 0])
        new_vel_row = jnp.where(vert_mask, 0.0, vel[:, 0])
        new_pf_row = jnp.where(vert_mask, 0.0, pf[:, 0])
        new_vel_col_v = jnp.where(
            vert_mask & (friction > 0),
            vel[:, 1] * (1.0 - friction),
            vel[:, 1])

        # ── Horizontal collision: flush col against nearest wall ──
        horiz_mask = mask & has_col_cross
        if type_b >= 0:
            h_dist = jnp.where(check_h, h_cdiff, 1e10)
            nearest_h = jnp.argmin(h_dist, axis=1)  # [max_n]
            wall_col = pos_b[nearest_h, 1]
            # Moving right (+col) → place at wall_col - 1.0; left (-col) → wall_col + 1.0
            flush_col = jnp.where(vel[:, 1] > 0, wall_col - 1.0, wall_col + 1.0)
            new_pos_col = jnp.where(horiz_mask, flush_col, pos[:, 1])
        else:
            new_pos_col = jnp.where(horiz_mask, prev[:, 1], pos[:, 1])
        new_vel_col = jnp.where(horiz_mask, 0.0, new_vel_col_v)
        new_pf_col = jnp.where(horiz_mask, 0.0, pf[:, 1])
        new_vel_row2 = jnp.where(
            horiz_mask & (friction > 0),
            new_vel_row * (1.0 - friction),
            new_vel_row)

        new_positions = jnp.stack([new_pos_row, new_pos_col], axis=-1)
        new_velocities = jnp.stack([new_vel_row2, new_vel_col], axis=-1)
        new_passive = jnp.stack([new_pf_row, new_pf_col], axis=-1)

        return state.replace(
            positions=state.positions.at[type_a].set(new_positions),
            velocities=state.velocities.at[type_a].set(new_velocities),
            passive_forces=state.passive_forces.at[type_a].set(new_passive),
            score=state.score + score_delta,
        )

    elif effect_type == 'wall_bounce':
        # Reflect velocity on collision axis, apply friction, revert position
        friction = kwargs.get('friction', 0.0)
        pos = state.positions[type_a]
        prev = prev_positions[type_a]
        vel = state.velocities[type_a]

        if type_b >= 0:
            # py-vgdl approach: center-to-center distance determines axis
            pos_b_all = state.positions[type_b]
            alive_b = state.alive[type_b]
            diff_ab = pos[:, None, :] - pos_b_all[None, :, :]
            dist_sq = jnp.sum(diff_ab ** 2, axis=-1)
            dist_sq = jnp.where(alive_b[None, :], dist_sq, 1e10)
            nearest_b = jnp.argmin(dist_sq, axis=1)
            nb_pos = pos_b_all[nearest_b]
            row_diff = jnp.abs(pos[:, 0] - nb_pos[:, 0])
            col_diff = jnp.abs(pos[:, 1] - nb_pos[:, 1])
            is_vertical = row_diff >= col_diff
        else:
            # Fallback: delta heuristic when no partner
            delta = pos - prev
            is_vertical = jnp.abs(delta[:, 0]) >= jnp.abs(delta[:, 1])

        # Reflect: negate velocity component on collision axis
        new_vel_row = jnp.where(
            mask & is_vertical, -vel[:, 0], vel[:, 0])
        new_vel_col = jnp.where(
            mask & ~is_vertical, -vel[:, 1], vel[:, 1])

        # Apply friction to speed
        speed = jnp.sqrt(new_vel_row ** 2 + new_vel_col ** 2)
        fric_scale = jnp.where(
            mask & (friction > 0) & (speed > 1e-6),
            jnp.maximum(1.0 - friction, 0.0), 1.0)
        new_vel_row = new_vel_row * fric_scale
        new_vel_col = new_vel_col * fric_scale

        # Revert to previous position
        new_pos = jnp.where(mask[:, None], prev, pos)

        # Update orientation from reflected velocity
        new_speed = jnp.sqrt(new_vel_row ** 2 + new_vel_col ** 2)
        new_ori_row = jnp.where(new_speed > 1e-6, new_vel_row / new_speed,
                                state.orientations[type_a][:, 0])
        new_ori_col = jnp.where(new_speed > 1e-6, new_vel_col / new_speed,
                                state.orientations[type_a][:, 1])

        new_velocities = jnp.stack([new_vel_row, new_vel_col], axis=-1)
        new_ori = jnp.stack([new_ori_row, new_ori_col], axis=-1)

        return state.replace(
            positions=state.positions.at[type_a].set(new_pos),
            velocities=state.velocities.at[type_a].set(new_velocities),
            orientations=state.orientations.at[type_a].set(new_ori),
            score=state.score + score_delta,
        )

    elif effect_type == 'bounce_direction':
        # Full vector reflection: v' = v - 2(v·n)n
        if type_b >= 0:
            pos_a = state.positions[type_a]
            pos_b = state.positions[type_b]
            vel = state.velocities[type_a]
            prev = prev_positions[type_a]

            # Find nearest type_b partner for each type_a
            # Use AABB-style: find closest type_b sprite
            diff = pos_a[:, None, :] - pos_b[None, :, :]  # [max_n_a, max_n_b, 2]
            dist_sq = jnp.sum(diff ** 2, axis=-1)  # [max_n_a, max_n_b]
            # Mask out dead type_b
            dist_sq = jnp.where(state.alive[type_b][None, :], dist_sq, 1e10)
            nearest_b = jnp.argmin(dist_sq, axis=1)  # [max_n_a]

            # Normal from a to b (unit vector)
            n = pos_a - pos_b[nearest_b]  # [max_n, 2]
            n_len = jnp.sqrt(jnp.sum(n ** 2, axis=-1, keepdims=True))
            n = jnp.where(n_len > 1e-6, n / n_len, jnp.array([0.0, 0.0]))

            # Reflection: v' = v - 2(v·n)n
            v_dot_n = jnp.sum(vel * n, axis=-1, keepdims=True)
            reflected = vel - 2.0 * v_dot_n * n

            # Apply friction to reflected velocity
            friction = kwargs.get('friction', 0.0)
            fric_scale = jnp.where(
                mask & (friction > 0),
                jnp.maximum(1.0 - friction, 0.0), 1.0)
            reflected = reflected * fric_scale[:, None]

            new_vel = jnp.where(mask[:, None], reflected, vel)
            new_pos = jnp.where(mask[:, None], prev, pos_a)

            # Update orientation from reflected velocity
            new_speed = jnp.sqrt(jnp.sum(new_vel ** 2, axis=-1, keepdims=True))
            new_ori = jnp.where(
                (new_speed > 1e-6) & mask[:, None],
                new_vel / new_speed,
                state.orientations[type_a])

            return state.replace(
                positions=state.positions.at[type_a].set(new_pos),
                velocities=state.velocities.at[type_a].set(new_vel),
                orientations=state.orientations.at[type_a].set(new_ori),
                score=state.score + score_delta,
            )
        return state.replace(score=state.score + score_delta)

    else:
        # null / unknown effect
        return state.replace(score=state.score + score_delta)


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
    new_pos = jnp.clip(new_pos, jnp.array([0.0, 0.0]),
                        jnp.array([height - 1, width - 1], dtype=jnp.float32))

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


def _update_npc(state, type_idx, cfg, height, width):
    """Update a single NPC type based on its sprite class."""
    sc = cfg['sprite_class']
    cooldown = cfg.get('cooldown', 1)

    if sc == SpriteClass.MISSILE:
        return update_missile(state, type_idx, cooldown)

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

    return state
