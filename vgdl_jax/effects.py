"""
Effect handlers for the VGDL-JAX step function.

Each handler implements one collision effect (kill, transform, resource change,
etc.). Handlers share a uniform keyword-argument interface — each declares the
params it needs and absorbs the rest via **_.

Sections:
    - Kill / removal
    - Position and orientation
    - Resources
    - Spawn and transform
    - Movement and conveying
    - Physics / wall interactions
    - Dispatch (EFFECT_DISPATCH dict + apply_masked_effect)
"""
import jax
import jax.numpy as jnp
from vgdl_jax.collision import in_bounds, AABB_EPS
from vgdl_jax.sprites import DIRECTION_DELTAS, prefix_sum_allocate


# ── Shared helpers ──────────────────────────────────────────────────────


def _get_partner_coords(state, type_b, height, width):
    """Get int32 positions, clipped coords, and in-bounds mask for type_b sprites."""
    ipos_b = state.positions[type_b].astype(jnp.int32)
    r_b = jnp.clip(ipos_b[:, 0], 0, height - 1)
    c_b = jnp.clip(ipos_b[:, 1], 0, width - 1)
    in_bounds_b = in_bounds(ipos_b, height, width)
    return ipos_b, r_b, c_b, in_bounds_b


# ── Kill / removal ─────────────────────────────────────────────────────


def kill_sprite(state, type_a, mask, score_delta, **_):
    return state.replace(
        alive=state.alive.at[type_a].set(state.alive[type_a] & ~mask),
        score=state.score + score_delta,
    )


def kill_both(state, type_a, type_b, mask, score_delta, height, width, **_):
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


def kill_if_alive(state, type_a, mask, score_delta, **_):
    """Kill type_a if type_b is alive (collision mask already requires type_b alive)."""
    return state.replace(
        alive=state.alive.at[type_a].set(state.alive[type_a] & ~mask),
        score=state.score + score_delta,
    )


def kill_if_slow(state, type_a, mask, score_change, kwargs, **_):
    limitspeed = kwargs.get('limitspeed', 0.0)
    is_slow = state.speeds[type_a] < limitspeed
    should_kill = mask & is_slow
    return state.replace(
        alive=state.alive.at[type_a].set(state.alive[type_a] & ~should_kill),
        score=state.score + should_kill.sum() * jnp.int32(score_change),
    )


def kill_others(state, type_a, mask, score_delta, kwargs, **_):
    kill_type = kwargs.get('kill_type_idx', -1)
    if kill_type >= 0:
        any_collision = mask.any()
        new_alive = jnp.where(any_collision,
                               jnp.zeros_like(state.alive[kill_type]),
                               state.alive[kill_type])
        return state.replace(
            alive=state.alive.at[kill_type].set(new_alive),
            score=state.score + score_delta,
        )
    return state.replace(score=state.score + score_delta)


def kill_if_from_above(state, prev_positions, type_a, type_b, mask,
                       score_change, height, width, **_):
    if type_b >= 0:
        ipos_a = state.positions[type_a].astype(jnp.int32)
        r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
        c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
        iprev_b = prev_positions[type_b].astype(jnp.int32)
        icurr_b = state.positions[type_b].astype(jnp.int32)
        fell_down = (icurr_b[:, 0] > iprev_b[:, 0]) & state.alive[type_b]
        from_above_grid = jnp.zeros((height, width), dtype=jnp.bool_)
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


# ── Position and orientation ───────────────────────────────────────────


def step_back(state, prev_positions, type_a, mask, score_delta, **_):
    new_pos = jnp.where(
        mask[:, None], prev_positions[type_a], state.positions[type_a])
    return state.replace(
        positions=state.positions.at[type_a].set(new_pos),
        score=state.score + score_delta,
    )


def reverse_direction(state, type_a, mask, score_delta, **_):
    new_ori = jnp.where(
        mask[:, None], -state.orientations[type_a], state.orientations[type_a])
    return state.replace(
        orientations=state.orientations.at[type_a].set(new_ori),
        score=state.score + score_delta,
    )


def turn_around(state, prev_positions, type_a, mask, score_delta, **_):
    new_ori = jnp.where(
        mask[:, None], -state.orientations[type_a], state.orientations[type_a])
    new_pos = jnp.where(
        mask[:, None], prev_positions[type_a], state.positions[type_a])
    return state.replace(
        orientations=state.orientations.at[type_a].set(new_ori),
        positions=state.positions.at[type_a].set(new_pos),
        score=state.score + score_delta,
    )


def flip_direction(state, type_a, mask, score_delta, max_n, **_):
    rng, key = jax.random.split(state.rng)
    dir_indices = jax.random.randint(key, (max_n,), 0, 4)
    random_ori = DIRECTION_DELTAS[dir_indices]
    new_ori = jnp.where(mask[:, None], random_ori, state.orientations[type_a])
    return state.replace(
        orientations=state.orientations.at[type_a].set(new_ori),
        score=state.score + score_delta, rng=rng,
    )


def undo_all(state, prev_positions, mask, score_delta, **_):
    any_triggered = mask.any()
    new_positions = jnp.where(any_triggered, prev_positions, state.positions)
    return state.replace(positions=new_positions, score=state.score + score_delta)


def wrap_around(state, type_a, mask, score_delta, height, width, **_):
    ori = state.orientations[type_a]
    pos = state.positions[type_a]
    row_axis = ori[:, 0] != 0
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


def attract_gaze(state, type_a, type_b, mask, score_delta, kwargs, max_n, **_):
    prob = kwargs.get('prob', 0.5)
    if type_b >= 0:
        rng, key = jax.random.split(state.rng)
        rolls = jax.random.uniform(key, (max_n,))
        should_attract = mask & (rolls < prob)
        partner_ori = state.orientations[type_b, 0]
        new_ori = jnp.where(should_attract[:, None],
                            jnp.broadcast_to(partner_ori, state.orientations[type_a].shape),
                            state.orientations[type_a])
        return state.replace(
            orientations=state.orientations.at[type_a].set(new_ori),
            score=state.score + score_delta, rng=rng,
        )
    return state.replace(score=state.score + score_delta)


# ── Resources ──────────────────────────────────────────────────────────


def change_resource(state, type_a, mask, score_delta, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    value = kwargs.get('value', 0)
    limit = kwargs.get('limit', 100)
    cur = state.resources[type_a, :, r_idx]
    new_val = jnp.where(mask, jnp.clip(cur + value, 0, limit), cur)
    return state.replace(
        resources=state.resources.at[type_a, :, r_idx].set(new_val),
        score=state.score + score_delta,
    )


def collect_resource(state, type_a, type_b, mask, score_delta,
                     kwargs, height, width, **_):
    r_idx = kwargs.get('resource_idx', 0)
    r_value = kwargs.get('resource_value', 1)
    limit = kwargs.get('limit', 100)
    if type_b >= 0:
        ipos_a = state.positions[type_a].astype(jnp.int32)
        r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
        c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
        grid_coll = jnp.zeros((height, width), dtype=jnp.bool_)
        grid_coll = grid_coll.at[r_a, c_a].max(mask)
        _, r_b, c_b, in_bounds_b = _get_partner_coords(state, type_b, height, width)
        b_mask = grid_coll[r_b, c_b] & state.alive[type_b] & in_bounds_b
        cur = state.resources[type_b, :, r_idx]
        new_val = jnp.where(b_mask, jnp.clip(cur + r_value, 0, limit), cur)
        state = state.replace(
            resources=state.resources.at[type_b, :, r_idx].set(new_val))
    return state.replace(score=state.score + score_delta)


def avatar_collect_resource(state, mask, score_delta, kwargs, **_):
    ati = kwargs.get('avatar_type_idx', 0)
    r_idx = kwargs.get('resource_idx', 0)
    r_val = kwargs.get('resource_value', 1)
    limit = kwargs.get('limit', 100)
    n_collected = mask.sum()
    cur = state.resources[ati, 0, r_idx]
    new_val = jnp.minimum(cur + n_collected * r_val, limit)
    return state.replace(
        resources=state.resources.at[ati, 0, r_idx].set(new_val),
        score=state.score + score_delta,
    )


def spend_resource(state, type_a, mask, score_delta, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    amount = kwargs.get('amount', 1)
    cur = state.resources[type_a, :, r_idx]
    spend = jnp.where(mask, jnp.minimum(cur, amount), 0)
    return state.replace(
        resources=state.resources.at[type_a, :, r_idx].add(-spend),
        score=state.score + score_delta,
    )


def spend_avatar_resource(state, mask, score_delta, kwargs, **_):
    ati = kwargs.get('avatar_type_idx', 0)
    r_idx = kwargs.get('resource_idx', 0)
    amount = kwargs.get('amount', 1)
    n_affected = mask.sum()
    total_spend = n_affected * amount
    cur = state.resources[ati, 0, r_idx]
    spend = jnp.minimum(cur, total_spend)
    return state.replace(
        resources=state.resources.at[ati, 0, r_idx].add(-spend),
        score=state.score + score_delta,
    )


def kill_if_has_less(state, type_a, mask, score_change, kwargs, **_):
    """Kill type_a if its resource <= limit (py-vgdl uses <=, not <)."""
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    cur = state.resources[type_a, :, r_idx]
    should_kill = mask & (cur <= limit)
    return state.replace(
        alive=state.alive.at[type_a].set(state.alive[type_a] & ~should_kill),
        score=state.score + should_kill.sum() * jnp.int32(score_change),
    )


def kill_if_has_more(state, type_a, mask, score_change, kwargs, **_):
    """Kill type_a if its resource >= limit (py-vgdl uses >=, not >)."""
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    cur = state.resources[type_a, :, r_idx]
    should_kill = mask & (cur >= limit)
    return state.replace(
        alive=state.alive.at[type_a].set(state.alive[type_a] & ~should_kill),
        score=state.score + should_kill.sum() * jnp.int32(score_change),
    )


def kill_if_other_has_more(state, type_a, type_b, mask, score_change,
                           kwargs, height, width, **_):
    """Kill type_a if type_b (partner) has resource >= limit."""
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


def kill_if_other_has_less(state, type_a, type_b, mask, score_change,
                           kwargs, height, width, **_):
    """Kill type_a if type_b (partner) has resource <= limit."""
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    if type_b >= 0:
        _, r_b, c_b, _ = _get_partner_coords(state, type_b, height, width)
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


def kill_if_avatar_without_resource(state, type_a, mask, score_change, kwargs, **_):
    ati = kwargs.get('avatar_type_idx', 0)
    r_idx = kwargs.get('resource_idx', 0)
    avatar_has = state.resources[ati, 0, r_idx] > 0
    should_kill = mask & ~avatar_has
    return state.replace(
        alive=state.alive.at[type_a].set(state.alive[type_a] & ~should_kill),
        score=state.score + should_kill.sum() * jnp.int32(score_change),
    )


# ── Spawn and transform ───────────────────────────────────────────────


def transform_to(state, prev_positions, type_a, mask, score_delta, kwargs, **_):
    new_type = kwargs['new_type_idx']
    state = state.replace(
        alive=state.alive.at[type_a].set(state.alive[type_a] & ~mask),
        score=state.score + score_delta,
    )
    should_fill, src_idx = prefix_sum_allocate(state.alive[new_type], mask)
    src_pos = prev_positions[type_a][src_idx]
    src_ori = state.orientations[type_a][src_idx]
    return state.replace(
        alive=state.alive.at[new_type].set(state.alive[new_type] | should_fill),
        positions=state.positions.at[new_type].set(
            jnp.where(should_fill[:, None], src_pos, state.positions[new_type])),
        orientations=state.orientations.at[new_type].set(
            jnp.where(should_fill[:, None], src_ori, state.orientations[new_type])),
        ages=state.ages.at[new_type].set(
            jnp.where(should_fill, 0, state.ages[new_type])),
    )


def clone_sprite(state, type_a, mask, score_delta, **_):
    should_fill, src_idx = prefix_sum_allocate(state.alive[type_a], mask)
    src_pos = state.positions[type_a][src_idx]
    src_ori = state.orientations[type_a][src_idx]
    return state.replace(
        alive=state.alive.at[type_a].set(state.alive[type_a] | should_fill),
        positions=state.positions.at[type_a].set(
            jnp.where(should_fill[:, None], src_pos, state.positions[type_a])),
        orientations=state.orientations.at[type_a].set(
            jnp.where(should_fill[:, None], src_ori, state.orientations[type_a])),
        ages=state.ages.at[type_a].set(
            jnp.where(should_fill, 0, state.ages[type_a])),
        score=state.score + score_delta,
    )


def spawn_if_has_more(state, type_a, mask, score_delta, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    spawn_type = kwargs.get('spawn_type_idx', -1)
    if spawn_type >= 0:
        cur = state.resources[type_a, :, r_idx]
        has_enough = mask & (cur >= limit)
        should_fill, src_idx = prefix_sum_allocate(state.alive[spawn_type], has_enough)
        src_pos = state.positions[type_a][src_idx]
        state = state.replace(
            alive=state.alive.at[spawn_type].set(
                state.alive[spawn_type] | should_fill),
            positions=state.positions.at[spawn_type].set(
                jnp.where(should_fill[:, None], src_pos,
                          state.positions[spawn_type])),
            ages=state.ages.at[spawn_type].set(
                jnp.where(should_fill, 0, state.ages[spawn_type])),
        )
    return state.replace(score=state.score + score_delta)


def transform_others_to(state, type_a, mask, score_delta, kwargs, **_):
    target_type = kwargs.get('target_type_idx', -1)
    new_type = kwargs.get('new_type_idx', -1)
    if target_type >= 0 and new_type >= 0:
        any_collision = mask.any()
        target_alive = state.alive[target_type] & any_collision
        new_target_alive = jnp.where(any_collision,
                                      jnp.zeros_like(state.alive[target_type]),
                                      state.alive[target_type])
        should_fill, src_idx = prefix_sum_allocate(state.alive[new_type], target_alive)
        src_pos = state.positions[target_type][src_idx]
        src_ori = state.orientations[target_type][src_idx]
        state = state.replace(
            alive=state.alive.at[target_type].set(new_target_alive)
                       .at[new_type].set(state.alive[new_type] | should_fill),
            positions=state.positions.at[new_type].set(
                jnp.where(should_fill[:, None], src_pos, state.positions[new_type])),
            orientations=state.orientations.at[new_type].set(
                jnp.where(should_fill[:, None], src_ori, state.orientations[new_type])),
            ages=state.ages.at[new_type].set(
                jnp.where(should_fill, 0, state.ages[new_type])),
        )
    return state.replace(score=state.score + score_delta)


# ── Movement and conveying ─────────────────────────────────────────────


def teleport_to_exit(state, type_a, mask, score_delta, kwargs, **_):
    exit_type = kwargs.get('exit_type_idx', -1)
    if exit_type >= 0:
        rng, key = jax.random.split(state.rng)
        exit_pos = state.positions[exit_type]
        exit_alive = state.alive[exit_type]
        n_exits = exit_alive.sum()
        rand_idx = jax.random.randint(key, (), 0, jnp.maximum(n_exits, 1))
        exit_rank = jnp.cumsum(exit_alive)
        chosen = exit_alive & (exit_rank == rand_idx + 1)
        chosen_idx = jnp.argmax(chosen)
        target_pos = exit_pos[chosen_idx]
        pos_a = state.positions[type_a]
        new_pos = jnp.where(mask[:, None] & (n_exits > 0), target_pos, pos_a)
        return state.replace(
            positions=state.positions.at[type_a].set(new_pos),
            score=state.score + score_delta, rng=rng,
        )
    return state.replace(score=state.score + score_delta)


def convey_sprite(state, type_a, type_b, mask, score_delta,
                  kwargs, height, width, **_):
    strength = kwargs.get('strength', 1.0)
    if type_b >= 0:
        ipos_b, r_b, c_b, in_bounds_b = _get_partner_coords(
            state, type_b, height, width)
        ori_grid = jnp.zeros((height, width, 2), dtype=jnp.float32)
        alive_b = state.alive[type_b] & in_bounds_b
        ori_grid = ori_grid.at[r_b, c_b].set(
            jnp.where(alive_b[:, None], state.orientations[type_b], jnp.zeros(2)))
        pos_a = state.positions[type_a]
        ipos_a = pos_a.astype(jnp.int32)
        r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
        c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
        partner_ori = ori_grid[r_a, c_a]
        new_pos = jnp.where(mask[:, None], pos_a + partner_ori * strength, pos_a)
    else:
        new_pos = state.positions[type_a]
    return state.replace(
        positions=state.positions.at[type_a].set(new_pos),
        score=state.score + score_delta,
    )


def wind_gust(state, type_a, type_b, mask, score_delta, max_n, **_):
    if type_b >= 0:
        rng, key = jax.random.split(state.rng)
        offsets = jax.random.randint(key, (max_n,), -1, 2)
        base_strength = state.speeds[type_b]
        per_sprite = (base_strength[:max_n] + offsets.astype(jnp.float32))
        partner_ori = state.orientations[type_b, 0]
        delta = partner_ori[None, :] * per_sprite[:, None]
        new_pos = jnp.where(mask[:, None],
                            state.positions[type_a] + delta,
                            state.positions[type_a])
        return state.replace(
            positions=state.positions.at[type_a].set(new_pos),
            score=state.score + score_delta, rng=rng,
        )
    return state.replace(score=state.score + score_delta)


def slip_forward(state, type_a, mask, score_delta, kwargs, max_n, **_):
    prob = kwargs.get('prob', 0.5)
    rng, key = jax.random.split(state.rng)
    rolls = jax.random.uniform(key, (max_n,))
    should_slip = mask & (rolls < prob)
    delta = state.orientations[type_a]
    new_pos = jnp.where(should_slip[:, None],
                        state.positions[type_a] + delta,
                        state.positions[type_a])
    return state.replace(
        positions=state.positions.at[type_a].set(new_pos),
        score=state.score + score_delta, rng=rng,
    )


# ── Physics / wall interactions ────────────────────────────────────────


def partner_delta(state, prev_positions, type_a, type_b, mask,
                  height, width, score_delta, **_):
    """Apply type_b's movement delta to type_a (bounceForward / pullWithIt)."""
    if type_b >= 0:
        b_delta = (state.positions[type_b] - prev_positions[type_b]).astype(jnp.int32)
        _, r_b, c_b, _ = _get_partner_coords(state, type_b, height, width)
        delta_grid = jnp.zeros((height, width, 2), dtype=jnp.int32)
        alive_b_expanded = (state.alive[type_b])[:, None]
        delta_grid = delta_grid.at[r_b, c_b].set(
            jnp.where(alive_b_expanded, b_delta, 0))
        pos_a = state.positions[type_a]
        ipos_a = pos_a.astype(jnp.int32)
        r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
        c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
        pd = delta_grid[r_a, c_a]
        new_pos = jnp.where(mask[:, None], pos_a + pd, pos_a)
        new_pos = jnp.clip(new_pos,
                           jnp.array([0, 0]),
                           jnp.array([height - 1, width - 1]))
    else:
        new_pos = state.positions[type_a]
    return state.replace(
        positions=state.positions.at[type_a].set(new_pos),
        score=state.score + score_delta,
    )


def wall_stop(state, prev_positions, type_a, type_b, mask,
              score_delta, kwargs, height, width, **_):
    friction = kwargs.get('friction', 0.0)
    pos = state.positions[type_a]
    prev = prev_positions[type_a]
    vel = state.velocities[type_a]
    pf = state.passive_forces[type_a]

    if type_b >= 0:
        pos_b = state.positions[type_b]
        alive_b = state.alive[type_b]
        threshold = 1.0 - AABB_EPS
        v_rdiff = jnp.abs(pos[:, None, 0] - pos_b[None, :, 0])
        v_cdiff = jnp.abs(prev[:, None, 1] - pos_b[None, :, 1])
        check_v = (v_rdiff < threshold) & (v_cdiff < threshold) & alive_b[None, :]
        h_rdiff = jnp.abs(prev[:, None, 0] - pos_b[None, :, 0])
        h_cdiff = jnp.abs(pos[:, None, 1] - pos_b[None, :, 1])
        check_h = (h_rdiff < threshold) & (h_cdiff < threshold) & alive_b[None, :]
        has_row_cross = jnp.any(check_v, axis=1)
        has_col_cross = jnp.any(check_h, axis=1)
    else:
        has_row_cross = jnp.zeros_like(mask)
        has_col_cross = jnp.zeros_like(mask)

    neither = mask & ~has_row_cross & ~has_col_cross
    delta = pos - prev
    is_vert_fb = jnp.abs(delta[:, 0]) >= jnp.abs(delta[:, 1])
    has_row_cross = has_row_cross | (neither & is_vert_fb)
    has_col_cross = has_col_cross | (neither & ~is_vert_fb)

    # Vertical collision
    vert_mask = mask & has_row_cross
    if type_b >= 0:
        v_dist = jnp.where(check_v, v_rdiff, 1e10)
        nearest_v = jnp.argmin(v_dist, axis=1)
        wall_row = pos_b[nearest_v, 0]
        flush_row = jnp.where(vel[:, 0] > 0, wall_row - 1.0, wall_row + 1.0)
        new_pos_row = jnp.where(vert_mask, flush_row, pos[:, 0])
    else:
        new_pos_row = jnp.where(vert_mask, prev[:, 0], pos[:, 0])
    new_vel_row = jnp.where(vert_mask, 0.0, vel[:, 0])
    new_pf_row = jnp.where(vert_mask, 0.0, pf[:, 0])
    new_vel_col_v = jnp.where(
        vert_mask & (friction > 0), vel[:, 1] * (1.0 - friction), vel[:, 1])

    # Horizontal collision
    horiz_mask = mask & has_col_cross
    if type_b >= 0:
        h_dist = jnp.where(check_h, h_cdiff, 1e10)
        nearest_h = jnp.argmin(h_dist, axis=1)
        wall_col = pos_b[nearest_h, 1]
        flush_col = jnp.where(vel[:, 1] > 0, wall_col - 1.0, wall_col + 1.0)
        new_pos_col = jnp.where(horiz_mask, flush_col, pos[:, 1])
    else:
        new_pos_col = jnp.where(horiz_mask, prev[:, 1], pos[:, 1])
    new_vel_col = jnp.where(horiz_mask, 0.0, new_vel_col_v)
    new_pf_col = jnp.where(horiz_mask, 0.0, pf[:, 1])
    new_vel_row2 = jnp.where(
        horiz_mask & (friction > 0), new_vel_row * (1.0 - friction), new_vel_row)

    return state.replace(
        positions=state.positions.at[type_a].set(
            jnp.stack([new_pos_row, new_pos_col], axis=-1)),
        velocities=state.velocities.at[type_a].set(
            jnp.stack([new_vel_row2, new_vel_col], axis=-1)),
        passive_forces=state.passive_forces.at[type_a].set(
            jnp.stack([new_pf_row, new_pf_col], axis=-1)),
        score=state.score + score_delta,
    )


def wall_bounce(state, prev_positions, type_a, type_b, mask,
                score_delta, kwargs, **_):
    friction = kwargs.get('friction', 0.0)
    pos = state.positions[type_a]
    prev = prev_positions[type_a]
    vel = state.velocities[type_a]

    if type_b >= 0:
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
        delta = pos - prev
        is_vertical = jnp.abs(delta[:, 0]) >= jnp.abs(delta[:, 1])

    new_vel_row = jnp.where(mask & is_vertical, -vel[:, 0], vel[:, 0])
    new_vel_col = jnp.where(mask & ~is_vertical, -vel[:, 1], vel[:, 1])

    speed = jnp.sqrt(new_vel_row ** 2 + new_vel_col ** 2)
    fric_scale = jnp.where(
        mask & (friction > 0) & (speed > 1e-6),
        jnp.maximum(1.0 - friction, 0.0), 1.0)
    new_vel_row = new_vel_row * fric_scale
    new_vel_col = new_vel_col * fric_scale

    new_pos = jnp.where(mask[:, None], prev, pos)

    new_speed = jnp.sqrt(new_vel_row ** 2 + new_vel_col ** 2)
    new_ori_row = jnp.where(new_speed > 1e-6, new_vel_row / new_speed,
                            state.orientations[type_a][:, 0])
    new_ori_col = jnp.where(new_speed > 1e-6, new_vel_col / new_speed,
                            state.orientations[type_a][:, 1])

    return state.replace(
        positions=state.positions.at[type_a].set(new_pos),
        velocities=state.velocities.at[type_a].set(
            jnp.stack([new_vel_row, new_vel_col], axis=-1)),
        orientations=state.orientations.at[type_a].set(
            jnp.stack([new_ori_row, new_ori_col], axis=-1)),
        score=state.score + score_delta,
    )


def bounce_direction(state, prev_positions, type_a, type_b, mask,
                     score_delta, kwargs, **_):
    if type_b >= 0:
        pos_a = state.positions[type_a]
        pos_b = state.positions[type_b]
        vel = state.velocities[type_a]
        prev = prev_positions[type_a]

        diff = pos_a[:, None, :] - pos_b[None, :, :]
        dist_sq = jnp.sum(diff ** 2, axis=-1)
        dist_sq = jnp.where(state.alive[type_b][None, :], dist_sq, 1e10)
        nearest_b = jnp.argmin(dist_sq, axis=1)

        n = pos_a - pos_b[nearest_b]
        n_len = jnp.sqrt(jnp.sum(n ** 2, axis=-1, keepdims=True))
        n = jnp.where(n_len > 1e-6, n / n_len, jnp.array([0.0, 0.0]))

        v_dot_n = jnp.sum(vel * n, axis=-1, keepdims=True)
        reflected = vel - 2.0 * v_dot_n * n

        friction = kwargs.get('friction', 0.0)
        fric_scale = jnp.where(
            mask & (friction > 0), jnp.maximum(1.0 - friction, 0.0), 1.0)
        reflected = reflected * fric_scale[:, None]

        new_vel = jnp.where(mask[:, None], reflected, vel)
        new_pos = jnp.where(mask[:, None], prev, pos_a)

        new_speed = jnp.sqrt(jnp.sum(new_vel ** 2, axis=-1, keepdims=True))
        new_ori = jnp.where(
            (new_speed > 1e-6) & mask[:, None],
            new_vel / new_speed, state.orientations[type_a])

        return state.replace(
            positions=state.positions.at[type_a].set(new_pos),
            velocities=state.velocities.at[type_a].set(new_vel),
            orientations=state.orientations.at[type_a].set(new_ori),
            score=state.score + score_delta,
        )
    return state.replace(score=state.score + score_delta)


# ── Null (unknown effect) ─────────────────────────────────────────────


def null(state, score_delta, **_):
    return state.replace(score=state.score + score_delta)


# ── Dispatch ───────────────────────────────────────────────────────────


EFFECT_DISPATCH = {
    # Kill / removal
    'kill_sprite':      kill_sprite,
    'kill_both':        kill_both,
    'kill_if_alive':    kill_if_alive,
    'kill_if_slow':     kill_if_slow,
    'kill_others':      kill_others,
    'kill_if_from_above': kill_if_from_above,
    # Position and orientation
    'step_back':          step_back,
    'reverse_direction':  reverse_direction,
    'turn_around':        turn_around,
    'flip_direction':     flip_direction,
    'undo_all':           undo_all,
    'wrap_around':        wrap_around,
    'attract_gaze':       attract_gaze,
    # Resources
    'change_resource':       change_resource,
    'collect_resource':      collect_resource,
    'avatar_collect_resource': avatar_collect_resource,
    'spend_resource':        spend_resource,
    'spend_avatar_resource': spend_avatar_resource,
    'kill_if_has_less':      kill_if_has_less,
    'kill_if_has_more':      kill_if_has_more,
    'kill_if_other_has_more':  kill_if_other_has_more,
    'kill_if_other_has_less':  kill_if_other_has_less,
    'kill_if_avatar_without_resource': kill_if_avatar_without_resource,
    # Spawn and transform
    'transform_to':       transform_to,
    'clone_sprite':       clone_sprite,
    'spawn_if_has_more':  spawn_if_has_more,
    'transform_others_to': transform_others_to,
    # Movement and conveying
    'teleport_to_exit':   teleport_to_exit,
    'convey_sprite':      convey_sprite,
    'wind_gust':          wind_gust,
    'slip_forward':       slip_forward,
    # Physics / wall interactions
    'bounce_forward':     partner_delta,
    'pull_with_it':       partner_delta,
    'wall_stop':          wall_stop,
    'wall_bounce':        wall_bounce,
    'bounce_direction':   bounce_direction,
}


def apply_masked_effect(state, prev_positions, type_a, type_b, mask,
                        effect_type, score_change, kwargs,
                        height, width, max_n):
    """Apply a single effect to all type_a sprites indicated by mask [max_n]."""
    n_affected = mask.sum()
    score_delta = n_affected * jnp.int32(score_change)
    handler = EFFECT_DISPATCH.get(effect_type, null)
    return handler(
        state,
        type_a=type_a, type_b=type_b, mask=mask,
        score_delta=score_delta, score_change=score_change,
        prev_positions=prev_positions, kwargs=kwargs,
        height=height, width=width, max_n=max_n,
    )
