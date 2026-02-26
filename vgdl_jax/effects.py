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


def kill_both(state, type_a, type_b, mask, score_delta, height, width, kwargs=None, **_):
    state = state.replace(
        alive=state.alive.at[type_a].set(state.alive[type_a] & ~mask))
    if kwargs is None:
        kwargs = {}
    static_b_grid_idx = kwargs.get('static_b_grid_idx')
    if static_b_grid_idx is not None:
        # type_b is a static grid — clear cells where type_a collided
        ipos_a = state.positions[type_a].astype(jnp.int32)
        r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
        c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
        grid_coll = jnp.zeros((height, width), dtype=jnp.bool_)
        grid_coll = grid_coll.at[r_a, c_a].max(mask)
        new_grids = state.static_grids.at[static_b_grid_idx].set(
            state.static_grids[static_b_grid_idx] & ~grid_coll)
        state = state.replace(static_grids=new_grids)
    elif type_b >= 0:
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


def _kill_if_resource(state, type_a, mask, score_change, kwargs, compare_fn):
    """Kill type_a sprites where compare_fn(resource, limit) is true."""
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    cur = state.resources[type_a, :, r_idx]
    should_kill = mask & compare_fn(cur, limit)
    return state.replace(
        alive=state.alive.at[type_a].set(state.alive[type_a] & ~should_kill),
        score=state.score + should_kill.sum() * jnp.int32(score_change),
    )


def kill_if_has_less(state, type_a, mask, score_change, kwargs, **_):
    """Kill type_a if its resource <= limit."""
    return _kill_if_resource(state, type_a, mask, score_change, kwargs, jnp.less_equal)


def kill_if_has_more(state, type_a, mask, score_change, kwargs, **_):
    """Kill type_a if its resource >= limit."""
    return _kill_if_resource(state, type_a, mask, score_change, kwargs, jnp.greater_equal)


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


def _fill_slots(state, target_type, source_mask, src_positions, src_orientations):
    """Allocate dead slots in target_type and fill with source data."""
    should_fill, src_idx = prefix_sum_allocate(state.alive[target_type], source_mask)
    src_pos = src_positions[src_idx]
    src_ori = src_orientations[src_idx]
    return state.replace(
        alive=state.alive.at[target_type].set(state.alive[target_type] | should_fill),
        positions=state.positions.at[target_type].set(
            jnp.where(should_fill[:, None], src_pos, state.positions[target_type])),
        orientations=state.orientations.at[target_type].set(
            jnp.where(should_fill[:, None], src_ori, state.orientations[target_type])),
        ages=state.ages.at[target_type].set(
            jnp.where(should_fill, 0, state.ages[target_type])),
    )


def transform_to(state, prev_positions, type_a, mask, score_delta, kwargs, **_):
    new_type = kwargs['new_type_idx']
    state = state.replace(
        alive=state.alive.at[type_a].set(state.alive[type_a] & ~mask),
        score=state.score + score_delta,
    )
    return _fill_slots(state, new_type, mask,
                        state.positions[type_a], state.orientations[type_a])


def clone_sprite(state, type_a, mask, score_delta, **_):
    state = _fill_slots(state, type_a, mask,
                         state.positions[type_a], state.orientations[type_a])
    return state.replace(score=state.score + score_delta)


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
              score_delta, kwargs, height, width,
              max_a=None, max_b=None, **_):
    global_max_n = state.alive.shape[1]
    eff_a = max_a if max_a is not None else global_max_n
    eff_b = max_b if max_b is not None else global_max_n

    friction = kwargs.get('friction', 0.0)
    static_b_grid_idx = kwargs.get('static_b_grid_idx')
    pos = state.positions[type_a, :eff_a]
    prev = prev_positions[type_a, :eff_a]
    vel = state.velocities[type_a, :eff_a]
    pf = state.passive_forces[type_a, :eff_a]
    m = mask[:eff_a]

    if static_b_grid_idx is not None:
        # Static grid path: find wall cell in movement direction
        grid_b = state.static_grids[static_b_grid_idx]
        delta_d = pos - prev
        # Movement direction (velocity fallback when delta is zero)
        going_down = jnp.where(delta_d[:, 0] != 0, delta_d[:, 0] > 0, vel[:, 0] >= 0)
        going_right = jnp.where(delta_d[:, 1] != 0, delta_d[:, 1] > 0, vel[:, 1] >= 0)
        # Wall cell: ceil in forward dir, floor in backward dir
        r_wall = jnp.where(going_down,
                           jnp.ceil(pos[:, 0]).astype(jnp.int32),
                           jnp.floor(pos[:, 0]).astype(jnp.int32))
        c_wall = jnp.where(going_right,
                           jnp.ceil(pos[:, 1]).astype(jnp.int32),
                           jnp.floor(pos[:, 1]).astype(jnp.int32))
        c_at_prev = jnp.round(prev[:, 1]).astype(jnp.int32)
        r_at_prev = jnp.round(prev[:, 0]).astype(jnp.int32)
        r_wall_c = jnp.clip(r_wall, 0, height - 1)
        c_wall_c = jnp.clip(c_wall, 0, width - 1)
        c_prev_c = jnp.clip(c_at_prev, 0, width - 1)
        r_prev_c = jnp.clip(r_at_prev, 0, height - 1)
        has_row_cross = grid_b[r_wall_c, c_prev_c]
        has_col_cross = grid_b[r_prev_c, c_wall_c]
        wall_row_v = r_wall.astype(jnp.float32)
        wall_col_h = c_wall.astype(jnp.float32)
    elif type_b >= 0:
        pos_b = state.positions[type_b, :eff_b]
        alive_b = state.alive[type_b, :eff_b]
        threshold = 1.0 - AABB_EPS
        v_rdiff = jnp.abs(pos[:, None, 0] - pos_b[None, :, 0])   # [eff_a, eff_b]
        v_cdiff = jnp.abs(prev[:, None, 1] - pos_b[None, :, 1])
        check_v = (v_rdiff < threshold) & (v_cdiff < threshold) & alive_b[None, :]
        h_rdiff = jnp.abs(prev[:, None, 0] - pos_b[None, :, 0])
        h_cdiff = jnp.abs(pos[:, None, 1] - pos_b[None, :, 1])
        check_h = (h_rdiff < threshold) & (h_cdiff < threshold) & alive_b[None, :]
        has_row_cross = jnp.any(check_v, axis=1)
        has_col_cross = jnp.any(check_h, axis=1)
    else:
        has_row_cross = jnp.zeros_like(m)
        has_col_cross = jnp.zeros_like(m)

    neither = m & ~has_row_cross & ~has_col_cross
    delta = pos - prev
    is_vert_fb = jnp.abs(delta[:, 0]) >= jnp.abs(delta[:, 1])
    has_row_cross = has_row_cross | (neither & is_vert_fb)
    has_col_cross = has_col_cross | (neither & ~is_vert_fb)

    # Vertical collision
    vert_mask = m & has_row_cross
    if static_b_grid_idx is not None:
        flush_row = jnp.where(going_down, wall_row_v - 1.0, wall_row_v + 1.0)
        new_pos_row = jnp.where(vert_mask, flush_row, pos[:, 0])
    elif type_b >= 0:
        v_dist = jnp.where(check_v, v_rdiff, 1e10)
        nearest_v = jnp.argmin(v_dist, axis=1)     # [eff_a] indices into [eff_b]
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
    horiz_mask = m & has_col_cross
    if static_b_grid_idx is not None:
        flush_col = jnp.where(going_right, wall_col_h - 1.0, wall_col_h + 1.0)
        new_pos_col = jnp.where(horiz_mask, flush_col, pos[:, 1])
    elif type_b >= 0:
        h_dist = jnp.where(check_h, h_cdiff, 1e10)
        nearest_h = jnp.argmin(h_dist, axis=1)     # [eff_a] indices into [eff_b]
        wall_col = pos_b[nearest_h, 1]
        flush_col = jnp.where(vel[:, 1] > 0, wall_col - 1.0, wall_col + 1.0)
        new_pos_col = jnp.where(horiz_mask, flush_col, pos[:, 1])
    else:
        new_pos_col = jnp.where(horiz_mask, prev[:, 1], pos[:, 1])
    new_vel_col = jnp.where(horiz_mask, 0.0, new_vel_col_v)
    new_pf_col = jnp.where(horiz_mask, 0.0, pf[:, 1])
    new_vel_row2 = jnp.where(
        horiz_mask & (friction > 0), new_vel_row * (1.0 - friction), new_vel_row)

    # Write sliced results back — only first eff_a entries modified
    new_pos_s = jnp.stack([new_pos_row, new_pos_col], axis=-1)
    new_vel_s = jnp.stack([new_vel_row2, new_vel_col], axis=-1)
    new_pf_s = jnp.stack([new_pf_row, new_pf_col], axis=-1)
    return state.replace(
        positions=state.positions.at[type_a, :eff_a].set(new_pos_s),
        velocities=state.velocities.at[type_a, :eff_a].set(new_vel_s),
        passive_forces=state.passive_forces.at[type_a, :eff_a].set(new_pf_s),
        score=state.score + score_delta,
    )


def wall_bounce(state, prev_positions, type_a, type_b, mask,
                score_delta, kwargs, max_a=None, max_b=None, height=0, width=0, **_):
    global_max_n = state.alive.shape[1]
    eff_a = max_a if max_a is not None else global_max_n
    eff_b = max_b if max_b is not None else global_max_n

    friction = kwargs.get('friction', 0.0)
    static_b_grid_idx = kwargs.get('static_b_grid_idx')
    pos = state.positions[type_a, :eff_a]
    prev = prev_positions[type_a, :eff_a]
    vel = state.velocities[type_a, :eff_a]
    m = mask[:eff_a]

    if static_b_grid_idx is not None:
        # Static grid path: determine bounce axis from grid cell position
        grid_b = state.static_grids[static_b_grid_idx]
        r_curr = jnp.clip(jnp.round(pos[:, 0]).astype(jnp.int32), 0, height - 1)
        c_curr = jnp.clip(jnp.round(pos[:, 1]).astype(jnp.int32), 0, width - 1)
        # Nearest wall cell is the one we're overlapping with
        row_diff = jnp.abs(pos[:, 0] - r_curr.astype(jnp.float32))
        col_diff = jnp.abs(pos[:, 1] - c_curr.astype(jnp.float32))
        is_vertical = row_diff >= col_diff
    elif type_b >= 0:
        pos_b_all = state.positions[type_b, :eff_b]
        alive_b = state.alive[type_b, :eff_b]
        diff_ab = pos[:, None, :] - pos_b_all[None, :, :]   # [eff_a, eff_b, 2]
        dist_sq = jnp.sum(diff_ab ** 2, axis=-1)
        dist_sq = jnp.where(alive_b[None, :], dist_sq, 1e10)
        nearest_b = jnp.argmin(dist_sq, axis=1)    # [eff_a] indices into [eff_b]
        nb_pos = pos_b_all[nearest_b]
        row_diff = jnp.abs(pos[:, 0] - nb_pos[:, 0])
        col_diff = jnp.abs(pos[:, 1] - nb_pos[:, 1])
        is_vertical = row_diff >= col_diff
    else:
        delta = pos - prev
        is_vertical = jnp.abs(delta[:, 0]) >= jnp.abs(delta[:, 1])

    new_vel_row = jnp.where(m & is_vertical, -vel[:, 0], vel[:, 0])
    new_vel_col = jnp.where(m & ~is_vertical, -vel[:, 1], vel[:, 1])

    speed = jnp.sqrt(new_vel_row ** 2 + new_vel_col ** 2)
    fric_scale = jnp.where(
        m & (friction > 0) & (speed > 1e-6),
        jnp.maximum(1.0 - friction, 0.0), 1.0)
    new_vel_row = new_vel_row * fric_scale
    new_vel_col = new_vel_col * fric_scale

    new_pos = jnp.where(m[:, None], prev, pos)

    new_speed = jnp.sqrt(new_vel_row ** 2 + new_vel_col ** 2)
    ori_a = state.orientations[type_a, :eff_a]
    new_ori_row = jnp.where(new_speed > 1e-6, new_vel_row / new_speed, ori_a[:, 0])
    new_ori_col = jnp.where(new_speed > 1e-6, new_vel_col / new_speed, ori_a[:, 1])

    return state.replace(
        positions=state.positions.at[type_a, :eff_a].set(new_pos),
        velocities=state.velocities.at[type_a, :eff_a].set(
            jnp.stack([new_vel_row, new_vel_col], axis=-1)),
        orientations=state.orientations.at[type_a, :eff_a].set(
            jnp.stack([new_ori_row, new_ori_col], axis=-1)),
        score=state.score + score_delta,
    )


def bounce_direction(state, prev_positions, type_a, type_b, mask,
                     score_delta, kwargs, max_a=None, max_b=None, height=0, width=0, **_):
    static_b_grid_idx = kwargs.get('static_b_grid_idx') if kwargs else None

    if static_b_grid_idx is not None or type_b >= 0:
        global_max_n = state.alive.shape[1]
        eff_a = max_a if max_a is not None else global_max_n
        eff_b = max_b if max_b is not None else global_max_n

        pos_a = state.positions[type_a, :eff_a]
        vel = state.velocities[type_a, :eff_a]
        prev = prev_positions[type_a, :eff_a]
        m = mask[:eff_a]

        if static_b_grid_idx is not None:
            # Static grid path: nearest wall is the grid cell we're overlapping
            r_curr = jnp.clip(jnp.round(pos_a[:, 0]).astype(jnp.int32), 0, height - 1)
            c_curr = jnp.clip(jnp.round(pos_a[:, 1]).astype(jnp.int32), 0, width - 1)
            nb_pos = jnp.stack([r_curr.astype(jnp.float32),
                                c_curr.astype(jnp.float32)], axis=-1)
        else:
            pos_b = state.positions[type_b, :eff_b]
            diff = pos_a[:, None, :] - pos_b[None, :, :]     # [eff_a, eff_b, 2]
            dist_sq = jnp.sum(diff ** 2, axis=-1)
            alive_b = state.alive[type_b, :eff_b]
            dist_sq = jnp.where(alive_b[None, :], dist_sq, 1e10)
            nearest_b = jnp.argmin(dist_sq, axis=1)  # [eff_a] indices into [eff_b]
            nb_pos = pos_b[nearest_b]

        n = pos_a - nb_pos
        n_len = jnp.sqrt(jnp.sum(n ** 2, axis=-1, keepdims=True))
        n = jnp.where(n_len > 1e-6, n / n_len, jnp.array([0.0, 0.0]))

        v_dot_n = jnp.sum(vel * n, axis=-1, keepdims=True)
        reflected = vel - 2.0 * v_dot_n * n

        friction = kwargs.get('friction', 0.0)
        fric_scale = jnp.where(
            m & (friction > 0), jnp.maximum(1.0 - friction, 0.0), 1.0)
        reflected = reflected * fric_scale[:, None]

        new_vel = jnp.where(m[:, None], reflected, vel)
        new_pos = jnp.where(m[:, None], prev, pos_a)

        new_speed = jnp.sqrt(jnp.sum(new_vel ** 2, axis=-1, keepdims=True))
        ori_a = state.orientations[type_a, :eff_a]
        new_ori = jnp.where(
            (new_speed > 1e-6) & m[:, None],
            new_vel / new_speed, ori_a)

        return state.replace(
            positions=state.positions.at[type_a, :eff_a].set(new_pos),
            velocities=state.velocities.at[type_a, :eff_a].set(new_vel),
            orientations=state.orientations.at[type_a, :eff_a].set(new_ori),
            score=state.score + score_delta,
        )
    return state.replace(score=state.score + score_delta)


# ── Null (unknown effect) ─────────────────────────────────────────────


def null(state, score_delta, **_):
    return state.replace(score=state.score + score_delta)


# ── Dispatch ───────────────────────────────────────────────────────────


# Single source of truth: VGDL name → (handler_fn, internal_key)
EFFECT_REGISTRY = {
    # Kill / removal
    'killSprite':     (kill_sprite,     'kill_sprite'),
    'killBoth':       (kill_both,       'kill_both'),
    'killIfAlive':    (kill_if_alive,   'kill_if_alive'),
    'killIfSlow':     (kill_if_slow,    'kill_if_slow'),
    'KillOthers':     (kill_others,     'kill_others'),
    'killIfFromAbove': (kill_if_from_above, 'kill_if_from_above'),
    # Position and orientation
    'stepBack':          (step_back,          'step_back'),
    'reverseDirection':  (reverse_direction,  'reverse_direction'),
    'turnAround':        (turn_around,        'turn_around'),
    'flipDirection':     (flip_direction,     'flip_direction'),
    'undoAll':           (undo_all,           'undo_all'),
    'wrapAround':        (wrap_around,        'wrap_around'),
    'attractGaze':       (attract_gaze,       'attract_gaze'),
    # Resources
    'changeResource':       (change_resource,       'change_resource'),
    'collectResource':      (collect_resource,      'collect_resource'),
    'AvatarCollectResource': (avatar_collect_resource, 'avatar_collect_resource'),
    'SpendResource':        (spend_resource,        'spend_resource'),
    'SpendAvatarResource':  (spend_avatar_resource, 'spend_avatar_resource'),
    'killIfHasLess':        (kill_if_has_less,      'kill_if_has_less'),
    'killIfHasMore':        (kill_if_has_more,      'kill_if_has_more'),
    'killIfOtherHasMore':   (kill_if_other_has_more,  'kill_if_other_has_more'),
    'killIfOtherHasLess':   (kill_if_other_has_less,  'kill_if_other_has_less'),
    'KillIfAvatarWithoutResource': (kill_if_avatar_without_resource, 'kill_if_avatar_without_resource'),
    # Spawn and transform
    'transformTo':       (transform_to,       'transform_to'),
    'cloneSprite':       (clone_sprite,       'clone_sprite'),
    'spawnIfHasMore':    (spawn_if_has_more,  'spawn_if_has_more'),
    'TransformOthersTo': (transform_others_to, 'transform_others_to'),
    # Movement and conveying
    'teleportToExit':   (teleport_to_exit,   'teleport_to_exit'),
    'conveySprite':     (convey_sprite,      'convey_sprite'),
    'windGust':         (wind_gust,          'wind_gust'),
    'slipForward':      (slip_forward,       'slip_forward'),
    # Physics / wall interactions
    'bounceForward':    (partner_delta,      'bounce_forward'),
    'pullWithIt':       (partner_delta,      'pull_with_it'),
    'wallStop':         (wall_stop,          'wall_stop'),
    'wallBounce':       (wall_bounce,        'wall_bounce'),
    'bounceDirection':  (bounce_direction,   'bounce_direction'),
}

# Derived mappings
EFFECT_DISPATCH = {v[1]: v[0] for v in EFFECT_REGISTRY.values()}
VGDL_TO_KEY = {k: v[1] for k, v in EFFECT_REGISTRY.items()}


def apply_static_a_effect(state, static_a_grid_idx, type_b, grid_mask,
                          effect_type, score_change, kwargs, height, width):
    """Apply an effect where type_a is stored as a static grid.

    Args:
        state: GameState
        static_a_grid_idx: index into state.static_grids for type_a
        type_b: dynamic type index of the collision partner
        grid_mask: [H, W] bool — cells where static type_a overlaps type_b
        effect_type: str
        score_change: int
        kwargs: dict
        height, width: grid dimensions

    Only a subset of effects make sense for static type_a:
    - kill_sprite: clear grid cells
    - kill_both: clear grid cells + kill type_b
    - kill_if_other_has_more/less: conditional clear
    - kill_if_avatar_without_resource: conditional clear
    - collect_resource / avatar_collect_resource: clear + add resource
    """
    sg_idx = static_a_grid_idx
    n_killed = grid_mask.sum()

    if effect_type == 'kill_sprite':
        new_grids = state.static_grids.at[sg_idx].set(
            state.static_grids[sg_idx] & ~grid_mask)
        return state.replace(
            static_grids=new_grids,
            score=state.score + n_killed * jnp.int32(score_change))

    elif effect_type == 'kill_both':
        new_grids = state.static_grids.at[sg_idx].set(
            state.static_grids[sg_idx] & ~grid_mask)
        # Also kill type_b sprites at collision cells
        if type_b >= 0:
            ipos_b = state.positions[type_b].astype(jnp.int32)
            r_b = jnp.clip(ipos_b[:, 0], 0, height - 1)
            c_b = jnp.clip(ipos_b[:, 1], 0, width - 1)
            ib_b = in_bounds(ipos_b, height, width)
            mask_b = grid_mask[r_b, c_b] & state.alive[type_b] & ib_b
            return state.replace(
                static_grids=new_grids,
                alive=state.alive.at[type_b].set(state.alive[type_b] & ~mask_b),
                score=state.score + n_killed * jnp.int32(score_change))
        return state.replace(
            static_grids=new_grids,
            score=state.score + n_killed * jnp.int32(score_change))

    elif effect_type == 'kill_if_other_has_more':
        r_idx = kwargs.get('resource_idx', 0)
        limit = kwargs.get('limit', 0)
        if type_b >= 0:
            # Check if type_b at collision cells has resource >= limit
            ipos_b = state.positions[type_b].astype(jnp.int32)
            r_b = jnp.clip(ipos_b[:, 0], 0, height - 1)
            c_b = jnp.clip(ipos_b[:, 1], 0, width - 1)
            ib_b = in_bounds(ipos_b, height, width)
            b_has_enough = (state.resources[type_b, :, r_idx] >= limit) & state.alive[type_b] & ib_b
            # Build grid of cells where type_b has enough
            res_grid = jnp.zeros((height, width), dtype=jnp.bool_)
            res_grid = res_grid.at[r_b, c_b].max(b_has_enough)
            kill_mask = grid_mask & res_grid
        else:
            kill_mask = jnp.zeros_like(grid_mask)
        n_cond = kill_mask.sum()
        new_grids = state.static_grids.at[sg_idx].set(
            state.static_grids[sg_idx] & ~kill_mask)
        return state.replace(
            static_grids=new_grids,
            score=state.score + n_cond * jnp.int32(score_change))

    elif effect_type == 'kill_if_other_has_less':
        r_idx = kwargs.get('resource_idx', 0)
        limit = kwargs.get('limit', 0)
        if type_b >= 0:
            ipos_b = state.positions[type_b].astype(jnp.int32)
            r_b = jnp.clip(ipos_b[:, 0], 0, height - 1)
            c_b = jnp.clip(ipos_b[:, 1], 0, width - 1)
            ib_b = in_bounds(ipos_b, height, width)
            b_has_less = (state.resources[type_b, :, r_idx] <= limit) & state.alive[type_b] & ib_b
            res_grid = jnp.zeros((height, width), dtype=jnp.bool_)
            res_grid = res_grid.at[r_b, c_b].max(b_has_less)
            kill_mask = grid_mask & res_grid
        else:
            kill_mask = jnp.zeros_like(grid_mask)
        n_cond = kill_mask.sum()
        new_grids = state.static_grids.at[sg_idx].set(
            state.static_grids[sg_idx] & ~kill_mask)
        return state.replace(
            static_grids=new_grids,
            score=state.score + n_cond * jnp.int32(score_change))

    elif effect_type == 'kill_if_avatar_without_resource':
        ati = kwargs.get('avatar_type_idx', 0)
        r_idx = kwargs.get('resource_idx', 0)
        avatar_has = state.resources[ati, 0, r_idx] > 0
        kill_mask = grid_mask & ~avatar_has
        n_cond = kill_mask.sum()
        new_grids = state.static_grids.at[sg_idx].set(
            state.static_grids[sg_idx] & ~kill_mask)
        return state.replace(
            static_grids=new_grids,
            score=state.score + n_cond * jnp.int32(score_change))

    elif effect_type in ('collect_resource', 'avatar_collect_resource'):
        # Kill static sprite + add resource to avatar
        new_grids = state.static_grids.at[sg_idx].set(
            state.static_grids[sg_idx] & ~grid_mask)
        r_idx = kwargs.get('resource_idx', 0)
        r_val = kwargs.get('resource_value', 1)
        limit = kwargs.get('limit', 100)
        ati = kwargs.get('avatar_type_idx', type_b)
        current = state.resources[ati, 0, r_idx]
        new_res = jnp.minimum(current + n_killed * r_val, limit)
        return state.replace(
            static_grids=new_grids,
            resources=state.resources.at[ati, 0, r_idx].set(new_res),
            score=state.score + n_killed * jnp.int32(score_change))

    else:
        # Unsupported effect on static type_a — no-op
        return state


def apply_masked_effect(state, prev_positions, type_a, type_b, mask,
                        effect_type, score_change, kwargs,
                        height, width, max_n,
                        max_a=None, max_b=None):
    """Apply a single effect to all type_a sprites indicated by mask [max_n].

    Both score_delta and score_change are passed to handlers:
      - score_delta = mask.sum() * score_change — pre-computed total for effects
        that apply to ALL colliding sprites (e.g. kill_sprite, step_back).
      - score_change — raw per-sprite value for handlers that conditionally kill a
        SUBSET of colliders (kill_if_slow, kill_if_from_above, kill_if_has_less/more,
        kill_if_other_has_more/less, kill_if_avatar_without_resource). These compute
        their own total from the narrowed mask.
    """
    n_affected = mask.sum()
    score_delta = n_affected * jnp.int32(score_change)
    handler = EFFECT_DISPATCH.get(effect_type, null)
    return handler(
        state,
        type_a=type_a, type_b=type_b, mask=mask,
        score_delta=score_delta, score_change=score_change,
        prev_positions=prev_positions, kwargs=kwargs,
        height=height, width=width, max_n=max_n,
        max_a=max_a, max_b=max_b,
    )
