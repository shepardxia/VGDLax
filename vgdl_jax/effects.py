"""Effect handlers for the VGDL-JAX step function.

Each handler implements one collision effect (kill, transform, resource change,
etc.). Handlers share a uniform keyword-argument interface — each declares the
params it needs and absorbs the rest via **_.

Sections:
    - JAX Primitives (prim_*)
    - Shared helpers
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


# ── JAX Primitives ───────────────────────────────────────────────────

def prim_kill(state, type_idx, mask):
    """Kill sprites where mask is True."""
    return state.replace(alive=state.alive.at[type_idx].set(state.alive[type_idx] & ~mask))


def prim_kill_partner(state, type_b, partner_idx, actor_mask):
    """Scatter-kill: kill type_b sprites that are partners of masked type_a sprites."""
    max_b = state.alive.shape[1]
    safe_idx = jnp.clip(partner_idx, 0, max_b - 1)
    b_kill = jnp.zeros(max_b, dtype=jnp.bool_).at[safe_idx].max(actor_mask & (partner_idx >= 0))
    return state.replace(alive=state.alive.at[type_b].set(state.alive[type_b] & ~b_kill))


def prim_restore_pos(state, type_idx, mask, prev_positions):
    """Restore positions from prev_positions where mask is True."""
    new = jnp.where(mask[:, None], prev_positions[type_idx], state.positions[type_idx])
    return state.replace(positions=state.positions.at[type_idx].set(new))


def prim_move(state, type_idx, mask, delta):
    """Add delta to positions where mask is True."""
    new = state.positions[type_idx] + mask[:, None] * delta
    return state.replace(positions=state.positions.at[type_idx].set(new))


def prim_set_orientation(state, type_idx, mask, new_ori):
    """Set orientations where mask is True."""
    cur = state.orientations[type_idx]
    new = jnp.where(mask[:, None], new_ori, cur)
    return state.replace(orientations=state.orientations.at[type_idx].set(new))


def prim_negate_orientation(state, type_idx, mask):
    """Negate orientations where mask is True."""
    cur = state.orientations[type_idx]
    new = jnp.where(mask[:, None], -cur, cur)
    return state.replace(orientations=state.orientations.at[type_idx].set(new))


def prim_clear_static(state, sg_idx, clear_mask):
    """Clear cells from a static grid."""
    return state.replace(static_grids=state.static_grids.at[sg_idx].set(
        state.static_grids[sg_idx] & ~clear_mask))


def _with_score(state, score_delta):
    """Add score_delta to state score."""
    return state.replace(score=state.score + score_delta)


def _partner_vals(field_slice, partner_idx):
    """Safely index field_slice[partner_idx] with clipping. field_slice: [max_n, ...]"""
    safe = jnp.clip(partner_idx, 0, field_slice.shape[0] - 1)
    return field_slice[safe]


def _partner_scatter_mask(actor_mask, partner_idx, max_b):
    """Build [max_b] bool mask: True for type_b slots that are partners of masked actors."""
    return jnp.zeros(max_b, dtype=jnp.bool_).at[jnp.clip(partner_idx, 0, max_b - 1)].max(
        actor_mask & (partner_idx >= 0))


# ── Shared helpers ──────────────────────────────────────────────────────

def _nearest_partner(pos_a, state, type_b, eff_b):
    """Find nearest alive type_b sprite for each pos_a sprite. Returns [eff_a, 2] positions."""
    pos_b = state.positions[type_b, :eff_b]
    alive_b = state.alive[type_b, :eff_b]
    diff = pos_a[:, None, :] - pos_b[None, :, :]
    dist_sq = jnp.sum(diff ** 2, axis=-1)
    dist_sq = jnp.where(alive_b[None, :], dist_sq, 1e10)
    nearest_b = jnp.argmin(dist_sq, axis=1)
    return pos_b[nearest_b]


def _fill_slots(state, target_type, source_mask, src_positions,
                src_orientations=None, target_speed=None, reset_cooldown=False):
    """Allocate dead slots in target_type and fill with source data."""
    should_fill, src_idx = prefix_sum_allocate(state.alive[target_type], source_mask)
    src_pos = src_positions[src_idx]
    state = state.replace(
        alive=state.alive.at[target_type].set(state.alive[target_type] | should_fill),
        positions=state.positions.at[target_type].set(
            jnp.where(should_fill[:, None], src_pos, state.positions[target_type])),
        ages=state.ages.at[target_type].set(
            jnp.where(should_fill, 0, state.ages[target_type])),
    )
    if src_orientations is not None:
        src_ori = src_orientations[src_idx]
        state = state.replace(
            orientations=state.orientations.at[target_type].set(
                jnp.where(should_fill[:, None], src_ori, state.orientations[target_type])))
    if target_speed is not None:
        state = state.replace(
            speeds=state.speeds.at[target_type].set(
                jnp.where(should_fill, target_speed, state.speeds[target_type])))
    if reset_cooldown:
        state = state.replace(
            cooldown_timers=state.cooldown_timers.at[target_type].set(
                jnp.where(should_fill, 0, state.cooldown_timers[target_type])))
    return state


# ── Kill / removal ─────────────────────────────────────────────────────

def kill_sprite(state, type_a, mask, score_delta, **_):
    return _with_score(prim_kill(state, type_a, mask), score_delta)


def kill_both(state, type_a, type_b, mask, score_delta, height, width,
              kwargs=None, partner_idx=None, **_):
    state = prim_kill(state, type_a, mask)
    if kwargs is None:
        kwargs = {}
    static_b_grid_idx = kwargs.get('static_b_grid_idx')
    if static_b_grid_idx is not None:
        ipos_a = state.positions[type_a].astype(jnp.int32)
        r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
        c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
        grid_coll = jnp.zeros((height, width), dtype=jnp.bool_).at[r_a, c_a].max(mask)
        state = prim_clear_static(state, static_b_grid_idx, grid_coll)
    elif type_b >= 0:
        state = prim_kill_partner(state, type_b, partner_idx, mask)
    return _with_score(state, score_delta)


def kill_if_slow(state, type_a, mask, score_change, kwargs, **_):
    limitspeed = kwargs.get('limitspeed', 0.0)
    should_kill = mask & (state.speeds[type_a] < limitspeed)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_others(state, type_a, mask, score_delta, kwargs, **_):
    kill_type = kwargs.get('kill_type_idx', -1)
    if kill_type >= 0:
        kill_all = jnp.broadcast_to(mask.any(), state.alive[kill_type].shape)
        state = prim_kill(state, kill_type, kill_all)
    return _with_score(state, score_delta)


def kill_if_from_above(state, prev_positions, type_a, type_b, mask,
                       score_change, partner_idx=None, **_):
    if type_b >= 0:
        icurr_b = _partner_vals(state.positions[type_b], partner_idx).astype(jnp.int32)
        iprev_b = _partner_vals(prev_positions[type_b], partner_idx).astype(jnp.int32)
        should_kill = mask & (partner_idx >= 0) & (icurr_b[:, 0] > iprev_b[:, 0])
    else:
        should_kill = jnp.zeros_like(mask)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_if_has_less(state, type_a, mask, score_change, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    should_kill = mask & (state.resources[type_a, :, r_idx] <= limit)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_if_has_more(state, type_a, mask, score_change, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    should_kill = mask & (state.resources[type_a, :, r_idx] >= limit)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_if_other_has_more(state, type_a, type_b, mask, score_change,
                           kwargs, partner_idx=None, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    if type_b >= 0:
        partner_res = _partner_vals(state.resources[type_b, :, r_idx], partner_idx)
        partner_res = jnp.where(partner_idx >= 0, partner_res, 0)
        should_kill = mask & jnp.greater_equal(partner_res, limit)
    else:
        should_kill = jnp.zeros_like(mask)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_if_other_has_less(state, type_a, type_b, mask, score_change,
                           kwargs, partner_idx=None, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    if type_b >= 0:
        partner_res = _partner_vals(state.resources[type_b, :, r_idx], partner_idx)
        partner_res = jnp.where(partner_idx >= 0, partner_res, -1)
        should_kill = mask & (partner_res >= 0) & (partner_res <= limit)
    else:
        should_kill = jnp.zeros_like(mask)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_if_avatar_without_resource(state, type_a, mask, score_change, kwargs, **_):
    ati = kwargs.get('avatar_type_idx', 0)
    r_idx = kwargs.get('resource_idx', 0)
    should_kill = mask & ~(state.resources[ati, 0, r_idx] > 0)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


# ── Position and orientation ───────────────────────────────────────────

def step_back(state, prev_positions, type_a, mask, score_delta, **_):
    return _with_score(prim_restore_pos(state, type_a, mask, prev_positions), score_delta)


def reverse_direction(state, type_a, mask, score_delta, **_):
    return _with_score(prim_negate_orientation(state, type_a, mask), score_delta)


def turn_around(state, prev_positions, type_a, mask, score_delta, **_):
    state = prim_negate_orientation(state, type_a, mask)
    return _with_score(prim_restore_pos(state, type_a, mask, prev_positions), score_delta)


def flip_direction(state, type_a, mask, score_delta, max_n, **_):
    rng, key = jax.random.split(state.rng)
    dir_indices = jax.random.randint(key, (max_n,), 0, 4)
    random_ori = DIRECTION_DELTAS[dir_indices]
    return _with_score(
        prim_set_orientation(state, type_a, mask, random_ori).replace(rng=rng),
        score_delta)


def undo_all(state, prev_positions, mask, score_delta, **_):
    new_positions = jnp.where(mask.any(), prev_positions, state.positions)
    return _with_score(state.replace(positions=new_positions), score_delta)


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
    return _with_score(state.replace(
        positions=state.positions.at[type_a].set(
            jnp.stack([new_row, new_col], axis=-1))), score_delta)


def attract_gaze(state, type_a, type_b, mask, score_delta, kwargs, max_n,
                 partner_idx=None, **_):
    prob = kwargs.get('prob', 0.5)
    if type_b >= 0:
        rng, key = jax.random.split(state.rng)
        rolls = jax.random.uniform(key, (max_n,))
        should_attract = mask & (rolls < prob)
        partner_ori = _partner_vals(state.orientations[type_b], partner_idx)
        return _with_score(
            prim_set_orientation(state, type_a, should_attract, partner_ori).replace(rng=rng),
            score_delta)
    return _with_score(state, score_delta)


# ── Resources ──────────────────────────────────────────────────────────

def change_resource(state, type_a, mask, score_delta, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    value = kwargs.get('value', 0)
    limit = kwargs.get('limit', 100)
    cur = state.resources[type_a, :, r_idx]
    new_val = jnp.where(mask, jnp.clip(cur + value, 0, limit), cur)
    return _with_score(state.replace(
        resources=state.resources.at[type_a, :, r_idx].set(new_val)), score_delta)


def collect_resource(state, type_a, type_b, mask, score_delta,
                     kwargs, partner_idx=None, **_):
    r_idx = kwargs.get('resource_idx', 0)
    r_value = kwargs.get('resource_value', 1)
    limit = kwargs.get('limit', 100)
    if type_b >= 0:
        b_affected = _partner_scatter_mask(mask, partner_idx, state.alive.shape[1])
        cur = state.resources[type_b, :, r_idx]
        new_val = jnp.where(b_affected, jnp.clip(cur + r_value, 0, limit), cur)
        state = state.replace(
            resources=state.resources.at[type_b, :, r_idx].set(new_val))
    return _with_score(state, score_delta)


def avatar_collect_resource(state, mask, score_delta, kwargs, **_):
    ati = kwargs.get('avatar_type_idx', 0)
    r_idx = kwargs.get('resource_idx', 0)
    r_val = kwargs.get('resource_value', 1)
    limit = kwargs.get('limit', 100)
    cur = state.resources[ati, 0, r_idx]
    new_val = jnp.minimum(cur + mask.sum() * r_val, limit)
    return _with_score(state.replace(
        resources=state.resources.at[ati, 0, r_idx].set(new_val)), score_delta)


def spend_resource(state, type_a, mask, score_delta, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    amount = kwargs.get('amount', 1)
    cur = state.resources[type_a, :, r_idx]
    spend = jnp.where(mask, jnp.minimum(cur, amount), 0)
    return _with_score(state.replace(
        resources=state.resources.at[type_a, :, r_idx].add(-spend)), score_delta)


def spend_avatar_resource(state, mask, score_delta, kwargs, **_):
    ati = kwargs.get('avatar_type_idx', 0)
    r_idx = kwargs.get('resource_idx', 0)
    amount = kwargs.get('amount', 1)
    cur = state.resources[ati, 0, r_idx]
    spend = jnp.minimum(cur, mask.sum() * amount)
    return _with_score(state.replace(
        resources=state.resources.at[ati, 0, r_idx].add(-spend)), score_delta)


# ── Spawn and transform ───────────────────────────────────────────────

def transform_to(state, type_a, mask, score_delta, kwargs, **_):
    new_type = kwargs['new_type_idx']
    target_speed = kwargs.get('target_speed', None)
    state = _with_score(prim_kill(state, type_a, mask), score_delta)
    return _fill_slots(state, new_type, mask,
                       state.positions[type_a], state.orientations[type_a],
                       target_speed=target_speed, reset_cooldown=True)


def clone_sprite(state, type_a, mask, score_delta, kwargs=None, **_):
    target_speed = kwargs.get('target_speed', None) if kwargs else None
    state = _fill_slots(state, type_a, mask,
                        state.positions[type_a], state.orientations[type_a],
                        target_speed=target_speed, reset_cooldown=True)
    return _with_score(state, score_delta)


def spawn_if_has_more(state, type_a, mask, score_delta, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    spawn_type = kwargs.get('spawn_type_idx', -1)
    target_speed = kwargs.get('target_speed', None)
    if spawn_type >= 0:
        has_enough = mask & (state.resources[type_a, :, r_idx] >= limit)
        state = _fill_slots(state, spawn_type, has_enough,
                            state.positions[type_a],
                            target_speed=target_speed, reset_cooldown=True)
    return _with_score(state, score_delta)


def transform_others_to(state, type_a, mask, score_delta, kwargs, **_):
    target_type = kwargs.get('target_type_idx', -1)
    new_type = kwargs.get('new_type_idx', -1)
    target_speed = kwargs.get('target_speed', None)
    if target_type >= 0 and new_type >= 0:
        any_collision = mask.any()
        target_alive = state.alive[target_type] & any_collision
        kill_all = jnp.broadcast_to(any_collision, state.alive[target_type].shape)
        state = prim_kill(state, target_type, kill_all)
        state = _fill_slots(state, new_type, target_alive,
                            state.positions[target_type],
                            state.orientations[target_type],
                            target_speed=target_speed, reset_cooldown=True)
    return _with_score(state, score_delta)


# ── Movement and conveying ─────────────────────────────────────────────

def teleport_to_exit(state, type_a, mask, score_delta, kwargs, **_):
    exit_type = kwargs.get('exit_type_idx', -1)
    if exit_type >= 0:
        rng, key = jax.random.split(state.rng)
        exit_pos = state.positions[exit_type]
        exit_alive = state.alive[exit_type]
        n_exits = exit_alive.sum()
        exit_rank = jnp.cumsum(exit_alive)         # [n_exit_slots]
        max_n = mask.shape[0]

        # Per-sprite independent random exit, fully vectorized (no vmap).
        # rand_indices: [max_n] ints in [0, n_exits)
        rand_indices = jax.random.randint(key, (max_n,), 0, jnp.maximum(n_exits, 1))
        # matches[i, j] = exit_alive[j] & (exit_rank[j] == rand_indices[i] + 1)
        matches = exit_alive[None, :] & (exit_rank[None, :] == rand_indices[:, None] + 1)
        chosen_slots = jnp.argmax(matches, axis=1)  # [max_n]
        targets = exit_pos[chosen_slots]             # [max_n, 2]

        pos_a = state.positions[type_a]
        new_pos = jnp.where(mask[:, None] & (n_exits > 0), targets, pos_a)
        return _with_score(state.replace(
            positions=state.positions.at[type_a].set(new_pos), rng=rng), score_delta)
    return _with_score(state, score_delta)


def convey_sprite(state, type_a, type_b, mask, score_delta,
                  kwargs, partner_idx=None, **_):
    strength = kwargs.get('strength', 1.0)
    if type_b >= 0:
        partner_ori = _partner_vals(state.orientations[type_b], partner_idx)
        valid = (partner_idx >= 0) & mask
        state = prim_move(state, type_a, valid, partner_ori * strength)
    return _with_score(state, score_delta)


def wind_gust(state, type_a, type_b, mask, score_delta, max_n,
              partner_idx=None, kwargs=None, **_):
    if kwargs is None:
        kwargs = {}
    if type_b >= 0:
        rng, key = jax.random.split(state.rng)
        strength = kwargs.get('strength', 1.0)
        offsets = jax.random.randint(key, (max_n,), -1, 2)
        per_sprite = strength + offsets.astype(jnp.float32)
        partner_ori = _partner_vals(state.orientations[type_b], partner_idx)
        state = prim_move(state, type_a, mask, partner_ori * per_sprite[:, None])
        return _with_score(state.replace(rng=rng), score_delta)
    return _with_score(state, score_delta)


def slip_forward(state, type_a, mask, score_delta, kwargs, max_n, **_):
    prob = kwargs.get('prob', 0.5)
    rng, key = jax.random.split(state.rng)
    rolls = jax.random.uniform(key, (max_n,))
    should_slip = mask & (rolls < prob)
    state = prim_move(state, type_a, should_slip, state.orientations[type_a])
    return _with_score(state.replace(rng=rng), score_delta)


# ── Physics / wall interactions ────────────────────────────────────────

def partner_delta(state, prev_positions, type_a, type_b, mask,
                  height, width, score_delta, partner_idx=None, **_):
    """Apply type_b's movement delta to type_a (bounceForward / pullWithIt)."""
    if type_b >= 0:
        b_curr = _partner_vals(state.positions[type_b], partner_idx)
        b_prev = _partner_vals(prev_positions[type_b], partner_idx)
        b_delta = (b_curr - b_prev).astype(jnp.int32)
        valid = (partner_idx >= 0) & mask
        new_pos = jnp.where(valid[:, None],
                            state.positions[type_a] + b_delta,
                            state.positions[type_a])
        new_pos = jnp.clip(new_pos,
                           jnp.array([0, 0]),
                           jnp.array([height - 1, width - 1]))
        state = state.replace(
            positions=state.positions.at[type_a].set(new_pos))
    return _with_score(state, score_delta)


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
        grid_b = state.static_grids[static_b_grid_idx]
        delta_d = pos - prev
        going_down = jnp.where(delta_d[:, 0] != 0, delta_d[:, 0] > 0, vel[:, 0] >= 0)
        going_right = jnp.where(delta_d[:, 1] != 0, delta_d[:, 1] > 0, vel[:, 1] >= 0)
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
        v_rdiff = jnp.abs(pos[:, None, 0] - pos_b[None, :, 0])
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

    vert_mask = m & has_row_cross
    if static_b_grid_idx is not None:
        flush_row = jnp.where(going_down, wall_row_v - 1.0, wall_row_v + 1.0)
        new_pos_row = jnp.where(vert_mask, flush_row, pos[:, 0])
    elif type_b >= 0:
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

    horiz_mask = m & has_col_cross
    if static_b_grid_idx is not None:
        flush_col = jnp.where(going_right, wall_col_h - 1.0, wall_col_h + 1.0)
        new_pos_col = jnp.where(horiz_mask, flush_col, pos[:, 1])
    elif type_b >= 0:
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
        grid_b = state.static_grids[static_b_grid_idx]
        r_curr = jnp.clip(jnp.round(pos[:, 0]).astype(jnp.int32), 0, height - 1)
        c_curr = jnp.clip(jnp.round(pos[:, 1]).astype(jnp.int32), 0, width - 1)
        row_diff = jnp.abs(pos[:, 0] - r_curr.astype(jnp.float32))
        col_diff = jnp.abs(pos[:, 1] - c_curr.astype(jnp.float32))
        is_vertical = row_diff >= col_diff
    elif type_b >= 0:
        nb_pos = _nearest_partner(pos, state, type_b, eff_b)
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
            r_curr = jnp.clip(jnp.round(pos_a[:, 0]).astype(jnp.int32), 0, height - 1)
            c_curr = jnp.clip(jnp.round(pos_a[:, 1]).astype(jnp.int32), 0, width - 1)
            nb_pos = jnp.stack([r_curr.astype(jnp.float32),
                                c_curr.astype(jnp.float32)], axis=-1)
        else:
            nb_pos = _nearest_partner(pos_a, state, type_b, eff_b)

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
    return _with_score(state, score_delta)


# ── Null (unknown effect) ─────────────────────────────────────────────

def null(state, score_delta, **_):
    return _with_score(state, score_delta)


# ── Dispatch ───────────────────────────────────────────────────────────

# Single source of truth: VGDL name → (handler_fn, internal_key)
EFFECT_REGISTRY = {
    # Kill / removal
    'killSprite':     (kill_sprite,     'kill_sprite'),
    'killBoth':       (kill_both,       'kill_both'),
    'killIfAlive':    (kill_sprite,     'kill_if_alive'),
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


# ── Static-A handlers ─────────────────────────────────────────────────

def _static_kill_if_other_resource(state, sg_idx, type_b, grid_mask,
                                    score_change, kwargs, height, width,
                                    compare_fn):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    if type_b >= 0:
        ipos_b = state.positions[type_b].astype(jnp.int32)
        r_b = jnp.clip(ipos_b[:, 0], 0, height - 1)
        c_b = jnp.clip(ipos_b[:, 1], 0, width - 1)
        ib_b = in_bounds(ipos_b, height, width)
        b_matches = compare_fn(state.resources[type_b, :, r_idx], limit) & state.alive[type_b] & ib_b
        res_grid = jnp.zeros((height, width), dtype=jnp.bool_).at[r_b, c_b].max(b_matches)
        kill_mask = grid_mask & res_grid
    else:
        kill_mask = jnp.zeros_like(grid_mask)
    state = prim_clear_static(state, sg_idx, kill_mask)
    return _with_score(state, kill_mask.sum() * jnp.int32(score_change))


def _static_kill_sprite(state, sg_idx, type_b, grid_mask, score_change, kwargs, height, width):
    n_killed = grid_mask.sum()
    state = prim_clear_static(state, sg_idx, grid_mask)
    return _with_score(state, n_killed * jnp.int32(score_change))


def _static_kill_both(state, sg_idx, type_b, grid_mask, score_change, kwargs, height, width):
    n_killed = grid_mask.sum()
    state = prim_clear_static(state, sg_idx, grid_mask)
    if type_b >= 0:
        ipos_b = state.positions[type_b].astype(jnp.int32)
        r_b = jnp.clip(ipos_b[:, 0], 0, height - 1)
        c_b = jnp.clip(ipos_b[:, 1], 0, width - 1)
        ib_b = in_bounds(ipos_b, height, width)
        mask_b = grid_mask[r_b, c_b] & state.alive[type_b] & ib_b
        state = prim_kill(state, type_b, mask_b)
    return _with_score(state, n_killed * jnp.int32(score_change))


def _static_kill_if_other_has_more(state, sg_idx, type_b, grid_mask,
                                    score_change, kwargs, height, width):
    return _static_kill_if_other_resource(
        state, sg_idx, type_b, grid_mask, score_change, kwargs,
        height, width, jnp.greater_equal)


def _static_kill_if_other_has_less(state, sg_idx, type_b, grid_mask,
                                    score_change, kwargs, height, width):
    return _static_kill_if_other_resource(
        state, sg_idx, type_b, grid_mask, score_change, kwargs,
        height, width, jnp.less_equal)


def _static_kill_if_avatar_without_resource(state, sg_idx, type_b, grid_mask,
                                             score_change, kwargs, height, width):
    ati = kwargs.get('avatar_type_idx', 0)
    r_idx = kwargs.get('resource_idx', 0)
    kill_mask = grid_mask & ~(state.resources[ati, 0, r_idx] > 0)
    state = prim_clear_static(state, sg_idx, kill_mask)
    return _with_score(state, kill_mask.sum() * jnp.int32(score_change))


def _static_collect_resource(state, sg_idx, type_b, grid_mask,
                              score_change, kwargs, height, width):
    n_killed = grid_mask.sum()
    state = prim_clear_static(state, sg_idx, grid_mask)
    r_idx = kwargs.get('resource_idx', 0)
    r_val = kwargs.get('resource_value', 1)
    limit = kwargs.get('limit', 100)
    ati = kwargs.get('avatar_type_idx', type_b)
    current = state.resources[ati, 0, r_idx]
    new_res = jnp.minimum(current + n_killed * r_val, limit)
    return _with_score(state.replace(
        resources=state.resources.at[ati, 0, r_idx].set(new_res)),
        n_killed * jnp.int32(score_change))


_STATIC_A_HANDLERS = {
    'kill_sprite': _static_kill_sprite,
    'kill_both': _static_kill_both,
    'kill_if_other_has_more': _static_kill_if_other_has_more,
    'kill_if_other_has_less': _static_kill_if_other_has_less,
    'kill_if_avatar_without_resource': _static_kill_if_avatar_without_resource,
    'collect_resource': _static_collect_resource,
    'avatar_collect_resource': _static_collect_resource,
}


def apply_static_a_effect(state, static_a_grid_idx, type_b, grid_mask,
                          effect_type, score_change, kwargs, height, width):
    """Apply an effect where type_a is stored as a static grid."""
    handler = _STATIC_A_HANDLERS.get(effect_type)
    if handler is not None:
        return handler(state, static_a_grid_idx, type_b, grid_mask,
                       score_change, kwargs, height, width)
    return state  # Unsupported effect — no-op


def apply_masked_effect(state, prev_positions, type_a, type_b, mask,
                        effect_type, score_change, kwargs,
                        height, width, max_n,
                        max_a=None, max_b=None,
                        partner_idx=None):
    """Apply a single effect to all type_a sprites indicated by mask [max_n].

    Both score_delta and score_change are passed to handlers:
      - score_delta = mask.sum() * score_change — pre-computed total for effects
        that apply to ALL colliding sprites.
      - score_change — raw per-sprite value for handlers that conditionally kill a
        SUBSET of colliders.

    partner_idx: optional [max_n] int32 array giving the type_b slot index for each
      type_a sprite (-1 if no collision).
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
        partner_idx=partner_idx,
    )
