import jax
import jax.numpy as jnp
from vgdl_jax.state import GameState

# UP, DOWN, LEFT, RIGHT
DIRECTION_DELTAS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.float32)


def update_missile(state: GameState, type_idx, cooldown):
    """Move along fixed orientation each tick (if cooldown met and alive)."""
    can_move = (state.cooldown_timers[type_idx] >= cooldown) & state.alive[type_idx]
    delta = state.orientations[type_idx]  # [max_n, 2] float32
    new_pos = state.positions[type_idx] + delta * can_move[:, None]
    new_timers = jnp.where(can_move, 0, state.cooldown_timers[type_idx])
    return state.replace(
        positions=state.positions.at[type_idx].set(new_pos),
        cooldown_timers=state.cooldown_timers.at[type_idx].set(new_timers),
    )


def update_random_npc(state: GameState, type_idx, cooldown):
    """Pick a random direction each move."""
    rng, key = jax.random.split(state.rng)
    can_move = (state.cooldown_timers[type_idx] >= cooldown) & state.alive[type_idx]
    max_n = state.alive.shape[1]
    # Random direction per instance
    dir_indices = jax.random.randint(key, (max_n,), 0, 4)
    deltas = DIRECTION_DELTAS[dir_indices]  # [max_n, 2]
    new_pos = state.positions[type_idx] + deltas * can_move[:, None]
    new_timers = jnp.where(can_move, 0, state.cooldown_timers[type_idx])
    return state.replace(
        positions=state.positions.at[type_idx].set(new_pos),
        cooldown_timers=state.cooldown_timers.at[type_idx].set(new_timers),
        rng=rng,
    )


def _manhattan_distance_field(target_pos, target_alive, height, width):
    """Compute Manhattan distance to nearest alive target for every grid cell.

    Uses iterative relaxation: O(H*W*(H+W)) total work, O(H+W) depth.
    Returns [H, W] int32 distance field.
    """
    INF = jnp.int32(height + width)
    # Initialize: 0 at alive target cells, INF elsewhere
    grid = jnp.full((height, width), INF, dtype=jnp.int32)
    itarget_pos = target_pos.astype(jnp.int32)
    in_bounds = (
        (itarget_pos[:, 0] >= 0) & (itarget_pos[:, 0] < height) &
        (itarget_pos[:, 1] >= 0) & (itarget_pos[:, 1] < width)
    )
    effective = target_alive & in_bounds
    r = jnp.clip(itarget_pos[:, 0], 0, height - 1)
    c = jnp.clip(itarget_pos[:, 1], 0, width - 1)
    grid = grid.at[r, c].min(jnp.where(effective, jnp.int32(0), INF))

    def relax(_, dist):
        up = jnp.concatenate([jnp.full((1, width), INF, jnp.int32),
                              dist[:-1]], axis=0) + 1
        down = jnp.concatenate([dist[1:],
                                jnp.full((1, width), INF, jnp.int32)], axis=0) + 1
        left = jnp.concatenate([jnp.full((height, 1), INF, jnp.int32),
                                dist[:, :-1]], axis=1) + 1
        right = jnp.concatenate([dist[:, 1:],
                                 jnp.full((height, 1), INF, jnp.int32)], axis=1) + 1
        return jnp.minimum(dist, jnp.minimum(
            jnp.minimum(up, down), jnp.minimum(left, right)))

    return jax.lax.fori_loop(0, height + width, relax, grid)


def update_chaser(state: GameState, type_idx, target_type_idx, cooldown,
                  fleeing=False, height=0, width=0):
    """Move toward (or away from) nearest target using grid distance field. O(H*W + N)."""
    rng, key = jax.random.split(state.rng)
    can_move = (state.cooldown_timers[type_idx] >= cooldown) & state.alive[type_idx]

    chaser_pos = state.positions[type_idx].astype(jnp.int32)  # [max_n, 2]
    target_pos = state.positions[target_type_idx]              # [max_n, 2]
    target_alive = state.alive[target_type_idx]     # [max_n]
    any_target_alive = jnp.any(target_alive)

    # Distance field: [H, W] Manhattan distance to nearest alive target
    dist_field = _manhattan_distance_field(target_pos, target_alive, height, width)

    # For each chaser, look up distance at each neighbor direction
    r = jnp.clip(chaser_pos[:, 0], 0, height - 1)
    c = jnp.clip(chaser_pos[:, 1], 0, width - 1)
    INF = jnp.int32(height + width)
    d_up    = jnp.where(r > 0,          dist_field[jnp.clip(r - 1, 0, height - 1), c], INF)
    d_down  = jnp.where(r < height - 1, dist_field[jnp.clip(r + 1, 0, height - 1), c], INF)
    d_left  = jnp.where(c > 0,          dist_field[r, jnp.clip(c - 1, 0, width - 1)], INF)
    d_right = jnp.where(c < width - 1,  dist_field[r, jnp.clip(c + 1, 0, width - 1)], INF)

    neighbor_dists = jnp.stack([d_up, d_down, d_left, d_right], axis=-1)  # [max_n, 4]
    if fleeing:
        best_dir = jnp.argmax(neighbor_dists, axis=-1)
    else:
        best_dir = jnp.argmin(neighbor_dists, axis=-1)
    delta = DIRECTION_DELTAS[best_dir]

    # If no targets alive, pick random direction
    rand_dirs = jax.random.randint(key, (chaser_pos.shape[0],), 0, 4)
    rand_delta = DIRECTION_DELTAS[rand_dirs]
    delta = jnp.where(any_target_alive, delta, rand_delta)

    new_pos = chaser_pos + delta * can_move[:, None]
    new_timers = jnp.where(can_move, 0, state.cooldown_timers[type_idx])
    return state.replace(
        positions=state.positions.at[type_idx].set(new_pos),
        cooldown_timers=state.cooldown_timers.at[type_idx].set(new_timers),
        rng=rng,
    )


def spawn_sprite(state: GameState, spawner_type, spawner_idx, target_type,
                 orientation, speed):
    """Create a new sprite of target_type at the spawner's position."""
    pos = state.positions[spawner_type, spawner_idx]
    available = ~state.alive[target_type]
    slot = jnp.argmax(available)
    has_slot = available[slot]
    state = state.replace(
        alive=state.alive.at[target_type, slot].set(has_slot),
        positions=state.positions.at[target_type, slot].set(pos),
        orientations=state.orientations.at[target_type, slot].set(orientation),
        speeds=state.speeds.at[target_type, slot].set(speed),
        ages=state.ages.at[target_type, slot].set(0),
        cooldown_timers=state.cooldown_timers.at[target_type, slot].set(0),
    )
    return state


def update_spawn_point(state: GameState, type_idx, cooldown, prob, total,
                       target_type, target_orientation, target_speed):
    """Conditionally spawn sprites — fully vectorized via prefix-sum slot allocation."""
    rng, key = jax.random.split(state.rng)
    max_n = state.alive.shape[1]

    # Vectorized spawn decision
    is_alive = state.alive[type_idx]
    timer_ready = state.cooldown_timers[type_idx] >= cooldown
    under_total = (total <= 0) | (state.spawn_counts[type_idx] < total)
    rand_ok = jax.random.uniform(key, (max_n,)) < prob
    should_spawn = is_alive & timer_ready & under_total & rand_ok

    n_spawns = should_spawn.sum()

    # Parallel slot allocation in target type
    available = ~state.alive[target_type]
    slot_rank = jnp.cumsum(available)  # 1-indexed rank per free slot
    should_fill = available & (slot_rank <= n_spawns)

    # Map each fill-slot to its source spawner
    source_order = jnp.argsort(~should_spawn)  # spawning indices first
    src_idx = source_order[jnp.clip(slot_rank - 1, 0, max_n - 1)]
    src_pos = state.positions[type_idx][src_idx]

    state = state.replace(
        alive=state.alive.at[target_type].set(
            state.alive[target_type] | should_fill),
        positions=state.positions.at[target_type].set(
            jnp.where(should_fill[:, None], src_pos,
                      state.positions[target_type])),
        orientations=state.orientations.at[target_type].set(
            jnp.where(should_fill[:, None], target_orientation,
                      state.orientations[target_type])),
        speeds=state.speeds.at[target_type].set(
            jnp.where(should_fill, target_speed,
                      state.speeds[target_type])),
        ages=state.ages.at[target_type].set(
            jnp.where(should_fill, 0, state.ages[target_type])),
        cooldown_timers=state.cooldown_timers.at[target_type].set(
            jnp.where(should_fill, 0, state.cooldown_timers[target_type])),
        rng=rng,
    )

    # Update spawn counts — only for spawners that actually got a slot
    n_available = available.sum()
    actual_n = jnp.minimum(n_spawns, n_available)
    spawn_rank = jnp.cumsum(should_spawn)
    actually_spawned = should_spawn & (spawn_rank <= actual_n)
    new_counts = state.spawn_counts[type_idx] + actually_spawned.astype(jnp.int32)
    state = state.replace(
        spawn_counts=state.spawn_counts.at[type_idx].set(new_counts))

    # Reset cooldown timers for spawners that actually fired
    new_timers = jnp.where(actually_spawned, 0, state.cooldown_timers[type_idx])
    state = state.replace(
        cooldown_timers=state.cooldown_timers.at[type_idx].set(new_timers))

    # Kill spawners that reached total
    total_reached = (total > 0) & (new_counts >= total)
    state = state.replace(
        alive=state.alive.at[type_idx].set(
            state.alive[type_idx] & ~(total_reached & actually_spawned)))

    return state


# ── Continuous physics avatar updates ─────────────────────────────────


def update_inertial_avatar(state: GameState, action, avatar_type, n_move,
                           mass, strength, height, width):
    """InertialAvatar: ContinuousPhysics, no gravity. Input = force direction.

    velocity += (direction * strength) / mass
    position += velocity
    orientation = velocity direction (for rendering)
    """
    is_move = action < n_move
    move_idx = jnp.clip(action, 0, 3)
    force = jax.lax.cond(
        is_move,
        lambda: DIRECTION_DELTAS[move_idx] * strength,
        lambda: jnp.array([0.0, 0.0], dtype=jnp.float32))

    vel = state.velocities[avatar_type, 0]
    new_vel = vel + force / mass
    new_pos = state.positions[avatar_type, 0] + new_vel

    # Orientation from velocity (if nonzero)
    speed = jnp.sqrt(jnp.sum(new_vel ** 2))
    new_ori = jnp.where(speed > 1e-6, new_vel / speed,
                        state.orientations[avatar_type, 0])

    return state.replace(
        positions=state.positions.at[avatar_type, 0].set(new_pos),
        velocities=state.velocities.at[avatar_type, 0].set(new_vel),
        orientations=state.orientations.at[avatar_type, 0].set(new_ori),
    )


def update_mario_avatar(state: GameState, action, avatar_type,
                        mass, strength, jump_strength, gravity,
                        airsteering, height, width):
    """MarioAvatar: GravityPhysics. 6-action space.

    Actions: LEFT=0, RIGHT=1, JUMP=2, JUMP_LEFT=3, JUMP_RIGHT=4, NOOP=5.

    Coordinate system: (row, col) where +row = down.
    Gravity = (+gravity_val, 0) in row direction.
    Jump = negative row velocity (upward).
    """
    # Decode action → (horizontal_dir, wants_jump)
    # LEFT=0: h=-1, RIGHT=1: h=+1, JUMP=2: h=0, JUMP_LEFT=3: h=-1, JUMP_RIGHT=4: h=+1, NOOP=5: h=0
    h = jnp.where(
        (action == 0) | (action == 3), -1.0,
        jnp.where((action == 1) | (action == 4), 1.0, 0.0))
    wants_jump = (action == 2) | (action == 3) | (action == 4)

    vel = state.velocities[avatar_type, 0]
    pf = state.passive_forces[avatar_type, 0]

    # Grounded: passive_forces[row] == 0 means wallStop zeroed it last frame
    grounded = (pf[0] == 0.0)

    # Active force computation (5 cases)
    # Case 1: grounded + jump → (-jump_strength, h * strength)
    # Case 2: airborne + airsteering → (0, h * strength)
    # Case 3: airborne + no steering → (0, 0)
    # Case 4: grounded + horizontal → (0, h * strength)
    # Case 5: else → (0, 0)
    active_row = jnp.where(grounded & wants_jump, -jump_strength, 0.0)
    active_col = jnp.where(
        grounded | airsteering, h * strength, 0.0)
    active_force = jnp.array([active_row, active_col])

    # Friction: on ground or with airsteering, apply horizontal friction
    friction_col = jnp.where(
        grounded | airsteering, -vel[1] / mass, 0.0)
    friction_force = jnp.array([0.0, friction_col])

    # Velocity update
    new_vel = vel + (active_force + friction_force) / mass

    # Apply gravity
    new_vel = new_vel.at[0].add(gravity)

    # Position update
    new_pos = state.positions[avatar_type, 0] + new_vel

    # Reset passive_forces to gravity (wallStop will zero on landing)
    new_pf = jnp.array([gravity * mass, 0.0])

    # Orientation: face movement direction (horizontal only for Mario)
    new_ori = jnp.where(
        jnp.abs(h) > 0.0,
        jnp.array([0.0, h]),
        state.orientations[avatar_type, 0])

    return state.replace(
        positions=state.positions.at[avatar_type, 0].set(new_pos),
        velocities=state.velocities.at[avatar_type, 0].set(new_vel),
        passive_forces=state.passive_forces.at[avatar_type, 0].set(new_pf),
        orientations=state.orientations.at[avatar_type, 0].set(new_ori),
    )
