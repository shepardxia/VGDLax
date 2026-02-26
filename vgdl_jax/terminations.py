import jax.numpy as jnp


def _count_sprites(state, type_indices, static_grid_indices=None):
    """Count total alive sprites across given dynamic type indices and static grid indices."""
    count = jnp.int32(0)
    for idx in type_indices:
        count = count + state.alive[idx].sum()
    if static_grid_indices:
        for sg_idx in static_grid_indices:
            count = count + state.static_grids[sg_idx].sum()
    return count


def check_sprite_counter(state, type_indices, limit, win,
                         static_grid_indices=None):
    """Game ends when total alive count of given types <= limit.

    Args:
        type_indices: list of dynamic type indices (counted from alive array)
        limit: termination threshold
        win: whether this is a win condition
        static_grid_indices: list of static grid indices (counted from static_grids)
    """
    count = _count_sprites(state, type_indices, static_grid_indices)
    ended = count <= limit
    return ended, jnp.bool_(win)


def check_multi_sprite_counter(state, type_indices_list, limit, win,
                               static_grid_indices_list=None):
    """Game ends when sum of alive counts across multiple stype groups == limit.

    Args:
        type_indices_list: list of lists of dynamic type indices
        limit: termination threshold
        win: whether this is a win condition
        static_grid_indices_list: list of lists of static grid indices
    """
    total = jnp.int32(0)
    for i, indices in enumerate(type_indices_list):
        sg_indices = None
        if static_grid_indices_list:
            sg_indices = static_grid_indices_list[i]
        total = total + _count_sprites(state, indices, sg_indices)
    ended = total == limit
    return ended, jnp.bool_(win)


def check_timeout(state, limit, win):
    """Game ends when step_count >= limit."""
    return state.step_count >= limit, jnp.bool_(win)


def check_resource_counter(state, avatar_type_idx, resource_idx, limit, win):
    """Game ends when avatar's resource count >= limit."""
    count = state.resources[avatar_type_idx, 0, resource_idx]
    ended = count >= limit
    return ended, jnp.bool_(win)


def check_all_terminations(state, compiled_terminations):
    """Check terminations in order. First matching condition wins."""
    done = jnp.bool_(False)
    win = jnp.bool_(False)
    for check_fn, term_score in compiled_terminations:
        ended, w = check_fn(state)
        first_fire = ended & ~done
        done = done | ended
        win = jnp.where(first_fire, w, win)
        state = state.replace(
            score=state.score + jnp.int32(term_score) * first_fire.astype(jnp.int32))
    return state, done, win
