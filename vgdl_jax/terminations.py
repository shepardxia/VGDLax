import jax.numpy as jnp


def check_sprite_counter(state, type_indices, limit, win):
    """Game ends when total alive count of given types <= limit."""
    count = jnp.int32(0)
    for idx in type_indices:
        count = count + state.alive[idx].sum()
    ended = count <= limit
    return ended, jnp.bool_(win)


def check_multi_sprite_counter(state, type_indices_list, limit, win):
    """Game ends when sum of alive counts across multiple stype groups == limit."""
    total = jnp.int32(0)
    for indices in type_indices_list:
        for idx in indices:
            total = total + state.alive[idx].sum()
    ended = total == limit
    return ended, jnp.bool_(win)


def check_timeout(state, limit, win):
    """Game ends when step_count >= limit."""
    return state.step_count >= limit, jnp.bool_(win)


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
