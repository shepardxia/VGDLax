import jax.numpy as jnp
from vgdl_jax.state import GameState


def apply_kill_sprite(state: GameState, type_idx, sprite_idx, score_change=0):
    return state.replace(
        alive=state.alive.at[type_idx, sprite_idx].set(False),
        score=state.score + jnp.int32(score_change),
    )


def apply_kill_both(state, type_a, idx_a, type_b, idx_b, score_change=0):
    state = state.replace(
        alive=state.alive.at[type_a, idx_a].set(False))
    return state.replace(
        alive=state.alive.at[type_b, idx_b].set(False),
        score=state.score + jnp.int32(score_change),
    )


def apply_step_back(state, prev_positions, type_idx, sprite_idx):
    return state.replace(
        positions=state.positions.at[type_idx, sprite_idx].set(
            prev_positions[type_idx, sprite_idx]))


def apply_transform_to(state, type_idx, sprite_idx, new_type_idx):
    pos = state.positions[type_idx, sprite_idx]
    ori = state.orientations[type_idx, sprite_idx]
    # Kill old
    state = state.replace(alive=state.alive.at[type_idx, sprite_idx].set(False))
    # Find first dead slot in target type
    available = ~state.alive[new_type_idx]
    slot = jnp.argmax(available)
    has_slot = available[slot]
    state = state.replace(
        alive=state.alive.at[new_type_idx, slot].set(has_slot),
        positions=state.positions.at[new_type_idx, slot].set(pos),
        orientations=state.orientations.at[new_type_idx, slot].set(ori),
        ages=state.ages.at[new_type_idx, slot].set(0),
    )
    return state


def apply_turn_around(state, type_idx, sprite_idx):
    """Reverse orientation and step back (like py-vgdl's turnAround)."""
    ori = state.orientations[type_idx, sprite_idx]
    return state.replace(
        orientations=state.orientations.at[type_idx, sprite_idx].set(-ori),
    )


def apply_reverse_direction(state, type_idx, sprite_idx):
    """Reverse orientation only."""
    ori = state.orientations[type_idx, sprite_idx]
    return state.replace(
        orientations=state.orientations.at[type_idx, sprite_idx].set(-ori),
    )
