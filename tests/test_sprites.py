import jax
import jax.numpy as jnp
from vgdl_jax.state import create_initial_state
from vgdl_jax.sprites import (
    update_missile, update_random_npc, update_chaser,
    spawn_sprite, DIRECTION_DELTAS,
)


def test_direction_deltas():
    assert jnp.array_equal(DIRECTION_DELTAS[0], jnp.array([-1, 0]))  # UP
    assert jnp.array_equal(DIRECTION_DELTAS[1], jnp.array([1, 0]))   # DOWN
    assert jnp.array_equal(DIRECTION_DELTAS[2], jnp.array([0, -1]))  # LEFT
    assert jnp.array_equal(DIRECTION_DELTAS[3], jnp.array([0, 1]))   # RIGHT


def test_update_missile():
    state = create_initial_state(n_types=1, max_n=3, height=10, width=10)
    # Place missile at (5, 5) going UP (dy=-1, dx=0)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([5, 5])),
        alive=state.alive.at[0, 0].set(True),
        orientations=state.orientations.at[0, 0].set(jnp.array([-1., 0.])),
        speeds=state.speeds.at[0, 0].set(1.0),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    state = update_missile(state, type_idx=0, cooldown=1)
    assert jnp.array_equal(state.positions[0, 0], jnp.array([4, 5]))


def test_update_missile_cooldown_not_met():
    state = create_initial_state(n_types=1, max_n=3, height=10, width=10)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([5, 5])),
        alive=state.alive.at[0, 0].set(True),
        orientations=state.orientations.at[0, 0].set(jnp.array([-1., 0.])),
        speeds=state.speeds.at[0, 0].set(1.0),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(0),  # not ready
    )
    state = update_missile(state, type_idx=0, cooldown=2)
    # Should NOT move — cooldown not met
    assert jnp.array_equal(state.positions[0, 0], jnp.array([5, 5]))


def test_update_missile_dead_doesnt_move():
    state = create_initial_state(n_types=1, max_n=3, height=10, width=10)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([5, 5])),
        alive=state.alive.at[0, 0].set(False),  # dead
        orientations=state.orientations.at[0, 0].set(jnp.array([-1., 0.])),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    state = update_missile(state, type_idx=0, cooldown=1)
    assert jnp.array_equal(state.positions[0, 0], jnp.array([5, 5]))


def test_update_random_npc():
    state = create_initial_state(n_types=1, max_n=2, height=10, width=10,
                                  rng_key=jax.random.PRNGKey(42))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([5, 5])),
        alive=state.alive.at[0, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    state = update_random_npc(state, type_idx=0, cooldown=1)
    pos = state.positions[0, 0]
    # Should have moved exactly 1 step in some direction
    dist = jnp.abs(pos - jnp.array([5, 5])).sum()
    assert dist == 1


def test_update_chaser_toward():
    state = create_initial_state(n_types=2, max_n=2, height=10, width=10,
                                  rng_key=jax.random.PRNGKey(0))
    # Chaser (type 0) at (2, 2), target (type 1) at (2, 5)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2, 2])),
        alive=state.alive.at[0, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2, 5])),
        alive=state.alive.at[1, 0].set(True),
    )
    state = update_chaser(state, type_idx=0, target_type_idx=1,
                          cooldown=1, fleeing=False, height=10, width=10)
    # Should move toward target: (2,2) → (2,3)
    assert jnp.array_equal(state.positions[0, 0], jnp.array([2, 3]))


def test_update_chaser_fleeing():
    state = create_initial_state(n_types=2, max_n=2, height=10, width=10,
                                  rng_key=jax.random.PRNGKey(0))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2, 2])),
        alive=state.alive.at[0, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2, 5])),
        alive=state.alive.at[1, 0].set(True),
    )
    state = update_chaser(state, type_idx=0, target_type_idx=1,
                          cooldown=1, fleeing=True, height=10, width=10)
    # Should move away from target: distance must increase
    new_pos = state.positions[0, 0]
    target_pos = jnp.array([2, 5])
    old_dist = jnp.abs(jnp.array([2, 2]) - target_pos).sum()
    new_dist = jnp.abs(new_pos - target_pos).sum()
    assert new_dist > old_dist


def test_spawn_sprite():
    state = create_initial_state(n_types=2, max_n=3, height=10, width=10)
    # Spawner at (3, 3), should create type 1 at same position
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([3, 3])),
        alive=state.alive.at[0, 0].set(True),
    )
    state = spawn_sprite(state, spawner_type=0, spawner_idx=0,
                         target_type=1, orientation=jnp.array([1., 0.]),
                         speed=1.0)
    assert state.alive[1, 0] == True
    assert jnp.array_equal(state.positions[1, 0], jnp.array([3, 3]))
    assert jnp.array_equal(state.orientations[1, 0], jnp.array([1., 0.]))
    assert state.speeds[1, 0] == 1.0


def test_spawn_sprite_fills_next_slot():
    state = create_initial_state(n_types=2, max_n=3, height=10, width=10)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([3, 3])),
        alive=state.alive.at[0, 0].set(True),
    )
    # Fill slot 0 of target type
    state = state.replace(alive=state.alive.at[1, 0].set(True))
    state = spawn_sprite(state, spawner_type=0, spawner_idx=0,
                         target_type=1, orientation=jnp.array([0., 1.]),
                         speed=1.0)
    # Should go into slot 1
    assert state.alive[1, 0] == True   # already was alive
    assert state.alive[1, 1] == True   # new sprite
    assert jnp.array_equal(state.positions[1, 1], jnp.array([3, 3]))
