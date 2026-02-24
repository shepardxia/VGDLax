import jax
import jax.numpy as jnp
from vgdl_jax.state import create_initial_state
from vgdl_jax.sprites import (
    update_missile, update_erratic_missile, update_random_npc,
    update_random_inertial, update_spreader, update_chaser,
    update_spawn_point, update_walk_jumper, spawn_sprite, DIRECTION_DELTAS,
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


def test_update_erratic_missile():
    """ErraticMissile moves like a missile but randomly changes direction."""
    state = create_initial_state(n_types=1, max_n=3, height=10, width=10,
                                  rng_key=jax.random.PRNGKey(42))
    # Place at (5, 5) facing RIGHT, cooldown met
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([5, 5])),
        alive=state.alive.at[0, 0].set(True),
        orientations=state.orientations.at[0, 0].set(jnp.array([0., 1.])),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    state = update_erratic_missile(state, type_idx=0, cooldown=1, prob=1.0)
    # Should have moved 1 step RIGHT (from original orientation)
    assert jnp.array_equal(state.positions[0, 0], jnp.array([5, 6]))
    # With prob=1.0, orientation should have changed to a random cardinal direction
    ori = state.orientations[0, 0]
    # Check it's a valid cardinal direction
    is_cardinal = jnp.any(jnp.all(DIRECTION_DELTAS == ori, axis=-1))
    assert is_cardinal


def test_erratic_missile_prob_zero_keeps_direction():
    """With prob=0, ErraticMissile never changes direction."""
    state = create_initial_state(n_types=1, max_n=3, height=10, width=10,
                                  rng_key=jax.random.PRNGKey(0))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([5, 5])),
        alive=state.alive.at[0, 0].set(True),
        orientations=state.orientations.at[0, 0].set(jnp.array([0., 1.])),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    state = update_erratic_missile(state, type_idx=0, cooldown=1, prob=0.0)
    # Moved right
    assert jnp.array_equal(state.positions[0, 0], jnp.array([5, 6]))
    # Orientation unchanged
    assert jnp.array_equal(state.orientations[0, 0], jnp.array([0., 1.]))


def test_update_random_inertial():
    """RandomInertial should accumulate velocity from random forces."""
    state = create_initial_state(n_types=1, max_n=3, height=20, width=20,
                                  rng_key=jax.random.PRNGKey(7))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([10.0, 10.0])),
        alive=state.alive.at[0, 0].set(True),
    )
    state = update_random_inertial(state, type_idx=0, mass=1.0, strength=1.0)
    # Position should have changed (not snapped to grid)
    pos = state.positions[0, 0]
    assert not jnp.array_equal(pos, jnp.array([10.0, 10.0]))
    # Velocity should be nonzero
    vel = state.velocities[0, 0]
    assert jnp.sum(vel ** 2) > 0

    # Run a second tick — velocity should accumulate
    state = update_random_inertial(state, type_idx=0, mass=1.0, strength=1.0)
    vel2 = state.velocities[0, 0]
    # Velocity magnitude should generally change (accumulation)
    assert not jnp.array_equal(vel, vel2)


def test_random_missile_random_orientations():
    """RandomMissile instances should get randomized orientations."""
    from vgdl_jax.parser import parse_vgdl_text
    from vgdl_jax.compiler import compile_game
    game_text = """
BasicGame
    SpriteSet
        avatar > MovingAvatar
        rm > RandomMissile
        wall > Immovable
    InteractionSet
        rm wall > stepBack
    LevelMapping
        A > avatar
        m > rm
        w > wall
    TerminationSet
        Timeout limit=100 win=True
"""
    level_text = "\n".join([
        "wwwwwww",
        "wA    w",
        "w mmm w",
        "wwwwwww",
    ])
    game_def = parse_vgdl_text(game_text, level_text)
    compiled = compile_game(game_def)
    rm_type = game_def.type_idx('rm')
    # Should have 3 missiles with potentially different orientations
    oris = compiled.init_state.orientations[rm_type, :3]
    # Each should be a valid cardinal direction
    for i in range(3):
        ori = oris[i]
        is_cardinal = jnp.any(jnp.all(DIRECTION_DELTAS == ori, axis=-1))
        assert is_cardinal


def test_spreader_replicates_at_age_2():
    """Spreader at age==2 with spreadprob=1.0 spawns 4 copies in adjacent cells."""
    state = create_initial_state(n_types=1, max_n=10, height=10, width=10,
                                  rng_key=jax.random.PRNGKey(42))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([5, 5])),
        alive=state.alive.at[0, 0].set(True),
        ages=state.ages.at[0, 0].set(2),  # ready to spread
    )
    state = update_spreader(state, type_idx=0, spreadprob=1.0)
    # Should have 4 new sprites + original = 5 alive
    n_alive = state.alive[0].sum()
    assert n_alive == 5
    # New sprites at (4,5), (6,5), (5,4), (5,6) (UP, DOWN, LEFT, RIGHT)
    new_positions = set()
    for i in range(1, 5):
        if state.alive[0, i]:
            p = (int(state.positions[0, i, 0]), int(state.positions[0, i, 1]))
            new_positions.add(p)
    assert new_positions == {(4, 5), (6, 5), (5, 4), (5, 6)}


def test_spreader_no_spread_at_age_1():
    """Spreader at age==1 should not spread."""
    state = create_initial_state(n_types=1, max_n=10, height=10, width=10,
                                  rng_key=jax.random.PRNGKey(42))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([5, 5])),
        alive=state.alive.at[0, 0].set(True),
        ages=state.ages.at[0, 0].set(1),
    )
    state = update_spreader(state, type_idx=0, spreadprob=1.0)
    assert state.alive[0].sum() == 1


def test_spawn_point_cooldown_exact_match():
    """SpawnPoint should only attempt spawn at exact cooldown match, not every tick after.

    With prob=0 (always fails) and >= comparison, the spawner retries every tick
    after the cooldown is met. With == comparison and timer reset on attempt,
    it only tries once per cooldown cycle, so timer always resets.
    """
    state = create_initial_state(n_types=2, max_n=10, height=10, width=10,
                                  rng_key=jax.random.PRNGKey(42))
    # Spawner (type 0) at (5, 5), alive, prob=0.0 (never succeeds)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([5, 5])),
        alive=state.alive.at[0, 0].set(True),
    )
    target_ori = jnp.array([1., 0.], dtype=jnp.float32)
    cooldown = 3

    timers = []
    for tick in range(7):
        state = state.replace(
            cooldown_timers=jnp.where(
                state.alive, state.cooldown_timers + 1, state.cooldown_timers))
        state = update_spawn_point(
            state, type_idx=0, cooldown=cooldown, prob=0.0, total=0,
            target_type=1, target_orientation=target_ori, target_speed=1.0)
        timers.append(int(state.cooldown_timers[0, 0]))

    # With correct behavior (== check + reset on attempt):
    # tick 0: timer 0→1, not ready → timer stays 1
    # tick 1: timer 1→2, not ready → timer stays 2
    # tick 2: timer 2→3, ready! attempt (prob=0, fails), reset timer → 0
    # tick 3: timer 0→1, not ready → 1
    # tick 4: timer 1→2, not ready → 2
    # tick 5: timer 2→3, ready! attempt (prob=0, fails), reset → 0
    # tick 6: timer 0→1 → 1
    assert timers == [1, 2, 0, 1, 2, 0, 1]


def test_walk_jumper_jumps_and_walks():
    """WalkJumper with prob=0.0 (always jumps when grounded) should gain upward velocity
    and continue horizontal movement."""
    GRAVITY = 1.0 / 24.0
    STRENGTH = 10.0 / 24.0
    MASS = 1.0

    state = create_initial_state(n_types=1, max_n=2, height=20, width=20,
                                  rng_key=jax.random.PRNGKey(0))
    # WalkJumper at (10, 5), alive, facing RIGHT
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([10.0, 5.0])),
        orientations=state.orientations.at[0, 0].set(jnp.array([0.0, 1.0])),
        alive=state.alive.at[0, 0].set(True),
        # Grounded: passive_forces row = 0 (wallStop zeroed it)
        passive_forces=state.passive_forces.at[0, 0].set(jnp.array([0.0, 0.0])),
        velocities=state.velocities.at[0, 0].set(jnp.array([0.0, 0.0])),
    )

    new_state = update_walk_jumper(state, type_idx=0, prob=0.0,
                                    strength=STRENGTH, gravity=GRAVITY, mass=MASS)

    # Should have moved: upward velocity from jump + horizontal from walking
    new_vel = new_state.velocities[0, 0]
    assert new_vel[0] < 0, "Should have upward (negative row) velocity from jump"
    assert new_vel[1] > 0, "Should have rightward (positive col) velocity from walk"

    # Position should have changed
    new_pos = new_state.positions[0, 0]
    assert new_pos[0] < 10.0, "Should have moved up"
    assert new_pos[1] > 5.0, "Should have moved right"


def test_walk_jumper_no_jump_when_prob_1():
    """WalkJumper with prob=1.0 never jumps (prob < random() is always false)."""
    GRAVITY = 1.0 / 24.0
    STRENGTH = 10.0 / 24.0
    MASS = 1.0

    state = create_initial_state(n_types=1, max_n=2, height=20, width=20,
                                  rng_key=jax.random.PRNGKey(42))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([10.0, 5.0])),
        orientations=state.orientations.at[0, 0].set(jnp.array([0.0, 1.0])),
        alive=state.alive.at[0, 0].set(True),
        passive_forces=state.passive_forces.at[0, 0].set(jnp.array([0.0, 0.0])),
        velocities=state.velocities.at[0, 0].set(jnp.array([0.0, 0.0])),
    )

    new_state = update_walk_jumper(state, type_idx=0, prob=1.0,
                                    strength=STRENGTH, gravity=GRAVITY, mass=MASS)

    new_vel = new_state.velocities[0, 0]
    # No jump: vertical velocity should only be gravity (positive = downward)
    assert new_vel[0] > 0, "Should only have downward gravity velocity, no jump"
    # Still walks horizontally
    assert new_vel[1] > 0, "Should still walk rightward"
