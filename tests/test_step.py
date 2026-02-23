import jax
import jax.numpy as jnp
from vgdl_jax.state import create_initial_state
from vgdl_jax.step import build_step_fn
from vgdl_jax.data_model import SpriteClass
from vgdl_jax.terminations import check_sprite_counter


def test_step_avatar_wall_stepback():
    """Avatar at (1,0) moves RIGHT into wall at (1,1). stepBack reverts."""
    n_types = 2
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=3, width=3)
    # Type 0 = avatar at (1, 0)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([1, 0])),
        alive=state.alive.at[0, 0].set(True),
        speeds=state.speeds.at[0, 0].set(1.0),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    # Type 1 = wall at (1, 1)
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([1, 1])),
        alive=state.alive.at[1, 0].set(True),
    )

    effects = [
        dict(type_a=0, type_b=1, is_eos=False,
             effect_type='step_back', score_change=0, kwargs={}),
    ]
    terminations = []
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config = dict(
        avatar_type_idx=0, n_move_actions=4, cooldown=1,
        can_shoot=False, shoot_action_idx=-1,
        projectile_type_idx=-1,
        projectile_orientation_from_avatar=False,
        projectile_default_orientation=[0., 0.],
        projectile_speed=0.0,
    )
    params = dict(n_types=n_types, max_n=max_n, height=3, width=3)

    step_fn = build_step_fn(effects, terminations, sprite_configs,
                            avatar_config, params)

    action = 3  # RIGHT
    new_state = step_fn(state, action)
    # Avatar should be back at (1, 0) due to stepBack
    assert jnp.array_equal(new_state.positions[0, 0], jnp.array([1, 0]))
    assert new_state.step_count == 1


def test_step_kill_on_collision():
    """Avatar walks into enemy → avatar dies."""
    n_types = 2
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=5, width=5)
    # Avatar at (2, 2)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2, 2])),
        alive=state.alive.at[0, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    # Enemy at (2, 3)
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2, 3])),
        alive=state.alive.at[1, 0].set(True),
    )

    effects = [
        dict(type_a=0, type_b=1, is_eos=False,
             effect_type='kill_sprite', score_change=-1, kwargs={}),
    ]
    terminations = [
        (lambda s: check_sprite_counter(s, [0], 0, False), 0),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config = dict(
        avatar_type_idx=0, n_move_actions=4, cooldown=1,
        can_shoot=False, shoot_action_idx=-1,
        projectile_type_idx=-1,
        projectile_orientation_from_avatar=False,
        projectile_default_orientation=[0., 0.],
        projectile_speed=0.0,
    )
    params = dict(n_types=n_types, max_n=max_n, height=5, width=5)

    step_fn = build_step_fn(effects, terminations, sprite_configs,
                            avatar_config, params)

    action = 3  # RIGHT → avatar moves to (2, 3) → collides with enemy
    new_state = step_fn(state, action)
    assert new_state.alive[0, 0] == False  # avatar killed
    assert new_state.score == -1
    assert new_state.done == True
    assert new_state.win == False


def test_step_eos_kills_missile():
    """Missile goes off screen → killed by EOS effect."""
    n_types = 2
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=5, width=5)
    # Avatar at (4, 2)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([4, 2])),
        alive=state.alive.at[0, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    # Missile at (0, 2) going UP → will go to (-1, 2) → OOB
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([0, 2])),
        alive=state.alive.at[1, 0].set(True),
        orientations=state.orientations.at[1, 0].set(jnp.array([-1., 0.])),
        cooldown_timers=state.cooldown_timers.at[1, 0].set(1),
    )

    effects = [
        dict(type_a=1, is_eos=True,
             effect_type='kill_sprite', score_change=0, kwargs={}),
    ]
    terminations = []
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.MISSILE, cooldown=1, flicker_limit=0),
    ]
    avatar_config = dict(
        avatar_type_idx=0, n_move_actions=4, cooldown=1,
        can_shoot=False, shoot_action_idx=-1,
        projectile_type_idx=-1,
        projectile_orientation_from_avatar=False,
        projectile_default_orientation=[0., 0.],
        projectile_speed=0.0,
    )
    params = dict(n_types=n_types, max_n=max_n, height=5, width=5)

    step_fn = build_step_fn(effects, terminations, sprite_configs,
                            avatar_config, params)

    action = 4  # NOOP
    new_state = step_fn(state, action)
    # Missile should be dead (went OOB)
    assert new_state.alive[1, 0] == False


def test_step_noop_increments_step():
    """NOOP action still increments step count."""
    n_types = 1
    max_n = 1
    state = create_initial_state(n_types=n_types, max_n=max_n, height=5, width=5)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2, 2])),
        alive=state.alive.at[0, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
    ]
    avatar_config = dict(
        avatar_type_idx=0, n_move_actions=4, cooldown=1,
        can_shoot=False, shoot_action_idx=-1,
        projectile_type_idx=-1,
        projectile_orientation_from_avatar=False,
        projectile_default_orientation=[0., 0.],
        projectile_speed=0.0,
    )
    params = dict(n_types=n_types, max_n=max_n, height=5, width=5)

    step_fn = build_step_fn([], [], sprite_configs, avatar_config, params)
    new_state = step_fn(state, 4)  # NOOP
    assert new_state.step_count == 1
    assert jnp.array_equal(new_state.positions[0, 0], jnp.array([2, 2]))
