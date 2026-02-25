import jax
import jax.numpy as jnp
from vgdl_jax.state import create_initial_state
from vgdl_jax.step import build_step_fn
from vgdl_jax.parser import parse_vgdl_text
from vgdl_jax.data_model import SpriteClass
from vgdl_jax.sprites import DIRECTION_DELTAS
from vgdl_jax.terminations import check_sprite_counter


def _make_avatar_config(n_types, max_n, h=5, w=5):
    """Helper to build minimal avatar_config + params for tests."""
    avatar_config = dict(
        avatar_type_idx=0, n_move_actions=4, cooldown=1,
        can_shoot=False, shoot_action_idx=-1,
        projectile_type_idx=-1,
        projectile_orientation_from_avatar=False,
        projectile_default_orientation=[0., 0.],
        projectile_speed=0.0,
    )
    params = dict(n_types=n_types, max_n=max_n, height=h, width=w)
    return avatar_config, params


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
    avatar_config, params = _make_avatar_config(n_types, max_n, h=3, w=3)

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
    avatar_config, params = _make_avatar_config(n_types, max_n)

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
    avatar_config, params = _make_avatar_config(n_types, max_n)

    step_fn = build_step_fn(effects, terminations, sprite_configs,
                            avatar_config, params)

    action = 4  # NOOP
    new_state = step_fn(state, action)
    # Missile should be dead (went OOB)
    assert new_state.alive[1, 0] == False


def test_step_avatar_no_clip():
    """Avatar at (0,0) moves UP → goes to (-1,0), not clipped to (0,0)."""
    n_types = 1
    max_n = 1
    state = create_initial_state(n_types=n_types, max_n=max_n, height=5, width=5)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([0, 0])),
        alive=state.alive.at[0, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
    ]
    avatar_config, params = _make_avatar_config(n_types, max_n)
    step_fn = build_step_fn([], [], sprite_configs, avatar_config, params)

    action = 0  # UP
    new_state = step_fn(state, action)
    # Avatar should be at (-1, 0), not clipped
    assert float(new_state.positions[0, 0, 0]) == -1.0
    assert float(new_state.positions[0, 0, 1]) == 0.0


def test_step_vertical_avatar():
    """VerticalAvatar: action=0 → UP, action=1 → DOWN."""
    n_types = 1
    max_n = 1
    state = create_initial_state(n_types=n_types, max_n=max_n, height=5, width=5)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2, 2])),
        alive=state.alive.at[0, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    sprite_configs = [
        dict(sprite_class=SpriteClass.VERTICAL_AVATAR, cooldown=1),
    ]
    avatar_config, params = _make_avatar_config(n_types, max_n)
    avatar_config['n_move_actions'] = 2
    avatar_config['direction_offset'] = 0
    step_fn = build_step_fn([], [], sprite_configs, avatar_config, params)

    # Action 0 = UP → (2,2) → (1,2)
    new_state = step_fn(state, 0)
    assert jnp.allclose(new_state.positions[0, 0], jnp.array([1, 2]))

    # Action 1 = DOWN → (1,2) → (2,2)
    new_state2 = step_fn(new_state, 1)
    assert jnp.allclose(new_state2.positions[0, 0], jnp.array([2, 2]))


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
    avatar_config, params = _make_avatar_config(n_types, max_n)

    step_fn = build_step_fn([], [], sprite_configs, avatar_config, params)
    new_state = step_fn(state, 4)  # NOOP
    assert new_state.step_count == 1
    assert jnp.array_equal(new_state.positions[0, 0], jnp.array([2, 2]))


def test_flip_direction_effect():
    """flipDirection randomizes sprite orientation on collision."""
    n_types = 2
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=5, width=5,
                                  rng_key=jax.random.PRNGKey(7))
    # Avatar (type 0) at (2,2), NPC (type 1) at (2,3) with orientation RIGHT
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2, 2])).at[1, 0].set(jnp.array([2, 3])),
        alive=state.alive.at[0, 0].set(True).at[1, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
        orientations=state.orientations.at[1, 0].set(jnp.array([0.5, 0.5])),  # non-cardinal
    )

    effects = [
        dict(type_a=1, type_b=0, is_eos=False,
             effect_type='flip_direction', score_change=0, kwargs={}),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_avatar_config(n_types, max_n)
    step_fn = build_step_fn(effects, [], sprite_configs, avatar_config, params)

    # Move avatar RIGHT to (2,3) → collides with NPC → flipDirection fires
    new_state = step_fn(state, 3)  # RIGHT
    new_ori = new_state.orientations[1, 0]
    # Orientation should be one of the 4 cardinal directions
    is_valid = False
    for i in range(4):
        if jnp.allclose(new_ori, DIRECTION_DELTAS[i]):
            is_valid = True
            break
    assert is_valid, f"Got invalid orientation: {new_ori}"


def test_kill_if_alive_kills_on_collision():
    """killIfAlive: type_a at same cell as alive type_b → type_a killed."""
    n_types = 3
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=5, width=5)
    # Avatar (type 0) at (2,2), type_a (type 1) at (2,3), type_b (type 2) at (2,3)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2, 2])).at[1, 0].set(jnp.array([2, 3])).at[2, 0].set(jnp.array([2, 3])),
        alive=state.alive.at[0, 0].set(True).at[1, 0].set(True).at[2, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )

    effects = [
        dict(type_a=1, type_b=2, is_eos=False,
             effect_type='kill_if_alive', score_change=0, kwargs={}),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_avatar_config(n_types, max_n)
    step_fn = build_step_fn(effects, [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 4)  # NOOP — collision already exists
    assert new_state.alive[1, 0] == False  # type_a killed
    assert new_state.alive[2, 0] == True   # type_b still alive


def test_kill_if_alive_spares_when_partner_dead():
    """killIfAlive: type_b dead → type_a survives."""
    n_types = 3
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=5, width=5)
    # type_a (type 1) at (2,3), type_b (type 2) at (2,3) but DEAD
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2, 2])).at[1, 0].set(jnp.array([2, 3])).at[2, 0].set(jnp.array([2, 3])),
        alive=state.alive.at[0, 0].set(True).at[1, 0].set(True).at[2, 0].set(False),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )

    effects = [
        dict(type_a=1, type_b=2, is_eos=False,
             effect_type='kill_if_alive', score_change=0, kwargs={}),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_avatar_config(n_types, max_n)
    step_fn = build_step_fn(effects, [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 4)  # NOOP
    assert new_state.alive[1, 0] == True  # type_a survives (partner dead)


def test_kill_if_slow_kills():
    """killIfSlow: type_a with speed=0.5 < limitspeed=1 → killed."""
    n_types = 3
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=5, width=5)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2, 2]))
            .at[1, 0].set(jnp.array([2, 3]))
            .at[2, 0].set(jnp.array([2, 3])),
        alive=state.alive.at[0, 0].set(True).at[1, 0].set(True).at[2, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
        speeds=state.speeds.at[1, 0].set(0.5),
    )
    effects = [
        dict(type_a=1, type_b=2, is_eos=False,
             effect_type='kill_if_slow', score_change=0,
             kwargs={'limitspeed': 1.0}),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_avatar_config(n_types, max_n)
    step_fn = build_step_fn(effects, [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 4)  # NOOP
    assert new_state.alive[1, 0] == False  # killed (speed < limitspeed)


def test_kill_if_slow_survives():
    """killIfSlow: type_a with speed=2 >= limitspeed=1 → survives."""
    n_types = 3
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=5, width=5)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2, 2]))
            .at[1, 0].set(jnp.array([2, 3]))
            .at[2, 0].set(jnp.array([2, 3])),
        alive=state.alive.at[0, 0].set(True).at[1, 0].set(True).at[2, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
        speeds=state.speeds.at[1, 0].set(2.0),
    )
    effects = [
        dict(type_a=1, type_b=2, is_eos=False,
             effect_type='kill_if_slow', score_change=0,
             kwargs={'limitspeed': 1.0}),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_avatar_config(n_types, max_n)
    step_fn = build_step_fn(effects, [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 4)
    assert new_state.alive[1, 0] == True  # survives (speed >= limitspeed)


def test_convey_sprite():
    """conveySprite: type_a at (3,3) on conveyor facing RIGHT with strength=1 → moves to (3,4)."""
    n_types = 3
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=7, width=7)
    # Avatar at (0,0), type_a (1) at (3,3), conveyor (2) at (3,3) facing RIGHT
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([0, 0]))
            .at[1, 0].set(jnp.array([3, 3]))
            .at[2, 0].set(jnp.array([3, 3])),
        alive=state.alive.at[0, 0].set(True).at[1, 0].set(True).at[2, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
        orientations=state.orientations.at[2, 0].set(jnp.array([0., 1.])),  # RIGHT
    )
    effects = [
        dict(type_a=1, type_b=2, is_eos=False,
             effect_type='convey_sprite', score_change=0,
             kwargs={'strength': 1.0}),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_avatar_config(n_types, max_n, h=7, w=7)
    step_fn = build_step_fn(effects, [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 4)  # NOOP
    assert jnp.allclose(new_state.positions[1, 0], jnp.array([3, 4]))


def test_parse_conveyor():
    """Parser resolves Conveyor to CONVEYOR class with correct orientation/strength."""
    game_text = """\
BasicGame
    SpriteSet
        avatar > MovingAvatar
        belt   > Conveyor orientation=RIGHT strength=2
    InteractionSet
        avatar belt > conveySprite
    TerminationSet
        Timeout limit=100 win=False
"""
    gd = parse_vgdl_text(game_text)
    belt = next(s for s in gd.sprites if s.key == 'belt')
    assert belt.sprite_class == SpriteClass.CONVEYOR
    assert belt.orientation == (0.0, 1.0)  # RIGHT
    assert belt.strength == 2.0
    assert belt.is_static == True


def test_clone_sprite():
    """cloneSprite: type_a at (3,3) collides with type_b → creates a clone of type_a."""
    n_types = 3
    max_n = 5
    state = create_initial_state(n_types=n_types, max_n=max_n, height=7, width=7)
    # Avatar (0) at (0,0), type_a (1) at (3,3), type_b (2) at (3,3)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([0, 0]))
            .at[1, 0].set(jnp.array([3, 3]))
            .at[2, 0].set(jnp.array([3, 3])),
        alive=state.alive.at[0, 0].set(True).at[1, 0].set(True).at[2, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
        orientations=state.orientations.at[1, 0].set(jnp.array([0., 1.])),
    )
    effects = [
        dict(type_a=1, type_b=2, is_eos=False,
             effect_type='clone_sprite', score_change=0, kwargs={}),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_avatar_config(n_types, max_n, h=7, w=7)
    step_fn = build_step_fn(effects, [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 4)  # NOOP
    # Original stays alive
    assert new_state.alive[1, 0] == True
    # Clone created in next slot
    assert new_state.alive[1, 1] == True
    assert jnp.allclose(new_state.positions[1, 1], jnp.array([3, 3]))
    assert jnp.allclose(new_state.orientations[1, 1], jnp.array([0., 1.]))


def test_spawn_if_has_more():
    """spawnIfHasMore: avatar has resource >= limit → spawns target sprite."""
    n_types = 3
    max_n = 5
    state = create_initial_state(n_types=n_types, max_n=max_n, height=7, width=7,
                                  n_resource_types=1)
    # Avatar (0) at (0,0), type_a (1) at (3,3) with resource[0]=3, trigger (2) at (3,3)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([0, 0]))
            .at[1, 0].set(jnp.array([3, 3]))
            .at[2, 0].set(jnp.array([3, 3])),
        alive=state.alive.at[0, 0].set(True).at[1, 0].set(True).at[2, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
        resources=state.resources.at[1, 0, 0].set(5),
    )
    # Effect: when type_a (1) collides with type_b (2), if resource[0] >= 3,
    # spawn a new type_a (1) at that position
    effects = [
        dict(type_a=1, type_b=2, is_eos=False,
             effect_type='spawn_if_has_more', score_change=0,
             kwargs={'resource_idx': 0, 'limit': 3, 'spawn_type_idx': 1}),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_avatar_config(n_types, max_n, h=7, w=7)
    params['n_resource_types'] = 1
    step_fn = build_step_fn(effects, [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 4)  # NOOP
    # Original stays alive
    assert new_state.alive[1, 0] == True
    # New sprite spawned in slot 1
    assert new_state.alive[1, 1] == True
    assert jnp.allclose(new_state.positions[1, 1], jnp.array([3, 3]))


def _make_rotating_avatar_config(n_types, max_n, is_flipping=False, noise_level=0.0,
                                  h=7, w=7):
    avatar_config = dict(
        avatar_type_idx=0, n_move_actions=4, cooldown=1,
        can_shoot=False, shoot_action_idx=-1,
        projectile_type_idx=-1,
        projectile_orientation_from_avatar=False,
        projectile_default_orientation=[0., 0.],
        projectile_speed=0.0,
        direction_offset=0,
        is_rotating=True,
        is_flipping=is_flipping,
        noise_level=noise_level,
    )
    params = dict(n_types=n_types, max_n=max_n, height=h, width=w)
    return avatar_config, params


def test_rotating_avatar_forward():
    """RotatingAvatar facing UP: action=0 (forward) moves to (row-1, col)."""
    n_types = 2
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=7, width=7)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([3, 3])),
        alive=state.alive.at[0, 0].set(True),
        orientations=state.orientations.at[0, 0].set(jnp.array([-1., 0.])),  # UP
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    sprite_configs = [
        dict(sprite_class=SpriteClass.ROTATING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_rotating_avatar_config(n_types, max_n)
    step_fn = build_step_fn([], [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 0)  # forward
    assert jnp.allclose(new_state.positions[0, 0], jnp.array([2, 3]))
    assert jnp.allclose(new_state.orientations[0, 0], jnp.array([-1., 0.]))


def test_rotating_avatar_rotate_ccw():
    """RotatingAvatar facing UP: action=2 (CCW) → now faces LEFT."""
    n_types = 2
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=7, width=7)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([3, 3])),
        alive=state.alive.at[0, 0].set(True),
        orientations=state.orientations.at[0, 0].set(jnp.array([-1., 0.])),  # UP
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    sprite_configs = [
        dict(sprite_class=SpriteClass.ROTATING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_rotating_avatar_config(n_types, max_n)
    step_fn = build_step_fn([], [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 2)  # CCW
    assert jnp.allclose(new_state.orientations[0, 0], jnp.array([0., -1.]))
    assert jnp.allclose(new_state.positions[0, 0], jnp.array([3, 3]))


def test_rotating_avatar_rotate_cw():
    """RotatingAvatar facing UP: action=3 (CW) → now faces RIGHT."""
    n_types = 2
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=7, width=7)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([3, 3])),
        alive=state.alive.at[0, 0].set(True),
        orientations=state.orientations.at[0, 0].set(jnp.array([-1., 0.])),  # UP
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    sprite_configs = [
        dict(sprite_class=SpriteClass.ROTATING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_rotating_avatar_config(n_types, max_n)
    step_fn = build_step_fn([], [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 3)  # CW
    assert jnp.allclose(new_state.orientations[0, 0], jnp.array([0., 1.]))


def test_rotating_flipping_avatar_flip():
    """RotatingFlippingAvatar facing UP: action=1 → flips to DOWN (no movement)."""
    n_types = 2
    max_n = 3
    state = create_initial_state(n_types=n_types, max_n=max_n, height=7, width=7)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([3, 3])),
        alive=state.alive.at[0, 0].set(True),
        orientations=state.orientations.at[0, 0].set(jnp.array([-1., 0.])),  # UP
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    sprite_configs = [
        dict(sprite_class=SpriteClass.ROTATING_FLIPPING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_rotating_avatar_config(n_types, max_n, is_flipping=True)
    step_fn = build_step_fn([], [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 1)  # flip
    assert jnp.allclose(new_state.orientations[0, 0], jnp.array([1., 0.]))
    assert jnp.allclose(new_state.positions[0, 0], jnp.array([3, 3]))


def test_shoot_everywhere_avatar():
    """ShootEverywhereAvatar fires 4 projectiles in all directions."""
    n_types = 3
    max_n = 10
    state = create_initial_state(n_types=n_types, max_n=max_n, height=7, width=7)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([3, 3])),
        alive=state.alive.at[0, 0].set(True),
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
        orientations=state.orientations.at[0, 0].set(jnp.array([-1., 0.])),
    )
    sprite_configs = [
        dict(sprite_class=SpriteClass.SHOOT_EVERYWHERE_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.MISSILE, cooldown=1, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config = dict(
        avatar_type_idx=0, n_move_actions=4, cooldown=1,
        can_shoot=True, shoot_action_idx=4,
        projectile_type_idx=1,
        projectile_orientation_from_avatar=False,
        projectile_default_orientation=[-1., 0.],
        projectile_speed=1.0,
        direction_offset=0,
        shoot_everywhere=True,
    )
    params = dict(n_types=n_types, max_n=max_n, height=7, width=7)
    step_fn = build_step_fn([], [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 4)  # shoot action
    # Should have 4 alive projectiles
    n_proj_alive = new_state.alive[1].sum()
    assert n_proj_alive == 4
    # All at avatar position (3,3)
    for i in range(4):
        assert jnp.allclose(new_state.positions[1, i], jnp.array([3, 3]))


def test_aimed_avatar_rotation():
    """AimedAvatar facing RIGHT: AIM_UP (action=0) rotates orientation CCW."""
    n_types = 3
    max_n = 5
    state = create_initial_state(n_types=n_types, max_n=max_n, height=7, width=7)
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([3, 3])),
        alive=state.alive.at[0, 0].set(True),
        orientations=state.orientations.at[0, 0].set(jnp.array([0., 1.])),  # RIGHT
        cooldown_timers=state.cooldown_timers.at[0, 0].set(1),
    )
    angle_diff = 0.1
    sprite_configs = [
        dict(sprite_class=SpriteClass.AIMED_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.MISSILE, cooldown=1, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config = dict(
        avatar_type_idx=0, n_move_actions=2, cooldown=1,
        can_shoot=True, shoot_action_idx=2,
        projectile_type_idx=1,
        projectile_orientation_from_avatar=True,
        projectile_default_orientation=[0., 1.],
        projectile_speed=1.0,
        direction_offset=0,
        is_aimed=True, can_move_aimed=False,
        angle_diff=angle_diff,
    )
    params = dict(n_types=n_types, max_n=max_n, height=7, width=7)
    step_fn = build_step_fn([], [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 0)  # AIM_UP → rotate CCW by -angle_diff
    new_ori = new_state.orientations[0, 0]
    import math
    expected_r = math.cos(-angle_diff) * 0 - math.sin(-angle_diff) * 1
    expected_c = math.sin(-angle_diff) * 0 + math.cos(-angle_diff) * 1
    assert jnp.allclose(new_ori, jnp.array([expected_r, expected_c]), atol=1e-5)
    # Position unchanged (AimedAvatar doesn't move)
    assert jnp.allclose(new_state.positions[0, 0], jnp.array([3, 3]))


# ── Niche effects tests ──────────────────────────────────────────────


def _make_step_fn_for_effect(effect_type, kwargs, n_types=2, max_n=3,
                              n_resource_types=1, height=5, width=5):
    """Helper: build step_fn with a single effect between type_a=0 and type_b=1."""
    effects = [
        dict(type_a=0, type_b=1, is_eos=False,
             effect_type=effect_type, score_change=0, kwargs=kwargs,
             use_aabb=False),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
    ] + [
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0)
        for _ in range(n_types - 1)
    ]
    avatar_config, params = _make_avatar_config(n_types, max_n, h=height, w=width)
    params['n_resource_types'] = max(n_resource_types, 1)
    return build_step_fn(effects, [], sprite_configs, avatar_config, params)


def test_slip_forward():
    """slipForward with prob=1.0 should move type_a forward by its orientation."""
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5,
                                  rng_key=jax.random.PRNGKey(0))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2.0, 2.0])),
        orientations=state.orientations.at[0, 0].set(jnp.array([0.0, 1.0])),
        alive=state.alive.at[0, 0].set(True),
    )
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[1, 0].set(True),
    )

    step_fn = _make_step_fn_for_effect('slip_forward', {'prob': 1.0})
    new_state = step_fn(state, 4)  # NOOP
    # Avatar should have slipped forward (right) by 1 step
    assert new_state.positions[0, 0, 1] > 2.0


def test_attract_gaze():
    """attractGaze with prob=1.0 should set type_a orientation to type_b's."""
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5,
                                  rng_key=jax.random.PRNGKey(0))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2.0, 2.0])),
        orientations=state.orientations.at[0, 0].set(jnp.array([0.0, 1.0])),
        alive=state.alive.at[0, 0].set(True),
    )
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2.0, 2.0])),
        orientations=state.orientations.at[1, 0].set(jnp.array([-1.0, 0.0])),
        alive=state.alive.at[1, 0].set(True),
    )

    step_fn = _make_step_fn_for_effect('attract_gaze', {'prob': 1.0})
    new_state = step_fn(state, 4)  # NOOP
    # Avatar should now face UP (type_b's orientation)
    assert jnp.allclose(new_state.orientations[0, 0], jnp.array([-1.0, 0.0]))


def test_spend_resource():
    """SpendResource deducts resource from type_a on collision."""
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5,
                                  n_resource_types=1,
                                  rng_key=jax.random.PRNGKey(0))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[0, 0].set(True),
        resources=state.resources.at[0, 0, 0].set(5),
    )
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[1, 0].set(True),
    )

    step_fn = _make_step_fn_for_effect('spend_resource',
                                        {'resource_idx': 0, 'amount': 2},
                                        n_resource_types=1)
    new_state = step_fn(state, 4)
    assert new_state.resources[0, 0, 0] == 3  # 5 - 2


def test_spend_avatar_resource():
    """SpendAvatarResource deducts from avatar regardless of collision actors."""
    state = create_initial_state(n_types=3, max_n=3, height=5, width=5,
                                  n_resource_types=1,
                                  rng_key=jax.random.PRNGKey(0))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([0.0, 0.0])),
        alive=state.alive.at[0, 0].set(True),
        resources=state.resources.at[0, 0, 0].set(10),
    )
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[1, 0].set(True),
    )
    state = state.replace(
        positions=state.positions.at[2, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[2, 0].set(True),
    )

    effects = [
        dict(type_a=1, type_b=2, is_eos=False,
             effect_type='spend_avatar_resource', score_change=0,
             kwargs={'avatar_type_idx': 0, 'resource_idx': 0, 'amount': 3},
             use_aabb=False),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_avatar_config(3, 3)
    params['n_resource_types'] = 1
    step_fn = build_step_fn(effects, [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 4)
    assert new_state.resources[0, 0, 0] == 7  # 10 - 3


def test_kill_others():
    """KillOthers kills all sprites of the target type on collision."""
    state = create_initial_state(n_types=3, max_n=3, height=5, width=5,
                                  rng_key=jax.random.PRNGKey(0))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[0, 0].set(True),
    )
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[1, 0].set(True),
    )
    state = state.replace(
        positions=state.positions.at[2, 0].set(jnp.array([0.0, 0.0])),
        alive=state.alive.at[2, 0].set(True),
    )
    state = state.replace(
        positions=state.positions.at[2, 1].set(jnp.array([4.0, 4.0])),
        alive=state.alive.at[2, 1].set(True),
    )

    effects = [
        dict(type_a=0, type_b=1, is_eos=False,
             effect_type='kill_others', score_change=0,
             kwargs={'kill_type_idx': 2}, use_aabb=False),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_avatar_config(3, 3)
    step_fn = build_step_fn(effects, [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 4)
    assert not new_state.alive[2, 0]
    assert not new_state.alive[2, 1]


def test_kill_if_avatar_without_resource():
    """KillIfAvatarWithoutResource kills type_a if avatar has no resource."""
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5,
                                  n_resource_types=1,
                                  rng_key=jax.random.PRNGKey(0))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[0, 0].set(True),
        resources=state.resources.at[0, 0, 0].set(0),
    )
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[1, 0].set(True),
    )

    step_fn = _make_step_fn_for_effect('kill_if_avatar_without_resource',
                                        {'avatar_type_idx': 0, 'resource_idx': 0},
                                        n_resource_types=1)
    new_state = step_fn(state, 4)
    assert not new_state.alive[0, 0]


def test_kill_if_avatar_with_resource_survives():
    """KillIfAvatarWithoutResource spares type_a if avatar HAS the resource."""
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5,
                                  n_resource_types=1,
                                  rng_key=jax.random.PRNGKey(0))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[0, 0].set(True),
        resources=state.resources.at[0, 0, 0].set(5),
    )
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[1, 0].set(True),
    )

    step_fn = _make_step_fn_for_effect('kill_if_avatar_without_resource',
                                        {'avatar_type_idx': 0, 'resource_idx': 0},
                                        n_resource_types=1)
    new_state = step_fn(state, 4)
    assert new_state.alive[0, 0]


def test_avatar_collect_resource():
    """AvatarCollectResource adds resource to avatar."""
    state = create_initial_state(n_types=2, max_n=3, height=5, width=5,
                                  n_resource_types=1,
                                  rng_key=jax.random.PRNGKey(0))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[0, 0].set(True),
        resources=state.resources.at[0, 0, 0].set(2),
    )
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[1, 0].set(True),
    )

    step_fn = _make_step_fn_for_effect('avatar_collect_resource',
                                        {'avatar_type_idx': 0, 'resource_idx': 0,
                                         'resource_value': 3, 'limit': 100},
                                        n_resource_types=1)
    new_state = step_fn(state, 4)
    assert new_state.resources[0, 0, 0] == 5  # 2 + 3


def test_transform_others_to():
    """TransformOthersTo transforms all sprites of target type into new type."""
    state = create_initial_state(n_types=4, max_n=3, height=5, width=5,
                                  rng_key=jax.random.PRNGKey(0))
    state = state.replace(
        positions=state.positions.at[0, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[0, 0].set(True),
    )
    state = state.replace(
        positions=state.positions.at[1, 0].set(jnp.array([2.0, 2.0])),
        alive=state.alive.at[1, 0].set(True),
    )
    state = state.replace(
        positions=state.positions.at[2, 0].set(jnp.array([0.0, 0.0])),
        alive=state.alive.at[2, 0].set(True),
    )
    state = state.replace(
        positions=state.positions.at[2, 1].set(jnp.array([4.0, 4.0])),
        alive=state.alive.at[2, 1].set(True),
    )

    effects = [
        dict(type_a=0, type_b=1, is_eos=False,
             effect_type='transform_others_to', score_change=0,
             kwargs={'target_type_idx': 2, 'new_type_idx': 3}, use_aabb=False),
    ]
    sprite_configs = [
        dict(sprite_class=SpriteClass.MOVING_AVATAR, cooldown=1),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
        dict(sprite_class=SpriteClass.IMMOVABLE, cooldown=0, flicker_limit=0),
    ]
    avatar_config, params = _make_avatar_config(4, 3)
    step_fn = build_step_fn(effects, [], sprite_configs, avatar_config, params)

    new_state = step_fn(state, 4)
    assert not new_state.alive[2, 0]
    assert not new_state.alive[2, 1]
    assert new_state.alive[3, 0]
    assert new_state.alive[3, 1]
