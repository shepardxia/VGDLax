import jax
import jax.numpy as jnp
from vgdl_jax.env import VGDLJaxEnv
import os

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')


def test_aliens_runs_100_steps():
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'aliens.txt'),
        os.path.join(GAMES_DIR, 'aliens_lvl0.txt'))
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)

    for i in range(100):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.step_count > 0


def test_aliens_flak_shoots_missile():
    """FlakAvatar's shoot action should create a missile."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'aliens.txt'),
        os.path.join(GAMES_DIR, 'aliens_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    gd = env.compiled.game_def
    sam_idx = gd.type_idx('sam')

    # Initially no sam missile
    assert state.alive[sam_idx].sum() == 0

    # FlakAvatar: actions are LEFT=0, RIGHT=1, SHOOT=2, NOOP=3
    shoot_action = 2
    obs, state, _, _, _ = env.step(state, shoot_action)

    # Sam missile should be spawned
    assert state.alive[sam_idx].sum() == 1


def test_aliens_missile_moves_up():
    """Sam missile should move upward (orientation=UP)."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'aliens.txt'),
        os.path.join(GAMES_DIR, 'aliens_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    gd = env.compiled.game_def
    sam_idx = gd.type_idx('sam')

    # Shoot to create missile
    obs, state, _, _, _ = env.step(state, 2)
    assert state.alive[sam_idx].sum() == 1
    sam_pos_0 = state.positions[sam_idx, 0].copy()

    # NOOP to let missile move
    obs, state, _, _, _ = env.step(state, 3)  # NOOP

    # Missile should move up (row decreases) or be dead (hit EOS or something)
    if state.alive[sam_idx, 0]:
        sam_pos_1 = state.positions[sam_idx, 0]
        assert sam_pos_1[0] < sam_pos_0[0]  # moved up


def test_aliens_vmap():
    """Batched execution."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'aliens.txt'),
        os.path.join(GAMES_DIR, 'aliens_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 8)
    obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    assert obs_batch.shape[0] == 8

    actions = jnp.zeros(8, dtype=jnp.int32)
    obs2, states2, rewards, dones, infos = jax.vmap(env.step)(state_batch, actions)
    assert obs2.shape[0] == 8


def test_aliens_flak_moves_horizontal():
    """FlakAvatar actions 0,1 should move LEFT,RIGHT (not UP,DOWN)."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'aliens.txt'),
        os.path.join(GAMES_DIR, 'aliens_lvl0.txt'))
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    gd = env.compiled.game_def
    avatar_idx = gd.type_idx('avatar')
    initial_pos = state.positions[avatar_idx, 0].copy()

    # Action 0 = LEFT for FlakAvatar
    obs, state, _, _, _ = env.step(state, 0)
    pos_after_left = state.positions[avatar_idx, 0]
    # Row should stay same, column should decrease (or stay if at wall)
    assert pos_after_left[0] == initial_pos[0], "LEFT should not change row"

    # Action 1 = RIGHT for FlakAvatar
    obs, state, _, _, _ = env.step(state, 1)
    pos_after_right = state.positions[avatar_idx, 0]
    assert pos_after_right[0] == initial_pos[0], "RIGHT should not change row"


def test_aliens_game_ends():
    """Game should eventually end with random play."""
    env = VGDLJaxEnv(
        os.path.join(GAMES_DIR, 'aliens.txt'),
        os.path.join(GAMES_DIR, 'aliens_lvl0.txt'))
    rng = jax.random.PRNGKey(7)
    obs, state = env.reset(rng)

    for i in range(2000):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.done == True
