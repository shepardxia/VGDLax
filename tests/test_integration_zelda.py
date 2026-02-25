import jax
from conftest import make_env


def test_zelda_shoot_creates_sword():
    """ShootAvatar's shoot action should create a sword sprite."""
    env = make_env('zelda')
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    gd = env.compiled.game_def
    sword_idx = gd.type_idx('sword')

    # Initially no sword alive
    assert state.alive[sword_idx].sum() == 0

    # Shoot action: ShootAvatar has UP=0,DOWN=1,LEFT=2,RIGHT=3,NOOP=4,SHOOT=5
    shoot_action = 5
    obs, state, _, _, _ = env.step(state, shoot_action)

    # Sword should be spawned
    assert state.alive[sword_idx].sum() == 1
