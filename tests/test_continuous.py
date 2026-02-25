"""Integration tests for ContinuousPhysics & GravityPhysics (InertialAvatar, MarioAvatar)."""
import jax
import jax.numpy as jnp
import pytest

from conftest import ALL_GAMES, GAMES_DIR
from vgdl_jax.parser import parse_vgdl_text
from vgdl_jax.compiler import compile_game


# ── Helper: compile inline game + level ──────────────────────────────────

def _compile(game_text, level_text):
    gd = parse_vgdl_text(game_text, level_text)
    return compile_game(gd)


# ── Game definitions ─────────────────────────────────────────────────────

MARIO_GAME = """\
BasicGame
    SpriteSet
        wall > Immovable
        goal > Immovable color=GREEN
        avatar > MarioAvatar
    InteractionSet
        avatar wall > wallStop
        avatar goal > killSprite scoreChange=1
        avatar EOS > killSprite
    LevelMapping
        G > goal
    TerminationSet
        SpriteCounter stype=avatar win=False
        SpriteCounter stype=goal win=True
"""

INERTIAL_GAME = """\
BasicGame
    SpriteSet
        wall > Immovable
        goal > Immovable color=GREEN
        avatar > InertialAvatar
    InteractionSet
        avatar wall > wallStop
        avatar goal > killSprite scoreChange=1
        avatar EOS > killSprite
    LevelMapping
        G > goal
    TerminationSet
        SpriteCounter stype=avatar win=False
        SpriteCounter stype=goal win=True
"""

# Simple level: avatar on floor with walls surrounding
# 7 wide x 5 tall
MARIO_LEVEL = """\
wwwwwww
w     w
w     w
wA   Gw
wwwwwww"""

# Open space for inertial avatar (no gravity)
# G placed far from avatar so we can test movement
INERTIAL_LEVEL = """\
wwwwwww
w    Gw
w  A  w
w     w
wwwwwww"""


# ── MarioAvatar Tests ────────────────────────────────────────────────────


class TestMarioGravity:
    """MarioAvatar should stay on floor when wallStop is present."""

    def test_grounded_stays_on_floor(self):
        """Avatar standing on floor with NOOP should not fall through."""
        cg = _compile(MARIO_GAME, MARIO_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')
        initial_pos = state.positions[avatar_type, 0].copy()

        # Step with NOOP (action=5) for 20 frames
        NOOP = 5
        for _ in range(20):
            state = cg.step_fn(state, NOOP)

        final_pos = state.positions[avatar_type, 0]
        # Avatar should remain at same row (floor), give or take float precision
        assert jnp.abs(final_pos[0] - initial_pos[0]) < 0.5, (
            f"Avatar fell through floor: {initial_pos[0]} -> {final_pos[0]}")

    def test_grounded_passive_forces(self):
        """When grounded via wallStop, passive_forces[row] should be 0."""
        cg = _compile(MARIO_GAME, MARIO_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')

        # Step a few frames for wallStop to fire and zero passive_forces
        NOOP = 5
        for _ in range(5):
            state = cg.step_fn(state, NOOP)

        pf = state.passive_forces[avatar_type, 0]
        # Row passive_force should be zeroed by wallStop (grounded indicator)
        assert pf[0] == 0.0, f"passive_forces[row] should be 0 when grounded, got {pf[0]}"


class TestMarioJump:
    """MarioAvatar jump trajectory: rises then falls."""

    def test_jump_rises_then_falls(self):
        """Pressing JUMP should cause avatar to rise (row decreases) then fall back."""
        cg = _compile(MARIO_GAME, MARIO_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')
        start_row = float(state.positions[avatar_type, 0, 0])

        # Jump = action 2
        JUMP = 2
        NOOP = 5

        # Apply jump
        state = cg.step_fn(state, JUMP)

        # Track min row (highest point) over the next frames
        min_row = float(state.positions[avatar_type, 0, 0])
        for _ in range(60):
            state = cg.step_fn(state, NOOP)
            row = float(state.positions[avatar_type, 0, 0])
            min_row = min(min_row, row)

        final_row = float(state.positions[avatar_type, 0, 0])

        # Should have risen above start
        assert min_row < start_row - 0.1, (
            f"Avatar didn't rise: start={start_row}, min={min_row}")
        # Should have returned near start (landed on floor)
        assert jnp.abs(final_row - start_row) < 0.5, (
            f"Avatar didn't land: start={start_row}, final={final_row}")

    def test_horizontal_movement(self):
        """LEFT and RIGHT should move avatar horizontally."""
        cg = _compile(MARIO_GAME, MARIO_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')

        # Let it settle
        NOOP = 5
        for _ in range(3):
            state = cg.step_fn(state, NOOP)

        start_col = float(state.positions[avatar_type, 0, 1])

        # Move RIGHT (action=1) several times
        RIGHT = 1
        for _ in range(10):
            state = cg.step_fn(state, RIGHT)

        end_col = float(state.positions[avatar_type, 0, 1])
        assert end_col > start_col, (
            f"Avatar didn't move right: {start_col} -> {end_col}")


class TestMarioActions:
    """MarioAvatar action space: 6 actions."""

    def test_action_count(self):
        """MarioAvatar should have 6 actions: LEFT, RIGHT, JUMP, J+L, J+R, NOOP."""
        cg = _compile(MARIO_GAME, MARIO_LEVEL)
        assert cg.n_actions == 6


# ── InertialAvatar Tests ─────────────────────────────────────────────────


class TestInertialVelocity:
    """InertialAvatar should accumulate velocity with inertia."""

    def test_velocity_accumulates(self):
        """Applying RIGHT force repeatedly should increase velocity."""
        cg = _compile(INERTIAL_GAME, INERTIAL_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')

        # RIGHT = action 3 (UP=0, DOWN=1, LEFT=2, RIGHT=3)
        RIGHT = 3
        for _ in range(5):
            state = cg.step_fn(state, RIGHT)

        vel = state.velocities[avatar_type, 0]
        # Velocity col component should be positive (moving right)
        assert vel[1] > 0.0, f"Velocity should be positive rightward, got {vel[1]}"

    def test_drift_continues_on_noop(self):
        """After applying force, NOOP should let avatar drift (no friction in open space)."""
        cg = _compile(INERTIAL_GAME, INERTIAL_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')

        RIGHT = 3
        NOOP = 4  # InertialAvatar: 4 move + NOOP = 5 actions

        # Apply right force for a few frames
        for _ in range(3):
            state = cg.step_fn(state, RIGHT)

        pos_before_noop = float(state.positions[avatar_type, 0, 1])

        # Now NOOP — should still drift
        for _ in range(3):
            state = cg.step_fn(state, NOOP)

        pos_after_noop = float(state.positions[avatar_type, 0, 1])
        assert pos_after_noop > pos_before_noop, (
            f"Avatar should drift: {pos_before_noop} -> {pos_after_noop}")

    def test_wallstop_zeroes_velocity(self):
        """Hitting a wall via wallStop should zero velocity on that axis."""
        cg = _compile(INERTIAL_GAME, INERTIAL_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')

        # Drive hard right until hitting wall
        RIGHT = 3
        for _ in range(50):
            state = cg.step_fn(state, RIGHT)

        # Velocity col component should be ~0 after wallStop
        vel_col = float(state.velocities[avatar_type, 0, 1])
        assert abs(vel_col) < 0.1, (
            f"Velocity should be ~0 after wallStop, got {vel_col}")


class TestInertialActions:
    """InertialAvatar action space: 5 actions."""

    def test_action_count(self):
        """InertialAvatar should have 5 actions: UP, DOWN, LEFT, RIGHT, NOOP."""
        cg = _compile(INERTIAL_GAME, INERTIAL_LEVEL)
        assert cg.n_actions == 5


# ── Grid regression tests ────────────────────────────────────────────────

@pytest.fixture(params=ALL_GAMES)
def grid_game_env(request):
    """Load a grid-based game and run 10 random steps to verify no crashes."""
    import os
    game_name = request.param
    game_file = os.path.join(GAMES_DIR, f'{game_name}.txt')
    level_file = os.path.join(GAMES_DIR, f'{game_name}_lvl0.txt')
    if not os.path.exists(game_file) or not os.path.exists(level_file):
        pytest.skip(f"Game files not found for {game_name}")
    return game_name, game_file, level_file


def test_grid_game_regression(grid_game_env):
    """Verify grid games still compile and run correctly with float32 positions."""
    from vgdl_jax.env import VGDLJaxEnv
    game_name, game_file, level_file = grid_game_env

    env = VGDLJaxEnv(game_file, level_file)
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)

    # Verify positions are float32
    assert state.positions.dtype == jnp.float32, (
        f"{game_name}: positions should be float32")

    # Run 10 steps without crash
    for i in range(10):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.step_count > 0
