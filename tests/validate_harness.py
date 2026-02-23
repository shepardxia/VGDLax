"""
Validation harness: orchestrates cross-engine comparison between py-vgdl and vgdl-jax.

Supports both single-engine validation and cross-engine trajectory comparison
with optional RNG replay for stochastic games.

Validation levels (inspired by PuzzleJAX validate_sols.py):
  0: Game loads
  1: Initial state extracted correctly
  2: Single NOOP step runs
  3: N-step trajectory (10, 50 steps)
  4: Terminal state (score, done, win)
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

PYVGDL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl')
if PYVGDL_DIR not in sys.path:
    sys.path.insert(0, PYVGDL_DIR)

from conftest import GAMES_DIR
BLOCK_SIZE = 10


@dataclass
class StepComparison:
    step: int
    action: int
    state_a: dict           # normalized state from engine A (py-vgdl)
    state_b: Optional[dict] # normalized state from engine B (vgdl-jax), None until ready
    matches: bool
    diffs: List[str]


@dataclass
class TrajectoryResult:
    game_name: str
    n_steps: int
    actions: List[int]
    steps: List[StepComparison]
    level: int  # highest passing validation level (0-4)


def compare_states(state_a, state_b):
    """Field-by-field comparison of two normalized state dicts.

    Args:
        state_a: from extract_pyvgdl_state or extract_jax_state
        state_b: from extract_pyvgdl_state or extract_jax_state

    Returns:
        (matches: bool, diffs: list[str])
    """
    diffs = []

    # Score
    if state_a['score'] != state_b['score']:
        diffs.append(f"score: {state_a['score']} vs {state_b['score']}")

    # Done
    if state_a['done'] != state_b['done']:
        diffs.append(f"done: {state_a['done']} vs {state_b['done']}")

    # Per-type comparison
    all_types = set(state_a['types'].keys()) | set(state_b['types'].keys())
    for tidx in sorted(all_types):
        ta = state_a['types'].get(tidx)
        tb = state_b['types'].get(tidx)
        if ta is None:
            diffs.append(f"type {tidx}: missing from state_a")
            continue
        if tb is None:
            diffs.append(f"type {tidx}: missing from state_b")
            continue

        key = ta.get('key', tb.get('key', f'type_{tidx}'))

        if ta['alive_count'] != tb['alive_count']:
            diffs.append(
                f"{key}(t{tidx}): alive_count {ta['alive_count']} vs {tb['alive_count']}")

        pos_a = ta['positions']
        pos_b = tb['positions']
        if pos_a != pos_b:
            diffs.append(
                f"{key}(t{tidx}): positions differ "
                f"(a={pos_a[:3]}{'...' if len(pos_a) > 3 else ''} "
                f"vs b={pos_b[:3]}{'...' if len(pos_b) > 3 else ''})")

    return (len(diffs) == 0, diffs)


# ── py-vgdl trajectory runner ────────────────────────────────────────


def _setup_pyvgdl_game(game_name):
    """Set up py-vgdl game without renderer (state extraction only).

    Returns:
        (game, action_keys, sprite_key_order)
        - game: BasicGameLevel instance
        - action_keys: list of Action objects (index = action int)
        - sprite_key_order: list of sprite keys matching type_idx order
    """
    import vgdl as pyvgdl

    game_file = os.path.join(GAMES_DIR, f'{game_name}.txt')
    level_file = os.path.join(GAMES_DIR, f'{game_name}_lvl0.txt')

    with open(game_file) as f:
        game_desc = f.read()
    with open(level_file) as f:
        level_desc = f.read()

    domain = pyvgdl.VGDLParser().parse_game(game_desc)
    game = domain.build_level(level_desc)

    # Get action keys (same ordering as VGDLEnv)
    from collections import OrderedDict
    action_dict = OrderedDict(game.get_possible_actions())
    action_keys = list(action_dict.values())

    # Sprite key order from registry (matches parser registration order)
    sprite_key_order = list(game.sprite_registry.sprite_keys)

    return game, action_keys, sprite_key_order


def run_pyvgdl_trajectory(game_name, actions, seed=42, rng_replay=None):
    """Run py-vgdl for an action sequence, return state at each step.

    Args:
        game_name: e.g. 'chase', 'zelda'
        actions: list of int action indices
        seed: random seed for py-vgdl
        rng_replay: optional ReplayRandomGenerator to inject

    Returns:
        list of normalized state dicts (one per step, including initial state)
    """
    from state_extractor import extract_pyvgdl_state

    game, action_keys, sprite_key_order = _setup_pyvgdl_game(game_name)
    game.set_seed(seed)

    if rng_replay is not None:
        game.set_random_generator(rng_replay)

    # Use actual block_size from the game (default=1 when no renderer)
    block_size = game.block_size

    # Initial state
    states = [extract_pyvgdl_state(game, sprite_key_order, block_size)]

    for action_idx in actions:
        if game.ended:
            break
        action = action_keys[action_idx]
        game.tick(action)
        states.append(extract_pyvgdl_state(game, sprite_key_order, block_size))

    return states


def validate_pyvgdl_loads(game_name):
    """Validation level 0: game loads without error.

    Returns:
        (success: bool, error_msg: str or None)
    """
    try:
        game, action_keys, sprite_key_order = _setup_pyvgdl_game(game_name)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_pyvgdl_state_extraction(game_name):
    """Validation level 1: initial state is extractable and well-formed.

    Returns:
        (success: bool, state_or_error)
    """
    from state_extractor import extract_pyvgdl_state

    try:
        game, action_keys, sprite_key_order = _setup_pyvgdl_game(game_name)
        state = extract_pyvgdl_state(game, sprite_key_order, game.block_size)

        # Basic sanity checks
        assert 'types' in state
        assert 'score' in state
        assert 'done' in state
        assert len(state['types']) == len(sprite_key_order)

        for tidx, info in state['types'].items():
            assert 'alive_count' in info
            assert 'positions' in info
            assert len(info['positions']) == info['alive_count']

        return True, state
    except Exception as e:
        return False, str(e)


def validate_pyvgdl_trajectory(game_name, n_steps=50, seed=42):
    """Validation levels 2+3: run trajectory, verify states extracted at each step.

    Uses NOOP action to minimize avatar-driven complexity.

    Returns:
        (success: bool, states_or_error)
    """
    try:
        game, action_keys, _ = _setup_pyvgdl_game(game_name)
        noop_idx = len(action_keys) - 1  # NOOP is always last
        actions = [noop_idx] * n_steps

        states = run_pyvgdl_trajectory(game_name, actions, seed=seed)

        assert len(states) >= 2, f"Expected at least 2 states, got {len(states)}"

        # Verify all states are well-formed
        for i, s in enumerate(states):
            assert 'types' in s, f"Step {i}: missing 'types'"
            assert 'score' in s, f"Step {i}: missing 'score'"

        return True, states
    except Exception as e:
        return False, str(e)


# ── vgdl-jax trajectory runner ─────────────────────────────────────


def _setup_jax_game(game_name):
    """Set up vgdl-jax compiled game from the shared game files.

    Uses max_sprites_per_type large enough for ALL sprites (including inert
    background tiles like 'floor'/'grass') so counts match py-vgdl exactly.

    Returns:
        (compiled, game_def)
    """
    from collections import Counter
    from vgdl_jax.parser import parse_vgdl
    from vgdl_jax.compiler import compile_game

    game_file = os.path.join(GAMES_DIR, f'{game_name}.txt')
    level_file = os.path.join(GAMES_DIR, f'{game_name}_lvl0.txt')

    game_def = parse_vgdl(game_file, level_file)
    # Compute max sprites needed across all types (including inert background)
    counts = Counter(t for t, r, c in game_def.level.initial_sprites)
    max_n = max(counts.values(), default=1) + 10  # headroom for spawns
    compiled = compile_game(game_def, max_sprites_per_type=max_n)
    return compiled, game_def


def run_jax_trajectory(game_name, actions, seed=42):
    """Run vgdl-jax for an action sequence, return state at each step.

    Returns list of normalized state dicts (same format as run_pyvgdl_trajectory).
    """
    import jax
    from state_extractor import extract_jax_state

    compiled, game_def = _setup_jax_game(game_name)
    state = compiled.init_state.replace(rng=jax.random.PRNGKey(seed))

    # Initial state
    states = [extract_jax_state(state, game_def)]

    for action_idx in actions:
        if bool(state.done):
            break
        state = compiled.step_fn(state, action_idx)
        states.append(extract_jax_state(state, game_def))

    return states


def run_comparison(game_name, actions, seed=42, use_rng_replay=False):
    """Run both engines on same actions, compare state at every step.

    Args:
        game_name: e.g. 'chase', 'zelda'
        actions: list of int action indices
        seed: random seed
        use_rng_replay: if True, record JAX RNG sequence and inject into py-vgdl

    Returns:
        TrajectoryResult with per-step StepComparison.
    """
    import jax
    from state_extractor import extract_pyvgdl_state, extract_jax_state
    from vgdl_jax.parser import parse_vgdl
    from vgdl_jax.compiler import compile_game

    # ── Set up JAX side ──
    compiled, game_def = _setup_jax_game(game_name)
    jax_state = compiled.init_state.replace(rng=jax.random.PRNGKey(seed))

    # ── Set up py-vgdl side ──
    game, action_keys, sprite_key_order = _setup_pyvgdl_game(game_name)
    game.set_seed(seed)

    # ── Optional RNG replay ──
    rng_replay = None
    recorder = None
    if use_rng_replay:
        from rng_replay import RNGRecorder, ReplayRandomGenerator

        sprite_configs = _get_sprite_configs_from_compiled(compiled)
        effects = _get_effects_from_compiled(compiled)

        recorder = RNGRecorder(sprite_configs, effects, game_def)
        rng_replay = ReplayRandomGenerator(game_def)
        game.set_random_generator(rng_replay)

    max_n = compiled.init_state.alive.shape[1]
    rng_key = jax.random.PRNGKey(seed)
    block_size = game.block_size

    # ── Compare initial state ──
    pv_state = extract_pyvgdl_state(game, sprite_key_order, block_size)
    jx_state = extract_jax_state(jax_state, game_def)
    matches, diffs = compare_states(pv_state, jx_state)

    step_comparisons = [StepComparison(
        step=0, action=-1,
        state_a=pv_state, state_b=jx_state,
        matches=matches, diffs=diffs,
    )]

    # ── Step through actions ──
    for i, action_idx in enumerate(actions):
        pv_done = game.ended
        jx_done = bool(jax_state.done)

        if pv_done and jx_done:
            break

        # RNG replay: record this step's draws before stepping
        if recorder is not None:
            record, rng_key = recorder.record_step(rng_key, max_n=max_n)
            rng_replay.set_step_record(record)

        # Step both engines
        if not pv_done:
            game.tick(action_keys[action_idx])
        if not jx_done:
            jax_state = compiled.step_fn(jax_state, action_idx)

        pv_state = extract_pyvgdl_state(game, sprite_key_order, block_size)
        jx_state = extract_jax_state(jax_state, game_def)
        matches, diffs = compare_states(pv_state, jx_state)

        step_comparisons.append(StepComparison(
            step=i + 1, action=action_idx,
            state_a=pv_state, state_b=jx_state,
            matches=matches, diffs=diffs,
        ))

    # ── Determine validation level ──
    all_match = all(sc.matches for sc in step_comparisons)
    init_match = step_comparisons[0].matches if step_comparisons else False
    level = 0
    if init_match:
        level = 1
    if len(step_comparisons) > 1:
        level = 2
    if len(step_comparisons) > 10:
        level = 3
    if all_match:
        level = 4

    return TrajectoryResult(
        game_name=game_name,
        n_steps=len(actions),
        actions=actions,
        steps=step_comparisons,
        level=level,
    )


# ── Helpers for extracting configs from compiled game ─────────────────


def _get_sprite_configs_from_compiled(compiled):
    """Extract sprite_configs list from a CompiledGame."""
    from vgdl_jax.data_model import SpriteClass

    gd = compiled.game_def
    sprite_configs = []
    for sd in gd.sprites:
        base_cooldown = max(sd.cooldown, 1)
        if sd.speed > 0 and sd.speed != 1.0:
            effective_cooldown = max(1, round(base_cooldown / sd.speed))
        else:
            effective_cooldown = base_cooldown

        cfg = dict(
            sprite_class=sd.sprite_class,
            cooldown=effective_cooldown,
            flicker_limit=sd.flicker_limit,
        )

        if sd.sprite_class in (SpriteClass.CHASER, SpriteClass.FLEEING):
            target_key = sd.spawner_stype
            target_indices = gd.resolve_stype(target_key) if target_key else []
            cfg['target_type_idx'] = target_indices[0] if target_indices else 0

        elif sd.sprite_class in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER):
            target_key = sd.spawner_stype
            target_indices = gd.resolve_stype(target_key) if target_key else []
            cfg['target_type_idx'] = target_indices[0] if target_indices else 0
            cfg['prob'] = sd.spawner_prob
            cfg['total'] = sd.spawner_total
            if target_indices:
                target_sd = gd.sprites[target_indices[0]]
                cfg['target_orientation'] = list(target_sd.orientation)
                cfg['target_speed'] = target_sd.speed
            else:
                cfg['target_orientation'] = [0., 0.]
                cfg['target_speed'] = 0.0

        sprite_configs.append(cfg)
    return sprite_configs


def _get_effects_from_compiled(compiled):
    """Extract compiled effects list from a CompiledGame."""
    from vgdl_jax.data_model import EffectType

    gd = compiled.game_def
    effects = []
    for ed in gd.effects:
        is_eos = (ed.actee_stype == 'EOS')
        actor_indices = gd.resolve_stype(ed.actor_stype)

        if is_eos:
            for ta_idx in actor_indices:
                effects.append(dict(
                    type_a=ta_idx,
                    type_b=-1,
                    is_eos=True,
                    effect_type=_effect_type_to_str(ed.effect_type),
                    score_change=ed.score_change,
                    kwargs=dict(ed.kwargs),
                ))
        else:
            actee_indices = gd.resolve_stype(ed.actee_stype)
            for ta_idx in actor_indices:
                for tb_idx in actee_indices:
                    eff = dict(
                        type_a=ta_idx,
                        type_b=tb_idx,
                        is_eos=False,
                        effect_type=_effect_type_to_str(ed.effect_type),
                        score_change=ed.score_change,
                        kwargs=dict(ed.kwargs),
                    )
                    if ed.effect_type == EffectType.TELEPORT_TO_EXIT:
                        exit_stype = ed.kwargs.get('stype')
                        if exit_stype:
                            exit_indices = gd.resolve_stype(exit_stype)
                            eff['kwargs']['exit_type_idx'] = (
                                exit_indices[0] if exit_indices else -1)
                    effects.append(eff)
    return effects


_EFFECT_NAMES = {
    0: 'kill_sprite', 1: 'kill_both', 2: 'step_back',
    3: 'transform_to', 4: 'turn_around', 5: 'reverse_direction',
    6: 'null', 7: 'change_resource', 8: 'collect_resource',
    9: 'kill_if_has_less', 10: 'kill_if_has_more',
    11: 'kill_if_other_has_more', 12: 'kill_if_other_has_less',
    13: 'kill_if_from_above', 14: 'wrap_around', 15: 'bounce_forward',
    16: 'undo_all', 17: 'teleport_to_exit', 18: 'pull_with_it',
    19: 'wall_stop', 20: 'wall_bounce', 21: 'bounce_direction',
}


def _effect_type_to_str(et):
    return _EFFECT_NAMES.get(et, f'unknown_{et}')
