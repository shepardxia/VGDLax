#!/usr/bin/env python
"""
Standalone CLI validation script for vgdl-jax vs py-vgdl cross-engine comparison.

For each of 9 games x 3 action sequences (NOOP, cycling, random), runs both engines
via the validation harness and classifies results. Writes structured output to
validation_results/ and optionally generates a LaTeX summary table.

Usage:
    python scripts/validate_all.py                  # run full validation
    python scripts/validate_all.py --latex-only     # regenerate tables from results.json
    python scripts/validate_all.py --game chase     # single game
    python scripts/validate_all.py --steps 50       # custom trajectory length
    python scripts/validate_all.py --render-diffs   # generate PNGs/GIFs on mismatch
"""

import argparse
import json
import os
import sys
import time
import traceback

from vgdl_jax.validate.constants import ALL_GAMES
from vgdl_jax.validate.harness import (
    run_comparison,
    setup_jax_game,
)

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")

TRAJECTORY_TYPES = ["noop", "cycling", "random"]

OUTPUT_DIR = os.path.join(PROJECT_DIR, "validation_results")


# ── Action sequence generators ───────────────────────────────────────────────


def _make_actions(traj_type, n_actions, n_steps, seed=42, noop_idx=None):
    """Generate an action sequence of the given type.

    Args:
        traj_type: one of 'noop', 'cycling', 'random'
        n_actions: total number of actions available
        n_steps: trajectory length
        seed: random seed for 'random' type
        noop_idx: explicit NOOP action index

    Returns:
        list[int] of action indices
    """
    if noop_idx is None:
        noop_idx = n_actions - 1
    if traj_type == "noop":
        return [noop_idx] * n_steps
    elif traj_type == "cycling":
        return [i % n_actions for i in range(n_steps)]
    elif traj_type == "random":
        import numpy as np
        rng = np.random.RandomState(seed)
        return rng.randint(0, n_actions, size=n_steps).tolist()
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")


# ── Classification ───────────────────────────────────────────────────────────


def _classify_result(result):
    """Classify a TrajectoryResult into a status string.

    Returns one of:
        'match'       - all steps match exactly
        'state_error' - at least one step diverges
    """
    if all(sc.matches for sc in result.steps):
        return "match"
    return "state_error"


# ── Per-game validation ─────────────────────────────────────────────────────


def validate_game(game_name, n_steps=30, seed=42, render_diffs=False,
                   output_dir=None):
    """Run all 3 trajectory types for a single game.

    Returns:
        dict with keys: status, trajectories, errors, timing
    """
    game_result = {
        "status": "match",
        "trajectories": {},
        "errors": [],
        "timing_s": 0.0,
    }

    t0 = time.time()

    # Get n_actions from JAX compiled game
    try:
        compiled, game_def = setup_jax_game(game_name)
        n_actions = compiled.n_actions
        noop_idx = compiled.noop_action
    except Exception as e:
        game_result["status"] = "compile_error"
        game_result["errors"].append(f"JAX compile failed: {e}")
        game_result["timing_s"] = time.time() - t0
        return game_result

    worst_status = "match"
    status_rank = {"match": 0, "state_error": 1, "compile_error": 2}

    for traj_type in TRAJECTORY_TYPES:
        traj_key = f"trajectory_{traj_type}"
        try:
            actions = _make_actions(traj_type, n_actions, n_steps, seed=seed,
                                    noop_idx=noop_idx)

            # Stochastic games use RNG replay; sokoban is deterministic
            use_rng = game_name != "sokoban"

            result = run_comparison(
                game_name, actions, seed=seed,
                use_rng_replay=use_rng,
            )

            status = _classify_result(result)

            # Collect per-step data
            step_data = []
            for sc in result.steps:
                step_data.append({
                    "step": sc.step,
                    "action": sc.action,
                    "matches": sc.matches,
                    "diffs": sc.diffs,
                })

            game_result["trajectories"][traj_type] = {
                "status": status,
                "n_steps": result.n_steps,
                "actual_steps": len(result.steps),
                "level": result.level,
                "steps": step_data,
            }

            # Render artifacts if requested (GIF always, PNGs at divergences)
            if render_diffs and output_dir:
                try:
                    render_diff_artifacts(
                        game_name, traj_type, result, actions,
                        seed=seed, output_dir=output_dir)
                except Exception as render_err:
                    game_result["errors"].append(
                        f"render {traj_type}: {render_err}")

            # Track worst status across trajectories
            if status_rank.get(status, 99) > status_rank.get(worst_status, 0):
                worst_status = status

        except Exception as e:
            tb = traceback.format_exc()
            game_result["trajectories"][traj_type] = {
                "status": "compile_error",
                "error": str(e),
            }
            game_result["errors"].append(f"{traj_type}: {e}")
            worst_status = "compile_error"

    game_result["status"] = worst_status
    game_result["timing_s"] = round(time.time() - t0, 2)
    return game_result


# ── Output writers ───────────────────────────────────────────────────────────


def _serialize_step_data(step_data):
    """Make step data JSON-serializable (convert sets, tuples, etc.)."""
    if isinstance(step_data, dict):
        return {str(k): _serialize_step_data(v) for k, v in step_data.items()}
    elif isinstance(step_data, (list, tuple)):
        return [_serialize_step_data(item) for item in step_data]
    elif isinstance(step_data, set):
        return sorted(step_data)
    else:
        return step_data


def write_results(all_results, output_dir):
    """Write structured results to output_dir/.

    Creates:
        results.json                           - aggregate stats
        per_game/{game}/trajectory_{type}.json - per-step comparison data
        per_game/{game}/errors.txt             - human-readable error log
    """
    os.makedirs(output_dir, exist_ok=True)

    # Aggregate summary
    total = len(all_results)
    matching = sum(1 for r in all_results.values() if r["status"] == "match")
    state_errors = sum(1 for r in all_results.values() if r["status"] == "state_error")
    compile_errors = sum(1 for r in all_results.values() if r["status"] == "compile_error")

    summary = {
        "total_games": total,
        "matching": matching,
        "state_errors": state_errors,
        "compile_errors": compile_errors,
    }

    # Build per_game dict for results.json (without step-level detail)
    per_game_summary = {}
    for game_name, result in all_results.items():
        per_game_summary[game_name] = {
            "status": result["status"],
            "timing_s": result["timing_s"],
            "trajectories": {
                ttype: {
                    "status": tdata.get("status", "unknown"),
                    "n_steps": tdata.get("n_steps", 0),
                    "actual_steps": tdata.get("actual_steps", 0),
                    "level": tdata.get("level", 0),
                    "failing_steps": sum(
                        1 for s in tdata.get("steps", []) if not s.get("matches", True)
                    ),
                }
                for ttype, tdata in result["trajectories"].items()
            },
        }

    output = {"summary": summary, "per_game": per_game_summary}
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Wrote {results_path}")

    # Per-game detailed files
    for game_name, result in all_results.items():
        game_dir = os.path.join(output_dir, "per_game", game_name)
        os.makedirs(game_dir, exist_ok=True)

        # Trajectory JSON files (with step-level data)
        for ttype, tdata in result["trajectories"].items():
            traj_path = os.path.join(game_dir, f"trajectory_{ttype}.json")
            with open(traj_path, "w") as f:
                json.dump(_serialize_step_data(tdata), f, indent=2, default=str)

        # Error log
        if result["errors"]:
            error_path = os.path.join(game_dir, "errors.txt")
            with open(error_path, "w") as f:
                f.write(f"Errors for {game_name}\n")
                f.write("=" * 60 + "\n\n")
                for err in result["errors"]:
                    f.write(f"- {err}\n")
            print(f"  Wrote {error_path}")

    print(f"  Wrote per-game results to {os.path.join(output_dir, 'per_game')}/")


# ── LaTeX table generation ───────────────────────────────────────────────────


_STATUS_SYMBOL = {
    "match": r"\checkmark",
    "state_error": r"$\times$",
    "compile_error": r"\textbf{ERR}",
    "unknown": "?",
}


def _traj_cell(traj_data):
    """Format a trajectory result for the LaTeX table.

    Returns a string like '\\checkmark' or '$\\times$ (3/30)' showing
    the number of failing steps.
    """
    status = traj_data.get("status", "unknown")
    symbol = _STATUS_SYMBOL.get(status, "?")
    if status in ("match", "compile_error", "unknown"):
        return symbol

    failing = traj_data.get("failing_steps", 0)
    actual = traj_data.get("actual_steps", 0)
    if failing > 0:
        return f"{symbol} ({failing}/{actual})"
    return symbol


def generate_latex(results_json_path, output_dir):
    """Generate LaTeX validation summary table from results.json.

    Output: validation_results/tables/validation_summary.tex
    """
    with open(results_json_path) as f:
        data = json.load(f)

    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    per_game = data["per_game"]
    summary = data["summary"]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-engine validation: py-vgdl vs vgdl-jax. "
                 r"\checkmark\ = match, "
                 r"$\times$ = state error. "
                 r"Parenthetical shows (failing steps / total steps).}")
    lines.append(r"\label{tab:validation-summary}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Game & Init State & NOOP & Cycling & Random & Overall \\")
    lines.append(r"\midrule")

    for game_name in ALL_GAMES:
        gdata = per_game.get(game_name)
        if gdata is None:
            lines.append(f"{game_name} & -- & -- & -- & -- & -- \\\\")
            continue

        # Init state: check step 0 of the NOOP trajectory
        noop_traj = gdata["trajectories"].get("noop", {})
        init_ok = True  # assume ok unless we find step-0 failure
        # We can infer from the noop trajectory: if level >= 1, init was ok
        init_level = noop_traj.get("level", 0)
        init_symbol = r"\checkmark" if init_level >= 1 else r"$\times$"

        # Per-trajectory cells
        noop_cell = _traj_cell(gdata["trajectories"].get("noop", {}))
        cycling_cell = _traj_cell(gdata["trajectories"].get("cycling", {}))
        random_cell = _traj_cell(gdata["trajectories"].get("random", {}))

        # Overall
        overall_symbol = _STATUS_SYMBOL.get(gdata["status"], "?")

        escaped_name = game_name.replace("_", r"\_")
        lines.append(
            f"{escaped_name} & {init_symbol} & {noop_cell} "
            f"& {cycling_cell} & {random_cell} & {overall_symbol} \\\\"
        )

    lines.append(r"\midrule")

    # Summary row
    total = summary["total_games"]
    matching = summary["matching"]
    errors = summary["state_errors"]
    lines.append(
        f"\\textbf{{Total}} & & & & & "
        f"{matching}/{total} match, {errors}/{total} diverge \\\\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = os.path.join(tables_dir, "validation_summary.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Wrote {tex_path}")
    return tex_path


# ── Console output ───────────────────────────────────────────────────────────


def print_summary(all_results):
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    status_icons = {
        "match": "[MATCH]  ",
        "state_error": "[FAIL]   ",
        "compile_error": "[ERROR]  ",
    }

    for game_name, result in all_results.items():
        icon = status_icons.get(result["status"], "[?]      ")
        timing = f"({result['timing_s']:.1f}s)"
        print(f"  {icon} {game_name:<20s} {timing}")

        for ttype, tdata in result["trajectories"].items():
            tstatus = tdata.get("status", "unknown")
            actual = tdata.get("actual_steps", "?")
            failing = sum(
                1 for s in tdata.get("steps", []) if not s.get("matches", True)
            )
            detail = f"  steps={actual}"
            if failing > 0:
                detail += f", failing={failing}"
            print(f"           {ttype:<10s} {tstatus:<16s} {detail}")

        if result["errors"]:
            for err in result["errors"]:
                print(f"           ERROR: {err}")

    # Totals
    total = len(all_results)
    matching = sum(1 for r in all_results.values() if r["status"] == "match")
    errors = sum(1 for r in all_results.values()
                 if r["status"] in ("state_error", "compile_error"))
    print()
    print(f"  Total: {total} games | Match: {matching} | Errors: {errors}")
    print("=" * 70)


# ── Visual diff rendering ───────────────────────────────────────────────────


def _render_jax_frames(game_name, actions, seed=42, block_size=24):
    """Re-run JAX trajectory and render each step to RGB.

    Returns:
        list of (step_idx, np.ndarray[H*bs, W*bs, 3] uint8) frames
    """
    import jax
    from vgdl_jax.render import render_pygame

    compiled, game_def = setup_jax_game(game_name)
    sgm = compiled.static_grid_map
    state = compiled.init_state.replace(rng=jax.random.PRNGKey(seed))
    step_fn = compiled.step_fn

    frames = []
    frames.append((0, render_pygame(state, game_def, block_size,
                                     render_sprites=False, static_grid_map=sgm)))

    for i, a in enumerate(actions):
        if bool(state.done):
            break
        state = step_fn(state, a)
        frames.append((i + 1, render_pygame(state, game_def, block_size,
                                             render_sprites=False, static_grid_map=sgm)))

    return frames


def _annotate_frame(frame, step_idx, diffs, is_divergent):
    """Add a colored border to a frame: green=match, red=divergent."""
    h, w = frame.shape[:2]
    border = 3
    annotated = frame.copy()
    color = [255, 60, 60] if is_divergent else [60, 200, 60]
    annotated[:border, :] = color
    annotated[-border:, :] = color
    annotated[:, :border] = color
    annotated[:, -border:] = color
    return annotated


def render_diff_artifacts(game_name, traj_type, comparison_result, actions,
                          seed, output_dir, block_size=24):
    """Render and save visual diff artifacts for a trajectory.

    Saves:
        per_game/{game}/step_NNN_jax.png   — at divergence points
        per_game/{game}/trajectory_{type}.gif — full animated trajectory
    """
    import imageio.v3 as iio

    game_dir = os.path.join(output_dir, "per_game", game_name)
    os.makedirs(game_dir, exist_ok=True)

    # Build lookup of divergent steps
    divergent_steps = {}
    for sc in comparison_result.steps:
        if not sc.matches:
            divergent_steps[sc.step] = sc.diffs

    # Render JAX trajectory
    frames = _render_jax_frames(game_name, actions, seed=seed,
                                block_size=block_size)

    # Save divergence PNGs and build GIF frames
    gif_frames = []
    for step_idx, frame in frames:
        is_div = step_idx in divergent_steps
        diffs = divergent_steps.get(step_idx, [])
        annotated = _annotate_frame(frame, step_idx, diffs, is_div)
        gif_frames.append(annotated)

        if is_div:
            png_path = os.path.join(
                game_dir, f"step_{step_idx:03d}_{traj_type}_jax.png")
            iio.imwrite(png_path, frame)

    # Save GIF
    if gif_frames:
        gif_path = os.path.join(game_dir, f"trajectory_{traj_type}.gif")
        iio.imwrite(gif_path, gif_frames, duration=200, loop=0)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Cross-engine validation: py-vgdl vs vgdl-jax"
    )
    parser.add_argument(
        "--game", type=str, default=None,
        help="Run validation for a single game (e.g. 'chase')",
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Number of steps per trajectory (default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--latex-only", action="store_true",
        help="Regenerate LaTeX table from existing results.json (no re-run)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--render-diffs", action="store_true",
        help="Generate PNG/GIF artifacts at divergence points",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or OUTPUT_DIR

    # ── LaTeX-only mode ──
    if args.latex_only:
        results_path = os.path.join(output_dir, "results.json")
        if not os.path.exists(results_path):
            print(f"ERROR: {results_path} not found. Run validation first.")
            sys.exit(1)
        generate_latex(results_path, output_dir)
        print("Done (LaTeX only).")
        return

    # ── Select games ──
    games = ALL_GAMES
    if args.game:
        if args.game not in ALL_GAMES:
            print(f"ERROR: Unknown game '{args.game}'. Available: {ALL_GAMES}")
            sys.exit(1)
        games = [args.game]

    # ── Run validation ──
    print(f"Running validation for {len(games)} game(s), "
          f"{args.steps} steps, seed={args.seed}")
    print(f"Trajectory types: {TRAJECTORY_TYPES}")
    print()

    all_results = {}
    for game_name in games:
        print(f"  Validating {game_name}...", end="", flush=True)
        result = validate_game(game_name, n_steps=args.steps, seed=args.seed,
                               render_diffs=args.render_diffs,
                               output_dir=output_dir)
        all_results[game_name] = result
        icon = {"match": "ok", "state_error": "FAIL", "compile_error": "ERROR"}
        print(f" {icon.get(result['status'], '?')} ({result['timing_s']:.1f}s)")

    # ── Write outputs ──
    print()
    print("Writing results...")
    write_results(all_results, output_dir)
    generate_latex(os.path.join(output_dir, "results.json"), output_dir)

    # ── Console summary ──
    print_summary(all_results)


if __name__ == "__main__":
    main()
