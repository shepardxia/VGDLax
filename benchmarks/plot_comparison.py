"""
Plot FPS comparison: vgdl-jax (varying batch sizes) vs py-vgdl (CPU baseline).
Mirrors PuzzleJAX's plot_rand_profile.py: grid of subplots, one per game.

Usage:
    python benchmarks/plot_comparison.py
    python benchmarks/plot_comparison.py --log-scale
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np

BENCH_DIR = os.path.dirname(__file__)
JAX_RESULTS_DIR = os.path.join(BENCH_DIR, 'results', 'jax')
PYVGDL_RESULTS_DIR = os.path.join(BENCH_DIR, 'results', 'pyvgdl')
PLOTS_DIR = os.path.join(BENCH_DIR, 'results', 'plots')

GAME_ORDER = [
    "Chase", "Zelda", "Aliens", "MissileCommand", "Sokoban",
    "Portals", "BoulderDash", "SurviveZombies", "Frogs",
]


def load_jax_results():
    """Load JAX profiling results. Returns {device: {game: {batch_size: fps}}}."""
    results = {}
    if not os.path.exists(JAX_RESULTS_DIR):
        return results
    for device in os.listdir(JAX_RESULTS_DIR):
        device_dir = os.path.join(JAX_RESULTS_DIR, device)
        if not os.path.isdir(device_dir):
            continue
        results[device] = {}
        for fname in os.listdir(device_dir):
            if not fname.endswith('.json'):
                continue
            game = fname[:-5]
            with open(os.path.join(device_dir, fname), 'r') as f:
                data = json.load(f)
            # data: {batch_size_str: [fps_loop0, fps_loop1, fps_loop2] or {error}}
            results[device][game] = {}
            for batch_str, val in data.items():
                if isinstance(val, list):
                    # Use the last measurement loop (most stable after warmup)
                    results[device][game][int(batch_str)] = val[-1]
        # else error dict, skip
    return results


def load_pyvgdl_results():
    """Load py-vgdl baseline results. Returns {game: fps}."""
    results = {}
    if not os.path.exists(PYVGDL_RESULTS_DIR):
        return results
    for fname in os.listdir(PYVGDL_RESULTS_DIR):
        if not fname.endswith('.json'):
            continue
        game = fname[:-5]
        with open(os.path.join(PYVGDL_RESULTS_DIR, fname), 'r') as f:
            data = json.load(f)
        results[game] = data.get('fps_last', data.get('fps_mean', 0))
    return results


def plot(log_scale=False):
    jax_results = load_jax_results()
    pyvgdl_results = load_pyvgdl_results()

    if not jax_results:
        print("No JAX results found. Run profile_jax.py first.")
        return
    if not pyvgdl_results:
        print("No py-vgdl results found. Run profile_pyvgdl.py first.")
        print("(Will plot JAX-only results)")

    # Use first device found
    device = list(jax_results.keys())[0]
    jax_data = jax_results[device]

    # Determine which games have results
    games = [g for g in GAME_ORDER if g in jax_data]
    if not games:
        print("No games with results found.")
        return

    n_games = len(games)
    n_cols = min(3, n_games)
    n_rows = (n_games + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    if n_games == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, game in enumerate(games):
        ax = axes[i // n_cols, i % n_cols]

        # JAX line: FPS vs batch size
        batch_fps = jax_data[game]
        batch_sizes = sorted(batch_fps.keys())
        fps_vals = [batch_fps[b] for b in batch_sizes]

        ax.plot(batch_sizes, fps_vals, marker='o', markersize=4,
                linestyle='-', color='C0', linewidth=1.5, label='VGDL-JAX')

        # py-vgdl baseline: horizontal dashed line
        if game in pyvgdl_results:
            baseline_fps = pyvgdl_results[game]
            ax.axhline(y=baseline_fps, color='C3', linestyle='--',
                        linewidth=1.5, label='py-vgdl (CPU)')

            # Annotate speedup at largest batch size
            if fps_vals:
                max_fps = fps_vals[-1]
                speedup = max_fps / baseline_fps
                ax.annotate(f'{speedup:.0f}x',
                            xy=(batch_sizes[-1], max_fps),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, fontweight='bold', color='C0')

        ax.set_title(game, fontsize=12, fontweight='bold')
        ax.set_xlabel('Batch size')
        ax.set_ylabel('Steps/sec')
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale('log')
        else:
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        ax.legend(fontsize=8)

    # Hide unused subplots
    for i in range(n_games, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    scale_label = "log" if log_scale else "linear"
    fig.suptitle(f'VGDL-JAX vs py-vgdl: Random Rollout Throughput ({device})',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, f'fps_comparison_{device}_{scale_label}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")

    # Also print a summary table
    print(f"\n{'Game':20s} {'py-vgdl':>12s} {'JAX@1':>12s} {'JAX@max':>12s} {'Speedup':>10s}")
    print("-" * 70)
    for game in games:
        baseline = pyvgdl_results.get(game, 0)
        batch_fps = jax_data[game]
        bs = sorted(batch_fps.keys())
        jax_1 = batch_fps.get(1, 0)
        jax_max = batch_fps[bs[-1]] if bs else 0
        speedup = jax_max / baseline if baseline > 0 else float('inf')
        print(f"{game:20s} {baseline:>12,.0f} {jax_1:>12,.0f} {jax_max:>12,.0f} {speedup:>9.0f}x")


def main():
    parser = argparse.ArgumentParser(description="Plot FPS comparison")
    parser.add_argument('--log-scale', action='store_true',
                        help="Use log scale for y-axis")
    args = parser.parse_args()
    plot(log_scale=args.log_scale)


if __name__ == '__main__':
    main()
