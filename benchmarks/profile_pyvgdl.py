"""
Profile py-vgdl (CPU baseline) environment speed with random actions.
Bypasses the gym wrapper and pygame to measure pure game-engine throughput.

This produces the flat horizontal baseline line in the comparison chart
(analogous to PuzzleJAX comparing against NodeJS PuzzleScript).
"""
import argparse
import json
import os
import random
import sys
from timeit import default_timer as timer

# Add py-vgdl to path
PYVGDL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl')
sys.path.insert(0, PYVGDL_DIR)

import vgdl

GAMES_DIR = os.path.join(PYVGDL_DIR, 'vgdl', 'games')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'pyvgdl')

GAMES = [
    ("Chase",           "chase.txt",           "chase_lvl0.txt"),
    ("Zelda",           "zelda.txt",           "zelda_lvl0.txt"),
    ("Aliens",          "aliens.txt",          "aliens_lvl0.txt"),
    ("MissileCommand",  "missilecommand.txt",  "missilecommand_lvl0.txt"),
    ("Sokoban",         "sokoban.txt",         "sokoban_lvl0.txt"),
    ("Portals",         "portals.txt",         "portals_lvl0.txt"),
    ("BoulderDash",     "boulderdash.txt",     "boulderdash_lvl0.txt"),
    ("SurviveZombies",  "survivezombies.txt",  "survivezombies_lvl0.txt"),
    ("Frogs",           "frogs.txt",           "frogs_lvl0.txt"),
]

N_STEPS = 2000  # more steps for stable CPU timing
N_MEASURE_LOOPS = 3


def profile_game(game_name, game_file, level_file, overwrite=False):
    """Profile a single game using py-vgdl's core engine (no pygame display)."""
    results_path = os.path.join(RESULTS_DIR, f'{game_name}.json')

    if os.path.exists(results_path) and not overwrite:
        with open(results_path, 'r') as f:
            return json.load(f)

    with open(game_file, 'r') as f:
        game_desc = f.read()
    with open(level_file, 'r') as f:
        level_desc = f.read()

    # Parse game and build level (no pygame needed)
    domain = vgdl.VGDLParser().parse_game(game_desc)
    game = domain.build_level(level_desc)

    action_keys = list(game.get_possible_actions().values())
    n_actions = len(action_keys)

    print(f"\n{'='*60}")
    print(f"  {game_name} (py-vgdl CPU)  |  {n_actions} actions  |  {N_STEPS} steps/loop")
    print(f"{'='*60}")

    # Warmup
    for _ in range(50):
        game.tick(random.choice(action_keys))
        if game.ended:
            game.reset()

    # Measurement loops
    times = []
    for i in range(N_MEASURE_LOOPS):
        game.reset()
        t0 = timer()
        for _ in range(N_STEPS):
            game.tick(random.choice(action_keys))
            if game.ended:
                game.reset()
        elapsed = timer() - t0
        times.append(elapsed)
        fps = N_STEPS / elapsed
        print(f"  loop{i}: {fps:,.0f} steps/sec ({elapsed:.3f}s)")

    fpss = [N_STEPS / t for t in times]
    results = {
        "fps_per_loop": fpss,
        "fps_mean": sum(fpss) / len(fpss),
        "fps_last": fpss[-1],
        "n_steps": N_STEPS,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Mean FPS: {results['fps_mean']:,.0f}")
    return results


def main():
    global N_STEPS

    parser = argparse.ArgumentParser(description="Profile py-vgdl CPU baseline FPS")
    parser.add_argument('--game', type=str, default=None,
                        help="Profile a single game (e.g. 'Chase')")
    parser.add_argument('--overwrite', action='store_true',
                        help="Re-run even if results exist")
    parser.add_argument('--n-steps', type=int, default=N_STEPS,
                        help=f"Steps per measurement loop (default: {N_STEPS})")
    args = parser.parse_args()

    N_STEPS = args.n_steps

    print(f"Profiling py-vgdl (CPU baseline)")
    print(f"Steps per loop: {N_STEPS}")

    games = GAMES
    if args.game:
        games = [(n, g, l) for n, g, l in GAMES if n == args.game]
        if not games:
            print(f"Unknown game '{args.game}'. Available: {[n for n,_,_ in GAMES]}")
            return

    all_results = {}
    for name, gf, lf in games:
        try:
            r = profile_game(
                name,
                os.path.join(GAMES_DIR, gf),
                os.path.join(GAMES_DIR, lf),
                overwrite=args.overwrite,
            )
            all_results[name] = r
        except Exception as e:
            print(f"  ERROR on {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("  Summary (py-vgdl CPU baseline):")
    for name, r in all_results.items():
        print(f"    {name:20s}: {r['fps_mean']:>10,.0f} steps/sec")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
