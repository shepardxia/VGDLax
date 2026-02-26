"""Count compiled effects per game with collision modes."""
import os
from collections import Counter, defaultdict
import math
from vgdl_jax.parser import parse_vgdl
from vgdl_jax.compiler import compile_game, AVATAR_INFO

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')
games = ['boulderdash', 'chase', 'zelda', 'aliens', 'sokoban',
         'portals', 'survivezombies', 'frogs', 'missilecommand']

for name in games:
    gd = parse_vgdl(
        os.path.join(GAMES_DIR, f'{name}.txt'),
        os.path.join(GAMES_DIR, f'{name}_lvl0.txt'))
    cg = compile_game(gd)

    n_types = len(gd.sprites)
    avatar_sd = None
    for sd in gd.sprites:
        if sd.sprite_class in AVATAR_INFO:
            avatar_sd = sd
            break

    continuous_types = {sd.type_idx for sd in gd.sprites
                        if sd.physics_type in ('continuous', 'gravity')}
    fractional_speed_types = {sd.type_idx for sd in gd.sprites
                               if sd.speed not in (0, 1.0) and sd.speed > 0}
    frac_or_cont = continuous_types | fractional_speed_types
    _speed_by_idx = {sd.type_idx: sd.speed for sd in gd.sprites}

    n_effects = 0
    modes = Counter()
    for ed in gd.effects:
        is_eos = (ed.actee_stype == 'EOS')
        actor_indices = gd.resolve_stype(ed.actor_stype)
        if is_eos:
            n_effects += len(actor_indices)
            modes['eos'] += len(actor_indices)
        else:
            actee_indices = gd.resolve_stype(ed.actee_stype)
            for ta in actor_indices:
                for tb in actee_indices:
                    n_effects += 1
                    sa = _speed_by_idx.get(ta, 1.0)
                    sb = _speed_by_idx.get(tb, 1.0)
                    af = ta in frac_or_cont
                    bf = tb in frac_or_cont
                    if sa > 1.0 or sb > 1.0:
                        modes['sweep'] += 1
                    elif af and not bf:
                        modes['exp_grid_a'] += 1
                    elif bf and not af:
                        modes['exp_grid_b'] += 1
                    elif af and bf:
                        modes['aabb'] += 1
                    else:
                        modes['grid'] += 1

    # Count per-type max_n
    active = set()
    for sd in gd.sprites:
        if sd.sprite_class in AVATAR_INFO:
            active.add(sd.type_idx)
    for ed in gd.effects:
        for idx in gd.resolve_stype(ed.actor_stype):
            active.add(idx)
        if ed.actee_stype != 'EOS':
            for idx in gd.resolve_stype(ed.actee_stype):
                active.add(idx)

    counts = defaultdict(int)
    for ti, _, _ in gd.level.initial_sprites:
        counts[ti] += 1

    type_info = []
    for sd in gd.sprites:
        base = counts.get(sd.type_idx, 0)
        tmn = max(base + 10, 10) if sd.type_idx in active else 1
        type_info.append(f"{sd.key}({base}→{tmn})")

    max_n = cg.init_state.alive.shape[1]
    print(f"\n{name}  n_types={n_types}  max_n={max_n}  effects={n_effects}")
    print(f"  modes: {dict(modes)}")
    print(f"  types: {', '.join(type_info)}")
