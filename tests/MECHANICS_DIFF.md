# Engine Mechanics Differences: py-vgdl vs vgdl-jax

Permanent record of known behavioral differences between the two engines.
These may cause state divergence during cross-engine validation.

| ID | Difference | Severity | Status | Games Affected |
|----|-----------|----------|--------|---------------|
| A1 | Chaser tie-breaking | MODERATE | **Accepted** | Chase, MissileCommand, SurviveZombies |
| A2 | SpawnPoint cooldown semantics | SIGNIFICANT | **Fixed** | ~~Aliens, Frogs, SurviveZombies~~ |
| A3 | Avatar position clipping | MINOR | **Fixed** | ~~(games without wall borders)~~ |
| A4 | Effect application timing | MODERATE | **Partially fixed** | Zelda (1 step in 20) |
| A5 | Speed→cooldown conversion | SIGNIFICANT | **Fixed** | ~~Games with speed != 1~~ |
| A6 | Collision detection method | MINOR | Accepted | None (grid-aligned only) |
| A7 | wallStop position correction | MINOR | Accepted | Continuous-physics games |
| A8 | wallStop friction (dead parameter) | MINOR | Accepted | Continuous-physics games |
| A9 | Continuous-physics collision threshold | MINOR | Accepted | Continuous-physics games |

---

## A1. Chaser Tie-Breaking (MODERATE)

**py-vgdl**: Finds all directions that reduce Manhattan distance to closest target. If multiple, picks **uniformly random** via `random_generator.choice(options)`. If none, falls back to `BASEDIRS`.

**vgdl-jax**: Builds global Manhattan distance field via iterative relaxation. Picks direction with `argmin` (deterministic, first-index wins on ties). Falls back to random only when no targets are alive.

**Impact**: Different chaser paths when multiple directions are equally good. Also, py-vgdl's greedy 1-step Manhattan check can't see around walls; vgdl-jax's distance field propagates globally.

**Files**:
- `py-vgdl/vgdl/ontology/sprites.py:215-258`
- `vgdl-jax/vgdl_jax/sprites.py:72-113`

---

## A2. SpawnPoint Cooldown Semantics — FIXED

**Was**: Per-sprite timer vs global clock multiples caused different spawn timing.

**Fix**: Timer resets on every spawn attempt (not just success), matching py-vgdl's per-cooldown-interval semantics.

---

## A3. Avatar Position Clipping — FIXED

**Was**: Grid-physics avatars hard-clipped to bounds; py-vgdl does not.

**Fix**: Removed clip from `_update_avatar`. Boundaries now handled by wall stepBack / EOS effects, matching py-vgdl.

---

## A4. Effect Application Timing (MODERATE) — Partially Fixed

**py-vgdl**: Effects applied **immediately** per collision. If effect A kills sprite X, subsequent effects see X as dead. Within one effect type, sprites are processed sequentially by iteration order.

**vgdl-jax**: Same-type effects (where `type_a == type_b`) now use `jax.lax.fori_loop` for sequential per-sprite processing with deferred kills, matching py-vgdl's iteration semantics. Cross-type effects remain batched (all collision masks computed before applying).

**Impact**: Most games now match. Residual difference only when cross-type kill ordering matters within the same step. In practice, only zelda shows 1 divergent step in 20 (step 20, a monsterNormal position difference).

**Files**:
- `py-vgdl/vgdl/core.py:782-846`
- `vgdl-jax/vgdl_jax/step.py` (`_apply_all_effects`, fori_loop path)

---

## A5. Speed→Cooldown Conversion — FIXED

**Was**: Speed converted to cooldown at compile time (`effective_cooldown = max(1, round(cooldown/speed))`), producing integer-cell movement. py-vgdl moves `delta * speed` per tick, producing fractional positions.

**Fix**: Movement functions now compute `delta * speed[:, None]` per tick, matching py-vgdl's fractional positions. AABB collision enabled for fractional-speed pairs. Sweep collision added for speed > 1 (marks all intermediate cells). EOS detection uses epsilon tolerance (0.01) to absorb float32 accumulation drift.

**Files**:
- `vgdl_jax/sprites.py` (update_missile, update_erratic_missile, update_random_npc, update_chaser)
- `vgdl_jax/compiler.py` (fractional_speed_types, use_aabb, needs_sweep)
- `vgdl_jax/step.py` (_build_swept_occupancy_grid, _collision_mask_sweep)
- `vgdl_jax/collision.py` (detect_eos with _EOS_EPS tolerance)

---

## A6. Collision Detection Method (MINOR)

**py-vgdl**: Pygame `rect.collidelistall()` — pixel-based rectangle overlap.

**vgdl-jax**: Two methods, selected per effect pair at compile time:
- **Grid-based** (default): Integer occupancy lookup. Used for pairs where neither type uses continuous/gravity physics.
- **AABB**: Float position overlap with `|diff| < 1.0 - 1e-3` threshold. Used when at least one type in the pair uses continuous or gravity physics.

**Impact**: Identical for grid-aligned sprites. For continuous-physics sprites, AABB approximates pygame rect overlap but may differ at sub-pixel boundaries (see A9).

---

## A7. wallStop Position Correction (MINOR)

**py-vgdl**: Uses pixel-precise `pygame.Rect.clip()` math to position sprite flush against the wall it collided with.

**vgdl-jax**: Computes flush position via `wall_pos ± 1.0` in grid-cell AABB coordinates. Finds the nearest wall on the collision axis and places the sprite exactly one cell-width away.

**Impact**: Both place sprite at the contact boundary. Sub-pixel difference possible for unusual velocities, but equivalent for typical gameplay. The gap of exactly 1.0 grid cell between sprite and wall center is >= the AABB threshold (0.999), preventing re-collision on the next frame.

**Files**:
- `vgdl-jax/vgdl_jax/step.py` (wall_stop effect block)

---

## A8. wallStop Friction (MINOR)

**py-vgdl**: Accepts `friction` parameter on wallStop but never uses it — the parameter is dead code in py-vgdl.

**vgdl-jax**: Applies friction to the surviving velocity axis when wallStop fires. E.g., vertical collision zeros row velocity and scales col velocity by `(1 - friction)`.

**Impact**: Invented behavior not present in py-vgdl. Harmless when no VGDL game specifies `friction` on wallStop, but would produce different behavior if a game did.

**Files**:
- `vgdl-jax/vgdl_jax/step.py` (wall_stop effect block)

---

## A9. Continuous-Physics Collision Threshold (MINOR)

**py-vgdl**: Uses pygame rect overlap — two sprites with `block_size × block_size` pixel rects overlap when their bounding boxes intersect (integer pixel math).

**vgdl-jax**: Uses AABB with `1.0 - 1e-3` threshold per axis in grid-cell coordinates. Two 1×1 sprites overlap when `|pos_a - pos_b| < 0.999` on both axes.

**Impact**: Results equivalent for 1×1 sprites at typical positions. May differ at exact sub-pixel boundaries where one engine detects overlap and the other doesn't due to the epsilon tolerance.

**Files**:
- `vgdl-jax/vgdl_jax/step.py` (`_collision_mask_aabb`, `_AABB_EPS`)
