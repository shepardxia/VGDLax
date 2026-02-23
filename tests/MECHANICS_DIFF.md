# Engine Mechanics Differences: py-vgdl vs vgdl-jax

Permanent record of known behavioral differences between the two engines.
These may cause state divergence during cross-engine validation.

| ID | Difference | Severity | Status | Games Affected |
|----|-----------|----------|--------|---------------|
| A1 | Chaser tie-breaking | MODERATE | Open | Chase, MissileCommand, SurviveZombies |
| A2 | SpawnPoint cooldown semantics | SIGNIFICANT | Open | Aliens, Frogs, SurviveZombies |
| A3 | Avatar position clipping | MINOR | Open | (games without wall borders) |
| A4 | Effect application timing | MODERATE | Open | All games with multiple effects |
| A5 | Speed→cooldown conversion | SIGNIFICANT | Open | Games with speed != 1 |
| A6 | Collision detection method | MINOR | Open | None (grid-aligned only) |
| A7 | wallStop position correction | MINOR | Open | Continuous-physics games |
| A8 | wallStop friction (dead parameter) | MINOR | Open | Continuous-physics games |
| A9 | Continuous-physics collision threshold | MINOR | Open | Continuous-physics games |

---

## A1. Chaser Tie-Breaking (MODERATE)

**py-vgdl**: Finds all directions that reduce Manhattan distance to closest target. If multiple, picks **uniformly random** via `random_generator.choice(options)`. If none, falls back to `BASEDIRS`.

**vgdl-jax**: Builds global Manhattan distance field via iterative relaxation. Picks direction with `argmin` (deterministic, first-index wins on ties). Falls back to random only when no targets are alive.

**Impact**: Different chaser paths when multiple directions are equally good. Also, py-vgdl's greedy 1-step Manhattan check can't see around walls; vgdl-jax's distance field propagates globally.

**Files**:
- `py-vgdl/vgdl/ontology/sprites.py:215-258`
- `vgdl-jax/vgdl_jax/sprites.py:72-113`

---

## A2. SpawnPoint Cooldown Semantics (SIGNIFICANT)

**py-vgdl**: `game.time % self.cooldown == 0` — checks at exact global clock multiples. If prob check fails at tick `N*cooldown`, next attempt is `(N+1)*cooldown`.

**vgdl-jax**: Per-sprite timer, `cooldown_timers >= cooldown`. Timer resets to 0 **only on successful spawn**. If prob check fails, timer stays above cooldown and sprite retries **every subsequent tick** until it succeeds.

**Impact**: vgdl-jax spawners are more aggressive — they retry every tick once eligible, while py-vgdl only tries at cooldown multiples.

**Files**:
- `py-vgdl/vgdl/ontology/sprites.py:111-112`
- `vgdl-jax/vgdl_jax/sprites.py:134-198`

---

## A3. Avatar Position Clipping (MINOR)

**py-vgdl**: No explicit position clipping. Avatar can move off-screen; EOS effects or wall stepBack prevent it in practice.

**vgdl-jax**: Grid-physics avatars (`_update_avatar`) hard clip position to `[0, height-1] x [0, width-1]`. Continuous-physics avatars (`update_inertial_avatar`, `update_mario_avatar`) do NOT clip — EOS effects and wallStop handle boundaries instead (matching py-vgdl). NPC positions are NOT clipped.

**Impact**: Only matters for grid-physics games without wall borders. Most VGDL games have walls.

**Files**:
- `vgdl-jax/vgdl_jax/step.py` (`_update_avatar`)
- `vgdl-jax/vgdl_jax/sprites.py` (`update_inertial_avatar`, `update_mario_avatar`)

---

## A4. Effect Application Timing (MODERATE)

**py-vgdl**: Effects applied **immediately** per collision. If effect A kills sprite X, subsequent effects see X as dead (`if sprite not in self.kill_list`).

**vgdl-jax**: All collision masks for one effect are computed before applying. Multiple effects in the same step can "see" the same sprite as alive even if a prior effect killed it.

**Impact**: Different kill ordering when multiple effects fire on the same sprite in the same step.

**Files**:
- `py-vgdl/vgdl/core.py:782-846`
- `vgdl-jax/vgdl_jax/step.py:173-197`

---

## A5. Speed→Cooldown Conversion (SIGNIFICANT)

**py-vgdl**: `speed=2` means move `2 * gridsize` pixels per step (can tunnel over obstacles).

**vgdl-jax**: Converts speed to cooldown: `effective_cooldown = max(1, round(cooldown / speed))`. Speed=2 means move 1 cell every ~0.5 cooldown periods — same rate but no tunneling.

**Impact**: Different movement distance per step for `speed != 1`. However, most standard VGDL games use `speed=1`.

**Files**:
- `vgdl-jax/vgdl_jax/compiler.py` (speed→cooldown conversion)

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
