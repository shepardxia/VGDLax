# Feature Parity: py-vgdl vs vgdl-jax

Tracks what's missing for full fidelity against py-vgdl, excluding py-vgdl
bugs (e.g. speed>1 tunneling). Companion to `MECHANICS_DIFF.md` which covers
behavioral differences in *shared* features.

## Overall Coverage

| Category | py-vgdl | vgdl-jax | Coverage |
|----------|---------|----------|----------|
| Sprite Classes | 22 | 14 | 64% |
| Avatar Classes | 14 | 7 | 50% |
| Effects | 37 | 22 | 59% |
| Terminations | 4 | 3 | 75% |
| Physics | 3 | 3 | 100% |

The 9 supported games (Chase, Zelda, Aliens, MissileCommand, Sokoban, Portals,
BoulderDash, SurviveZombies, Frogs) use only the implemented subset. The gaps
below block other VGDL games.

---

## Behavioral Differences (shared features, different results)

Both engines implement these, but produce different outputs. See
`MECHANICS_DIFF.md` for full details (A1-A9).

| ID | What | Impact | Fixable? |
|----|------|--------|----------|
| A1 | Chaser: distance field (global BFS) vs greedy Manhattan | Different paths around obstacles | Could match by switching to greedy, but distance field is arguably *better* |
| A2 | SpawnPoint: retries every tick once eligible vs only at cooldown multiples | vgdl-jax spawns more aggressively | Fixable -- add modulo check |
| A3 | Grid avatar clips to bounds; py-vgdl doesn't | Differs only in wall-less games | Fixable -- remove clip from `_update_avatar` |
| A4 | Effects: per-effect-batch mask-then-apply vs immediate per-collision | Different kill ordering when multiple effects hit same sprite | Fundamental to JAX vectorized design -- hard to fix without losing performance |
| A5 | Speed->cooldown conversion vs per-step distance scaling | Same rate over time, different per-step position | By design (avoids tunneling) |

**A4 is the biggest fidelity gap** -- it's structural. The rest are fixable or
intentional tradeoffs.

---

## Missing Sprite Classes

### Used in real VGDL games

| Class | What it does | Difficulty |
|-------|-------------|-----------|
| `Spreader` | Flicker that replicates to adjacent cells | Medium -- neighbor spawning logic |
| `ErraticMissile` | Missile that randomly changes direction | Easy -- RandomNPC + Missile hybrid |
| `Conveyor` | Static sprite that pushes overlapping sprites | Easy -- effect-like behavior |
| `WalkJumper` | Walker that can jump | Medium -- needs gravity integration |
| `RandomInertial` | RandomNPC with ContinuousPhysics | Easy -- continuous physics exists now |
| `RandomMissile` | Missile with random speed/direction at spawn | Easy |

### Rarely used / niche

| Class | Notes |
|-------|-------|
| `AStarChaser` | Distance field already handles this better |
| `OrientedSprite` | Base class, not directly instantiated in games |

---

## Missing Avatar Classes (7 of 14)

| Class | What it does | Difficulty |
|-------|-------------|-----------|
| `VerticalAvatar` | UP/DOWN only (mirror of HorizontalAvatar) | Trivial |
| `RotatingAvatar` | UP=forward, DOWN=backward, L/R=rotate orientation | Medium -- different action semantics |
| `RotatingFlippingAvatar` | RotatingAvatar + DOWN=180 spin | Medium |
| `NoisyRotatingFlippingAvatar` | Above + stochastic noise | Medium |
| `ShootEverywhereAvatar` | Shoots all 4 directions simultaneously | Easy |
| `AimedAvatar` | Aim/shoot only, no movement | Easy |
| `AimedFlakAvatar` | AimedAvatar + horizontal movement | Easy |

---

## Missing Effects (15 of 37)

### Likely needed for real games

| Effect | What | Difficulty |
|--------|------|-----------|
| `cloneSprite` | Duplicate sprite at same position | Medium |
| `flipDirection` | Flip one axis of orientation | Easy |
| `killIfAlive` | Kill actor if partner is alive | Easy |
| `killIfSlow` | Kill if speed below threshold | Easy |
| `spawnIfHasMore` | Spawn sprite if resource > limit | Medium |
| `conveySprite` | Move sprite in conveyor's direction | Easy |

### Niche / uncommon

| Effect | What |
|--------|------|
| `windGust` | Push sprite with random force |
| `slipForward` | Move sprite forward on conveyor |
| `attractGaze` | Rotate sprite to face partner |
| `SpendResource` | Kill sprite if avatar lacks resource |
| `SpendAvatarResource` | Deduct resource from avatar on collision |
| `KillOthers` | Kill all sprites of a given type |
| `KillIfAvatarWithoutResource` | Kill if avatar resource below threshold |
| `AvatarCollectResource` | Avatar-specific resource collection |
| `TransformOthersTo` | Transform all sprites of a type |

---

## Missing Terminations

| Class | What | Difficulty |
|-------|------|-----------|
| `ResourceCounter` | Win/lose when avatar resource count reaches limit | Easy |

---

## Priority Ranking

For matching py-vgdl across the broadest set of real VGDL games:

1. **A2 fix** (SpawnPoint cooldown semantics) -- easy, affects Aliens/Frogs/SurviveZombies
2. **ResourceCounter termination** -- easy, blocks some games entirely
3. **VerticalAvatar** -- trivial
4. **flipDirection, killIfAlive, killIfSlow** -- easy effects
5. **ErraticMissile, RandomInertial, Conveyor** -- easy sprite classes
6. **cloneSprite, spawnIfHasMore** -- medium effects used in some games
7. **Rotating avatar family** -- medium, needed for rotation-based games
8. **A4 (effect timing)** -- hardest, requires rearchitecting the effect loop
