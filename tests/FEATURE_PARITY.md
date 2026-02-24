# Feature Parity: py-vgdl vs vgdl-jax

Tracks what's missing for full fidelity against py-vgdl, excluding py-vgdl
bugs (e.g. speed>1 tunneling). Companion to `MECHANICS_DIFF.md` which covers
behavioral differences in *shared* features.

## Overall Coverage

| Category | py-vgdl | vgdl-jax | Coverage |
|----------|---------|----------|----------|
| Sprite Classes | 22 | 22 | 100% |
| Avatar Classes | 14 | 14 | 100% |
| Effects | 37 | 37 | 100% |
| Terminations | 4 | 4 | 100% |
| Physics | 3 | 3 | 100% |

The 9 core games (Chase, Zelda, Aliens, MissileCommand, Sokoban, Portals,
BoulderDash, SurviveZombies, Frogs) use only the original subset. All remaining
sprite classes, effects, avatar types, and terminations are now implemented,
unblocking additional VGDL games.

---

## Behavioral Differences (shared features, different results)

Both engines implement these, but produce different outputs. See
`MECHANICS_DIFF.md` for full details (A1-A9).

| ID | What | Impact | Status |
|----|------|--------|--------|
| A1 | Chaser: distance field (global BFS) vs greedy Manhattan | Different paths around obstacles | **Accepted** -- distance field is arguably *better* |
| A2 | SpawnPoint cooldown semantics | vgdl-jax spawns more aggressively | **Fixed** -- exact cooldown match, timer resets on attempt |
| A3 | Grid avatar clips to bounds | Differs in wall-less games | **Fixed** -- clip removed from `_update_avatar` |
| A4 | Effects: per-effect-batch mask-then-apply vs immediate per-collision | Different kill ordering | **Accepted tradeoff** -- fundamental to JAX vectorized design |
| A5 | Speed->cooldown conversion vs per-step distance scaling | Same rate over time | **By design** -- avoids tunneling |

**A4** is the only remaining fidelity gap, and is a structural tradeoff.

---

## Implemented Sprite Classes (all 22)

| Class | Status | Notes |
|-------|--------|-------|
| `Immovable` / `Immutable` | Original | |
| `Passive` | Original | |
| `Resource` / `ResourcePack` | Original | |
| `Missile` | Original | |
| `RandomNPC` | Original | |
| `Chaser` / `AStarChaser` | Original | Distance field relaxation |
| `Fleeing` | Original | |
| `Flicker` / `OrientedFlicker` | Original | |
| `SpawnPoint` | Original | Cooldown fix applied |
| `Bomber` | Original | |
| `Walker` | Original | |
| `Portal` | Original | |
| `Conveyor` | **New** | Static sprite with orientation |
| `ErraticMissile` | **New** | Missile + random direction change |
| `RandomInertial` | **New** | ContinuousPhysics NPC |
| `RandomMissile` | **New** | Random orientation at init |
| `Spreader` | **New** | Flicker that replicates to 4 neighbors |
| `WalkJumper` | **New** | Walker + random jump under gravity |

## Implemented Avatar Classes (all 14)

| Class | Status | Notes |
|-------|--------|-------|
| `MovingAvatar` | Original | 4-directional |
| `FlakAvatar` | Original | Horizontal + shoot |
| `ShootAvatar` | Original | 4-dir + oriented shoot |
| `HorizontalAvatar` | Original | LEFT/RIGHT only |
| `OrientedAvatar` | Original | |
| `InertialAvatar` | Original | ContinuousPhysics |
| `MarioAvatar` | Original | GravityPhysics |
| `VerticalAvatar` | **New** | UP/DOWN only |
| `RotatingAvatar` | **New** | Ego-centric: forward/backward/rotate |
| `RotatingFlippingAvatar` | **New** | Ego-centric: forward/flip/rotate |
| `NoisyRotatingFlippingAvatar` | **New** | Above + stochastic noise |
| `ShootEverywhereAvatar` | **New** | Fires all 4 directions |
| `AimedAvatar` | **New** | Continuous-angle aim + shoot |
| `AimedFlakAvatar` | **New** | Aim + horizontal movement |

## Implemented Effects (all 37)

| Effect | Status | Notes |
|--------|--------|-------|
| `killSprite` | Original | |
| `killBoth` | Original | |
| `stepBack` | Original | |
| `transformTo` | Original | Prefix-sum slot allocation |
| `turnAround` | Original | |
| `reverseDirection` | Original | |
| `changeResource` | Original | |
| `collectResource` | Original | |
| `killIfHasLess` / `killIfHasMore` | Original | |
| `killIfOtherHasMore` / `killIfOtherHasLess` | Original | |
| `killIfFromAbove` | Original | |
| `wrapAround` | Original | |
| `bounceForward` | Original | |
| `undoAll` | Original | |
| `teleportToExit` | Original | |
| `pullWithIt` | Original | |
| `wallStop` | Original | |
| `wallBounce` | Original | |
| `bounceDirection` | Original | |
| `flipDirection` | **New** | Randomizes orientation |
| `killIfAlive` | **New** | Kill if partner alive |
| `killIfSlow` | **New** | Kill if speed < limit |
| `conveySprite` | **New** | Move in partner's orientation |
| `cloneSprite` | **New** | Duplicate sprite |
| `spawnIfHasMore` | **New** | Conditional spawn |
| `windGust` | **New** | Random push |
| `slipForward` | **New** | Probabilistic forward step |
| `attractGaze` | **New** | Adopt partner orientation |
| `SpendResource` | **New** | Deduct from collider |
| `SpendAvatarResource` | **New** | Deduct from avatar |
| `KillOthers` | **New** | Kill all of target type |
| `KillIfAvatarWithoutResource` | **New** | Conditional kill |
| `AvatarCollectResource` | **New** | Avatar resource pickup |
| `TransformOthersTo` | **New** | Mass transform |

## Implemented Terminations (all 4)

| Class | Status |
|-------|--------|
| `SpriteCounter` | Original |
| `MultiSpriteCounter` | Original |
| `Timeout` | Original |
| `ResourceCounter` | **New** |

---

## Remaining Known Limitations

1. **A4 (effect timing)**: Effects are applied in batch per effect-type, not
   immediately per-collision. This means if effect A kills sprite X, and effect B
   checks if X is alive, B still sees X as alive within the same step. This is
   inherent to the JAX vectorized approach.

2. **A1 (chaser pathfinding)**: Distance field relaxation produces globally
   optimal paths vs py-vgdl's greedy Manhattan approach. This means chasers
   may take different routes around obstacles.

3. **A5 (speed handling)**: Speed is converted to cooldown at compile time.
   Fractional speeds use integer cooldowns (speed=0.5 → cooldown=2), avoiding
   the sub-pixel tunneling issues in py-vgdl.
