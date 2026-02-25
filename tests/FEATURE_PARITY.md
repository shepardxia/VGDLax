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
| A2 | SpawnPoint cooldown semantics | vgdl-jax spawns more aggressively | **Fixed** |
| A3 | Grid avatar clips to bounds | Differs in wall-less games | **Fixed** |
| A4 | Effects: batch vs immediate per-collision | Different kill ordering | **Partially fixed** -- same-type effects now sequential via fori_loop; cross-type still batched |
| A5 | Speed→cooldown vs fractional movement | Integer vs fractional positions | **Fixed** -- sprites now move `delta * speed` per tick, matching py-vgdl |

**A1** (accepted design choice) and **A4** (residual cross-type batching) are the only remaining gaps.

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

## Cross-Engine Validation Results

73 of 74 cross-engine tests pass (with RNG replay synchronization).

| Game | Status | Notes |
|------|--------|-------|
| Chase | **PASS** | All 20 NOOP steps match |
| Zelda | 1/20 diverged | Step 20 monsterNormal position (A4 residual) |
| Aliens | **PASS** | |
| MissileCommand | **PASS** | |
| Sokoban | **PASS** | Exact deterministic match |
| Portals | **PASS** | Fractional-speed RandomNPC matches |
| BoulderDash | **PASS** | Fractional-speed boulders match |
| SurviveZombies | **PASS** | |
| Frogs | **PASS** | Fractional-speed trucks/logs match |

---

## Remaining Known Limitations

1. **A4 (effect timing)**: Cross-type effects are still applied in batch
   (mask-then-apply), not immediately per-collision. Same-type effects are now
   sequential via `fori_loop`. In practice, only zelda shows 1 divergent step.

2. **A1 (chaser pathfinding)**: Distance field relaxation produces globally
   optimal paths vs py-vgdl's greedy Manhattan approach. This means chasers
   may take different routes around obstacles. Accepted as a design improvement.
