"""
Renderers for vgdl-jax: convert game state to RGB images.

Two modes:
- render_rgb: Pure JAX, jittable, uses occupancy grids + color lookup
- render_pygame: Uses pygame for pixel-exact match with py-vgdl (added later)
"""
import jax.numpy as jnp


def render_rgb(obs, colors, block_size, bg_color=None):
    """
    JAX-native renderer. Converts occupancy grid to RGB image.

    Args:
        obs: [n_types, H, W] bool — occupancy grid from _get_obs()
        colors: [n_types, 3] uint8 — RGB color per sprite type
        block_size: int — pixels per grid cell
        bg_color: optional [3] uint8 — background color (default white)

    Returns:
        [H*block_size, W*block_size, 3] uint8 RGB image
    """
    if bg_color is None:
        bg_color = jnp.array([255, 255, 255], dtype=jnp.uint8)

    n_types, H, W = obs.shape

    # Find top-z-order type per cell: highest type_idx that's occupied.
    type_indices = jnp.arange(n_types)[:, None, None]  # [n_types, 1, 1]
    weighted = jnp.where(obs, type_indices, -1)         # [n_types, H, W]
    top_type = jnp.max(weighted, axis=0)                # [H, W]

    # Build extended color table: append bg_color at index n_types
    extended_colors = jnp.concatenate(
        [colors, bg_color[None]], axis=0
    )  # [n_types+1, 3]

    # Map -1 (empty) to bg_color index
    color_idx = jnp.where(top_type >= 0, top_type, n_types)  # [H, W]

    # Look up colors
    canvas = extended_colors[color_idx]  # [H, W, 3]

    # Scale up by block_size
    canvas = jnp.repeat(jnp.repeat(canvas, block_size, axis=0),
                        block_size, axis=1)

    return canvas


def _load_sprite(img_name, block_size, shrinkfactor, sprites_path, cache):
    """Load a sprite image, scale it, cache it."""
    import pygame
    effective_size = int((1 - shrinkfactor) * block_size)
    cache_key = (img_name, effective_size)
    if cache_key not in cache:
        from pathlib import Path
        path = Path(sprites_path) / (img_name + '.png')
        img = pygame.image.load(str(path))
        img = img.convert_alpha()
        if effective_size != max(img.get_rect().size):
            img = pygame.transform.smoothscale(img, (effective_size, effective_size))
        cache[cache_key] = img
    return cache[cache_key]


def _calculate_render_rect(row, col, block_size, shrinkfactor):
    """Match py-vgdl's calculate_render_rect.

    py-vgdl uses: sprite_rect.inflate(*(-Vector2(sprite_rect.size) * shrinkfactor))
    We replicate this exactly so pixel output matches.
    """
    import pygame
    sprite_rect = pygame.Rect(col * block_size, row * block_size, block_size, block_size)
    if shrinkfactor != 0:
        shrink = pygame.math.Vector2(sprite_rect.size) * shrinkfactor
        sprite_rect = sprite_rect.inflate(-shrink.x, -shrink.y)
    return sprite_rect


def _apply_rotation(img, orientation):
    """Rotate sprite image based on orientation, matching py-vgdl."""
    import pygame
    ori_row, ori_col = orientation
    if ori_row == 0 and ori_col == 0:
        return img
    RIGHT = pygame.math.Vector2(1, 0)
    screen_ori = pygame.math.Vector2(float(ori_col), float(ori_row))
    angle = RIGHT.angle_to(screen_ori)
    if abs(angle / 180) == 1:
        return pygame.transform.flip(img, True, False)
    elif angle != 0:
        return pygame.transform.rotate(img, -angle)
    return img


def render_pygame(state, game_def, block_size, sprites_path=None,
                  render_sprites=True):
    """
    Pygame-exact renderer. Matches py-vgdl's rendering output.

    Requires pygame to be installed. NOT jittable.

    Args:
        state: GameState (JAX)
        game_def: GameDef with sprite definitions and level info
        block_size: pixels per grid cell
        sprites_path: path to directory containing sprite PNGs (e.g. oryx/, newset/).
            If None, auto-discovers ../../py-vgdl/vgdl/sprites/ relative to this file.
        render_sprites: if False, always draw solid color rectangles (no images).
            Useful for cross-engine validation against color-only renderers.

    Returns:
        [H*block_size, W*block_size, 3] uint8 numpy array
    """
    import os
    import pygame
    import numpy as np

    H = game_def.level.height
    W = game_def.level.width
    screen_h = H * block_size
    screen_w = W * block_size

    # Auto-discover sprites_path (only if sprite rendering is enabled)
    if render_sprites and sprites_path is None:
        from pathlib import Path
        candidate = Path(__file__).resolve().parent.parent.parent / 'py-vgdl' / 'vgdl' / 'sprites'
        if candidate.exists():
            sprites_path = str(candidate)

    # Disable sprite path when render_sprites=False
    if not render_sprites:
        sprites_path = None

    # Initialize pygame with a dummy display (needed for convert_alpha)
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    if not pygame.get_init():
        pygame.init()
    pygame.display.set_mode((1, 1))

    # Create offscreen surface
    surface = pygame.Surface((screen_w, screen_h))
    surface.fill((255, 255, 255))  # white background

    # Convert JAX arrays to numpy for indexing
    positions = np.asarray(state.positions)      # [n_types, max_n, 2]
    alive = np.asarray(state.alive)              # [n_types, max_n]
    orientations = np.asarray(state.orientations)  # [n_types, max_n, 2]

    sprite_cache = {}

    # Draw sprites in type order (z-order: definition order in SpriteSet)
    for sd in game_def.sprites:
        t = sd.type_idx
        for i in range(alive.shape[1]):
            if not alive[t, i]:
                continue
            row, col = round(float(positions[t, i, 0])), round(float(positions[t, i, 1]))
            # Skip out-of-bounds sprites
            if row < 0 or row >= H or col < 0 or col >= W:
                continue

            sprite_rect = _calculate_render_rect(row, col, block_size, sd.shrinkfactor)

            if sd.img is not None and sprites_path is not None:
                img = _load_sprite(sd.img, block_size, sd.shrinkfactor,
                                   sprites_path, sprite_cache)
                ori = orientations[t, i]
                img = _apply_rotation(img, (float(ori[0]), float(ori[1])))
                surface.blit(img, sprite_rect)
            else:
                surface.fill(sd.color, sprite_rect)

    # Convert to numpy array (same transform as py-vgdl's get_image())
    img = np.flipud(np.rot90(
        pygame.surfarray.array3d(surface).astype(np.uint8)
    ))

    return img
