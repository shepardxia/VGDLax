import jax
import jax.numpy as jnp
from vgdl_jax.render import render_rgb


def test_render_rgb_shape():
    """Output image has correct shape: [H*bs, W*bs, 3]."""
    obs = jnp.zeros((2, 3, 4), dtype=jnp.bool_)
    colors = jnp.array([[255, 0, 0], [0, 0, 255]], dtype=jnp.uint8)
    img = render_rgb(obs, colors, block_size=10)
    assert img.shape == (30, 40, 3)
    assert img.dtype == jnp.uint8


def test_render_rgb_background():
    """Empty grid renders as white background."""
    obs = jnp.zeros((2, 3, 4), dtype=jnp.bool_)
    colors = jnp.array([[255, 0, 0], [0, 0, 255]], dtype=jnp.uint8)
    img = render_rgb(obs, colors, block_size=1)
    assert jnp.all(img == 255)


def test_render_rgb_single_sprite():
    """A single sprite at (1, 2) renders its color in the correct block."""
    obs = jnp.zeros((2, 3, 4), dtype=jnp.bool_)
    obs = obs.at[0, 1, 2].set(True)
    colors = jnp.array([[255, 0, 0], [0, 0, 255]], dtype=jnp.uint8)
    img = render_rgb(obs, colors, block_size=1)
    assert jnp.array_equal(img[1, 2], jnp.array([255, 0, 0]))
    assert jnp.array_equal(img[0, 0], jnp.array([255, 255, 255]))


def test_render_rgb_z_order():
    """Higher type_idx draws on top."""
    obs = jnp.zeros((2, 3, 4), dtype=jnp.bool_)
    obs = obs.at[0, 1, 1].set(True)  # type 0 (red)
    obs = obs.at[1, 1, 1].set(True)  # type 1 (blue) — on top
    colors = jnp.array([[255, 0, 0], [0, 0, 255]], dtype=jnp.uint8)
    img = render_rgb(obs, colors, block_size=1)
    assert jnp.array_equal(img[1, 1], jnp.array([0, 0, 255]))


def test_render_rgb_block_size():
    """block_size > 1 scales each cell to a block."""
    obs = jnp.zeros((1, 2, 2), dtype=jnp.bool_)
    obs = obs.at[0, 0, 0].set(True)
    colors = jnp.array([[255, 0, 0]], dtype=jnp.uint8)
    img = render_rgb(obs, colors, block_size=4)
    assert img.shape == (8, 8, 3)
    assert jnp.all(img[0:4, 0:4, 0] == 255)
    assert jnp.all(img[0:4, 0:4, 1] == 0)
    assert jnp.all(img[4:8, 4:8] == 255)


def test_render_rgb_jittable():
    """render_rgb works under jax.jit."""
    obs = jnp.zeros((2, 3, 4), dtype=jnp.bool_)
    obs = obs.at[0, 1, 2].set(True)
    colors = jnp.array([[255, 0, 0], [0, 0, 255]], dtype=jnp.uint8)
    jitted = jax.jit(render_rgb, static_argnums=(2,))
    img = jitted(obs, colors, 1)
    assert jnp.array_equal(img[1, 2], jnp.array([255, 0, 0]))
