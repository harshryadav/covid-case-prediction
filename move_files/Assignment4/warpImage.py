"""
Warp an input image into the reference image plane using inverse mapping (no holes from forward splat).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _as_uint8_rgb(im: np.ndarray) -> np.ndarray:
    """Convert image to uint8 HxWx3 for consistent warping and blending."""
    x = np.asarray(im)
    if x.dtype == np.float32 or x.dtype == np.float64:
        if x.max() <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
    else:
        x = x.astype(np.uint8)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    else:
        x = x[..., :3].copy()
    return x


def _apply_homography_cols(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 3×3 H to 3×N points without batched matmul (avoids spurious OpenBLAS warnings on some platforms)."""
    x = pts[0, :]
    y = pts[1, :]
    w = pts[2, :]
    out = np.empty_like(pts, dtype=np.float64)
    for i in range(3):
        out[i, :] = H[i, 0] * x + H[i, 1] * y + H[i, 2] * w
    return out


def warpImage(
    inputIm: np.ndarray,
    refIm: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp inputIm into refIm's coordinate frame using homography H (maps points from input to ref:
    p_ref ~ H @ p_input).

    Uses inverse warping: for each destination pixel, sample inputIm at H^{-1} p.

    Returns:
        warpIm: warped input on a canvas aligned with the mosaic (uint8 RGB).
        mergeIm: blend of warped input and reference on the same canvas.
    """
    inputIm = _as_uint8_rgb(inputIm)
    refIm = _as_uint8_rgb(refIm)

    H = np.asarray(H, dtype=np.float64)
    H_inv = np.linalg.inv(H)

    in_h, in_w = inputIm.shape[:2]
    ref_h, ref_w = refIm.shape[:2]

    corners_in = np.array(
        [[0, in_w - 1, in_w - 1, 0], [0, 0, in_h - 1, in_h - 1]], dtype=np.float64
    )
    ones = np.ones((1, 4), dtype=np.float64)
    homo = np.vstack([corners_in, ones])
    warped_corners = _apply_homography_cols(H, homo)
    warped_corners /= warped_corners[2:3, :]
    wx = warped_corners[0, :]
    wy = warped_corners[1, :]

    xmin = float(np.floor(min(0.0, np.min(wx))))
    ymin = float(np.floor(min(0.0, np.min(wy))))
    xmax = float(np.ceil(max(ref_w, np.max(wx))))
    ymax = float(np.ceil(max(ref_h, np.max(wy))))

    canvas_w = int(np.ceil(xmax - xmin))
    canvas_h = int(np.ceil(ymax - ymin))
    if canvas_w < 1 or canvas_h < 1:
        raise ValueError("Invalid mosaic canvas size.")

    mx = np.arange(canvas_w, dtype=np.float64)
    my = np.arange(canvas_h, dtype=np.float64)
    mx_grid, my_grid = np.meshgrid(mx, my)
    xref = mx_grid + xmin
    yref = my_grid + ymin

    ones_g = np.ones_like(xref)
    pref = np.stack([xref, yref, ones_g], axis=0).reshape(3, -1)
    pin = _apply_homography_cols(H_inv, pref)
    with np.errstate(divide="ignore", invalid="ignore"):
        pin = pin / pin[2:3, :]
    pin = np.nan_to_num(pin, nan=-1.0, posinf=-1.0, neginf=-1.0)
    map_x = pin[0, :].reshape(canvas_h, canvas_w)
    map_y = pin[1, :].reshape(canvas_h, canvas_w)

    warp_channels = []
    for c in range(3):
        ch = inputIm[..., c].astype(np.float64)
        sampled = ndimage.map_coordinates(
            ch,
            [map_y, map_x],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        warp_channels.append(sampled)
    warpIm = np.stack(warp_channels, axis=-1)
    warpIm = np.clip(np.round(warpIm), 0, 255).astype(np.uint8)

    ox = int(round(-xmin))
    oy = int(round(-ymin))
    ref_layer = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    h_end = min(oy + ref_h, canvas_h)
    w_end = min(ox + ref_w, canvas_w)
    ref_y0 = max(0, -oy)
    ref_x0 = max(0, -ox)
    out_y0 = max(0, oy)
    out_x0 = max(0, ox)
    ref_y1 = ref_y0 + (h_end - out_y0)
    ref_x1 = ref_x0 + (w_end - out_x0)
    if h_end > out_y0 and w_end > out_x0:
        ref_layer[out_y0:h_end, out_x0:w_end, :] = refIm[ref_y0:ref_y1, ref_x0:ref_x1, :]

    warp_f = warpIm.astype(np.float64)
    ref_f = ref_layer.astype(np.float64)
    warp_mask = warp_f.sum(axis=-1) > 1e-3
    ref_mask = ref_f.sum(axis=-1) > 1e-3
    overlap = warp_mask & ref_mask

    mergeIm = np.zeros_like(warpIm, dtype=np.float64)
    mergeIm[ref_mask & ~warp_mask] = ref_f[ref_mask & ~warp_mask]
    mergeIm[warp_mask & ~ref_mask] = warp_f[warp_mask & ~ref_mask]
    mergeIm[overlap] = 0.5 * ref_f[overlap] + 0.5 * warp_f[overlap]
    mergeIm = np.clip(np.round(mergeIm), 0, 255).astype(np.uint8)

    return warpIm, mergeIm
