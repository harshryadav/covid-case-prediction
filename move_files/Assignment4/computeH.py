"""
Homography from point correspondences using the Direct Linear Transform (DLT) and SVD.
Maps homogeneous coordinates from the first view to the second: x2 ~ H @ x1.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Hartley normalization for 2xN points: translate centroid to origin, average distance sqrt(2).
    Returns (pts_norm 3xN, T 3x3).
    """
    x = pts[0, :].astype(np.float64)
    y = pts[1, :].astype(np.float64)
    c_x = np.mean(x)
    c_y = np.mean(y)
    d = np.mean(np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2))
    if d < 1e-12:
        d = 1.0
    s = np.sqrt(2) / d
    T = np.array([[s, 0, -s * c_x], [0, s, -s * c_y], [0, 0, 1]], dtype=np.float64)
    ones = np.ones((1, pts.shape[1]), dtype=np.float64)
    homo = np.vstack([x, y, ones])
    homo_n = T @ homo
    return homo_n, T


def computeH(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """
    Compute 3x3 homography H such that (homogeneous) points in view 2 satisfy
    x2 ~ H @ x1, where t1, t2 are 2xN arrays (row 0: x, row 1: y).

    Uses normalized DLT and SVD (numpy.linalg.svd). At least 4 correspondences required.
    """
    if t1.shape != t2.shape or t1.shape[0] != 2:
        raise ValueError("t1 and t2 must be 2xN matrices with the same N.")
    n = t1.shape[1]
    if n < 4:
        raise ValueError("At least 4 point pairs are required.")

    homo1, T1 = _normalize_points(t1)
    homo2, T2 = _normalize_points(t2)
    # Cartesian coordinates in normalized frames (w is 1 after T-normalization)
    x1 = homo1[0, :] / homo1[2, :]
    y1 = homo1[1, :] / homo1[2, :]
    x2 = homo2[0, :] / homo2[2, :]
    y2 = homo2[1, :] / homo2[2, :]

    rows = []
    for i in range(n):
        x, y = x1[i], y1[i]
        xp, yp = x2[i], y2[i]
        rows.append([-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp])
        rows.append([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
    A = np.asarray(rows, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    Hn = h.reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def apply_homography(H: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply H to 2xN points t; returns 2xN Cartesian coordinates in the target plane."""
    x = t[0, :].astype(np.float64)
    y = t[1, :].astype(np.float64)
    ones = np.ones_like(x)
    ph = H @ np.vstack([x, y, ones])
    ph /= ph[2:3, :]
    return np.vstack([ph[0, :], ph[1, :]])


def verify_homography_overlay(
    t1: np.ndarray,
    t2: np.ndarray,
    H: np.ndarray,
    image_ref: np.ndarray,
    title: str = "Homography verification: mapped t1 (crosses) vs true t2 (circles)",
) -> None:
    """
    Map t1 through H and plot predicted positions on the reference image together with t2.
    """
    pred = apply_homography(H, t1)
    fig, ax = plt.subplots(figsize=(10, 8))
    if image_ref.ndim == 2:
        ax.imshow(image_ref, cmap="gray")
    else:
        ax.imshow(image_ref)
    ax.scatter(t2[0, :], t2[1, :], s=120, facecolors="none", edgecolors="lime", linewidths=2, label="True t2")
    ax.scatter(pred[0, :], pred[1, :], s=80, c="red", marker="+", linewidths=2, label="H @ t1")
    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
