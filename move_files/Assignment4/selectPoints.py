"""
Manual correspondence selection for image mosaics.
Click a point on the left image, then the matching point on the right (same order).
Each pair is drawn in a unique color on both views.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from pathlib import Path


def pair2_image_paths(base: Path) -> tuple[Path, Path]:
    """
    Pair 2 from the download is often 3.jpeg / 4.jpeg; handout also says 3.jpg / 4.jpg.
    Returns (input, reference); point files remain 3.npy, 4.npy per assignment.
    """
    jpeg = (base / "3.jpeg", base / "4.jpeg")
    jpg = (base / "3.jpg", base / "4.jpg")
    if jpeg[0].is_file() and jpeg[1].is_file():
        return jpeg
    return jpg


def pair2_names_for_display(paths: tuple[Path, Path]) -> str:
    return f"{paths[0].name} (input) -> {paths[1].name} (reference)"


def _distinct_colors(n: int):
    try:
        cmap = plt.colormaps["tab10"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def select_correspondences(
    image_path_left: str | Path,
    image_path_right: str | Path,
    out_npy_left: str | Path,
    out_npy_right: str | Path,
    title_left: str = "Image 1 (click first)",
    title_right: str = "Image 2 (click match)",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interactive paired clicks: left image, then right image, for each correspondence.
    Saves two 2xN float arrays (rows: x, y) to the given .npy paths.
    """
    image_path_left = Path(image_path_left)
    image_path_right = Path(image_path_right)
    im1 = plt.imread(str(image_path_left))
    im2 = plt.imread(str(image_path_right))
    if im1.dtype == np.float32 or im1.dtype == np.float64:
        im1 = np.clip(im1, 0, 1) * 255 if im1.max() <= 1.0 else np.clip(im1, 0, 255)
        im1 = im1.astype(np.uint8)
    if im2.dtype == np.float32 or im2.dtype == np.float64:
        im2 = np.clip(im2, 0, 1) * 255 if im2.max() <= 1.0 else np.clip(im2, 0, 255)
        im2 = im2.astype(np.uint8)
    if im1.ndim == 3 and im1.shape[2] == 4:
        im1 = im1[:, :, :3]
    if im2.ndim == 3 and im2.shape[2] == 4:
        im2 = im2[:, :, :3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(im1)
    ax2.imshow(im2)
    ax1.set_title(title_left)
    ax2.set_title(title_right)
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    ax2.set_xlabel("x (pixels)")
    ax2.set_ylabel("y (pixels)")

    Cursor(ax1, useblit=True, color="white", linewidth=1)
    Cursor(ax2, useblit=True, color="white", linewidth=1)

    pts1: list[tuple[float, float]] = []
    pts2: list[tuple[float, float]] = []
    waiting_on_right = False
    pair_index = 0
    colors = _distinct_colors(32)

    status = fig.suptitle(
        "Step: click a point on the LEFT image, then its match on the RIGHT. Close window when done.",
        fontsize=11,
    )

    def on_click(event):
        nonlocal waiting_on_right, pair_index
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes == ax1 and not waiting_on_right:
            x, y = float(event.xdata), float(event.ydata)
            pts1.append((x, y))
            c = colors[pair_index % len(colors)]
            ax1.plot(x, y, "o", color=c, markersize=10, markeredgecolor="white", markeredgewidth=1)
            waiting_on_right = True
            status.set_text(f"Pair {pair_index + 1}: now click the matching point on the RIGHT.")
            fig.canvas.draw_idle()
        elif event.inaxes == ax2 and waiting_on_right:
            x, y = float(event.xdata), float(event.ydata)
            pts2.append((x, y))
            c = colors[pair_index % len(colors)]
            ax2.plot(x, y, "o", color=c, markersize=10, markeredgecolor="white", markeredgewidth=1)
            waiting_on_right = False
            pair_index += 1
            status.set_text(
                f"Recorded {pair_index} pair(s). Next: LEFT image, or close window to save."
            )
            fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if len(pts1) != len(pts2):
        raise RuntimeError(
            f"Unpaired clicks: {len(pts1)} on left vs {len(pts2)} on right. "
            "Each left click must be followed by a right click."
        )
    if len(pts1) < 4:
        raise RuntimeError("Need at least 4 point pairs for a homography.")

    t1 = np.array([[p[0] for p in pts1], [p[1] for p in pts1]], dtype=np.float64)
    t2 = np.array([[p[0] for p in pts2], [p[1] for p in pts2]], dtype=np.float64)

    out_npy_left = Path(out_npy_left)
    out_npy_right = Path(out_npy_right)
    np.save(str(out_npy_left), t1)
    np.save(str(out_npy_right), t2)
    print(f"Saved {out_npy_left} and {out_npy_right} with shape {t1.shape}.")
    return t1, t2


def prompt_generate_or_load() -> str:
    """
    Ask whether to collect new correspondences or use existing .npy files.
    Returns 'manual' or 'existing'.
    """
    print("\nCorrespondence points:")
    print("  [m] Manually select new corresponding points (runs interactive tool)")
    print("  [e] Use pre-generated .npy files from this folder")
    while True:
        choice = input("Enter choice [m/e]: ").strip().lower()
        if choice in ("m", "manual"):
            return "manual"
        if choice in ("e", "existing", ""):
            return "existing"
        print("Please enter 'm' or 'e'.")


def prompt_image_pair() -> str:
    """Return '12' for 1.jpg/2.jpg or '34' for 3.jpg/4.jpg or 'custom'."""
    print("\nWhich image pair for selection?")
    print("  [1] 1.jpg and 2.jpg -> saves 1.npy, 2.npy")
    print("  [2] Pair 2 images (3.jpeg/4.jpeg or 3.jpg/4.jpg) -> saves 3.npy, 4.npy")
    print("  [c] custom1.jpg and custom2.jpg -> saves custom1.npy, custom2.npy")
    while True:
        c = input("Enter choice [1/2/c]: ").strip().lower()
        if c in ("1", "12"):
            return "12"
        if c in ("2", "34"):
            return "34"
        if c in ("c", "custom"):
            return "custom"
        print("Please enter 1, 2, or c.")


def run_interactive_selection(base_dir: str | Path | None = None) -> None:
    """CLI entry: choose pair, then manual selection and save."""
    base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent

    pair = prompt_image_pair()
    if pair == "12":
        select_correspondences(
            base_dir / "1.jpg",
            base_dir / "2.jpg",
            base_dir / "1.npy",
            base_dir / "2.npy",
            title_left="1.jpg — click first point of pair",
            title_right="2.jpg — click matching point (same order)",
        )
    elif pair == "34":
        p_in, p_ref = pair2_image_paths(base_dir)
        if not p_in.is_file() or not p_ref.is_file():
            raise FileNotFoundError(
                "Pair 2: need both 3.jpeg & 4.jpeg, or both 3.jpg & 4.jpg in this folder."
            )
        select_correspondences(
            p_in,
            p_ref,
            base_dir / "3.npy",
            base_dir / "4.npy",
            title_left=f"{p_in.name} — click first point of pair",
            title_right=f"{p_ref.name} — click matching point (same order)",
        )
    else:
        select_correspondences(
            base_dir / "custom1.jpg",
            base_dir / "custom2.jpg",
            base_dir / "custom1.npy",
            base_dir / "custom2.npy",
            title_left="custom1.jpg — click first point of pair",
            title_right="custom2.jpg — click matching point (same order)",
        )


if __name__ == "__main__":
    run_interactive_selection()
