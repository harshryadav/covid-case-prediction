"""
MSML640 Assignment 4 — Image mosaics (main entry).
Imports helper modules, optionally collects correspondences, builds homographies, warps, and displays results.
Run from the folder that contains the images and .npy files (no absolute paths).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import computeH
import selectPoints
import warpImage


def _base_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path.name} (place it next to this script).")
    return plt.imread(str(path))


def _load_points(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Run manual point selection or add the provided .npy files."
        )
    t = np.load(str(path))
    if t.ndim != 2 or t.shape[0] != 2:
        raise ValueError(f"{path.name} must be a 2xN array (rows: x, y).")
    if t.shape[1] < 4:
        raise ValueError(f"{path.name} must contain at least 4 points.")
    return t.astype(np.float64)


def run_stitch(
    input_path: Path,
    ref_path: Path,
    npy_input: Path,
    npy_ref: Path,
    section_title: str,
) -> None:
    """
    inputIm is warped into refIm's plane; npy files are correspondences (same order):
    t_input from input image, t_ref from reference image.
    """
    input_im = _load_image(input_path)
    ref_im = _load_image(ref_path)
    t_in = _load_points(npy_input)
    t_ref = _load_points(npy_ref)

    H = computeH.computeH(t_in, t_ref)

    ref_display = ref_im
    if ref_display.dtype == np.float32 or ref_display.dtype == np.float64:
        if ref_display.max() <= 1.0:
            ref_display = (np.clip(ref_display, 0, 1) * 255).astype(np.uint8)

    computeH.verify_homography_overlay(
        t_in,
        t_ref,
        H,
        ref_display,
        title=f"{section_title}: true reference points vs H mapped from input",
    )

    warp_im, merge_im = warpImage.warpImage(input_im, ref_im, H)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(input_im)
    axes[0, 0].set_title(f"{section_title}: input image ({input_path.name})")
    axes[0, 0].set_xlabel("x (pixels)")
    axes[0, 0].set_ylabel("y (pixels)")

    axes[0, 1].imshow(ref_im)
    axes[0, 1].set_title(f"Reference image ({ref_path.name})")
    axes[0, 1].set_xlabel("x (pixels)")
    axes[0, 1].set_ylabel("y (pixels)")

    axes[1, 0].imshow(warp_im)
    axes[1, 0].set_title("Warped input in reference frame (inverse warp)")
    axes[1, 0].set_xlabel("x (pixels)")
    axes[1, 0].set_ylabel("y (pixels)")

    axes[1, 1].imshow(merge_im)
    axes[1, 1].set_title("Mosaic (blended overlap)")
    axes[1, 1].set_xlabel("x (pixels)")
    axes[1, 1].set_ylabel("y (pixels)")
    plt.suptitle(section_title, fontsize=12)
    plt.tight_layout()
    plt.show()


def main() -> None:
    base = _base_dir()

    mode = selectPoints.prompt_generate_or_load()
    if mode == "manual":
        selectPoints.run_interactive_selection(base)

    print("\n--- Assignment image pair ---")
    print("Choose which pair to stitch (uses matching .npy files).")
    print("  [1] 1.jpg (input) -> 2.jpg (reference): 1.npy, 2.npy")
    p2_in, p2_ref = selectPoints.pair2_image_paths(base)
    print(f"  [2] {selectPoints.pair2_names_for_display((p2_in, p2_ref))}: 3.npy, 4.npy")
    while True:
        p = input("Enter choice [1/2]: ").strip()
        if p == "1":
            run_stitch(
                base / "1.jpg",
                base / "2.jpg",
                base / "1.npy",
                base / "2.npy",
                section_title="Pair 1: mosaic of 1.jpg warped to 2.jpg",
            )
            break
        if p == "2":
            if not p2_in.is_file() or not p2_ref.is_file():
                print(
                    "Pair 2 images not found. Add 3.jpeg & 4.jpeg (or 3.jpg & 4.jpg) to this folder."
                )
                continue
            run_stitch(
                p2_in,
                p2_ref,
                base / "3.npy",
                base / "4.npy",
                section_title=f"Pair 2: mosaic of {p2_in.name} warped to {p2_ref.name}",
            )
            break
        print("Please enter 1 or 2.")

    custom_in = base / "custom1.jpg"
    custom_ref = base / "custom2.jpg"
    custom_n1 = base / "custom1.npy"
    custom_n2 = base / "custom2.npy"

    if custom_in.is_file() and custom_ref.is_file():
        if custom_n1.is_file() and custom_n2.is_file():
            print("\n--- Custom mosaic ---")
            run_stitch(
                custom_in,
                custom_ref,
                custom_n1,
                custom_n2,
                section_title="Custom pair: custom1.jpg warped to custom2.jpg",
            )
        else:
            print(
                "\nCustom images found but custom1.npy / custom2.npy are missing. "
                "Run this script, choose manual selection, then option [c] to create them."
            )
    else:
        print(
            "\nSkipping custom mosaic: add custom1.jpg and custom2.jpg (and matching .npy) to this folder."
        )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
