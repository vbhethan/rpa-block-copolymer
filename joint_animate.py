"""
Animate the joint dynamics+box-optimization trajectory from a HDF5 file.

Reads the output of joint_phi_box_optimize.py and produces a density-map
animation where the plot aspect ratio updates each frame to match the
current physical box dimensions (Lx × Ly).  The figure canvas stays the
same size; whitespace is added as needed when the box is non-square.

Usage
-----
    python joint_animate.py                              # display
    python joint_animate.py joint_optimization.h5
    python joint_animate.py joint_optimization.h5 -o movie.gif --fps 5
    python joint_animate.py joint_optimization.h5 -o movie.mp4 --fps 10
    python joint_animate.py joint_optimization.h5 --no-tile
"""

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


def compute_dominant_component(
    delta_phi: np.ndarray, block_fractions: np.ndarray
) -> np.ndarray:
    """Return the index of the locally dominant species at every grid point."""
    rho = delta_phi + block_fractions[:, np.newaxis, np.newaxis]
    return np.argmax(rho, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Animate joint dynamics+box-opt trajectory with variable aspect ratio."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="joint_optimization.h5",
        help="HDF5 file from joint_phi_box_optimize.py (default: joint_optimization.h5)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="joint_movie.gif",
        help="Output file path (mp4 or gif, default: joint_movie.gif)",
    )
    parser.add_argument(
        "--fps", type=int, default=5, help="Frames per second (default: 5)"
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="Output resolution (default: 150)"
    )
    parser.add_argument(
        "--no-tile",
        action="store_true",
        help="Show single unit cell instead of 2×2 tiling",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load trajectory
    # ------------------------------------------------------------------
    with h5py.File(args.input, "r") as f:
        block_fractions = np.array(f.attrs["block_fractions"])
        n_components = int(f.attrs["n_components"])
        outer_idx = f["outer_idx"][:]  # (n_frames,)
        box_lengths = f["box_lengths"][:]  # (n_frames, ndim)
        F_after_box = f["F_after_box"][:]  # (n_frames,)
        phi = f["phi"][:]  # (n_frames, n_comp, Nx, Ny)

    n_frames = phi.shape[0]
    n_tile = 1 if args.no_tile else 2

    # Pre-compute dominant-component maps for all frames
    dom_frames = []
    for i in range(n_frames):
        d = compute_dominant_component(phi[i], block_fractions)
        if n_tile > 1:
            d = np.tile(d, (n_tile, n_tile))
        dom_frames.append(d)

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    fig_size = 5.5  # inches — fixed canvas size
    fig = plt.figure(figsize=(fig_size, fig_size))
    # leave a margin so axis labels/title don't get clipped
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])

    # Equal physical aspect; 'box' means the axes box (not data limits) resizes
    ax.set_aspect("equal", adjustable="box")
    # ax.set_xticks([])
    # ax.set_yticks([])

    # Initial frame
    Lx0 = box_lengths[0, 0] * n_tile
    Ly0 = box_lengths[0, 1] * n_tile

    im = ax.imshow(
        dom_frames[0],
        cmap="tab10",
        vmin=0,
        vmax=n_components - 1,
        interpolation="nearest",
        origin="lower",
        extent=[0, Lx0, 0, Ly0],
    )
    ax.set_xlim(0, Lx0)
    ax.set_ylim(0, Ly0)

    title = ax.set_title(
        _frame_title(outer_idx[0], box_lengths[0], F_after_box[0]),
        fontsize=10,
    )

    # ------------------------------------------------------------------
    # Update function
    # ------------------------------------------------------------------
    def update(i):
        Lx_i = box_lengths[i, 0] * n_tile
        Ly_i = box_lengths[i, 1] * n_tile

        im.set_data(dom_frames[i])
        im.set_extent([0, Lx_i, 0, Ly_i])
        ax.set_xlim(0, Lx_i)
        ax.set_ylim(0, Ly_i)
        title.set_text(_frame_title(outer_idx[i], box_lengths[i], F_after_box[i]))
        return (im, title)

    # blit=False because axes geometry changes each frame
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 // args.fps,
        blit=False,
    )

    # ------------------------------------------------------------------
    # Save or display
    # ------------------------------------------------------------------
    if args.output:
        ext = args.output.rsplit(".", 1)[-1].lower()
        writer = (
            PillowWriter(fps=args.fps) if ext == "gif" else FFMpegWriter(fps=args.fps)
        )
        anim.save(args.output, writer=writer, dpi=args.dpi)
        print(f"Saved {n_frames}-frame animation → {args.output}")
    else:
        plt.show()


def _frame_title(outer: int, box: np.ndarray, F: float) -> str:
    Lx, Ly = box[0], box[1]
    return f"Outer {outer}   Lx={Lx:.3f}  Ly={Ly:.3f}   F={F:.6f}"


if __name__ == "__main__":
    main()
