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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from simulation_io import SimulationData


def compute_dominant_component(
    delta_phi: np.ndarray, block_fractions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rho = delta_phi + block_fractions[:, np.newaxis, np.newaxis]
    dominant_component = np.argmax(rho, axis=0)
    # Works for any spatial dimensionality (1D, 2D, 3D)
    spatial_shape = dominant_component.shape
    flat_dom = dominant_component.ravel()
    flat_rho = rho.reshape(rho.shape[0], -1)
    dominant_rho = flat_rho[flat_dom, np.arange(flat_dom.size)].reshape(spatial_shape)
    alpha = (dominant_rho - dominant_rho.min()) / (
        dominant_rho.max() - dominant_rho.min() + 1e-12
    )
    return dominant_component, alpha


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
    data = SimulationData.from_hdf5(args.input)
    block_fractions = data.block_fractions
    box_lengths = data.box_lengths
    F_values = data.F
    phi = data.phi

    n_frames = data.n_frames
    n_tile = 1 if args.no_tile else 2

    # Pre-compute dominant-component maps for all frames
    dom_frames = []
    alpha_frames = []
    for i in range(n_frames):
        d, alpha = compute_dominant_component(phi[i], block_fractions)
        if n_tile > 1:
            d = np.tile(d, (n_tile, n_tile))
            alpha = np.tile(alpha, (n_tile, n_tile))
        dom_frames.append(d)
        alpha_frames.append(alpha)

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
        vmax=10,
        interpolation="nearest",
        origin="lower",
        extent=[0, Lx0, 0, Ly0],
        alpha=alpha_frames[0],
    )
    ax.set_xlim(0, Lx0)
    ax.set_ylim(0, Ly0)

    title = ax.set_title(
        _frame_title(0, box_lengths[0], F_values[0]),
        fontsize=10,
    )

    # ------------------------------------------------------------------
    # Update function
    # ------------------------------------------------------------------
    def update(i):
        Lx_i = box_lengths[i, 0] * n_tile
        Ly_i = box_lengths[i, 1] * n_tile

        im.set_data(dom_frames[i])
        im.set_alpha(alpha_frames[i])
        im.set_extent([0, Lx_i, 0, Ly_i])
        ax.set_xlim(0, Lx_i)
        ax.set_ylim(0, Ly_i)
        title.set_text(_frame_title(i, box_lengths[i], F_values[i]))
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


def _frame_title(frame: int, box: np.ndarray, F: float) -> str:
    Lx, Ly = box[0], box[1]
    return f"Frame {frame}   Lx={Lx:.3f}  Ly={Ly:.3f}   F={F:.6f}"


if __name__ == "__main__":
    main()
