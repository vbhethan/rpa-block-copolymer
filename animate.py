"""
Animate the density profile evolution from a Cahn-Hilliard simulation HDF5 file.

Usage
-----
    python animate.py                          # display animation
    python animate.py simulation.h5 -o movie.mp4 --fps 20
    python animate.py simulation.h5 -o movie.gif --fps 10
"""

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


def compute_dominant_component(
    delta_phi: np.ndarray, block_fractions: np.ndarray
) -> np.ndarray:
    """Map density fluctuations to the index of the locally dominant species."""
    rho = delta_phi + block_fractions[:, np.newaxis, np.newaxis]
    return np.argmax(rho, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Animate density profile evolution.")
    parser.add_argument(
        "input", nargs="?", default="simulation.h5", help="HDF5 trajectory file"
    )
    parser.add_argument(
        "-o", "--output", default="movie.gif", help="Save animation to file (mp4/gif)"
    )
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument(
        "--dpi", type=int, default=150, help="Resolution for saved file"
    )
    parser.add_argument(
        "--no-tile",
        action="store_true",
        help="Show single unit cell instead of 2x2 tiling",
    )
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        block_fractions = f.attrs["block_fractions"]
        n_components = f.attrs["n_components"]
        box_lengths = f["box_lengths"][:]
        t = f["t"][:]
        F = f["F"][:]
        phi = f["phi"][:]

    n_frames = phi.shape[0]
    Lx = box_lengths[0] * 2
    Ly = box_lengths[1] * 2

    # Pre-compute all dominant-component maps
    dom = np.stack(
        [compute_dominant_component(phi[i], block_fractions) for i in range(n_frames)]
    )
    if not args.no_tile:
        dom = np.tile(dom, (1, 2, 2))

    # Set up figure
    fig, (ax_img, ax_F) = plt.subplots(
        1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1, 1]}
    )
    fig.subplots_adjust(wspace=0.35)

    im = ax_img.imshow(
        dom[0],
        cmap="tab10",
        vmin=0,
        vmax=n_components - 1,
        interpolation="nearest",
        extent=[0, Ly, 0, Lx],
    )
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    title = ax_img.set_title(f"t = {t[0]:.3e}", fontsize=11)

    # Free energy trace
    ax_F.plot(t, F, color="0.75", lw=1)
    (marker,) = ax_F.plot(t[0], F[0], "o", color="tab:red", ms=6, zorder=5)
    ax_F.set_xlabel("t")
    ax_F.set_ylabel("F")
    ax_F.set_title("Free energy", fontsize=11)

    def update(frame_idx):
        im.set_data(dom[frame_idx])
        title.set_text(f"t = {t[frame_idx]:.3e}")
        marker.set_data([t[frame_idx]], [F[frame_idx]])
        return im, title, marker

    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // args.fps, blit=True
    )

    if args.output:
        ext = args.output.rsplit(".", 1)[-1].lower()
        if ext == "gif":
            writer = PillowWriter(fps=args.fps)
        else:
            writer = FFMpegWriter(fps=args.fps)
        anim.save(args.output, writer=writer, dpi=args.dpi)
        print(f"Saved {n_frames}-frame animation to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
