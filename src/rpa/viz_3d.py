import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from .simulation_io import SimulationData


def _extract_isosurface(
    data: SimulationData,
    frame: int,
    level: float,
    n_tiles: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared helper: compute rho_A, tile, run marching cubes.

    Returns (verts, faces, normals, box_dims) where box_dims = (Lx, Ly, Lz)
    after tiling.
    """
    phi = data.phi[frame]
    bf = data.block_fractions
    box_len = data.box_lengths[frame]

    rho_A = phi[0] + bf[0]

    if any(n > 1 for n in n_tiles):
        rho_A = np.tile(rho_A, n_tiles)

    nx, ny, nz = rho_A.shape
    Lx = box_len[0] * n_tiles[0]
    Ly = box_len[1] * n_tiles[1]
    Lz = box_len[2] * n_tiles[2]

    spacing = (Lx / nx, Ly / ny, Lz / nz)
    verts, faces, normals, _ = marching_cubes(rho_A, level=level, spacing=spacing)

    return verts, faces, normals, np.array([Lx, Ly, Lz])


def _setup_axes(
    fig: plt.Figure,
    ax: plt.Axes,
    box_dims: np.ndarray,
    level: float,
    title_prefix: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Set limits, labels, aspect ratio and title on a 3D axis."""
    Lx, Ly, Lz = box_dims

    if fig is None or ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((Lx, Ly, Lz))
    ax.set_title(rf"{title_prefix} $\phi_A = {level}$")

    return fig, ax


def visualize_3d_wireframe(
    data: SimulationData,
    frame: int = -1,
    level: float = 0.5,
    color: str = "green",
    linewidth: float = 0.3,
    n_tiles: tuple[int, int, int] = (1, 1, 1),
    fig: plt.Figure = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Wireframe rendering of the rho_A = *level* isosurface.

    Uses marching cubes to extract the mesh, then draws only the
    triangle edges.  Much faster to interact with than the filled
    isosurface because matplotlib only needs to depth-sort line
    segments, not alpha-blend polygons.
    """
    verts, faces, normals, box_dims = _extract_isosurface(data, frame, level, n_tiles)
    fig, ax = _setup_axes(fig, ax, box_dims, level, "Wireframe")

    mesh = Poly3DCollection(verts[faces], linewidths=linewidth)
    mesh.set_facecolor((0, 0, 0, 0))
    mesh.set_edgecolor(to_rgba(color, alpha=0.6))
    ax.add_collection3d(mesh)

    return fig, ax


def visualize_3d_isosurface(
    data: SimulationData,
    frame: int = -1,
    level: float = 0.5,
    color: str = "green",
    alpha: float = 0.6,
    n_tiles: tuple[int, int, int] = (1, 1, 1),
    fig: plt.Figure = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Filled isosurface rendering of the rho_A = *level* contour.

    Simple diffuse shading is applied so the surface reads well
    when rotated.
    """
    verts, faces, normals, box_dims = _extract_isosurface(data, frame, level, n_tiles)
    fig, ax = _setup_axes(fig, ax, box_dims, level, "Isosurface")

    light_dir = np.array([1.0, 1.0, 2.0])
    light_dir /= np.linalg.norm(light_dir)
    face_normals = normals[faces].mean(axis=1)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = np.divide(
        face_normals, norms, out=np.zeros_like(face_normals), where=norms > 1e-12
    )
    shade = np.clip(np.abs(face_normals @ light_dir), 0.25, 1.0)

    base_rgb = np.array(to_rgba(color)[:3])
    face_colors = np.empty((len(faces), 4))
    face_colors[:, :3] = base_rgb[None, :] * shade[:, None]
    face_colors[:, 3] = alpha

    mesh = Poly3DCollection(verts[faces])
    mesh.set_facecolor(face_colors)
    mesh.set_edgecolor(to_rgba(color, alpha=0.05))
    ax.add_collection3d(mesh)

    return fig, ax


