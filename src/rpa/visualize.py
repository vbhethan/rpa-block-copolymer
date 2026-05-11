"""
Interactive plotly isosurface visualization for block copolymer density fields.

The returned ``go.Figure`` is directly renderable in a Jupyter notebook (either
as the last expression in a cell, or via ``fig.show()``) and can be written to
a self-contained HTML file by passing ``output="path/to/file.html"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .simulation_io import SimulationData


def _extract_volume(
    data: SimulationData | np.ndarray,
    component: int,
    frame: int,
) -> tuple[np.ndarray, tuple[float, ...] | None]:
    """
    Return (scalar_volume, box_lengths) from any supported input type.

    Accepted forms for ``data``:
    - SimulationData: uses phi[frame, component] (delta_phi field, interface at 0)
    - ndarray shape (n_components, Nx, Ny, Nz): selects data[component]
    - ndarray shape (Nx, Ny, Nz): used as-is (component ignored)

    box_lengths is None when data is a raw array (axes will be voxel indices).
    """
    if isinstance(data, SimulationData):
        if data.ndim != 3:
            raise ValueError(
                f"plot_diblock_isosurface requires a 3-D SimulationData "
                f"(got ndim={data.ndim})"
            )
        vol = data.phi[frame, component].astype(np.float32)
        box_lengths = tuple(float(v) for v in data.box_lengths[frame])
        return vol, box_lengths

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3:
        return arr, None
    if arr.ndim == 4:
        return arr[component], None
    raise ValueError(
        f"Raw array must be 3-D (Nx, Ny, Nz) or 4-D (n_components, Nx, Ny, Nz), "
        f"got shape {arr.shape}"
    )


def _expand_periodic(vol: np.ndarray, pad: int, tile: int | tuple[int, int, int]) -> np.ndarray:
    if isinstance(tile, int):
        tile = (tile, tile, tile)
    if any(t > 1 for t in tile):
        vol = np.tile(vol, tile)
    if pad > 0:
        vol = np.pad(vol, pad_width=pad, mode="wrap")
    return vol


def plot_diblock_isosurface(
    data: SimulationData | np.ndarray,
    frame: int = -1,
    component: int = 0,
    isovalue: float = 0.0,
    surface_count: int = 1,
    isomin: float | None = None,
    isomax: float | None = None,
    periodic_pad: int = 1,
    tile: int | tuple[int, int, int] = 1,
    colorscale: str | list | None = None,
    opacity: float = 0.6,
    title: str | None = None,
    output: str | None = None,
) -> go.Figure:
    """
    Render an interactive plotly isosurface of a single density component.

    Parameters
    ----------
    data : SimulationData or np.ndarray
        Source of the density field. A ``SimulationData`` uses the stored
        ``phi`` (delta_phi) at the given ``frame``, so the natural interface
        isovalue is 0.0 (the mean-density contour).  A raw array of shape
        ``(n_components, Nx, Ny, Nz)`` or ``(Nx, Ny, Nz)`` is also accepted.
    frame : int
        Trajectory frame to visualize (only used with SimulationData).
    component : int
        Which density component to render (default 0 = A-block for diblock).
    isovalue : float
        Isosurface level drawn when ``surface_count == 1``.
    surface_count : int
        Number of isosurfaces drawn between ``isomin`` and ``isomax``.
    isomin, isomax : float, optional
        Range for isosurface levels. Defaults to ``isovalue`` (single-surface
        mode) or the field min/max (multi-surface mode).
    periodic_pad : int
        Wrap-pad voxels added per axis so surfaces close across the periodic
        boundary.
    tile : int or (int, int, int)
        Number of unit-cell copies to show along each axis.
    colorscale : str or list, optional
        Plotly colorscale name (e.g. ``"RdBu"``) or an explicit scale list.
        Defaults to a solid royal-blue surface with the colorbar hidden.
    opacity : float
        Surface opacity (0–1).
    title : str, optional
        Figure title. Auto-generated from ``isovalue`` if not provided.
    output : str, optional
        If given, write the figure to this path as a self-contained HTML file.
        The figure is always returned regardless.

    Returns
    -------
    go.Figure
        Plotly figure. Display inline in Jupyter with ``fig.show()`` or by
        making it the last expression in a cell. Save to HTML with
        ``fig.write_html("path.html")`` or via the ``output`` argument.
    """
    vol, box_lengths = _extract_volume(data, component, frame)

    tile_tuple = (tile, tile, tile) if isinstance(tile, int) else tuple(tile)
    vol = _expand_periodic(vol, periodic_pad, tile_tuple)

    nz, ny, nx = vol.shape

    # Physical or voxel coordinate axes
    if box_lengths is not None:
        Lx = box_lengths[0] * tile_tuple[0]
        Ly = box_lengths[1] * tile_tuple[1]
        Lz = box_lengths[2] * tile_tuple[2]
        # Shift so the unpadded cell starts at 0
        dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
        x = np.arange(nx, dtype=np.float32) * dx - periodic_pad * dx
        y = np.arange(ny, dtype=np.float32) * dy - periodic_pad * dy
        z = np.arange(nz, dtype=np.float32) * dz - periodic_pad * dz
        axis_labels = ("x", "y", "z")
    else:
        x = np.arange(nx, dtype=np.float32) - periodic_pad
        y = np.arange(ny, dtype=np.float32) - periodic_pad
        z = np.arange(nz, dtype=np.float32) - periodic_pad
        axis_labels = ("i", "j", "k")

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    if surface_count == 1:
        _isomin = isovalue if isomin is None else isomin
        _isomax = isovalue if isomax is None else isomax
    else:
        _isomin = float(vol.min()) if isomin is None else isomin
        _isomax = float(vol.max()) if isomax is None else isomax

    solid_blue = colorscale is None
    _colorscale = [[0.0, "royalblue"], [1.0, "royalblue"]] if solid_blue else colorscale

    iso = go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=vol.flatten(),
        isomin=_isomin,
        isomax=_isomax,
        surface_count=surface_count,
        colorscale=_colorscale,
        showscale=not solid_blue,
        opacity=opacity,
        caps=dict(x_show=False, y_show=False, z_show=False),
    )

    _title = title or (
        f"component {component}  iso={isovalue:.3f}  "
        f"tile={tile_tuple}  pad={periodic_pad}"
    )

    fig = go.Figure(data=iso)
    fig.update_layout(
        title=_title,
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            aspectmode="data",
        ),
    )

    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out)

    return fig
