import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import os

    return (os,)


@app.cell
def _(os):
    #! export KMP_DUPLICATE_LIB_OK=TRUE
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return


@app.cell
def _():
    from rpa.simulation_io import SimulationData
    from rpa.visualize import plot_diblock_isosurface
    import numpy as np
    import torch
    from rpa.optimizers import scipy_optimize_box_only, optimize_box_only

    return SimulationData, optimize_box_only, plot_diblock_isosurface, torch


@app.cell
def _(SimulationData, optimize_box_only):
    data = SimulationData.from_hdf5("./examples/input.h5")
    print(data.grid_shape)
    print(data.chi_matrix)
    print(data.block_fractions)
    print(data.box_lengths)
    # Optimize the box first to get a better starting point
    _model = data.build_model(optimize_box=True)
    data = optimize_box_only(_model)
    return (data,)


@app.cell
def _(data, plot_diblock_isosurface):
    plot_diblock_isosurface(data, tile=1, periodic_pad=1)
    return


@app.cell
def _(data, plot_diblock_isosurface, torch):
    from rpa.optimizers import scipy_optimize_joint
    _model = data.build_model(optimize_box=True, dtype=torch.float32)
    result_scipy = scipy_optimize_joint(_model, patience_outer=25, method_phi="CG", method_box="CG")
    plot_diblock_isosurface(result_scipy, frame=-1, tile=1)
    return (result_scipy,)


@app.cell
def _(result_scipy):
    from rpa.pscf_io import write_C_RGRID_from_array
    _fields = result_scipy.phi + result_scipy.block_fractions[None, :, None, None, None]
    _cell_params = result_scipy.box_lengths[-1]
    _mesh = result_scipy.grid_shape
    write_C_RGRID_from_array("./example_scp.rf", _fields[-1], _mesh, _cell_params)
    return


if __name__ == "__main__":
    app.run()
