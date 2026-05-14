import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import subprocess
    import os

    return (os,)


@app.cell
def _(os):
    #! export KMP_DUPLICATE_LIB_OK=TRUE
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return


@app.cell
def _():
    from rpa.free_energy import BlockCopolymerFreeEnergy
    from rpa.simulation_io import SimulationData
    from rpa.visualize import plot_diblock_isosurface
    import torch

    return SimulationData, plot_diblock_isosurface, torch


@app.cell
def _(SimulationData):
    data = SimulationData.from_hdf5("./examples/input.h5")
    return (data,)


@app.cell
def _(data, plot_diblock_isosurface):
    fig = plot_diblock_isosurface(data, tile=2, periodic_pad=1)
    fig
    return


@app.cell
def _(data, torch):
    from rpa.optimizers import optimize_box_only
    model = data.build_model(optimize_box=True, dtype=torch.float32)
    result_box = optimize_box_only(model, n_steps=500, lr_box=1.0, box_grad_scale=10.0, tol_grad=1e-4)
    return (result_box,)


@app.cell
def _(result_box):
    print(result_box.box_lengths[0])
    print(result_box.box_lengths[-1])
    return


@app.cell
def _(plot_diblock_isosurface, result_box):
    fig_1 = plot_diblock_isosurface(result_box, frame=-1, tile=2, periodic_pad=1)
    fig_1
    return


@app.cell
def _(result_box, torch):
    from rpa.optimizers import optimize_phi_only
    model_1 = result_box.build_model(optimize_box=False, dtype=torch.float32)
    result_phi = optimize_phi_only(model_1, n_steps=6000, lr_phi=1.0, tol_F=0.0001)
    return (result_phi,)


@app.cell
def _(plot_diblock_isosurface, result_phi):
    fig_2 = plot_diblock_isosurface(result_phi, frame=-1, tile=1, periodic_pad=1)
    fig_2
    return


if __name__ == "__main__":
    app.run()
