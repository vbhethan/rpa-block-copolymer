import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return


@app.cell
def _():
    import numpy as np
    import torch
    from rpa import BlockCopolymerFreeEnergy
    from rpa.simulation_io import SimulationData
    from rpa.visualize import plot_diblock_isosurface
    from rpa.optimizers import optimize_box_only, optimize_phi_only

    return (
        BlockCopolymerFreeEnergy,
        SimulationData,
        np,
        optimize_box_only,
        optimize_phi_only,
        plot_diblock_isosurface,
        torch,
    )


@app.cell
def _(torch):
    N = 100
    f_A = 0.2
    f_B = 1.0 - f_A
    CHI_AB = 30.0
    phi_bar = 1.0
    grid_shape = (32, 32, 32)
    box_lengths = (20, 20.0, 20.0)
    DTYPE = torch.float32
    block_fractions = torch.tensor([f_A, f_B], dtype=DTYPE)
    chi_matrix = torch.tensor([[0.0, CHI_AB], [CHI_AB, 0.0]], dtype=DTYPE)
    l_ij_matrix = torch.zeros((2, 2), dtype=DTYPE)
    return (
        DTYPE,
        N,
        block_fractions,
        box_lengths,
        chi_matrix,
        grid_shape,
        l_ij_matrix,
        phi_bar,
    )


@app.cell
def _(
    BlockCopolymerFreeEnergy,
    DTYPE,
    N,
    SimulationData,
    block_fractions,
    box_lengths,
    chi_matrix,
    grid_shape,
    l_ij_matrix,
    np,
    phi_bar,
    torch,
):
    samples = torch.load("./examples/load-density/samples.pt", weights_only=False)
    FRAME = 0

    # samples are in [-1, 1]; map to physical density in [0, 1]
    phi_A_raw = (samples[FRAME, 0] + 1.0) / 2.0   # (32, 32, 32)
    phi_B_raw = 1.0 - phi_A_raw
    phi = torch.stack([phi_A_raw, phi_B_raw], dim=0)  # (2, 32, 32, 32)

    model = BlockCopolymerFreeEnergy(
        N=N,
        block_fractions=block_fractions,
        chi_matrix=chi_matrix,
        l_ij_matrix=l_ij_matrix,
        phi_bar=phi_bar,
        grid_shape=grid_shape,
        box_lengths=box_lengths,
        dtype=DTYPE,
    )
    model.set_density(phi)

    # Package as a one-frame SimulationData for visualization and chaining
    initial_data = SimulationData.from_model(model)
    initial_data.phi = model.get_order_parameters().numpy()[np.newaxis]
    initial_data.F = np.array([model().item()])
    initial_data.box_lengths = np.array([model.L.detach().cpu().tolist()])

    print(f"Loaded frame {FRAME} | F = {initial_data.F[0]:.4f}")
    return (initial_data,)


@app.cell
def _(initial_data, plot_diblock_isosurface):
    fig_initial = plot_diblock_isosurface(
        initial_data, frame=0, tile=2, periodic_pad=1, title="Loaded density"
    )
    fig_initial
    return


@app.cell
def _(DTYPE, initial_data, optimize_box_only):
    model_box = initial_data.build_model(optimize_box=True, dtype=DTYPE)
    result_box = optimize_box_only(
        model_box, n_steps=200, lr_box=1.0, box_grad_scale=10.0, tol_grad=1e-4
    )
    print(f"Box: {result_box.box_lengths[0].tolist()} → {result_box.box_lengths[-1].tolist()}")
    return (result_box,)


@app.cell
def _(plot_diblock_isosurface, result_box):
    fig_box = plot_diblock_isosurface(
        result_box, frame=-1, tile=2, periodic_pad=1, title="After box optimization"
    )
    fig_box
    return


@app.cell
def _(DTYPE, optimize_phi_only, result_box):
    model_phi = result_box.build_model(optimize_box=False, dtype=DTYPE)
    result_phi = optimize_phi_only(model_phi, n_steps=2000, lr_phi=1.0, tol_F=1e-4)
    return (result_phi,)


@app.cell
def _(plot_diblock_isosurface, result_phi):
    fig_phi = plot_diblock_isosurface(
        result_phi, frame=-1, tile=2, periodic_pad=1, title="After phi optimization"
    )
    fig_phi
    return


if __name__ == "__main__":
    app.run()
