import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Ordered morphologies from the Picard solver

    This notebook uses the grand-canonical **fixed-point (Picard) solver**
    `picard_optimize_phi_only` to relax seeded initial conditions into ordered
    block-copolymer morphologies at fixed box: **lamellae**, **hexagonal
    cylinders**, and **BCC spheres**.

    The Picard iteration is
    `rho_i = f_i * phi_bar * exp(-f_i * n_grid * dF_res/d(delta_phi_i))`,
    Fourier-**preconditioned** to tame the `k^2` stiffness of the ideal-chain
    vertex. Because the `exp` map keeps every density positive, it reaches
    strongly-ordered states where projected gradient descent's `log rho` term
    diverges. It relaxes to the **metastable fixed point of the seed** (like
    SCFT), so we seed each target symmetry with reciprocal-lattice stars from
    `rpa.initial_conditions`.

    > Note: incompressibility `sum_i rho_i = phi_bar` is *not* enforced (pure
    > grand canonical), so a modest `drift` is expected and reported per run.
    """)
    return


@app.cell
def _():
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import math

    import numpy as np
    import torch

    from rpa import BlockCopolymerFreeEnergy
    from rpa.optimizers import picard_optimize_phi_only
    from rpa.initial_conditions import (
        generate_fourier_mode_initial_conditions,
        LAMELLAR_Z,
        BCC_110_STAR,
    )
    from rpa.visualize import plot_diblock_isosurface

    # Radius of gyration sets the natural length scale for the box.
    Rg = math.sqrt(100 / 6)

    # Hexagonal first-star modes in the rectangular (Ly = sqrt(3) Lx) unit cell.
    HEX_STAR = [(1, 1, 0), (1, -1, 0), (0, 2, 0)]
    return (
        BCC_110_STAR,
        BlockCopolymerFreeEnergy,
        HEX_STAR,
        LAMELLAR_Z,
        Rg,
        generate_fourier_mode_initial_conditions,
        math,
        np,
        picard_optimize_phi_only,
        plot_diblock_isosurface,
        torch,
    )


@app.cell
def _(
    BlockCopolymerFreeEnergy,
    generate_fourier_mode_initial_conditions,
    np,
    picard_optimize_phi_only,
    torch,
):
    def relax_morphology(f_A, chi, box_lengths, star, grid_shape, n_steps=4000, alpha=0.05):
        """Build a diblock model, seed a symmetry, and relax it with Picard.

        Returns the populated ``SimulationData`` (its last frame is the relaxed
        morphology). ``star`` is a list of reciprocal-lattice Miller indices.
        """
        chi_matrix = torch.tensor([[0.0, chi], [chi, 0.0]], dtype=torch.float64)
        model = BlockCopolymerFreeEnergy(
            N=100,
            chi_matrix=chi_matrix,
            l_ij_matrix=torch.zeros((2, 2), dtype=torch.float64),
            block_fractions=torch.tensor([f_A, 1.0 - f_A], dtype=torch.float64),
            grid_shape=grid_shape,
            box_lengths=box_lengths,
            optimize_box=False,
            dtype=torch.float64,
        )
        seed = generate_fourier_mode_initial_conditions(model, star, amplitude=0.3)
        model.delta_phi.data.copy_(torch.tensor(seed.phi[0]))

        result = picard_optimize_phi_only(
            model,
            n_steps=n_steps,
            alpha=alpha,
            precondition=True,
            tol=1e-7,
            patience=200,
            record_every=200,
            silent=True,
        )
        amp = float(np.abs(result.phi[-1]).max())
        print(
            f"F = {result.F[-1]:.3f} | amp = {amp:.3f} | "
            f"drift = {result.incompressibility_drift[-1]:.2e} | "
            f"converged = {result.converged}"
        )
        return result

    return (relax_morphology,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Lamellae — symmetric diblock (f_A = 0.5, chi = 40)

    Seeded with a single `(0,0,1)` mode in a box two lamellar periods deep.
    """)
    return


@app.cell
def _(LAMELLAR_Z, Rg, relax_morphology):
    lam = relax_morphology(
        f_A=0.5,
        chi=40.0,
        box_lengths=(Rg, Rg, 2.0 * Rg),
        grid_shape=(16, 16, 32),
        star=LAMELLAR_Z,
    )
    return (lam,)


@app.cell
def _(lam, plot_diblock_isosurface):
    plot_diblock_isosurface(
        lam, frame=-1, component=0, tile=(1, 1, 2), periodic_pad=1, title="Lamellae"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Hexagonal cylinders — asymmetric diblock (f_A = 0.35, chi = 30)

    A-block cylinders in a B matrix, seeded with the three first-star modes of
    the 2-D hexagonal lattice in a rectangular `Ly = sqrt(3) Lx` cell.
    """)
    return


@app.cell
def _(HEX_STAR, Rg, math, relax_morphology):
    hex_result = relax_morphology(
        f_A=0.35,
        chi=30.0,
        box_lengths=(2.5 * Rg, math.sqrt(3) * 2.5 * Rg, 2.5 * Rg),
        grid_shape=(32, 32, 32),
        star=HEX_STAR,
    )
    return (hex_result,)


@app.cell
def _(hex_result, plot_diblock_isosurface):
    plot_diblock_isosurface(
        hex_result,
        frame=-1,
        component=0,
        tile=(2, 2, 1),
        periodic_pad=1,
        title="Hexagonal cylinders",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## BCC spheres — asymmetric diblock (f_A = 0.25, chi = 40)

    A-block spheres on a body-centered-cubic lattice, seeded with the `{110}`
    star in a cubic cell.
    """)
    return


@app.cell
def _(BCC_110_STAR, Rg, relax_morphology):
    bcc = relax_morphology(
        f_A=0.25,
        chi=40.0,
        box_lengths=(2.5 * Rg, 2.5 * Rg, 2.5 * Rg),
        grid_shape=(32, 32, 32),
        star=BCC_110_STAR,
    )
    return (bcc,)


@app.cell
def _(bcc, plot_diblock_isosurface):
    plot_diblock_isosurface(
        bcc, frame=-1, component=0, tile=(2, 2, 2), periodic_pad=1, title="BCC spheres"
    )
    return


if __name__ == "__main__":
    app.run()
