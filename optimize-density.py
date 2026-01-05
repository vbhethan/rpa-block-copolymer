from scipy.optimize import minimize
import numpy as np
from rpa import ABCBlockCopolymer, phi_grid
import matplotlib.pyplot as plt
import pickle


def create_optimization_problem(copolymer):
    """
    Create an optimization problem for minimizing free energy with constraints.

    Parameters:
    -----------
    copolymer : ABCBlockCopolymer
        The copolymer instance to optimize

    Returns:
    --------
    objective_function : callable
        Function that takes flattened delta_phi and returns free energy
    constraint_functions : list
        List of constraint dictionaries for scipy.optimize.minimize
    initial_guess : np.ndarray
        Flattened initial delta_phi values
    """
    grid_shape = copolymer.phi_grid.grid_shape
    n_components = copolymer.n_components
    block_fractions = copolymer.block_fractions

    # Store original copolymer parameters for use in objective
    original_chi_matrix = copolymer.chi_matrix.copy()
    original_block_lengths = copolymer.block_lengths.copy()
    original_kuhn_length = copolymer.kuhn_length
    original_box_length = copolymer.phi_grid.box_length

    def objective_function(flattened_delta_phi):
        """
        Objective function: compute free energy for given delta_phi values.

        Parameters:
        -----------
        flattened_delta_phi : np.ndarray
            Flattened array of shape (n_components * grid_size,)

        Returns:
        --------
        free_energy : float
            The free energy functional value
        """
        # Reshape to (n_components, *grid_shape)
        delta_phi_reshaped = flattened_delta_phi.reshape(n_components, *grid_shape)

        # Compute phi_values = delta_phi + block_fractions
        # Reshape block_fractions for broadcasting
        block_fractions_reshaped = block_fractions.reshape(
            n_components, *([1] * len(grid_shape))
        )
        phi_values = delta_phi_reshaped + block_fractions_reshaped

        # Create new phi_grid with updated phi_values
        new_phi_grid = phi_grid(
            grid_shape=grid_shape,
            block_fractions=block_fractions,
            phi_values=phi_values,
            n_components=n_components,
            box_length=original_box_length,
        )

        # Create new copolymer instance with updated phi_grid
        new_copolymer = ABCBlockCopolymer(
            n_components=n_components,
            block_lengths=original_block_lengths,
            block_fractions=block_fractions,
            chi_matrix=original_chi_matrix,
            phi_grid=new_phi_grid,
            kuhn_length=original_kuhn_length,
        )

        # Compute and return free energy
        return new_copolymer.free_energy_functional()

    # Create constraints: sum(delta_phi[i]) = 0 for each component i
    constraints = []
    for i in range(n_components):

        def make_constraint(component_idx):
            """Create a constraint function for component component_idx."""

            # Use default argument to capture component_idx by value (not by reference)
            def constraint(flattened_delta_phi, idx=component_idx):
                delta_phi_reshaped = flattened_delta_phi.reshape(
                    n_components, *grid_shape
                )
                # Sum over all grid points for this component
                return np.sum(delta_phi_reshaped[idx])

            return constraint

        constraints.append({"type": "eq", "fun": make_constraint(i)})

    # Initial guess: use current delta_phi values, flattened
    initial_guess = copolymer.phi_grid.delta_phi_i.flatten()

    return objective_function, constraints, initial_guess


def optimize_delta_phi(copolymer, method="SLSQP", options=None, **kwargs):
    """
    Optimize delta_phi values to minimize free energy subject to constraints.

    Parameters:
    -----------
    copolymer : ABCBlockCopolymer
        The copolymer instance to optimize
    method : str, optional
        Optimization method (default: 'SLSQP')
    options : dict, optional
        Options to pass to scipy.optimize.minimize
    **kwargs : dict
        Additional arguments to pass to scipy.optimize.minimize

    Returns:
    --------
    result : scipy.optimize.OptimizeResult
        The optimization result
    optimized_copolymer : ABCBlockCopolymer
        Copolymer instance with optimized delta_phi values
    """
    # Create optimization problem
    objective_function, constraints, initial_guess = create_optimization_problem(
        copolymer
    )

    # Default options
    if options is None:
        options = {"maxiter": 1000, "ftol": 1e-6}

    # Create callback to print free energy at each iteration
    iteration_count = [0]  # Use list to allow modification in nested function

    def callback(xk):
        """Callback function called at each iteration."""
        iteration_count[0] += 1
        free_energy = objective_function(xk)
        print(f"Iteration {iteration_count[0]}: Free energy = {free_energy:.6f}")

    # Run optimization
    result = minimize(
        objective_function,
        initial_guess,
        method=method,
        constraints=constraints,
        options=options,
        callback=callback,
        **kwargs,
    )

    # Create optimized copolymer with result
    grid_shape = copolymer.phi_grid.grid_shape
    n_components = copolymer.n_components
    block_fractions = copolymer.block_fractions

    # Reshape optimized delta_phi
    optimized_delta_phi = result.x.reshape(n_components, *grid_shape)

    # Compute optimized phi_values
    block_fractions_reshaped = block_fractions.reshape(
        n_components, *([1] * len(grid_shape))
    )
    optimized_phi_values = optimized_delta_phi + block_fractions_reshaped

    # Create optimized phi_grid and copolymer
    optimized_phi_grid = phi_grid(
        grid_shape=grid_shape,
        block_fractions=block_fractions,
        phi_values=optimized_phi_values,
        n_components=n_components,
        box_length=copolymer.phi_grid.box_length,
    )

    optimized_copolymer = ABCBlockCopolymer(
        n_components=n_components,
        block_lengths=copolymer.block_lengths,
        block_fractions=block_fractions,
        chi_matrix=copolymer.chi_matrix,
        phi_grid=optimized_phi_grid,
        kuhn_length=copolymer.kuhn_length,
    )

    return result, optimized_copolymer


def initialize_phi_values_satisfying_constraints(
    grid_shape, block_fractions, n_components, noise_scale=0.1
):
    """
    Initialize phi_values such that sum(delta_phi[i]) = 0 for each component i.

    Parameters:
    -----------
    grid_shape : tuple
        Shape of the grid
    block_fractions : np.ndarray
        Block fractions for each component
    n_components : int
        Number of components
    noise_scale : float, optional
        Scale of random noise to add (default: 0.1)

    Returns:
    --------
    phi_values : np.ndarray
        Array of shape (n_components, *grid_shape) with phi values
        that satisfy the constraint sum(delta_phi[i]) = 0
    """
    grid_size = np.prod(grid_shape)
    phi_values = np.zeros((n_components, *grid_shape))

    # For each component, generate random values and adjust so sum = block_fraction * grid_size
    for i in range(n_components):
        # Generate random values around the block fraction
        phi_i = block_fractions[i] + noise_scale * (np.random.rand(*grid_shape) - 0.5)

        # Adjust so that sum(phi_i) = block_fractions[i] * grid_size
        # This ensures sum(delta_phi[i]) = sum(phi_i - block_fractions[i]) = 0
        current_sum = np.sum(phi_i)
        target_sum = block_fractions[i] * grid_size
        correction = (target_sum - current_sum) / grid_size
        phi_i = phi_i + correction

        # Ensure non-negative (clip to small positive value if needed)
        phi_i = np.maximum(phi_i, 1e-10)

        # Re-normalize to ensure sum is exactly target_sum (in case clipping changed it)
        current_sum = np.sum(phi_i)
        if current_sum > 0:
            phi_i = phi_i * (target_sum / current_sum)

        phi_values[i] = phi_i

    return phi_values


# Example usage
if __name__ == "__main__":
    # Example parameters (you can modify these)
    n_components = 3
    block_lengths = np.array([10.0, 20.0, 30.0])
    block_fractions = np.array([0.33, 0.33, 0.34])
    chi_matrix = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.15],
            [0.2, 0.15, 0.0],
        ]
    )
    grid_shape = (32, 32)
    box_length = 10.0

    # Initialize phi_values that satisfy constraints (sum(delta_phi[i]) = 0 for each i)
    initial_phi_values = initialize_phi_values_satisfying_constraints(
        grid_shape=grid_shape,
        block_fractions=block_fractions,
        n_components=n_components,
        noise_scale=0.1,  # Adjust this to control initial variation
    )

    # Create initial phi_grid and copolymer
    initial_grid = phi_grid(
        grid_shape=grid_shape,
        block_fractions=block_fractions,
        phi_values=initial_phi_values,
        n_components=n_components,
        box_length=box_length,
    )

    initial_copolymer = ABCBlockCopolymer(
        n_components=n_components,
        block_lengths=block_lengths,
        block_fractions=block_fractions,
        chi_matrix=chi_matrix,
        phi_grid=initial_grid,
        kuhn_length=1.0,
    )

    # Print initial free energy
    initial_free_energy = initial_copolymer.free_energy_functional()
    print(f"Initial free energy: {initial_free_energy:.6f}")

    # Verify initial constraints
    print("\nInitial constraint values (should be ~0):")
    for i in range(n_components):
        sum_delta_phi = np.sum(initial_copolymer.phi_grid.delta_phi_i[i])
        print(f"  Component {i}: sum(delta_phi) = {sum_delta_phi:.6e}")

    # Optimize
    print("\nOptimizing...")
    result, optimized_copolymer = optimize_delta_phi(
        initial_copolymer,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-6, "disp": True},
    )

    # Print results
    print(f"\nOptimization {'successful' if result.success else 'failed'}")
    print(f"Final free energy: {optimized_copolymer.free_energy_functional():.6f}")
    print(f"Number of iterations: {result.nit}")
    print(f"Message: {result.message}")

    # Verify constraints are satisfied
    print("\nFinal constraint values (should be ~0):")
    for i in range(n_components):
        sum_delta_phi = np.sum(optimized_copolymer.phi_grid.delta_phi_i[i])
        print(f"  Component {i}: sum(delta_phi) = {sum_delta_phi:.6e}")

    # Save the density profile to disk
    npy_filename = "density_profile.npy"
    np.save(npy_filename, optimized_copolymer.phi_grid.phi_values)
    print(f"Density profile saved to {npy_filename}")

    # Save the optimized copolymer to disk
    copolymer_filename = "optimized_copolymer.pkl"
    with open(copolymer_filename, "wb") as f:
        pickle.dump(optimized_copolymer, f)
    print(f"Optimized copolymer saved to {copolymer_filename}")

    # Plot the density profile of the 3 components
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(n_components):
        axs[i].imshow(optimized_copolymer.phi_grid.phi_values[i], cmap="viridis")
        axs[i].set_title(f"Component {i}")
    plt.savefig("density_profile.png")
    plt.close()
