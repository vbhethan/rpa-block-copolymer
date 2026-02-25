from rpa_torch import BlockCopolymerFreeEnergy
import torch

# Define the system parameters
N = 100
chi_matrix = torch.tensor(
    [
        [0.0, 26.0, 26.0],
        [26.0, 0.0, 26.0],
        [26.0, 26.0, 0.0],
    ],
    dtype=torch.float64,
)

# Define the chain architecture
# Linear triblock with equal fractions
l_ij_matrix = torch.zeros((3, 3), dtype=torch.float64)
block_fractions = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64) / 3.0

R_g = (100**0.5) * 1.0
box_length = 0.1

grid_res = 32

print(f"box_length = {box_length}")
print(f"dx = {box_length / grid_res}")

# Create the free energy model
free_energy = BlockCopolymerFreeEnergy(
    N=N,
    chi_matrix=chi_matrix,
    l_ij_matrix=l_ij_matrix,
    block_fractions=block_fractions,
    optimize_box=False,
    grid_shape=(grid_res, grid_res),
    box_lengths=(box_length, box_length),
)

# Define the initial order parameter field with small fluctuations
delta_phi = 0.1 * torch.randn(
    free_energy.n_components,
    *free_energy.grid_shape,
    dtype=torch.float64,
    requires_grad=True,
)

delta_phi = free_energy._project_order_parameter(delta_phi.detach()).requires_grad_(
    True
)

# Check the adherance to the local and global constraints
rho = free_energy.get_densities(delta_phi)
print("max |sum_i rho_i - 1|:", (rho.sum(dim=0) - 1.0).abs().max().item())

# Check the adherance to the global constraints
print("component means:", delta_phi.mean(dim=tuple(range(1, free_energy.ndim + 1))))


def pgd_step(delta_phi: torch.Tensor, step_size: float) -> torch.Tensor:
    F = free_energy(delta_phi)
    grad = torch.autograd.grad(F, delta_phi)[0]
    with torch.no_grad():
        delta_phi_new = delta_phi - step_size * grad

    F_new = free_energy(delta_phi_new)

    return F_new, delta_phi_new.detach().requires_grad_(True)


def pgd_step_line_search(
    delta_phi: torch.Tensor,
    alpha_prev: float,
    c: float = 1e-4,
    beta: float = 0.5,
    grow: float = 1.05,
    max_ls: int = 20,
):
    F = free_energy(delta_phi)
    grad = torch.autograd.grad(F, delta_phi)[0]
    grad_sq = (grad * grad).sum()

    # Warm start from last accepted step size
    alpha = min(alpha_prev * grow, 1.0)

    with torch.no_grad():
        for _ in range(max_ls):
            cand = delta_phi - alpha * grad
            F_cand = free_energy(cand)

            # Armijo-style sufficient decrease
            if F_cand <= F - c * alpha * grad_sq:
                return (
                    F_cand.detach(),
                    cand.detach().requires_grad_(True),
                    alpha,
                    grad.norm().item(),
                )

            alpha *= beta

    # If nothing accepted, keep current point
    return F.detach(), delta_phi.detach().requires_grad_(True), 0.0, grad.norm().item()


max_steps = 10000
patience = 2500
step_size = 0.001
e_tol = 1e-6
grad_tol = 1e-7

F = free_energy(delta_phi)
F_prev = F.item()
for step in range(max_steps):
    F, delta_phi, step_size, grad_norm = pgd_step_line_search(delta_phi, step_size)
    if step % 100 == 0:
        print(f"Step {step} completed, F = {free_energy(delta_phi).item():.6f}")
    if abs(F.item() - F_prev) < e_tol and grad_norm < grad_tol and step > patience:
        print(
            f"Convergence threshold reached on step {step}, F = {free_energy(delta_phi).item():.6f}"
        )
        break
    F_prev = F.item()

print(f"Final F = {F.item():.6f}")

# Check the adherance to the local and global constraints
rho = free_energy.get_densities(delta_phi)
print("max |sum_i rho_i - 1|:", (rho.sum(dim=0) - 1.0).abs().max().item())

# Check the adherance to the global constraints
print("component means:", delta_phi.mean(dim=tuple(range(1, free_energy.ndim + 1))))

# Save the output phi to a file
torch.save(delta_phi, "delta_phi.pt")

# save the BlockCopolymer to a file
torch.save(free_energy, "block_copolymer_model.pt")
