from rpa_torch import BlockCopolymerFreeEnergy
import torch

# Define the system parameters
chi_matrix = torch.tensor(
    [
        [0.0, 26.0, 26.0],
        [26.0, 0.0, 26.0],
        [26.0, 26.0, 0.0],
    ],
    dtype=torch.float32,
)
l_ij_matrix = torch.zeros((3, 3), dtype=torch.float32)  # Star copolymer architecture
block_fractions = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32) / 3.0

R_g = (100**0.5) * 1.0
box_length = R_g * 2.0


grid_res = 32
dx = box_length / grid_res
dy = box_length / grid_res

print(f"box_length = {box_length}")
print(f"dx = {dx}")
print(f"dy = {dy}")

# Create the free energy model
free_energy = BlockCopolymerFreeEnergy(
    chi_matrix=chi_matrix,
    l_ij_matrix=l_ij_matrix,
    block_fractions=block_fractions,
    optimize_box=False,
    Nx=grid_res,
    Ny=grid_res,
    dx=dx,
    dy=dy,
)

# Define the initial order parameter field with small fluctuations
delta_phi = 0.01 * torch.randn(
    free_energy.n_components,
    free_energy.Nx,
    free_energy.Ny,
    dtype=torch.float32,
    requires_grad=True,
)

delta_phi = free_energy._project_order_parameter(delta_phi.detach()).requires_grad_(
    True
)

# Check the adherance to the local and global constraints
rho = free_energy.get_densities(delta_phi)
print("max |sum_i rho_i - 1|:", (rho.sum(dim=0) - 1.0).abs().max().item())

# Check the adherance to the global constraints
print("component means:", delta_phi.mean(dim=(-2, -1)))


def pgd_step(delta_phi: torch.Tensor, step_size: float) -> torch.Tensor:
    F = free_energy(delta_phi)
    grad = torch.autograd.grad(F, delta_phi)[0]
    with torch.no_grad():
        delta_phi_new = delta_phi - step_size * grad
        delta_phi_new = free_energy._project_order_parameter(delta_phi_new.detach())

    F_new = free_energy(delta_phi_new)

    return F_new, delta_phi_new.detach().requires_grad_(True)


max_steps = 100000
step_size = 0.01
tol = 1e-10

F = free_energy(delta_phi)
F_prev = F.item()
for step in range(max_steps):
    F, delta_phi = pgd_step(delta_phi, step_size)
    if step % 10000 == 0:
        print(f"Step {step} completed, F = {free_energy(delta_phi).item():.6f}")
    # if abs(F.item() - F_prev) < tol:
    #     print(
    #         f"Convergence threshold reached on step {step}, F = {free_energy(delta_phi).item():.6f}"
    #     )
    F_prev = F.item()

print(f"Final F = {F.item():.6f}")

# Check the adherance to the local and global constraints
rho = free_energy.get_densities(delta_phi)
print("max |sum_i rho_i - 1|:", (rho.sum(dim=0) - 1.0).abs().max().item())

# Check the adherance to the global constraints
print("component means:", delta_phi.mean(dim=(-2, -1)))

# Save the output phi to a file
torch.save(delta_phi, "delta_phi.pt")

# save the BlockCopolymer to a file
torch.save(free_energy, "block_copolymer_model.pt")
