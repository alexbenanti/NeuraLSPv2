import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# 1. LOSS FUNCTIONS
# ==============================================================================

def nested_lora_loss(Q, S):
    """
    The Proposed Method: Forces Ordered Singular Vectors.
    """
    projections = torch.bmm(Q.transpose(1, 2), S)
    vector_energies = torch.norm(projections, dim=2)**2
    cum_energies = torch.cumsum(vector_energies, dim=1)
    total_S_energy = torch.norm(S, dim=(1, 2), keepdim=True)**2
    residuals = total_S_energy - cum_energies
    loss_per_rank = residuals / (total_S_energy + 1e-8)
    return loss_per_rank.mean()

def subspace_loss(Q, S):
    """
    The Ablation Baseline: Standard Subspace Learning.
    """
    projections = torch.bmm(Q.transpose(1, 2), S)
    captured_energy = torch.norm(projections, dim=(1, 2))**2
    total_energy = torch.norm(S, dim=(1, 2))**2
    loss = 1.0 - (captured_energy / (total_energy + 1e-8))
    return loss.mean()

def error_propagation_loss(P, A_csr):
    """
    The Loss from Luz et al. (2020).
    Minimizes ||M||_F^2 where M = (I - P (P^T A P)^-1 P^T A) S_relax.
    
    Args:
        P: (B, N, r) Learned Prolongator
        A_csr: Scipy CSR matrix (single instance, not batched for now)
    """
    # Convert A to dense torch tensor for this loss calculation

    
    device = P.device
    N = P.shape[1]
    
    # Convert CSR A to Dense Tensor A (1, N, N)
    A_dense = torch.tensor(A_csr.todense(), dtype=torch.float32, device=device).unsqueeze(0)
    
    # 1. Compute Coarse Operator: Ac = P^T A P
    # (B, r, N) @ (B, N, N) -> (B, r, N)
    # (B, r, N) @ (B, N, r) -> (B, r, r)
    AP = torch.bmm(A_dense, P)
    Ac = torch.bmm(P.transpose(1, 2), AP)
    
    # 2. Compute Projection Operator: Pi = P Ac^-1 P^T A
    # Add jitter to Ac for stability
    Ac_reg = Ac + 1e-6 * torch.eye(Ac.shape[1], device=device).unsqueeze(0)
    
    # We need M = (I - Pi).
    
    # Ac_inv: (B, r, r)
    Ac_inv = torch.linalg.inv(Ac_reg)
    
    # Pi = P @ Ac_inv @ P.T @ A
    # (B, N, r) @ (B, r, r) -> (B, N, r)
    temp = torch.bmm(P, Ac_inv)
    # (B, N, r) @ (B, r, N) -> (B, N, N)
    Pi = torch.bmm(temp, P.transpose(1, 2))
    Pi = torch.bmm(Pi, A_dense)
    
    I = torch.eye(N, device=device).unsqueeze(0)
    Coarse_Correction = I - Pi
    
    # 3. Add Relaxation S (Jacobi)
    # S = I - w D^-1 A
    diag_inv = 1.0 / (A_dense.diagonal(dim1=1, dim2=2) + 1e-8)
    D_inv = torch.diag_embed(diag_inv)
    omega = 0.66
    Smoother = I - omega * torch.bmm(D_inv, A_dense)
    
    # M_total = Smoother @ Coarse @ Smoother (S C S)
    M = torch.bmm(Smoother, torch.bmm(Coarse_Correction, Smoother))
    
    # Loss = ||M||_F^2
    loss = torch.norm(M, dim=(1, 2))**2
    return loss.mean()

# ==============================================================================
# 2. NEURAL ARCHITECTURES
# ==============================================================================

class ProlongationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_nodes, rank):
        super().__init__()
        self.n_nodes = n_nodes
        self.rank = rank

        flat_input_dim = n_nodes*input_dim

        flat_output_dim = n_nodes*rank

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flat_output_dim)
        )

    def forward(self, x):
        y = self.net(x)
        y = y.view(-1, self.n_nodes, self.rank)
        q, r = torch.linalg.qr(y)
        return q
    
class ProlongationMLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, n_nodes, rank):
        super().__init__()
        self.n_nodes = n_nodes
        self.rank = rank

        flat_input_dim = n_nodes*input_dim

        flat_output_dim = n_nodes*rank

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.GELU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.GELU(),
            nn.Linear(hidden_dim1, flat_output_dim)
        )

    def forward(self, x):
        y = self.net(x)
        y = y.view(-1, self.n_nodes, self.rank)
        q, r = torch.linalg.qr(y)
        return q
    
