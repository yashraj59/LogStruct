"""
Utility functions for LogStruct.
"""

from __future__ import annotations

import torch


def pick_device() -> torch.device:
    """
    Auto-select best available device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns
    -------
    device : torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    if mps_available:
        return torch.device("mps")
    
    return torch.device("cpu")


def normalize_rows(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Row-normalize a matrix: A_ij / sum_j(A_ij).
    
    Parameters
    ----------
    A : torch.Tensor
        Input matrix.
    eps : float
        Small constant for numerical stability.
    
    Returns
    -------
    A_norm : torch.Tensor
        Row-normalized matrix.
    """
    row_sums = A.sum(dim=1, keepdim=True)
    return A / (row_sums + eps)


def normalize_symmetric(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}.
    
    Parameters
    ----------
    A : torch.Tensor
        Symmetric adjacency matrix.
    eps : float
        Small constant for numerical stability.
    
    Returns
    -------
    A_norm : torch.Tensor
        Symmetrically normalized matrix.
    """
    d = A.sum(dim=1)
    d_inv_sqrt = (d + eps).pow(-0.5)
    return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)


def compute_laplacian(A: torch.Tensor, normalized: bool = False) -> torch.Tensor:
    """
    Compute graph Laplacian.
    
    Parameters
    ----------
    A : torch.Tensor
        Adjacency matrix.
    normalized : bool
        If True, return normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
        If False, return combinatorial Laplacian L = D - A.
    
    Returns
    -------
    L : torch.Tensor
        Laplacian matrix.
    """
    if normalized:
        A_norm = normalize_symmetric(A)
        return torch.eye(A.size(0), device=A.device) - A_norm
    else:
        D = torch.diag(A.sum(dim=1))
        return D - A
