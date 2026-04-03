"""
Triviality checker for inhomogeneous deformations of osp(1|2n).

This module checks if a given inhomogeneous deformation (specified by matrix B)
represents a trivial abelian extension. A deformation is trivial if there exists
an odd linear map f: L -> L such that:

    γ(a,b) = (-1)^{p(a)} [a,f(b)] - (-1)^{(p(a)+1)p(b)} [b,f(a)] - f([a,b])

where γ is the 2-cocycle computed from B.

Mathematical Background: Bakalov-Sullivan (arXiv:1612.09400) Section 4, Eq.(trivial).
"""

from typing import Dict, List, Tuple, Union, Any
import numpy as np
from pathlib import Path


def check_triviality(
    n: int,
    B: Union[np.ndarray, List[List[float]], List[float]],
    tol: float = 1e-9
) -> Dict[str, Any]:
    """
    Check if the inhomogeneous deformation specified by B is trivial.
    
    A deformation is trivial if the 2-cocycle γ(a,b) can be written as a coboundary,
    i.e., there exists an odd linear map f such that γ = df.
    
    Args:
        n: Size parameter for osp(1|2n)
        B: Matrix B specifying the inhomogeneous deformation.
           Can be provided as:
           - np.ndarray of shape (2n,) or (2n, 1)
           - List[float] of length 2n
           - List[List[float]] representing column vector
        tol: Numerical tolerance for triviality check.
             - Default: 1e-9 (numerical tolerance)
             - Set to 0 for exact check (residual must be exactly 0)
             
    Returns:
        Dictionary with the following keys:
        - 'is_trivial': bool - True if residual_norm < tol
        - 'residual_norm': float - L2 norm of residual from Eq.(trivial)
        - 'f_map': Dict[str, Dict[str, complex]] - Odd linear map f (if trivial)
        - 'gamma': Dict - Computed gamma coefficients
        - 'tolerance': float - Tolerance used for the check
        - 'n': int - Size parameter
        - 'B': np.ndarray - Input B matrix (normalized)
        
    Example:
        >>> # Check triviality for osp(1|2) with B_1 = [1, 0]
        >>> result = check_triviality(n=1, B=[1, 0])
        >>> print(f"Trivial: {result['is_trivial']}")
        >>> print(f"Residual: {result['residual_norm']:.2e}")
        >>> if result['is_trivial']:
        >>>     print(f"f map: {result['f_map']}")
    """
    from .gamma_from_gb import compute_gamma_from_gb
    from .adjoint_from_json import build_adjoint_from_json  # v5 (B_0_n) schema
    from .cohomology_solver import solve_odd_f_generic
    
    # Normalize B to numpy array
    B_array = _normalize_B(n, B)
    
    # Step 1: Convert B to gb matrix (v5 convention: gb is 1 x 2n, B is 2n x 1)
    # gb matrix for v5: shape (1, 2n) for n >= 1 (matches JSON)
    if B_array.shape[0] == 2 * n:
        gb = [B_array.flatten().tolist()]
    else:
        raise ValueError(f"B must have length 2n={2*n}, got {B_array.shape[0]}")
    
    # Step 2: Compute gamma from gb using v5 JSON
    gamma_json_path = f"data/gamma_structures/B_0_{n}_gamma.json"
    gamma = compute_gamma_from_gb(gamma_json_path, gb)
    
    # Step 3: Load algebra structure and build adjoint matrices
    basis, parity, adjoint_matrices = build_adjoint_from_json(n)  # Now uses v5 (B_0_n) schema
    
    # Step 4: Solve for f
    f_map, residual_norm = solve_odd_f_generic(
        basis=basis,
        parity=parity,
        adjoint_matrices=adjoint_matrices,
        gamma=gamma,
        tol=tol
    )
    
    # Step 5: Determine triviality
    is_trivial = (residual_norm < tol)
    
    return {
        'is_trivial': is_trivial,
        'residual_norm': residual_norm,
        'f_map': f_map,
        'gamma': gamma,
        'tolerance': tol,
        'n': n,
        'B': B_array.flatten(),
    }


def _normalize_B(n: int, B: Union[np.ndarray, List, List[List[float]]]) -> np.ndarray:
    """
    Normalize B to standard numpy array format (2n, 1).
    
    Args:
        n: Size parameter
        B: Input B in various formats
        
    Returns:
        B as numpy array of shape (2n, 1)
        
    Raises:
        ValueError: If B has incorrect shape or dimensions
    """
    # Convert to numpy array
    if not isinstance(B, np.ndarray):
        B = np.array(B)
    
    # Handle 1D array
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    
    expected_shape = (2 * n, 1)
    if B.shape != expected_shape:
        raise ValueError(
            f"B matrix has incorrect shape. Expected {expected_shape}, got {B.shape}. "
            f"For osp(1|{2*n}), B should be a column vector with {2*n} entries."
        )
    
    return B


def format_f_map(f_map: Dict[str, Dict[str, complex]], tol: float = 1e-9) -> str:
    """
    Format the f map for readable output.
    
    Args:
        f_map: Odd linear map f as nested dictionary
        tol: Tolerance for displaying coefficients
        
    Returns:
        Formatted string representation
    """
    lines = []
    for src, dst_dict in f_map.items():
        if not dst_dict:
            continue
        terms = []
        for dst, coeff in dst_dict.items():
            if abs(coeff) > tol:
                if abs(coeff.imag) < tol:
                    # Real coefficient
                    coeff_str = f"{coeff.real:.6f}"
                else:
                    # Complex coefficient
                    coeff_str = f"({coeff.real:.6f} + {coeff.imag:.6f}i)"
                terms.append(f"{coeff_str} {dst}")
        if terms:
            lines.append(f"f({src}) = {' + '.join(terms)}")
    
    return "\n".join(lines) if lines else "f = 0 (zero map)"


def print_triviality_result(result: Dict[str, Any], verbose: bool = True):
    """
    Print triviality check result in a readable format.
    
    Args:
        result: Result dictionary from check_triviality()
        verbose: If True, print detailed information including f map
    """
    print("="*60)
    print(f"Triviality Check for osp(1|{2*result['n']})")
    print("="*60)
    print(f"B = {result['B']}")
    print(f"Tolerance = {result['tolerance']:.2e}")
    print()
    print(f"Result: {'TRIVIAL' if result['is_trivial'] else 'NON-TRIVIAL'}")
    print(f"Residual norm: {result['residual_norm']:.6e}")
    print()
    
    if verbose and result['is_trivial']:
        print("Odd linear map f:")
        print("-"*60)
        f_str = format_f_map(result['f_map'], tol=result['tolerance'])
        print(f_str)
        print("-"*60)
    
    print("="*60)
