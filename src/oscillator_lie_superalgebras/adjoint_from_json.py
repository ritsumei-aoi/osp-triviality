"""
Build adjoint representation matrices from Algebra Structure JSON files.

This module provides a JSON-based approach to constructing adjoint matrices,
ensuring consistency with the data stored in algebra_structures/*.json files.

Version: 1.0 (2026-03-07)
"""

from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path


def build_adjoint_from_json(n: int) -> Tuple[List[str], Dict[str, int], Dict[str, np.ndarray]]:
    """
    Build adjoint representation matrices from Algebra Structure JSON (v3.0).
    
    The adjoint action is defined as ad(x) y = [x, y].
    For a basis {e_0, ..., e_{N-1}}, the matrix representation ad_x of ad(x)
    satisfies:
        [x, e_j] = sum_i (ad_x)_{i, j} e_i
    
    This function reads the structure_constants section from the algebra JSON
    file to construct the adjoint matrices, ensuring consistency with the
    stored data.
    
    Args:
        n: The size parameter for osp(1|2n).
        
    Returns:
        basis: The ordered list of generator names used as the basis.
        parity: Dictionary mapping generator names to parity (0=even, 1=odd).
        adjoint_matrices: A dictionary mapping each generator name to its
                          adjoint representation matrix (numpy array of shape (N, N)).
                          
    Raises:
        FileNotFoundError: If algebra structure JSON for this n is not found.
        
    Example:
        >>> basis, parity, adjoint = build_adjoint_from_json(1)
        >>> # Check that [M_0_0, Q_0] = Q_0
        >>> ad_M00 = adjoint["M_0_0"]
        >>> Q0_idx = basis.index("Q_0")
        >>> result = ad_M00[:, Q0_idx]
        >>> # result should have coefficient 1 at Q_0 position
    """
    from .algebra_parser import OSpAlgebraParser
    
    # Get path to algebra structure JSON
    json_path = _get_algebra_structure_path(n)
    
    # Load algebra structure
    parser = OSpAlgebraParser(str(json_path))
    
    # Extract basis and parity
    basis = parser.basis
    parity = parser.parity
    dim = len(basis)
    
    # Create index mapping
    basis_to_idx = {name: idx for idx, name in enumerate(basis)}
    
    # Build adjoint matrices
    adjoint_matrices: Dict[str, np.ndarray] = {}
    
    for x in basis:
        # Initialize an N x N matrix for ad(x)
        ad_x = np.zeros((dim, dim), dtype=complex)
        
        for j, y in enumerate(basis):
            # Get [x, y] from parser
            bracket_result = parser.get_bracket(x, y)
            
            # Fill in the column corresponding to y
            for res_gen, coeff in bracket_result.items():
                i = basis_to_idx[res_gen]
                ad_x[i, j] = complex(coeff)
        
        adjoint_matrices[x] = ad_x
    
    return basis, parity, adjoint_matrices


def _get_algebra_structure_path(n: int) -> Path:
    """
    Get the path to the algebra structure JSON file for osp(1|2n).
    
    Args:
        n: Size parameter
        
    Returns:
        Path to algebra structure JSON
        
    Raises:
        FileNotFoundError: If algebra structure file doesn't exist
    """
    # Determine project root (assuming this file is in src/oscillator_lie_superalgebras/)
    module_path = Path(__file__).resolve()
    project_root = module_path.parent.parent.parent
    
    algebra_file = project_root / "data" / "algebra_structures" / f"osp_1_{2*n}_structure.json"
    
    if not algebra_file.exists():
        raise FileNotFoundError(
            f"Algebra structure file not found: {algebra_file}\n"
            f"Please generate algebra structure for n={n} first using "
            f"experiments/generate_osp_structure_json.py"
        )
    
    return algebra_file


def get_available_algebra_structures() -> List[int]:
    """
    Get list of n values for which algebra structures are available.
    
    Returns:
        List of n values (e.g., [1, 2, 3])
    """
    module_path = Path(__file__).resolve()
    project_root = module_path.parent.parent.parent
    algebra_dir = project_root / "data" / "algebra_structures"
    
    available = []
    if algebra_dir.exists():
        for file in algebra_dir.glob("osp_1_*_structure.json"):
            # Extract 2n from filename
            two_n = int(file.stem.split('_')[2])
            n = two_n // 2
            available.append(n)
    
    return sorted(available)

def build_adjoint_from_json_v41(m: int, n: int) -> Tuple[List[str], Dict[str, int], Dict[str, np.ndarray]]:
    """
    Build adjoint representation matrices from Algebra Structure JSON (v4.1).

    The adjoint action is defined as ad(x) y = [x, y].
    For a basis {e_0, ..., e_{N-1}}, the matrix representation ad_x of ad(x)
    satisfies:
        [x, e_j] = sum_i (ad_x)_{i, j} e_i

    This function reads the structure_constants section from the v4.1 algebra JSON
    file directly (without OSpAlgebraParser) to construct the adjoint matrices.

    Args:
        m: Number of standard fermionic oscillator pairs (B(m,n) type).
        n: Number of bosonic oscillator pairs (B(m,n) type).

    Returns:
        basis: The ordered list of generator names used as the basis.
        parity: Dictionary mapping generator names to parity (0=even, 1=odd).
        adjoint_matrices: A dictionary mapping each generator name to its
                          adjoint representation matrix (numpy array of shape (N, N)).

    Raises:
        FileNotFoundError: If algebra structure JSON for this (m, n) is not found.
    """
    import json

    json_path = _get_algebra_structure_path_v41(m, n)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract basis and parity
    basis: List[str] = data["basis"]["even"] + data["basis"]["odd"]
    parity: Dict[str, int] = data["parity"]
    dim = len(basis)

    # Create index mapping
    basis_to_idx = {name: idx for idx, name in enumerate(basis)}

    # Parse brackets
    import sympy as sp
    raw_brackets = data["structure_constants"]["brackets"]
    brackets: Dict[Tuple[str, str], Dict[str, complex]] = {}
    for pair_str, result_dict in raw_brackets.items():
        # Handle cases where generator names might contain commas (should not happen in B(m,n) standard labels)
        # B(m,n) labels like E_eps1_del1_pp, E_del1_p are safe with split(',')
        parts = pair_str.split(',')
        if len(parts) == 2:
            g1, g2 = parts
            brackets[(g1, g2)] = {k: complex(sp.sympify(v).evalf()) for k, v in result_dict.items()}
        else:
            # Fallback for complex labels if needed
            pass

    # Build adjoint matrices
    adjoint_matrices: Dict[str, np.ndarray] = {}

    for x in basis:
        ad_x = np.zeros((dim, dim), dtype=complex)
        for j, y in enumerate(basis):
            bracket_result = brackets.get((x, y), {})
            for res_gen, coeff in bracket_result.items():
                i = basis_to_idx[res_gen]
                ad_x[i, j] = coeff
        adjoint_matrices[x] = ad_x

    return basis, parity, adjoint_matrices


def _get_algebra_structure_path_v41(m: int, n: int) -> Path:
    """
    Get the path to the v4.1 algebra structure JSON file for B(m,n).

    Args:
        m: Number of standard fermionic oscillator pairs.
        n: Number of bosonic oscillator pairs.

    Returns:
        Path to algebra structure JSON.

    Raises:
        FileNotFoundError: If algebra structure file doesn't exist.
    """
    module_path = Path(__file__).resolve()
    project_root = module_path.parent.parent.parent

    algebra_file = project_root / "data" / "algebra_structures" / f"B_{m}_{n}_structure.json"

    if not algebra_file.exists():
        raise FileNotFoundError(
            f"Algebra structure file not found: {algebra_file}\n"
            f"Please generate algebra structure for B({m},{n}) first using "
            f"experiments/generate_B_structure_json.py"
        )

    return algebra_file


def get_available_algebra_structures_v41() -> List[Tuple[int, int]]:
    """
    Get list of (m, n) pairs for which v4.1 algebra structures are available.

    Returns:
        List of (m, n) tuples (e.g., [(0, 1), (0, 2), (1, 1)]).
    """
    module_path = Path(__file__).resolve()
    project_root = module_path.parent.parent.parent
    algebra_dir = project_root / "data" / "algebra_structures"

    available = []
    if algebra_dir.exists():
        for file in algebra_dir.glob("B_*_*_structure.json"):
            parts = file.stem.split('_')  # ['B', m, n, 'structure']
            try:
                m_val, n_val = int(parts[1]), int(parts[2])
                available.append((m_val, n_val))
            except (IndexError, ValueError):
                continue

    return sorted(available)
