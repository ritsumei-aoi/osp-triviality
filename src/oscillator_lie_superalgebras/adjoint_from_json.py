"""
Build adjoint representation matrices from Algebra Structure JSON files.

This module provides a JSON-based approach to constructing adjoint matrices,
ensuring consistency with the data stored in algebra_structures/*.json files.

Version: 1.0 (2026-03-07)
"""

from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path


def get_available_algebra_structures() -> list[tuple[int, int]]:
    """
    Get list of (m, n) values for which B_{m}_{n}_structure.json algebra structures are available.
    Returns:
        List of (m, n) tuples (e.g., [(0, 1), (0, 2), ...])
    """
    module_path = Path(__file__).resolve()
    project_root = module_path.parent.parent.parent
    algebra_dir = project_root / "data" / "algebra_structures"
    available = []
    if algebra_dir.exists():
        for file in algebra_dir.glob("B_*_*_structure.json"):
            parts = file.stem.split('_')  # ['B', m, n, 'structure']
            if len(parts) == 4 and parts[0] == 'B' and parts[3] == 'structure':
                try:
                    m = int(parts[1])
                    n = int(parts[2])
                    available.append((m, n))
                except ValueError:
                    continue
    return available

def build_adjoint_from_json(n: int, m: int = 0) -> Tuple[List[str], Dict[str, int], Dict[str, np.ndarray]]:
    """
    Build adjoint representation matrices from Algebra Structure JSON (v5.0).

    The adjoint action is defined as ad(x) y = [x, y].
    For a basis {e_0, ..., e_{N-1}}, the matrix representation ad_x of ad(x)
    satisfies:
        [x, e_j] = sum_i (ad_x)_{i, j} e_i

    This function reads the structure_constants section from the v5.0 algebra JSON
    file directly to construct the adjoint matrices.

    Args:
        n: The size parameter for osp(1|2n) (or B(m,n) type).
        m: Number of standard fermionic oscillator pairs (default 0 for B(0,n)).

    Returns:
        basis: The ordered list of generator names used as the basis.
        parity: Dictionary mapping generator names to parity (0=even, 1=odd).
        adjoint_matrices: A dictionary mapping each generator name to its
                          adjoint representation matrix (numpy array of shape (N, N)).

    Raises:
        FileNotFoundError: If algebra structure JSON for this (m, n) is not found.
    """
    import json

    json_path = _get_algebra_structure_path(m, n)

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


def _get_algebra_structure_path(m: int, n: int) -> Path:
    """
    Get the path to the v5.0 algebra structure JSON file for B(m,n).

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

