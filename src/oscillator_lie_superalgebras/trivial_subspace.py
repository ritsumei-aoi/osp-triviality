"""
Analytical computation of the trivial gb subspace for B(m,n).

Mathematical background:
- γ(gb) depends linearly on gb: γ(gb) = L · vec(gb)
  where L is a fixed matrix determined by the algebra structure (Schema 2 gamma_matrix).
- The coboundary operator δ is linear: δf = A · vec(f)
  where A is determined by the adjoint representation (Schema 1 structure_constants).
- A deformation is trivial ⟺ γ(gb) ∈ Im(A).
- Let P_⊥ be the orthogonal projector onto the complement of Im(A).
  Then: trivial ⟺ P_⊥ · γ(gb) = 0 ⟺ (P_⊥ · L) · vec(gb) = 0.
- The trivial gb subspace is null(P_⊥ · L), computed via SVD.

Usage:
    from oscillator_lie_superalgebras.trivial_subspace import trivial_gb_subspace
    result = trivial_gb_subspace(0, 1)
    print(result['null_space_basis'])  # basis of trivial gb directions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_gamma_json_path(m: int, n: int) -> Path:
    p = PROJECT_ROOT / "data" / "gamma_structures" / f"B_{m}_{n}_gamma.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Gamma structure JSON not found: {p}\n"
            f"Run: python experiments/generate_B_gamma_json.py --m {m} --n {n}"
        )
    return p


def _get_algebra_json_path(m: int, n: int) -> Path:
    p = PROJECT_ROOT / "data" / "algebra_structures" / f"B_{m}_{n}_structure.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Algebra structure JSON not found: {p}\n"
            f"Run: python experiments/generate_B_structure_json.py --m {m} --n {n}"
        )
    return p


def build_coboundary_matrix(
    m: int,
    n: int,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """
    Build the coboundary matrix A such that δf = A · vec(f).

    vec(f) is indexed over all valid odd pairs (src, dst) in PBW order:
    f(even) → odd, f(odd) → even.

    Rows of A correspond to independent (a,b) pairs in the same convention
    as solve_odd_f_generic, flattened over output components t.

    Returns:
        A: ndarray of shape (num_equations, num_f_vars)
        basis: ordered basis list
        parity: dict {name: 0/1}
    """
    import sympy as sp

    algebra_path = _get_algebra_json_path(m, n)
    with open(algebra_path, encoding="utf-8") as f:
        data = json.load(f)

    basis: List[str] = data["basis"]["even"] + data["basis"]["odd"]
    parity: Dict[str, int] = data["parity"]
    dim = len(basis)
    basis_to_idx = {name: i for i, name in enumerate(basis)}

    # Parse brackets
    raw_brackets = data["structure_constants"]["brackets"]
    brackets: Dict[Tuple[str, str], Dict[str, complex]] = {}
    for pair_str, result_dict in raw_brackets.items():
        parts = pair_str.split(",")
        if len(parts) == 2:
            g1, g2 = parts
            brackets[(g1, g2)] = {
                k: complex(sp.sympify(v).evalf()) for k, v in result_dict.items()
            }

    def ad(x: str, j: int) -> np.ndarray:
        """Column j of ad(x): [x, e_j] expanded in basis."""
        col = np.zeros(dim, dtype=complex)
        y = basis[j]
        for res_gen, coeff in brackets.get((x, y), {}).items():
            col[basis_to_idx[res_gen]] = coeff
        return col

    # Valid f-variable pairs: f(src) -> dst, parity(src) != parity(dst)
    valid_f_pairs = [
        (src, dst)
        for src in basis
        for dst in basis
        if parity[src] != parity[dst]
    ]
    var_to_idx = {pair: i for i, pair in enumerate(valid_f_pairs)}
    num_vars = len(valid_f_pairs)

    equations = []

    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            pa = parity[a]
            pb = parity[b]
            if pa == 0 and pb == 0 and i >= j:
                continue
            if pa == 1 and pb == 1 and i > j:
                continue
            if pa == 1 and pb == 0:
                continue

            sign_a = (-1) ** pa
            sign_ab = (-1) ** ((pa + 1) * pb)
            b_idx = basis_to_idx[b]

            ad_a = np.zeros((dim, dim), dtype=complex)
            ad_b = np.zeros((dim, dim), dtype=complex)
            for jj in range(dim):
                ad_a[:, jj] = ad(a, jj)
                ad_b[:, jj] = ad(b, jj)

            for t_idx in range(dim):
                row = np.zeros(num_vars, dtype=complex)

                # Term 1: (-1)^{p(a)} [a, f(b)]_t
                for dst in basis:
                    if (b, dst) in var_to_idx:
                        dst_idx = basis_to_idx[dst]
                        c = ad_a[t_idx, dst_idx]
                        if abs(c) > tol:
                            row[var_to_idx[(b, dst)]] += sign_a * c

                # Term 2: - (-1)^{(p(a)+1)*p(b)} [b, f(a)]_t
                for dst in basis:
                    if (a, dst) in var_to_idx:
                        dst_idx = basis_to_idx[dst]
                        c = ad_b[t_idx, dst_idx]
                        if abs(c) > tol:
                            row[var_to_idx[(a, dst)]] -= sign_ab * c

                # Term 3: - f([a,b])_t
                ab_col = ad_a[:, b_idx]
                for k_idx, k in enumerate(basis):
                    c_ab = ab_col[k_idx]
                    if abs(c_ab) > tol:
                        if (k, basis[t_idx]) in var_to_idx:
                            row[var_to_idx[(k, basis[t_idx])]] -= c_ab

                equations.append(row)

    if not equations:
        A = np.zeros((0, num_vars), dtype=complex)
    else:
        A = np.array(equations, dtype=complex)

    return A, basis, parity


def _gb_var_names_from_json(m: int, n: int) -> List[str]:
    """Read gb variable names from the gamma structure JSON (row-major order)."""
    gamma_path = _get_gamma_json_path(m, n)
    with open(gamma_path, encoding="utf-8") as f:
        data = json.load(f)
    sympy_repr = data["inhomogeneous_deformation"]["inhomogeneous_matrix"]["sympy_repr"]
    return [v for row in sympy_repr for v in row]


def build_gamma_linear_map(
    m: int,
    n: int,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, List[str], List[str], List[str], Dict[str, int]]:
    """
    Build the linear map L such that vec(γ) = L · vec(gb).

    γ is represented as a vector indexed by (independent pair, output basis element),
    using the same pair convention as solve_odd_f_generic.

    Returns:
        L: ndarray of shape (num_gamma_components, num_gb_vars)
        gamma_index: list of (pair_str, output_gen) describing each row of L
        gb_vars: ordered list of gb variable names
        basis: ordered basis list
        parity: dict {name: 0/1}
    """
    gamma_path = _get_gamma_json_path(m, n)
    algebra_path = _get_algebra_json_path(m, n)

    with open(gamma_path, encoding="utf-8") as f:
        gamma_data = json.load(f)
    with open(algebra_path, encoding="utf-8") as f:
        alg_data = json.load(f)

    basis: List[str] = alg_data["basis"]["even"] + alg_data["basis"]["odd"]
    parity: Dict[str, int] = alg_data["parity"]

    # gb variable names in canonical order
    gb_vars = _gb_var_names_from_json(m, n)
    gb_var_to_idx = {v: i for i, v in enumerate(gb_vars)}
    num_gb = len(gb_vars)

    # Parse gamma_matrix from Schema 2: γ(a,b)_t = Σ_v coeff[v] * gb_v
    # gamma_data["gamma_matrix"]["brackets"] = { "a,b": { "t": { "gb_var": "coeff_str" } } }
    raw_gamma = gamma_data["gamma_matrix"]["brackets"]

    # Build index of independent (a,b) pairs consistent with solver convention
    basis_to_idx = {name: i for i, name in enumerate(basis)}
    independent_pairs: List[Tuple[str, str]] = []
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            pa = parity[a]
            pb = parity[b]
            if pa == 0 and pb == 0 and i >= j:
                continue
            if pa == 1 and pb == 1 and i > j:
                continue
            if pa == 1 and pb == 0:
                continue
            independent_pairs.append((a, b))

    # Build L matrix and gamma_index
    rows = []
    gamma_index = []

    for a, b in independent_pairs:
        pair_str = f"{a},{b}"
        gamma_entry = raw_gamma.get(pair_str, {})
        for t in basis:
            gb_coeffs = gamma_entry.get(t, {})
            row = np.zeros(num_gb, dtype=complex)
            for gb_var, coeff_str in gb_coeffs.items():
                if gb_var in gb_var_to_idx:
                    row[gb_var_to_idx[gb_var]] = complex(sp.sympify(coeff_str).evalf())
            # Include row even if all zero (keeps index consistent with A rows)
            rows.append(row)
            gamma_index.append((pair_str, t))

    if rows:
        L = np.array(rows, dtype=complex)
    else:
        L = np.zeros((0, num_gb), dtype=complex)

    return L, gamma_index, gb_vars, basis, parity


def trivial_gb_subspace(
    m: int,
    n: int,
    tol: float = 1e-9,
) -> dict:
    """
    Compute the trivial gb subspace for B(m,n) analytically.

    A deformation with parameter gb is trivial iff γ(gb) ∈ Im(δ), i.e.,
    (P_⊥ · L) · vec(gb) = 0, where P_⊥ projects onto Im(A)^⊥.

    The trivial subspace is null(P_⊥ · L), computed via SVD.

    Args:
        m: number of fermionic oscillator pairs.
        n: number of bosonic oscillator pairs.
        tol: numerical tolerance for SVD null-space threshold.

    Returns:
        dict with keys:
          - 'gb_vars': list of gb variable names (column order of L)
          - 'gb_shape': [rows, cols] of gb matrix
          - 'null_space_basis': list of null vectors (each is a list of floats,
                                length = len(gb_vars)); trivial gb = linear combo
          - 'null_dim': int, dimension of trivial subspace
          - 'gb_total_dim': int, total number of gb parameters
          - 'coboundary_rank': int, rank of coboundary matrix A
          - 'gamma_map_rank': int, rank of gamma linear map L
          - 'PL_rank': int, rank of P_⊥ · L
          - 'singular_values_PL': list of singular values of P_⊥ · L
    """
    # Build coboundary matrix A
    A, basis, parity = build_coboundary_matrix(m, n, tol=tol * 1e-3)

    # Build gamma linear map L
    L, gamma_index, gb_vars, _, _ = build_gamma_linear_map(m, n, tol=tol * 1e-3)

    num_gb = len(gb_vars)
    num_rows = L.shape[0]

    # Orthogonal projector P_⊥ onto Im(A)^⊥
    if A.shape[0] > 0 and A.shape[1] > 0:
        # SVD of A to get orthonormal basis of Im(A)
        U_A, s_A, _ = np.linalg.svd(A, full_matrices=True)
        rank_A = int(np.sum(s_A > tol))
        # P_Im = U_A[:, :rank_A] @ U_A[:, :rank_A].conj().T
        U_col = U_A[:, :rank_A]
        P_perp = np.eye(num_rows, dtype=complex) - U_col @ U_col.conj().T
    else:
        rank_A = 0
        P_perp = np.eye(num_rows, dtype=complex)

    # Compute P_⊥ · L
    PL = P_perp @ L  # shape: (num_rows, num_gb)

    # SVD of PL to find null space
    if PL.shape[0] > 0 and PL.shape[1] > 0:
        _, s_PL, Vt_PL = np.linalg.svd(PL, full_matrices=True)
        rank_PL = int(np.sum(s_PL > tol))
        # Null space = rows of Vt_PL[rank_PL:]
        null_vecs = Vt_PL[rank_PL:]  # shape: (null_dim, num_gb)
    else:
        s_PL = np.array([])
        rank_PL = 0
        null_vecs = np.eye(num_gb, dtype=complex)

    # Rank of L itself
    _, s_L, _ = np.linalg.svd(L, full_matrices=False)
    rank_L = int(np.sum(s_L > tol))

    null_dim = num_gb - rank_PL
    gb_shape = [2 * m + 1, 2 * n]

    # Convert null vectors to real if imaginary parts are negligible
    null_basis_out = []
    for v in null_vecs:
        if np.max(np.abs(v.imag)) < tol:
            null_basis_out.append(v.real.tolist())
        else:
            null_basis_out.append(v.tolist())

    return {
        "gb_vars": gb_vars,
        "gb_shape": gb_shape,
        "null_space_basis": null_basis_out,
        "null_dim": null_dim,
        "gb_total_dim": num_gb,
        "coboundary_rank": rank_A,
        "gamma_map_rank": rank_L,
        "PL_rank": rank_PL,
        "singular_values_PL": s_PL.real.tolist() if len(s_PL) > 0 else [],
    }
