from typing import Dict, List, Tuple
import numpy as np


def reconstruct_gamma_from_f(
    basis: List[str],
    parity: Dict[str, int],
    adjoint_matrices: Dict[str, np.ndarray],
    f_map: Dict[str, Dict[str, complex]],
    tol: float = 1e-9,
) -> Dict[Tuple[str, str], Dict[str, complex]]:
    """
    Reconstruct gamma(a,b) from an odd linear map f via the coboundary equation
    (Eq. 4.5 from Bakalov-Sullivan):

        gamma(a,b)_t = (-1)^{p(a)} [a, f(b)]_t
                     - (-1)^{(p(a)+1)*p(b)} [b, f(a)]_t
                     - f([a,b])_t

    Independent pairs (a,b) selected with the same convention as solve_odd_f_generic:
      - even-even: i < j
      - odd-odd:   i <= j
      - even-odd:  all (i, j)
      - odd-even:  skipped (redundant)

    Args:
        basis: List of basis element names.
        parity: Dict mapping basis names to parity (0=even, 1=odd).
        adjoint_matrices: Dict mapping basis names to their adjoint action matrices.
        f_map: Odd linear map as {src: {dst: coeff}}.
        tol: Numerical tolerance for filtering near-zero components.

    Returns:
        gamma_rec: Reconstructed cocycle as {(a, b): {t: coeff}}.
    """
    basis_to_idx = {g: i for i, g in enumerate(basis)}
    dim = len(basis)
    gamma_rec: Dict[Tuple[str, str], Dict[str, complex]] = {}

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

            sign_a  = (-1) ** pa
            sign_ab = (-1) ** ((pa + 1) * pb)
            ad_a = adjoint_matrices[a]
            ad_b = adjoint_matrices[b]
            b_idx = basis_to_idx[b]

            result = np.zeros(dim, dtype=complex)

            # Term 1: (-1)^{p(a)} [a, f(b)]
            for dst, coeff in f_map.get(b, {}).items():
                dst_idx = basis_to_idx[dst]
                result += sign_a * coeff * ad_a[:, dst_idx]

            # Term 2: - (-1)^{(p(a)+1)*p(b)} [b, f(a)]
            for dst, coeff in f_map.get(a, {}).items():
                dst_idx = basis_to_idx[dst]
                result -= sign_ab * coeff * ad_b[:, dst_idx]

            # Term 3: - f([a,b])
            ab_col = ad_a[:, b_idx]
            for k_idx, k in enumerate(basis):
                c_ab = ab_col[k_idx]
                if abs(c_ab) > tol:
                    for dst, coeff in f_map.get(k, {}).items():
                        dst_idx = basis_to_idx[dst]
                        result[dst_idx] -= c_ab * coeff

            entry = {basis[t]: result[t] for t in range(dim) if abs(result[t]) > tol}
            if entry:
                gamma_rec[(a, b)] = entry

    return gamma_rec


def solve_odd_f_generic(
    basis: List[str],
    parity: Dict[str, int],
    adjoint_matrices: Dict[str, np.ndarray],
    gamma: Dict[Tuple[str, str], Dict[str, complex]],
    tol: float = 1e-9,
) -> Tuple[Dict[str, Dict[str, complex]], float]:
    """
    Solves Eq. (4.5) from the paper to find an odd linear map f: L -> L such that
        gamma(a,b) = (-1)^{p(a)} [a,f(b)] - (-1)^{(p(a)+1)p(b)} [b,f(a)] - f([a,b])

    The equation system is built over all independent pairs (a,b), exploiting
    the skew-supersymmetry of gamma:
        gamma(a,b) = -(-1)^{p(a)*p(b)} gamma(b,a)

    Independent pair selection (basis indices i <= j or i < j):
      - even-even: i < j only  (diagonal: gamma(a,a)=0 guaranteed, no constraint)
      - odd-odd:   i <= j      (diagonal: gamma(a,a) is free, valid constraint)
      - even-odd:  all (i, j)  (odd-even is a distinct slot, not redundant)

    Args:
        basis: List of basis element names
        parity: Dict mapping basis names to parity (0=even, 1=odd)
        adjoint_matrices: Dict mapping basis names to their adjoint action matrices
        gamma: The cocycle gamma(a,b) as nested dict (zero pairs may be omitted)
        tol: Numerical tolerance

    Returns:
        (f_map, residual_norm) where:
        - f_map: Dict representing f as {src: {dst: coeff}}
        - residual_norm: L2 norm of the residual Ax - b

    Note:
        This equation system may be underdetermined (infinite solutions).
        This function returns a least-norm solution via numpy.linalg.lstsq.
    """
    basis_to_idx = {name: idx for idx, name in enumerate(basis)}

    # Identify valid (src, dst) pairs for the odd map f
    # f must be odd: f(even) -> odd, f(odd) -> even
    valid_f_pairs = []
    for src in basis:
        for dst in basis:
            if parity[src] != parity[dst]:
                valid_f_pairs.append((src, dst))

    var_to_idx = {pair: i for i, pair in enumerate(valid_f_pairs)}
    num_vars = len(valid_f_pairs)

    equations = []
    rhs_values = []

    # Loop over all independent pairs (a, b)
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            pa = parity[a]
            pb = parity[b]

            # even-even: i < j only (diagonal gamma=0 guaranteed)
            if pa == 0 and pb == 0 and i >= j:
                continue
            # odd-odd: i <= j only (diagonal is a valid constraint)
            if pa == 1 and pb == 1 and i > j:
                continue
            # odd-even: skip (dependent on even-odd via skew-supersymmetry)
            if pa == 1 and pb == 0:
                continue
            # even-odd: use all (i, j)

            sign_a  = (-1) ** pa
            sign_ab = (-1) ** ((pa + 1) * pb)

            gamma_coeffs = gamma.get((a, b), {})

            ad_a = adjoint_matrices[a]
            ad_b = adjoint_matrices[b]
            b_idx = basis_to_idx[b]

            # Loop over all basis elements t as the output component
            for t_idx, t in enumerate(basis):
                row = np.zeros(num_vars, dtype=complex)

                # Term 1: (-1)^{p(a)} [a, f(b)]_t
                for dst in basis:
                    if (b, dst) in var_to_idx:
                        dst_idx = basis_to_idx[dst]
                        coeff = ad_a[t_idx, dst_idx]
                        if abs(coeff) > tol:
                            row[var_to_idx[(b, dst)]] += sign_a * coeff

                # Term 2: - (-1)^{(p(a)+1)*p(b)} [b, f(a)]_t
                for dst in basis:
                    if (a, dst) in var_to_idx:
                        dst_idx = basis_to_idx[dst]
                        coeff = ad_b[t_idx, dst_idx]
                        if abs(coeff) > tol:
                            row[var_to_idx[(a, dst)]] -= sign_ab * coeff

                # Term 3: - f([a,b])_t
                # [a,b] = ad_a[:, b_idx], then f applied to each component k
                for k_idx, k in enumerate(basis):
                    c_ab = ad_a[k_idx, b_idx]
                    if abs(c_ab) > tol:
                        if (k, t) in var_to_idx:
                            row[var_to_idx[(k, t)]] -= c_ab

                rhs = gamma_coeffs.get(t, 0+0j)

                if np.any(np.abs(row) > tol) or abs(rhs) > tol:
                    equations.append(row)
                    rhs_values.append(rhs)

    if not equations:
        return {b: {} for b in basis}, 0.0

    A = np.array(equations, dtype=complex)
    B = np.array(rhs_values, dtype=complex)

    # Solve using least squares (system may be underdetermined)
    x, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

    # Compute residual norm
    if len(residuals) > 0:
        residual_norm = float(np.sqrt(residuals[0].real))
    else:
        residual_norm = float(np.linalg.norm(A @ x - B))

    # Build f_map from solution
    f_map: Dict[str, Dict[str, complex]] = {b: {} for b in basis}
    for (src, dst), idx in var_to_idx.items():
        coeff = complex(x[idx])
        if abs(coeff) > tol:
            f_map[src][dst] = coeff

    return f_map, residual_norm
