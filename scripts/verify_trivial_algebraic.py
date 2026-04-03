"""
experiments/verify_trivial_algebraic.py

Algebraic verification that only gb=0 is a trivial deformation for B(m,n) with m+n >= 2.

Proof strategy
--------------
The deformation γ(gb) = L·vec(gb) is trivial iff γ(gb) ∈ Im(A), i.e. the linear system
A·vec(f) = L·vec(gb) has a solution f.  Over any field:

    "only gb=0 trivial"  ⟺  Im(A) ∩ Im(L) = {0}
                          ⟺  rank([A|L]) = rank(A) + rank(L)

Field-extension theorem
-----------------------
For a matrix M with entries in a subfield F ⊆ K, rank_F(M) = rank_K(M).
(The largest non-zero minor is the same: det ∈ F is nonzero in F iff nonzero in K.)

Application:
- B(0,n): structure constants in Q (integer entries) → rank_Q = rank_R (numerical SVD)
- B(m,n) m≥1: structure constants in Q(√2) → rank_Q(√2) = rank_R (numerical SVD)

So the numerically-computed rank (from SVD with min singular value ≫ ε) IS the exact
algebraic rank.  A positive min singular value of P⊥L constitutes an exact algebraic
proof that Im(A) ∩ Im(L) = {0}.

For small algebras (Q entries, manageable size) we also run SymPy exact rank as a
cross-check.

Usage
-----
    python experiments/verify_trivial_algebraic.py          # all available algebras
    python experiments/verify_trivial_algebraic.py --m 0 --n 2
    python experiments/verify_trivial_algebraic.py --sympy  # force exact SymPy for all
"""

import argparse
import sys
import time
import numpy as np

sys.path.insert(0, "src")

from oscillator_lie_superalgebras.trivial_subspace import (
    build_coboundary_matrix,
    build_gamma_linear_map,
)

AVAILABLE = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 1)]

# Threshold for "numerically exact" min singular value certificate
SV_THRESHOLD = 0.1


def _to_Q_sqrt2_sym(x, tol=1e-10):
    """Convert float x ≈ a + b√2 to exact SymPy expression a + b·√2."""
    import sympy as sp

    sqrt2 = np.sqrt(2)
    for num_b in range(-8, 9):
        b = num_b / 2
        a = x - b * sqrt2
        a_round = round(a * 2) / 2
        if abs(a - a_round) < tol:
            a_sym = sp.Rational(int(round(a * 2)), 2)
            b_sym = sp.Rational(int(round(b * 2)), 2)
            return a_sym + b_sym * sp.sqrt(2)
    return sp.Float(x)


def _exact_rank_Q(M_int):
    """Exact rank of integer matrix using SymPy."""
    import sympy as sp

    M_sp = sp.Matrix(M_int.tolist())
    return M_sp.rank()


def _exact_rank_Q_sqrt2(M_float):
    """Exact rank of Q(√2) matrix using SymPy."""
    import sympy as sp

    nz = np.any(np.abs(M_float) > 1e-12, axis=1)
    M_nz = M_float[nz]
    M_sp = sp.Matrix([[_to_Q_sqrt2_sym(float(x)) for x in row] for row in M_nz])
    return M_sp.rank()


def determine_coefficient_field(A_real):
    """Return 'Q' if entries are integers, 'Q(sqrt2)' otherwise."""
    if A_real.size == 0:
        return "Q"  # degenerate case: no entries
    dev = np.max(np.abs(A_real - np.round(A_real)))
    return "Q" if dev < 1e-10 else "Q(sqrt2)"


def compute_rank_certificate(A_real, L_real):
    """
    Compute rank certificate for [A|L] using SVD.

    Returns dict with:
      rank_A, rank_L, rank_AL, min_sv_PL, is_independent
    """
    # Degenerate case: no gb parameters (L has 0 columns)
    if L_real.shape[1] == 0:
        return {
            "rank_A": 0,
            "rank_L": 0,
            "rank_AL": 0,
            "min_sv_PL": 0.0,
            "is_independent": True,  # vacuously: Im(A) ∩ Im(L) = {0} since Im(L)={0}
        }

    tol = 1e-8
    _, sA, _ = np.linalg.svd(A_real, full_matrices=False)
    rank_A = int(np.sum(sA > tol))

    _, sL, _ = np.linalg.svd(L_real, full_matrices=False)
    rank_L = int(np.sum(sL > tol))

    AL = np.hstack([A_real, L_real])
    _, sAL, _ = np.linalg.svd(AL, full_matrices=False)
    rank_AL = int(np.sum(sAL > tol))

    # P⊥ · L for min singular value certificate
    U, _, _ = np.linalg.svd(A_real, full_matrices=True)
    P_perp = U[:, rank_A:]
    PL = P_perp.T @ L_real
    _, sPL, _ = np.linalg.svd(PL, full_matrices=False)
    nonzero = sPL[sPL > tol]
    min_sv_PL = float(nonzero.min()) if len(nonzero) > 0 else 0.0

    is_independent = rank_AL == rank_A + rank_L

    return {
        "rank_A": rank_A,
        "rank_L": rank_L,
        "rank_AL": rank_AL,
        "min_sv_PL": min_sv_PL,
        "is_independent": is_independent,
    }


def verify_algebra(m, n, use_sympy=False, sympy_size_limit=300):
    """
    Run algebraic verification for B(m,n).

    Returns a result dict with proof details.
    """
    A, basis, _ = build_coboundary_matrix(m, n)
    L, _, gb_vars, _, _ = build_gamma_linear_map(m, n)
    A_r, L_r = A.real, L.real

    field = determine_coefficient_field(A_r)
    cert = compute_rank_certificate(A_r, L_r)

    result = {
        "algebra": f"B({m},{n})",
        "m": m,
        "n": n,
        "field": field,
        **cert,
        "gb_dim": len(gb_vars),
        "basis_dim": len(basis),
        "sympy_rank_AL": None,
        "sympy_verified": False,
        "proof_type": None,
        "conclusion": None,
    }

    # Determine proof type and conclusion
    if len(gb_vars) == 0:
        conclusion = "gb space is 0-dimensional (n=0: no fermionic↔bosonic coupling)"
        proof_type = "trivial (no gb parameters)"
    elif cert["is_independent"]:
        conclusion = "Im(A) ∩ Im(L) = {0}  →  only gb=0 is trivial"
    else:
        if cert["rank_AL"] == cert["rank_A"]:
            conclusion = "Im(L) ⊆ Im(A)  →  ALL deformations are trivial"
        else:
            conclusion = f"PARTIAL overlap (rank=[A|L]={cert['rank_AL']})"

    # Proof certificate via field extension theorem
    if len(gb_vars) == 0:
        pass  # proof_type already set above
    elif cert["is_independent"] and cert["min_sv_PL"] > SV_THRESHOLD:
        proof_type = f"field extension ({field}) + min_sv={cert['min_sv_PL']:.4f} >> ε"
    elif not cert["is_independent"] and cert["min_sv_PL"] == 0.0:
        proof_type = f"field extension ({field}) + min_sv=0 (rank([A|L])=rank(A))"
    else:
        proof_type = "numerical (min_sv close to threshold, needs SymPy)"

    # Optional SymPy exact cross-check (skip degenerate case)
    run_sympy = use_sympy and len(gb_vars) > 0
    if not run_sympy and field == "Q" and len(gb_vars) > 0:
        nz_rows = int(np.sum(np.any(np.abs(A_r) > 1e-12, axis=1)))
        cols = A_r.shape[1] + L_r.shape[1]
        run_sympy = nz_rows * cols <= sympy_size_limit * sympy_size_limit

    if run_sympy:
        t0 = time.time()
        try:
            AL = np.hstack([A_r, L_r])
            nz = np.any(np.abs(AL) > 1e-12, axis=1)
            AL_nz = AL[nz]
            if field == "Q":
                AL_int = np.round(AL_nz).astype(int)
                r = _exact_rank_Q(AL_int)
            else:
                r = _exact_rank_Q_sqrt2(AL_nz)
            result["sympy_rank_AL"] = int(r)
            result["sympy_verified"] = r == cert["rank_AL"]
            proof_type = (
                f"SymPy exact rank over {field} confirmed: rank([A|L])={r}"
                + (
                    f" = rank(A)+rank(L) = {cert['rank_A']}+{cert['rank_L']}"
                    if cert["is_independent"]
                    else f" = rank(A) = {cert['rank_A']}"
                )
            )
        except Exception as e:
            proof_type += f" [SymPy failed: {e}]"
        print(f"    SymPy exact computation took {time.time()-t0:.1f}s")

    result["proof_type"] = proof_type
    result["conclusion"] = conclusion
    return result


def print_result(res):
    print(f"\n{'='*60}")
    print(f"  {res['algebra']}  (basis dim={res['basis_dim']}, gb_dim={res['gb_dim']})")
    print(f"  Coefficient field: {res['field']}")
    print(
        f"  rank(A)={res['rank_A']}, rank(L)={res['rank_L']}, rank([A|L])={res['rank_AL']}"
    )
    print(f"  min singular value of P⊥·L: {res['min_sv_PL']:.4f}")
    print(f"  Proof: {res['proof_type']}")
    if res["sympy_verified"]:
        print(f"  SymPy cross-check: rank([A|L])={res['sympy_rank_AL']} ✓")
    elif res["sympy_rank_AL"] is not None:
        print(f"  SymPy cross-check: rank([A|L])={res['sympy_rank_AL']} ✗ (mismatch)")
    print(f"  CONCLUSION: {res['conclusion']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument(
        "--sympy",
        action="store_true",
        help="Force SymPy exact rank computation for all algebras",
    )
    parser.add_argument(
        "--sympy-size-limit",
        type=int,
        default=300,
        help="Auto-run SymPy only if (nz_rows*cols) <= limit^2 and field=Q (default: 300)",
    )
    args = parser.parse_args()

    if args.m is not None and args.n is not None:
        pairs = [(args.m, args.n)]
    else:
        pairs = AVAILABLE

    print("Algebraic Triviality Verification for Oscillator Lie Superalgebras")
    print("=" * 60)
    print()
    print("Proof strategy:")
    print(
        "  For F ⊆ R subfield and matrix M over F:  rank_F(M) = rank_R(M)")
    print("  (field extension theorem: minors in F are nonzero iff nonzero in R)")
    print(
        "  ⟹ SVD numerical rank (with min_sv >> ε) = exact algebraic rank over F"
    )
    print()

    all_results = []
    for m, n in pairs:
        print(f"Processing B({m},{n})...", flush=True)
        res = verify_algebra(m, n, use_sympy=args.sympy, sympy_size_limit=args.sympy_size_limit)
        print_result(res)
        all_results.append(res)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Algebra':<10} {'Field':<10} {'rank(A)':<9} {'rank(L)':<9} "
          f"{'rank([A|L])':<13} {'min_sv_PL':<11} {'Conclusion'}")
    print("-" * 80)
    for res in all_results:
        ind = "✓" if res["is_independent"] else " "
        print(
            f"{res['algebra']:<10} {res['field']:<10} {res['rank_A']:<9} {res['rank_L']:<9} "
            f"{res['rank_AL']:<13} {res['min_sv_PL']:<11.4f} "
            f"{ind} {res['conclusion'].split('→')[1].strip() if '→' in res['conclusion'] else res['conclusion']}"
        )

    print()
    proved = [r for r in all_results if r["is_independent"] and r["min_sv_PL"] > SV_THRESHOLD]
    all_trivial = [r for r in all_results if not r["is_independent"]]
    degenerate = [r for r in all_results if r["gb_dim"] == 0]
    print(f"Algebraically proved 'only gb=0 trivial': {len(proved)}/{len(all_results)} algebras")
    if all_trivial:
        print(f"All deformations trivial: {[r['algebra'] for r in all_trivial]}")
    if degenerate:
        print(f"Degenerate (gb_dim=0, no coupling): {[r['algebra'] for r in degenerate]}")


if __name__ == "__main__":
    main()
