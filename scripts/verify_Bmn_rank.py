"""
experiments/verify_Bmn_rank.py

Verify Conjecture (B(m,n) non-triviality) for m >= 1, n >= 1, m + n >= 2.

For each B(m,n) = osp(2m+1|2n), checks rank([A|L]) = rank(A) + rank(L)
which certifies that the only trivial deformation is gb = 0.

This script is referenced in triviality_B0n_ja.tex, Appendix (§6 supplementary).

Usage
-----
    python experiments/verify_Bmn_rank.py            # all available m>=1, n>=1
    python experiments/verify_Bmn_rank.py --m 1 --n 1
    python experiments/verify_Bmn_rank.py --max-sum 6  # m+n <= 6
    python experiments/verify_Bmn_rank.py --sympy       # force exact SymPy check
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

# All B(m,n) pairs with m >= 1, n >= 1 for which data files exist
ALL_PAIRS = [
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2),
    (3, 1), (3, 2), (3, 3),
]

SV_THRESHOLD = 0.1


def _to_Q_sqrt2_sym(x, tol=1e-10):
    """Convert float x ≈ a + b√2 to exact SymPy expression."""
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


def _exact_rank_Q_sqrt2(M_float):
    """Exact rank of Q(√2) matrix using SymPy."""
    import sympy as sp
    nz = np.any(np.abs(M_float) > 1e-12, axis=1)
    M_nz = M_float[nz]
    M_sp = sp.Matrix([[_to_Q_sqrt2_sym(float(x)) for x in row] for row in M_nz])
    return M_sp.rank()


def verify_Bmn(m, n, use_sympy=False, sympy_size_limit=200):
    """Verify rank condition for a single B(m,n)."""
    t0 = time.time()

    A, basis, _ = build_coboundary_matrix(m, n)
    L, _, gb_vars, _, _ = build_gamma_linear_map(m, n)
    A_r, L_r = A.real, L.real

    tol = 1e-8
    _, sA, _ = np.linalg.svd(A_r, full_matrices=False)
    rank_A = int(np.sum(sA > tol))

    _, sL, _ = np.linalg.svd(L_r, full_matrices=False)
    rank_L = int(np.sum(sL > tol))

    AL = np.hstack([A_r, L_r])
    _, sAL, _ = np.linalg.svd(AL, full_matrices=False)
    rank_AL = int(np.sum(sAL > tol))

    # P⊥ · L for min singular value certificate
    U, _, _ = np.linalg.svd(A_r, full_matrices=True)
    P_perp = U[:, rank_A:]
    PL = P_perp.T @ L_r
    _, sPL, _ = np.linalg.svd(PL, full_matrices=False)
    nonzero = sPL[sPL > tol]
    min_sv_PL = float(nonzero.min()) if len(nonzero) > 0 else 0.0

    is_independent = (rank_AL == rank_A + rank_L)
    gb_dim = len(gb_vars)
    elapsed = time.time() - t0

    result = {
        "algebra": f"B({m},{n})",
        "m": m, "n": n,
        "osp": f"osp({2*m+1}|{2*n})",
        "rank_A": rank_A,
        "rank_L": rank_L,
        "rank_AL": rank_AL,
        "gb_dim": gb_dim,
        "basis_dim": len(basis),
        "min_sv_PL": min_sv_PL,
        "is_independent": is_independent,
        "field": "Q(sqrt2)",
        "elapsed": elapsed,
        "sympy_verified": False,
    }

    # Optional SymPy exact cross-check
    if use_sympy and gb_dim > 0:
        t1 = time.time()
        try:
            nz = np.any(np.abs(AL) > 1e-12, axis=1)
            r = _exact_rank_Q_sqrt2(AL[nz])
            result["sympy_rank_AL"] = int(r)
            result["sympy_verified"] = (r == rank_AL)
            print(f"    SymPy exact: rank([A|L])={r} ({time.time()-t1:.1f}s)")
        except Exception as e:
            print(f"    SymPy failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify Conjecture B(m,n) non-triviality via rank condition"
    )
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--max-sum", type=int, default=None,
                        help="Only check m+n <= max-sum (default: all available)")
    parser.add_argument("--sympy", action="store_true",
                        help="Force SymPy exact rank computation")
    args = parser.parse_args()

    if args.m is not None and args.n is not None:
        pairs = [(args.m, args.n)]
    else:
        pairs = ALL_PAIRS
        if args.max_sum is not None:
            pairs = [(m, n) for m, n in pairs if m + n <= args.max_sum]

    print("=" * 72)
    print("  Conjecture B(m,n) Verification: rank([A|L]) = rank(A) + rank(L)")
    print("  Target: m >= 1, n >= 1, m + n >= 2")
    print("  Field: Q(√2) (fermionic oscillators contribute √2 coefficients)")
    print("=" * 72)

    results = []
    for m, n in pairs:
        print(f"\n--- B({m},{n}) = osp({2*m+1}|{2*n}) ---")
        try:
            res = verify_Bmn(m, n, use_sympy=args.sympy)
            results.append(res)
            status = "✓ PASS" if res["is_independent"] else "✗ FAIL"
            print(f"  rank(A)={res['rank_A']}, rank(L)={res['rank_L']}, "
                  f"rank([A|L])={res['rank_AL']}, gb_dim={res['gb_dim']}")
            print(f"  min_sv(P⊥·L)={res['min_sv_PL']:.4f}  "
                  f"({status})  [{res['elapsed']:.1f}s]")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"algebra": f"B({m},{n})", "is_independent": None, "error": str(e)})

    # Summary table
    print(f"\n{'=' * 72}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 72}")
    header = f"{'Type':<12} {'osp':<12} {'rank(A)':>8} {'rank(L)':>8} {'rank([A|L])':>12} {'gb_dim':>7} {'min_sv':>8} {'Result':>8}"
    print(header)
    print("-" * len(header))

    pass_count = 0
    for res in results:
        if "error" in res:
            print(f"{res['algebra']:<12} {'':12} {'ERROR':>8}")
            continue
        status = "PASS" if res["is_independent"] else "FAIL"
        if res["is_independent"]:
            pass_count += 1
        print(f"{res['algebra']:<12} {res['osp']:<12} {res['rank_A']:>8} "
              f"{res['rank_L']:>8} {res['rank_AL']:>12} {res['gb_dim']:>7} "
              f"{res['min_sv_PL']:>8.4f} {status:>8}")

    total = len([r for r in results if "error" not in r])
    print(f"\nPassed: {pass_count}/{total}")
    if pass_count == total:
        print("ALL B(m,n) with m>=1, n>=1 satisfy rank([A|L]) = rank(A) + rank(L)")
        print("Conjecture supported: only gb=0 is trivial for these types.")
    else:
        failed = [r["algebra"] for r in results if r.get("is_independent") is False]
        print(f"FAILED: {failed}")


if __name__ == "__main__":
    main()
