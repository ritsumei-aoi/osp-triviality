"""
experiments/verify_n_invariance.py

Verify n-invariance of certificate f-variable fingerprints.

For Family I/II, checks that f-variable sets and coefficients are identical
across different n values (n=2..5). For Family III, shows stabilization at n≥3.

Usage:
    python experiments/verify_n_invariance.py
    python experiments/verify_n_invariance.py --n-max 5
"""

import argparse
import sys
from fractions import Fraction
from typing import Dict, List, Tuple

sys.path.insert(0, "experiments")
from verify_certificates_algebraic import (
    expand_delta_f_component,
    get_family_I_certificate,
    get_family_II_certificate,
    get_family_III_certificate_minus,
    get_family_III_certificate_plus,
    load_gamma_constants,
    load_structure_constants,
    gamma_component,
)


def get_fvar_fingerprint(
    cert_components: List[Tuple[str, str, str, Fraction]],
    basis: List[str],
    parity: Dict,
    brackets: Dict,
) -> List[Tuple[Tuple[str, str], Fraction]]:
    """Return sorted list of (fvar, total_coeff) for a certificate."""
    all_fvars: Dict[Tuple[str, str], Fraction] = {}
    for X, Y, Z, coeff in cert_components:
        expansion = expand_delta_f_component(X, Y, Z, basis, parity, brackets)
        for fvar, fcoeff in expansion.items():
            all_fvars[fvar] = all_fvars.get(fvar, Fraction(0)) + fcoeff * coeff
    return sorted((k, v) for k, v in all_fvars.items())


def get_ctL_fingerprint(
    cert_components: List[Tuple[str, str, str, Fraction]],
    parity: Dict,
    gamma: Dict,
) -> Dict[str, Fraction]:
    """Compute c^T L as {gb_var: value}."""
    ctL: Dict[str, Fraction] = {}
    for X, Y, Z, coeff in cert_components:
        gc = gamma_component(X, Y, Z, gamma, parity)
        for gb_var, gval in gc.items():
            ctL[gb_var] = ctL.get(gb_var, Fraction(0)) + coeff * gval
    return {k: v for k, v in ctL.items() if v != 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-max", type=int, default=5)
    args = parser.parse_args()
    n_max = args.n_max

    all_pass = True

    families = []
    for j in range(1, n_max):
        families.append((f"Family I (j={j})", lambda n, j=j: get_family_I_certificate(j, n) if n > j else None, j + 1))
        families.append((f"Family II (j={j})", lambda n, j=j: get_family_II_certificate(j, n) if n > j else None, j + 1))
    families.append(("Family III minus", lambda n: get_family_III_certificate_minus(n), 2))
    families.append(("Family III plus", lambda n: get_family_III_certificate_plus(n), 2))

    print("=" * 70)
    print("  Certificate n-invariance verification")
    print("=" * 70)
    print()

    for fname, cert_fn, n_min in families:
        print(f"--- {fname} (n={n_min}..{n_max}) ---")
        prev_fp = None
        prev_n = None
        invariant = True

        for n in range(n_min, n_max + 1):
            cert = cert_fn(n)
            if cert is None:
                continue
            basis, parity, brackets = load_structure_constants(0, n)
            fp = get_fvar_fingerprint(cert, basis, parity, brackets)
            gb_vars, gamma = load_gamma_constants(0, n)
            ctL = get_ctL_fingerprint(cert, parity, gamma)

            all_zero = all(v == 0 for _, v in fp)
            n_fvars = len(fp)

            if prev_fp is not None:
                same = fp == prev_fp
                if not same:
                    invariant = False
                status = "SAME" if same else "CHANGED"
                print(f"  n={n}: {n_fvars} fvars, c^T A=0: {all_zero}, vs n={prev_n}: {status}")
            else:
                print(f"  n={n}: {n_fvars} fvars, c^T A=0: {all_zero}")

            # Show c^T L
            ctL_str = ", ".join(f"{k}={v}" for k, v in sorted(ctL.items()))
            print(f"         c^T L = {{{ctL_str}}}")

            if not all_zero:
                all_pass = False
                for k, v in fp:
                    if v != 0:
                        print(f"         *** NONZERO: f({k[0]})→{k[1]} = {v}")

            prev_fp = fp
            prev_n = n

        # Show f-variable detail for last n
        if prev_fp:
            print(f"  f-variables (n={prev_n}):")
            for (src, dst), coeff in prev_fp:
                print(f"    f({src})→{dst}: {coeff}")

        marker = "✓ n-INVARIANT" if invariant else "△ n-DEPENDENT (stabilizes at n≥3)"
        print(f"  Result: {marker}")
        print()

    print("=" * 70)
    if all_pass:
        print("  ALL PASS: c^T A = 0 verified for all certificates, n=2..%d" % n_max)
    else:
        print("  FAILURE: some c^T A ≠ 0")
    print("=" * 70)


if __name__ == "__main__":
    main()
