#!/usr/bin/env python3
"""
Check triviality of first-order deformations of B(0,n) = osp(1|2n).

Usage:
    python scripts/check_triviality.py --n 2
    python scripts/check_triviality.py --n 3 --B "0.5 -0.3 1.0 0.0 0.7 -0.2"
    python scripts/check_triviality.py --n 1 --B "1.0 0.0" --tol 1e-12
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oscillator_lie_superalgebras.triviality_checker import check_triviality


def main():
    parser = argparse.ArgumentParser(
        description="Check triviality of B(0,n) deformations"
    )
    parser.add_argument("--n", type=int, required=True,
                        help="Parameter n for B(0,n) = osp(1|2n)")
    parser.add_argument("--B", type=str, default=None,
                        help="Deformation parameters as space-separated floats "
                             "(2n values). Default: random.")
    parser.add_argument("--tol", type=float, default=1e-9,
                        help="Numerical tolerance (default: 1e-9)")
    args = parser.parse_args()

    n = args.n
    dim_B = 2 * n

    if args.B is not None:
        B = [float(x) for x in args.B.split()]
        if len(B) != dim_B:
            print(f"Error: B must have {dim_B} components for n={n}, got {len(B)}")
            sys.exit(1)
    else:
        import numpy as np
        rng = np.random.default_rng(42)
        B = rng.standard_normal(dim_B).tolist()

    print(f"=== Triviality Check for B(0,{n}) = osp(1|{2*n}) ===")
    print(f"  B = {B}")
    print(f"  tolerance = {args.tol}")
    print()

    result = check_triviality(n=n, B=B, tol=args.tol)

    if result["is_trivial"]:
        print(f"  Result: TRIVIAL ✓")
    else:
        print(f"  Result: NON-TRIVIAL ✗")
    print(f"  Residual norm: {result['residual_norm']:.2e}")

    if result["f_map"]:
        nonzero = sum(1 for v in result["f_map"].values()
                      if any(abs(c) > args.tol for c in v.values()))
        print(f"  Non-zero f-map entries: {nonzero}")

    print()
    return 0 if result["is_trivial"] else 1


if __name__ == "__main__":
    sys.exit(main())
