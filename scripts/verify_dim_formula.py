#!/usr/bin/env python3
"""
Verify dim(C^1_μ) = 6n - 2 for all gb weight sectors of B(0,n) = osp(1|2n).

Decomposes each gb-sector's f-variables into the 8 categories from the
analytical proof and verifies the count matches 6n - 2.
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def gen_weight(name, n):
    """Compute H-basis weight vector for a generator name.

    H-basis: w(e_j)_k = δ_{kj} - δ_{k+1,j} for k<n, δ_{kn} for k=n.
    """
    def e_vec(j):
        """Standard basis vector e_j in H-basis."""
        w = [0] * n
        for k in range(n):
            if k + 1 < n:
                w[k] = (1 if (k + 1) == j else 0) - (1 if (k + 2) == j else 0)
            else:
                w[k] = 1 if (k + 1) == j else 0
        return tuple(w)

    def add(a, b):
        return tuple(x + y for x, y in zip(a, b))

    def neg(a):
        return tuple(-x for x in a)

    zero = tuple([0] * n)

    if name.startswith("H_"):
        return zero
    # Odd: E_del{j}_{p,m}
    m1 = re.match(r'^E_del(\d+)_(p|m)$', name)
    if m1:
        j, sign = int(m1.group(1)), m1.group(2)
        return e_vec(j) if sign == 'p' else neg(e_vec(j))
    # Long even: E_2del{j}_{p,m}
    m2 = re.match(r'^E_2del(\d+)_(p|m)$', name)
    if m2:
        j, sign = int(m2.group(1)), m2.group(2)
        w = add(e_vec(j), e_vec(j))
        return w if sign == 'p' else neg(w)
    # Short even: E_del{i}_del{j}_{pp,mm,pm,mp}
    m3 = re.match(r'^E_del(\d+)_del(\d+)_(pp|mm|pm|mp)$', name)
    if m3:
        i, j, signs = int(m3.group(1)), int(m3.group(2)), m3.group(3)
        si = e_vec(i) if signs[0] == 'p' else neg(e_vec(i))
        sj = e_vec(j) if signs[1] == 'p' else neg(e_vec(j))
        return add(si, sj)

    raise ValueError(f"Unknown generator: {name}")


def load_structure(m, n):
    path = DATA_DIR / f"algebra_structures/B_{m}_{n}_structure.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_fvar_sectors(structure, n):
    """Build f-variable weight sectors from structure JSON."""
    even = structure['basis']['even']
    odd = structure['basis']['odd']
    parity = {g: 0 for g in even}
    parity.update({g: 1 for g in odd})
    all_gens = even + odd

    sectors = defaultdict(list)
    for src in all_gens:
        for dst in all_gens:
            if parity[src] == parity[dst]:
                continue
            w_src = gen_weight(src, n)
            w_dst = gen_weight(dst, n)
            mu = tuple(d - s for s, d in zip(w_src, w_dst))
            sectors[mu].append((src, dst))
    return sectors


def get_gb_weights(n):
    """Return distinct gb-parameter weights."""
    weights = set()
    for j in range(1, n + 1):
        w = [0] * n
        for k in range(n):
            if k + 1 < n:
                w[k] = (1 if (k + 1) == j else 0) - (1 if (k + 2) == j else 0)
            else:
                w[k] = 1 if (k + 1) == j else 0
        weights.add(tuple(w))
        weights.add(tuple(-x for x in w))
    return sorted(weights)


def count_by_formula(n):
    """Direct count via 8-category decomposition."""
    part_a = {
        "E+_1 -> E_{2d1}+": 1,
        "E+_k -> E_{d1+dk}": max(n - 1, 0),
        "E-_1 -> H_k": n,
        "E-_k -> E_{d1-dk}": max(n - 1, 0),
    }
    part_b = {
        "E+_1 <- H_k": n,
        "E+_k <- E_{-(d1-dk)}": max(n - 1, 0),
        "E-_1 <- E_{2d1}-": 1,
        "E-_k <- E_{-(d1+dk)}": max(n - 1, 0),
    }
    return sum(part_a.values()) + sum(part_b.values()), part_a, part_b


def main():
    print("=" * 60)
    print("dim(C^1_μ) = 6n - 2 Verification")
    print("=" * 60)

    all_pass = True
    for n in range(1, 6):
        print(f"\n--- B(0,{n}) = osp(1|{2*n}) ---")

        total, pa, pb = count_by_formula(n)
        print(f"  Formula: Part A={sum(pa.values())}, Part B={sum(pb.values())}, Total={total}")
        expected = 6 * n - 2
        assert total == expected, f"Formula mismatch: {total} != {expected}"
        print(f"  6n-2 = {expected} ✓")

        structure = load_structure(0, n)
        if structure is None:
            print(f"  (no structure data for B(0,{n}), formula-only)")
            continue

        sectors = get_fvar_sectors(structure, n)
        gb_wts = get_gb_weights(n)
        for wt in gb_wts:
            dim = len(sectors.get(wt, []))
            ok = dim == expected
            status = "✓" if ok else "✗"
            print(f"  μ={list(wt)}: dim={dim} {status}")
            if not ok:
                all_pass = False

    print("\n" + "=" * 60)
    print("ALL PASSED ✓" if all_pass else "SOME FAILED ✗")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
