"""
experiments/verify_certificates_algebraic.py

Phase 3: Algebraic verification and expansion of left null space certificates.

For each certificate family (I, II, III), this script:
1. Expands (δf)(X,Y)|_Z using the coboundary definition and structure constants
2. Collects terms by f-variable to verify c^T A = 0
3. Computes c^T L from γ structure constants
4. Displays the cancellation mechanism in detail

The algebraic proof that c^T A = 0 reduces to identities among osp(1|2n)
structure constants (eigenvalue relations and Jacobi-like identities).

Usage:
    python experiments/verify_certificates_algebraic.py
    python experiments/verify_certificates_algebraic.py --n 3 --family I --verbose
    python experiments/verify_certificates_algebraic.py --n-max 4 --all
"""

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "src")


def load_structure_constants(m: int, n: int) -> Tuple[List[str], Dict, Dict]:
    """Load basis, parity, and bracket structure constants from JSON."""
    root = Path(__file__).resolve().parent.parent
    alg_path = root / "data" / "algebra_structures" / f"B_{m}_{n}_structure.json"
    with open(alg_path, encoding="utf-8") as f:
        data = json.load(f)

    basis = data["basis"]["even"] + data["basis"]["odd"]
    parity = data["parity"]

    brackets: Dict[Tuple[str, str], Dict[str, Fraction]] = {}
    for pair_str, result_dict in data["structure_constants"]["brackets"].items():
        a, b = pair_str.split(",")
        parsed = {}
        for gen, coeff_str in result_dict.items():
            parsed[gen] = Fraction(coeff_str)
        brackets[(a, b)] = parsed

    return basis, parity, brackets


def load_gamma_constants(m: int, n: int) -> Tuple[List[str], Dict]:
    """Load gamma structure constants from JSON."""
    root = Path(__file__).resolve().parent.parent
    gamma_path = root / "data" / "gamma_structures" / f"B_{m}_{n}_gamma.json"
    with open(gamma_path, encoding="utf-8") as f:
        data = json.load(f)

    gh_vars = []
    inh = data["inhomogeneous_deformation"]["inhomogeneous_matrix"]
    for row in inh["matrix"]:
        gh_vars.extend(row)

    # Parse gamma brackets: gamma[(X,Y)][Z] = {gh_var: coeff}
    gamma: Dict[Tuple[str, str], Dict[str, Dict[str, Fraction]]] = {}
    for pair_str, out_dict in data["gamma_matrix"]["brackets"].items():
        a, b = pair_str.split(",")
        gamma[(a, b)] = {}
        for out_gen, gh_dict in out_dict.items():
            gamma[(a, b)][out_gen] = {}
            for gh_var, coeff_str in gh_dict.items():
                gamma[(a, b)][out_gen][gh_var] = Fraction(coeff_str)

    return gh_vars, gamma


def bracket(a: str, b: str, brackets: Dict) -> Dict[str, Fraction]:
    """Compute [a, b] using structure constants. Returns {gen: coeff}."""
    if (a, b) in brackets:
        return dict(brackets[(a, b)])
    # Try (b, a) with sign: [a, b] = -(-1)^{p(a)p(b)} [b, a]
    if (b, a) in brackets:
        result = {}
        for gen, coeff in brackets[(b, a)].items():
            result[gen] = -coeff  # For simplicity: handle parity sign below
        return result
    return {}


def bracket_with_sign(a: str, b: str, parity: Dict, brackets: Dict) -> Dict[str, Fraction]:
    """Compute [a, b] with proper super-sign convention.

    For super bracket: [a,b] = -(-1)^{p(a)p(b)} [b,a]
    """
    if (a, b) in brackets:
        return dict(brackets[(a, b)])
    if (b, a) in brackets:
        pa = parity[a]
        pb = parity[b]
        sign = -(-1) ** (pa * pb)
        result = {}
        for gen, coeff in brackets[(b, a)].items():
            result[gen] = Fraction(sign) * coeff
        return result
    return {}


def expand_delta_f_component(
    X: str, Y: str, Z: str,
    basis: List[str], parity: Dict, brackets: Dict
) -> Dict[Tuple[str, str], Fraction]:
    """
    Expand (δf)(X,Y)|_Z in terms of f-variables.

    δf(X,Y) = (-1)^{p(X)} [X, f(Y)] - (-1)^{(p(X)+1)p(Y)} [Y, f(X)] - f([X,Y])

    Returns: dict mapping (src, dst) -> coefficient of f(src)->dst
    """
    pX = parity[X]
    pY = parity[Y]

    result: Dict[Tuple[str, str], Fraction] = {}

    # f maps even→odd and odd→even. Y is either even or odd.
    # f(Y) has opposite parity to Y.
    f_Y_parity = 1 - parity[Y]
    f_X_parity = 1 - parity[X]

    # Term 1: (-1)^{p(X)} [X, f(Y)]|_Z
    # f(Y) = Σ_d f(Y→d) · d, where d has parity f_Y_parity
    sign1 = Fraction((-1) ** pX)
    for d in basis:
        if parity[d] != f_Y_parity:
            continue
        # [X, d]|_Z
        br = bracket_with_sign(X, d, parity, brackets)
        if Z in br:
            key = (Y, d)
            result[key] = result.get(key, Fraction(0)) + sign1 * br[Z]

    # Term 2: -(-1)^{(p(X)+1)p(Y)} [Y, f(X)]|_Z
    sign2 = Fraction(-(-1) ** ((pX + 1) * pY))
    for d in basis:
        if parity[d] != f_X_parity:
            continue
        # [Y, d]|_Z
        br = bracket_with_sign(Y, d, parity, brackets)
        if Z in br:
            key = (X, d)
            result[key] = result.get(key, Fraction(0)) + sign2 * br[Z]

    # Term 3: -f([X,Y])|_Z
    # [X,Y] = Σ_g c_g · g, and f(g)|_Z contributes
    br_XY = bracket_with_sign(X, Y, parity, brackets)
    for g, c_g in br_XY.items():
        # f(g)|_Z means f(g) → Z, i.e., the (g, Z) variable
        key = (g, Z)
        if parity[g] != parity[Z]:  # f maps between parities
            result[key] = result.get(key, Fraction(0)) - c_g

    # Remove zero entries
    return {k: v for k, v in result.items() if v != 0}


def gamma_component(
    X: str, Y: str, Z: str,
    gamma: Dict, parity: Dict
) -> Dict[str, Fraction]:
    """
    Get γ(X,Y)|_Z as dict {gh_var: coeff}.
    Handles symmetry: γ(X,Y) = -(-1)^{p(X)p(Y)} γ(Y,X)
    """
    if (X, Y) in gamma and Z in gamma[(X, Y)]:
        return dict(gamma[(X, Y)][Z])
    if (Y, X) in gamma and Z in gamma[(Y, X)]:
        pX = parity[X]
        pY = parity[Y]
        sign = -(-1) ** (pX * pY)
        result = {}
        for gh, c in gamma[(Y, X)][Z].items():
            result[gh] = Fraction(sign) * c
        return result
    return {}


# ============================================================
# Certificate definitions for each family
# ============================================================

def get_family_I_certificate(j: int, n: int) -> List[Tuple[str, str, str, Fraction]]:
    """
    Family I certificate for gh_{a0, b_j^+} (diagonal type, 4 terms).

    For j=1: uses H_1, E_{2δ_1}^±, E_{δ_1}^±, output H_2
    For j≥2: uses H_{j-1}, E_{2δ_j}^±, E_{δ_j}^±, output H_{j+1}
    Requires: n ≥ j+1 (i.e., H_{j+1} exists)

    Returns: list of (X, Y, Z, coeff) for certificate components
    """
    assert 1 <= j <= n - 1, f"Family I requires 1 ≤ j ≤ n-1, got j={j}, n={n}"

    if j == 1:
        H = "H_1"
        E2p = "E_2del1_p"
        E2m = "E_2del1_m"
        Ep = "E_del1_p"
        Em = "E_del1_m"
        Hout = "H_2"
        return [
            (H, E2p, Ep, Fraction(1)),
            (H, Ep, Hout, Fraction(2)),
            (H, Em, E2m, Fraction(-4)),
            (E2p, Em, Hout, Fraction(1)),
        ]
    else:
        H = f"H_{j-1}"
        E2p = f"E_2del{j}_p"
        E2m = f"E_2del{j}_m"
        Ep = f"E_del{j}_p"
        Em = f"E_del{j}_m"
        Hout = f"H_{j+1}"
        return [
            (H, E2p, Ep, Fraction(-1)),
            (H, Ep, Hout, Fraction(-2)),
            (H, Em, E2m, Fraction(4)),
            (E2p, Em, Hout, Fraction(1)),
        ]


def get_family_II_certificate(j: int, n: int) -> List[Tuple[str, str, str, Fraction]]:
    """
    Family II certificate for gh_{a0, b_j^-} (difference type, 3 terms).

    For j=1: uses H_1, E_{δ_1}^-, H_1/H_2, E_{2δ_2}^+
    For j≥2: uses H_{j-1}, E_{δ_j}^-, H_j/H_{j+1}, E_{2δ_{j+1}}^+
    Requires: n ≥ j+1

    Returns: list of (X, Y, Z, coeff) for certificate components
    """
    assert 1 <= j <= n - 1, f"Family II requires 1 ≤ j ≤ n-1, got j={j}, n={n}"

    if j == 1:
        H = "H_1"
        Em = "E_del1_m"
        Hj = "H_1"
        Hjp1 = "H_2"
        E2next = "E_2del2_p"
        return [
            (H, Em, Hj, Fraction(-2)),
            (H, Em, Hjp1, Fraction(2)),
            (E2next, Em, E2next, Fraction(1)),
        ]
    else:
        H = f"H_{j-1}"
        Em = f"E_del{j}_m"
        Hj = f"H_{j}"
        Hjp1 = f"H_{j+1}"
        E2next = f"E_2del{j+1}_p"
        return [
            (H, Em, Hj, Fraction(2)),
            (H, Em, Hjp1, Fraction(-2)),
            (E2next, Em, E2next, Fraction(1)),
        ]


def get_family_III_certificate_minus(n: int) -> List[Tuple[str, str, str, Fraction]]:
    """
    Family III certificate for gh_{a0, b_n^-} (cross-term type, 4 terms).

    Uses H_{n-1}, E_{δ_1-δ_n}^{--}, E_{δ_1+δ_n}^{++}, E_{δ_1}^±, E_{δ_n}^-, output H_2
    Requires: n ≥ 2

    For n=2: H_{n-1}=H_1 acts on E_{δ_1}^± with eigenvalue ±1/2 → coeff=2
    For n≥3: H_{n-1} acts trivially on E_{δ_1}^± → coeff=1

    Returns: list of (X, Y, Z, coeff) for certificate components
    """
    assert n >= 2, f"Family III requires n ≥ 2, got n={n}"

    H = f"H_{n-1}"
    E_cross_mm = f"E_del1_del{n}_mm"
    E_cross_pp = f"E_del1_del{n}_pp"
    E1p = "E_del1_p"
    E1m = "E_del1_m"
    Enm = f"E_del{n}_m"
    Hout = "H_2"

    # Coefficient depends on whether H_{n-1} has nontrivial action on δ_1 generators
    c2 = Fraction(2) if n == 2 else Fraction(1)

    return [
        (H, E_cross_mm, E1m, Fraction(1)),
        (H, E1p, E_cross_pp, c2),
        (H, Enm, Hout, Fraction(-1)),
        (E_cross_mm, E1p, Hout, Fraction(1)),
    ]


def get_family_III_certificate_plus(n: int) -> List[Tuple[str, str, str, Fraction]]:
    """
    Family III certificate for gh_{a0, b_n^+} (cross-term type, 4 terms).

    For n=2: coeff=2 on third term (H_1 eigenvalue effect)
    For n≥3: coeff=1

    Returns: list of (X, Y, Z, coeff) for certificate components
    """
    assert n >= 2, f"Family III requires n ≥ 2, got n={n}"

    H = f"H_{n-1}"
    E_cross_pp = f"E_del1_del{n}_pp"
    E_cross_mm = f"E_del1_del{n}_mm"
    E1p = "E_del1_p"
    E1m = "E_del1_m"
    Enp = f"E_del{n}_p"
    Hout = "H_2"

    c3 = Fraction(2) if n == 2 else Fraction(1)

    return [
        (H, E_cross_pp, E1p, Fraction(-1)),
        (H, Enp, Hout, Fraction(-1)),
        (H, E1m, E_cross_mm, c3),
        (E_cross_pp, E1m, Hout, Fraction(1)),
    ]


# ============================================================
# Verification engine
# ============================================================

def verify_certificate(
    cert_components: List[Tuple[str, str, str, Fraction]],
    basis: List[str], parity: Dict, brackets: Dict,
    gamma: Dict, gh_vars: List[str],
    label: str = "",
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Fraction]]:
    """
    Verify that a certificate satisfies c^T A = 0 and compute c^T L.

    Args:
        cert_components: list of (X, Y, Z, coeff) defining the certificate
        basis, parity, brackets: algebra structure constants
        gamma, gh_vars: gamma deformation data
        label: display label
        verbose: show detailed expansion

    Returns:
        (ctA_is_zero, ctL_values): whether c^T A = 0, and c^T L as {gh_var: value}
    """
    # Accumulate c^T A: sum of coeff * (δf)(X,Y)|_Z over f-variables
    ctA: Dict[Tuple[str, str], Fraction] = {}

    # Accumulate c^T L: sum of coeff * γ(X,Y)|_Z over gh variables
    ctL: Dict[str, Fraction] = {}

    if verbose:
        print(f"\n  --- Expanding certificate: {label} ---")

    for X, Y, Z, coeff in cert_components:
        if verbose:
            print(f"\n  Component: {coeff} · (δf)({X}, {Y})|_{Z}")

        # Expand δf(X,Y)|_Z
        expansion = expand_delta_f_component(X, Y, Z, basis, parity, brackets)
        if verbose and expansion:
            for (src, dst), c in sorted(expansion.items()):
                print(f"    {coeff*c:>6} · f({src})→{dst}")

        for key, val in expansion.items():
            ctA[key] = ctA.get(key, Fraction(0)) + coeff * val

        # Expand γ(X,Y)|_Z
        gcomp = gamma_component(X, Y, Z, gamma, parity)
        for gh, val in gcomp.items():
            ctL[gh] = ctL.get(gh, Fraction(0)) + coeff * val

    # Clean up zeros
    ctA = {k: v for k, v in ctA.items() if v != 0}
    ctL = {k: v for k, v in ctL.items() if v != 0}

    ctA_is_zero = len(ctA) == 0

    return ctA_is_zero, ctL


def verify_all_certificates(m: int, n: int, verbose: bool = False):
    """Verify all certificate families for B(m, n)."""
    basis, parity, brackets = load_structure_constants(m, n)
    gh_vars, gamma = load_gamma_constants(m, n)

    print(f"\n{'='*74}")
    print(f"  B({m},{n}) = osp(1|{2*n}) — Algebraic certificate verification")
    print(f"{'='*74}")

    results = {}
    all_ok = True

    # Family I: b_j^+ for j = 1, ..., n-1
    for j in range(1, n):
        label = f"Family I: b_{j}^+ (j={j})"
        cert = get_family_I_certificate(j, n)
        ctA_ok, ctL = verify_certificate(
            cert, basis, parity, brackets, gamma, gh_vars,
            label=label, verbose=verbose
        )
        gh_target = f"gh_a0_b{j}p"
        ctL_val = ctL.get(gh_target, Fraction(0))

        status = "✓" if ctA_ok and ctL_val != 0 else "✗"
        if not ctA_ok:
            all_ok = False
            status = "✗ c^T A ≠ 0!"
        if ctL_val == 0:
            all_ok = False
            status = "✗ c^T L = 0!"

        results[label] = {"ctA_zero": ctA_ok, "ctL": ctL, "ctL_target": ctL_val}
        print(f"\n  {status} {label}")
        print(f"    c^T A = 0: {ctA_ok}")
        print(f"    c^T L = {dict(ctL)}  (target gh: {gh_target} = {ctL_val})")

        if not ctA_ok:
            print(f"    ⚠ NONZERO c^T A entries:")
            ctA_debug = {}
            for X, Y, Z, coeff in cert:
                exp = expand_delta_f_component(X, Y, Z, basis, parity, brackets)
                for key, val in exp.items():
                    ctA_debug[key] = ctA_debug.get(key, Fraction(0)) + coeff * val
            ctA_debug = {k: v for k, v in ctA_debug.items() if v != 0}
            for (src, dst), val in sorted(ctA_debug.items()):
                print(f"      f({src})→{dst}: {val}")

    # Family II: b_j^- for j = 1, ..., n-1
    for j in range(1, n):
        label = f"Family II: b_{j}^- (j={j})"
        cert = get_family_II_certificate(j, n)
        ctA_ok, ctL = verify_certificate(
            cert, basis, parity, brackets, gamma, gh_vars,
            label=label, verbose=verbose
        )
        gh_target = f"gh_a0_b{j}m"
        ctL_val = ctL.get(gh_target, Fraction(0))

        status = "✓" if ctA_ok and ctL_val != 0 else "✗"
        if not ctA_ok:
            all_ok = False
        if ctL_val == 0:
            all_ok = False

        results[label] = {"ctA_zero": ctA_ok, "ctL": ctL, "ctL_target": ctL_val}
        print(f"\n  {status} {label}")
        print(f"    c^T A = 0: {ctA_ok}")
        print(f"    c^T L = {dict(ctL)}  (target gh: {gh_target} = {ctL_val})")

        if not ctA_ok:
            print(f"    ⚠ NONZERO c^T A entries:")
            ctA_debug = {}
            for X, Y, Z, coeff in cert:
                exp = expand_delta_f_component(X, Y, Z, basis, parity, brackets)
                for key, val in exp.items():
                    ctA_debug[key] = ctA_debug.get(key, Fraction(0)) + coeff * val
            ctA_debug = {k: v for k, v in ctA_debug.items() if v != 0}
            for (src, dst), val in sorted(ctA_debug.items()):
                print(f"      f({src})→{dst}: {val}")

    # Family III: b_n^- and b_n^+
    if n >= 2:
        for sign, getter, suffix in [
            ("-", get_family_III_certificate_minus, "m"),
            ("+", get_family_III_certificate_plus, "p"),
        ]:
            label = f"Family III: b_{n}^{sign} (n={n})"
            cert = getter(n)
            ctA_ok, ctL = verify_certificate(
                cert, basis, parity, brackets, gamma, gh_vars,
                label=label, verbose=verbose
            )
            gh_target = f"gh_a0_b{n}{suffix}"
            ctL_val = ctL.get(gh_target, Fraction(0))

            status = "✓" if ctA_ok and ctL_val != 0 else "✗"
            if not ctA_ok:
                all_ok = False
            if ctL_val == 0:
                all_ok = False

            results[label] = {"ctA_zero": ctA_ok, "ctL": ctL, "ctL_target": ctL_val}
            print(f"\n  {status} {label}")
            print(f"    c^T A = 0: {ctA_ok}")
            print(f"    c^T L = {dict(ctL)}  (target gh: {gh_target} = {ctL_val})")

            if not ctA_ok:
                print(f"    ⚠ NONZERO c^T A entries:")
                ctA_debug = {}
                for X, Y, Z, coeff in cert:
                    exp = expand_delta_f_component(X, Y, Z, basis, parity, brackets)
                    for key, val in exp.items():
                        ctA_debug[key] = ctA_debug.get(key, Fraction(0)) + coeff * val
                ctA_debug = {k: v for k, v in ctA_debug.items() if v != 0}
                for (src, dst), val in sorted(ctA_debug.items()):
                    print(f"      f({src})→{dst}: {val}")

    # Summary
    total = len(results)
    passed = sum(1 for r in results.values() if r["ctA_zero"] and r["ctL_target"] != 0)
    print(f"\n  {'='*60}")
    print(f"  Summary: {passed}/{total} certificates verified ({'ALL PASS' if all_ok else 'FAILURES'})")
    print(f"  {'='*60}")

    # c^T L value summary
    print(f"\n  c^T L values:")
    for label, r in results.items():
        print(f"    {label}: {r['ctL_target']}")

    return all_ok, results


def detailed_expansion(m: int, n: int, family: str, j: int = 1):
    """Show detailed δf expansion for a specific certificate."""
    basis, parity, brackets = load_structure_constants(m, n)
    gh_vars, gamma = load_gamma_constants(m, n)

    if family == "I":
        cert = get_family_I_certificate(j, n)
        label = f"Family I: b_{j}^+"
    elif family == "II":
        cert = get_family_II_certificate(j, n)
        label = f"Family II: b_{j}^-"
    elif family == "IIIm":
        cert = get_family_III_certificate_minus(n)
        label = f"Family III: b_{n}^-"
    elif family == "IIIp":
        cert = get_family_III_certificate_plus(n)
        label = f"Family III: b_{n}^+"
    else:
        raise ValueError(f"Unknown family: {family}")

    print(f"\n{'='*74}")
    print(f"  B(0,{n}) — Detailed expansion: {label}")
    print(f"{'='*74}")
    print(f"\n  Certificate components:")
    for X, Y, Z, coeff in cert:
        print(f"    {str(coeff):>3} · (δf)({X}, {Y})|_{Z}")

    # Expand each component
    all_f_vars: Dict[Tuple[str, str], Dict[int, Fraction]] = {}

    for idx, (X, Y, Z, coeff) in enumerate(cert):
        print(f"\n  --- Component {idx+1}: {coeff} · (δf)({X}, {Y})|_{Z} ---")

        pX = parity[X]
        pY = parity[Y]
        print(f"  p({X}) = {pX}, p({Y}) = {pY}")
        print(f"  Term 1: (-1)^{pX} [{X}, f({Y})]|_{Z}")
        print(f"  Term 2: -(-1)^{((pX+1)*pY)} [{Y}, f({X})]|_{Z}")
        print(f"  Term 3: -f([{X},{Y}])|_{Z}")

        expansion = expand_delta_f_component(X, Y, Z, basis, parity, brackets)

        for (src, dst), val in sorted(expansion.items()):
            print(f"    {str(val):>6} · f({src})→{dst}")
            if (src, dst) not in all_f_vars:
                all_f_vars[(src, dst)] = {}
            all_f_vars[(src, dst)][idx] = coeff * val

    # Show cancellation
    print(f"\n  --- Cancellation analysis (c^T A per f-variable) ---")
    ctA_total: Dict[Tuple[str, str], Fraction] = {}
    for (src, dst), contribs in sorted(all_f_vars.items()):
        total = sum(contribs.values())
        ctA_total[(src, dst)] = total
        parts = " + ".join(f"({v})" for v in contribs.values())
        status = "= 0 ✓" if total == 0 else f"= {total} ✗"
        print(f"    f({src})→{dst}: {parts} {status}")

    nonzero = {k: v for k, v in ctA_total.items() if v != 0}
    if not nonzero:
        print(f"\n  ✓ All f-variable coefficients cancel: c^T A = 0")
    else:
        print(f"\n  ✗ {len(nonzero)} nonzero entries in c^T A!")

    # c^T L
    ctL: Dict[str, Fraction] = {}
    print(f"\n  --- c^T L computation ---")
    for idx, (X, Y, Z, coeff) in enumerate(cert):
        gcomp = gamma_component(X, Y, Z, gamma, parity)
        if gcomp:
            for gh, val in gcomp.items():
                print(f"    Component {idx+1}: {coeff} · γ({X},{Y})|_{Z} → {coeff*val} · {gh}")
                ctL[gh] = ctL.get(gh, Fraction(0)) + coeff * val
        else:
            print(f"    Component {idx+1}: γ({X},{Y})|_{Z} = 0")

    ctL = {k: v for k, v in ctL.items() if v != 0}
    print(f"\n  c^T L = {dict(ctL)}")


def main():
    parser = argparse.ArgumentParser(
        description="Algebraic verification of left null space certificates"
    )
    parser.add_argument("--n", type=int, default=None, help="Specific n value")
    parser.add_argument("--n-max", type=int, default=4, help="Max n for range check")
    parser.add_argument("--family", choices=["I", "II", "IIIm", "IIIp", "all"], default="all")
    parser.add_argument("--j", type=int, default=1, help="Index j for Family I/II")
    parser.add_argument("--verbose", action="store_true", help="Detailed expansion")
    parser.add_argument("--detail", action="store_true", help="Show full cancellation")
    parser.add_argument("--all", action="store_true", help="Run all n values")
    args = parser.parse_args()

    if args.detail:
        n = args.n or 2
        if args.family == "all":
            for fam in ["I", "II", "IIIm", "IIIp"]:
                j = min(args.j, n - 1) if fam in ["I", "II"] else n
                detailed_expansion(0, n, fam, j)
        else:
            detailed_expansion(0, n, args.family, args.j)
        return

    if args.n:
        all_ok, _ = verify_all_certificates(0, args.n, verbose=args.verbose)
    elif args.all:
        overall_ok = True
        for n_val in range(2, args.n_max + 1):
            ok, _ = verify_all_certificates(0, n_val, verbose=args.verbose)
            if not ok:
                overall_ok = False
        print(f"\n{'='*74}")
        if overall_ok:
            print(f"  ALL CERTIFICATES VERIFIED for n = 2..{args.n_max}")
        else:
            print(f"  SOME CERTIFICATES FAILED")
        print(f"{'='*74}")
    else:
        # Default: run all n from 2 to n_max
        overall_ok = True
        for n_val in range(2, args.n_max + 1):
            ok, _ = verify_all_certificates(0, n_val, verbose=args.verbose)
            if not ok:
                overall_ok = False
        print(f"\n{'='*74}")
        if overall_ok:
            print(f"  ALL CERTIFICATES VERIFIED for n = 2..{args.n_max}")
        else:
            print(f"  SOME CERTIFICATES FAILED")
        print(f"{'='*74}")


if __name__ == "__main__":
    main()
