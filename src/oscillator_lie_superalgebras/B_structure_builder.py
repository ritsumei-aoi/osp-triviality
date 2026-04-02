"""
B_structure_builder.py

JSON section builders for B(m,n) = osp(2m+1|2n) structure files (schema v5.0).
Provides metadata/basis/realization sections independent of structure constant engine.
"""

from typing import Any, Dict

from .B_generators import (
    build_B0n_basis,
    build_Bmn_basis,
    build_B0n_structure_constants,  # _legacy用・直接は使わないが依存関係のため残す
)
from .osp_generators import standard_symplectic_matrix

def build_algebra_section(m: int, n: int) -> Dict[str, Any]:
    osp_p = 2 * m + 1
    osp_q = 2 * n
    even_dim = n * (2 * n + 1) + m * (2 * m + 1)
    odd_dim = osp_p * osp_q
    return {
        "family": "B",
        "m": m,
        "n": n,
        "cartan_type": f"B({m},{n})",
        "alternative_notation": {
            "osp": f"osp({osp_p}|{osp_q})",
            "dimension_formula": "osp(2m+1|2n)",
        },
        "dimension": {
            "total": even_dim + odd_dim,
            "even": even_dim,
            "odd": odd_dim,
        },
    }

def build_oscillator_generators_section(m: int, n: int) -> Dict[str, Any]:
    boson_labels = []
    for i in range(1, n + 1):
        boson_labels += [f"b_{i}_p", f"b_{i}_m"]

    fermion_labels = []
    for j in range(1, m + 1):
        fermion_labels += [f"a_{j}_p", f"a_{j}_m"]

    return {
        "supplementary_fermion": {
            "label": "a_0",
            "description": (
                "Supplementary real fermionic oscillator (Frappat 2000, p.219). "
                + ("In B(0,n), this is the sole fermionic oscillator "
                   "(no standard fermionic pairs, m=0)."
                   if m == 0 else
                   "Replaces grading operator (-1)^N in B(m,n) for m>=1.")
            ),
            "role": "Sole fermionic oscillator for m=0" if m == 0
                    else "Grading operator replacement",
            "reference": "Frappat et al., Dictionary on Lie Algebras and Superalgebras (2000)",
        },
        "fermions": {
            "count": 2 * m,
            "m": m,
            "labels": fermion_labels,
            "description": "No standard fermionic oscillators for m=0" if m == 0
                           else f"Standard fermionic oscillators a_j^± with j=1,...,{m}",
        },
        "bosons": {
            "count": 2 * n,
            "n": n,
            "labels": boson_labels,
            "description": f"Bosonic oscillators b_i^± with i=1,...,{n}",
        },
    }

def build_oscillator_relations_section(m: int, n: int) -> Dict[str, Any]:
    J = standard_symplectic_matrix(n).tolist()
    basis_order = []
    for i in range(1, n + 1):
        basis_order += [f"b_{i}_p", f"b_{i}_m"]

    canonical_pairs = [
        {
            "index": i,
            "annihilation": f"b_{i}_m",
            "creation": f"b_{i}_p",
            "commutator": 1,
        }
        for i in range(1, n + 1)
    ]

    idx_map = []
    for i in range(n):
        idx_map.append(i)
        idx_map.append(i + n)
    J_reordered = [
        [J[idx_map[r]][idx_map[c]] for c in range(2 * n)]
        for r in range(2 * n)
    ]

    return {
        "supplementary_fermion_relations": {
            "label": "a_0",
            "description": "Relations for supplementary real fermionic oscillator",
            "self_anticommutator": {
                "relation": "{a_0, a_0} = 1",
                "note": (
                    "Real fermionic element with {a_0, a_0} = 1. "
                    "See metadata for correspondence with Frappat's e."
                ),
            },
            "commutes_with_bosons": {
                "relation": "[b_i^±, a_0] = 0",
                "description": "Commutes with all bosonic oscillators",
            },
            "anticommutes_with_fermions": {
                "relation": "{a_i^±, a_0} = 0",
                "description": (
                    "No standard fermionic oscillators for m=0; vacuously true."
                    if m == 0 else
                    "Anticommutes with all standard fermionic oscillators"
                ),
            },
        },
        "standard_fermion_anticommutators": {
            "description": (
                "No standard fermionic oscillators for m=0" if m == 0
                else "Canonical anticommutation relations for standard fermionic oscillators"
            ),
            "relations": {
                "same_type": "N/A (m=0)" if m == 0
                             else "{a_i^±, a_j^±} = 0 for all i, j",
                "conjugate_pair": "N/A (m=0)" if m == 0
                                  else "{a_i^-, a_j^+} = δ_{ij}",
            },
            "canonical_pairs": [] if m == 0 else [
                {
                    "index": j,
                    "annihilation": f"a_{j}_m",
                    "creation": f"a_{j}_p",
                    "anticommutator": 1,
                }
                for j in range(1, m + 1)
            ],
            "metric_matrix": {
                "description": (
                    "Empty metric matrix for m=0" if m == 0
                    else "Metric g_{ij} in basis [a_1^-, a_1^+, ..., a_m^-, a_m^+]."
                ),
                "basis_order": [] if m == 0 else [
                    label
                    for j in range(1, m + 1)
                    for label in [f"a_{j}_m", f"a_{j}_p"]
                ],
                "m": m,
                "matrix": [],
            },
        },
        "bosonic_commutators": {
            "description": "Canonical commutation relations for bosonic oscillators",
            "relations": {
                "same_type": "[b_i^±, b_j^±] = 0 for all i, j",
                "conjugate_pair": "[b_i^-, b_j^+] = δ_{ij}",
            },
            "canonical_pairs": canonical_pairs,
            "symplectic_matrix": {
                "description": (
                    f"Symplectic form J_{{ij}} in basis {basis_order} "
                    f"with [b_i, b_j] = J_{{ij}}."
                ),
                "basis_order": basis_order,
                "n": n,
                "matrix": J_reordered,
            },
        },
        "mixed_commutators": {
            "description": "Commutation relations between different oscillator types",
            "boson_fermion": {
                "relation": "[b_i^±, a_j^±] = 0",
                "description": (
                    "N/A for m=0 (no standard fermionic oscillators)" if m == 0
                    else "Bosons commute with standard fermions (homogeneous case, κ=0)"
                ),
            },
            "boson_supplementary": {
                "relation": "[b_i^±, a_0] = 0",
                "description": "Bosons commute with supplementary fermion",
            },
            "fermion_supplementary": {
                "relation": "{a_i^±, a_0} = 0",
                "description": (
                    "N/A for m=0 (no standard fermionic oscillators); vacuously true."
                    if m == 0 else
                    "Standard fermions anticommute with supplementary fermion"
                ),
            },
        },
    }

def build_basis_section(m: int, n: int) -> Dict[str, Any]:
    if m == 0:
        even_basis, odd_basis, _ = build_B0n_basis(n)
        ordering = (
            "Physical ordering: "
            "(1) Cartan: H_1,...,H_n; "
            "(2) Positive even roots: E_2del{k}_p (long), E_del{k}_del{l}_pp (k<l, short); "
            "(3) Negative even roots: E_2del{k}_m (long), E_del{k}_del{l}_mm (k<l, short); "
            "(4) Other even roots: E_del{k}_del{l}_pm (k!=l); "
            "(5) Positive odd roots: E_del{k}_p; "
            "(6) Negative odd roots: E_del{k}_m."
        )
    else:
        even_basis, odd_basis, _ = build_Bmn_basis(m, n)
        ordering = (
            "Physical ordering: "
            "(1) Cartan: H_1,...,H_n (bosonic), H_{n+1},...,H_{n+m} (fermionic); "
            "(2) Positive even roots: bosonic long/short, fermionic long/short, mixed; "
            "(3) Negative even roots: same pattern; "
            "(4) Mixed even roots: bosonic, fermionic, boson-fermion; "
            "(5) Positive odd roots: E_del{k}_p (bosonic), E_eps{i}_p (fermionic); "
            "(6) Negative odd roots: E_del{k}_m (bosonic), E_eps{i}_m (fermionic)."
        )
    return {
        "even": even_basis,
        "odd": odd_basis,
        "ordering_convention": ordering,
    }


def build_parity_section(m: int, n: int) -> Dict[str, int]:
    if m == 0:
        _, _, parity = build_B0n_basis(n)
    else:
        _, _, parity = build_Bmn_basis(m, n)
    return parity


def build_generator_realization_section(m: int, n: int) -> Dict[str, Any]:
    if m == 0:
        realizations: Dict[str, Any] = {}
        realizations.update(_cartan_standard_form_B0n(n))
        realizations.update(_non_cartan_realization_B0n(n))
        ordering = "a_0, " + ", ".join(
            f"b_{i}_p, b_{i}_m" for i in range(1, n + 1)
        )
        description = (
            "Standard form expressions with supplementary fermion-first PBW ordering. "
            "Frappat forms provided for reference. "
            "Odd generator coefficients use implementation convention "
            "(E_{±δ_k} = a_0 b_k^±); see metadata for Frappat correspondence."
        )
    else:
        realizations = _build_Bmn_realizations(m, n)
        ordering = (
            "a_0, "
            + ", ".join(f"a_{j}_p, a_{j}_m" for j in range(1, m + 1))
            + ", "
            + ", ".join(f"b_{i}_p, b_{i}_m" for i in range(1, n + 1))
        )
        description = (
            "Standard form with PBW ordering: a_0, a_j^±, b_k^±. "
            "E_{+ε_i} = -sqrt(2)*a_0*a_i^+ (sign from {a_i^+,a_0}=0). "
            "E_{-ε_i} = sqrt(2)*a_0*a_i^-. "
            "E_{±δ_k} = a_0*b_k^± (B(0,n) convention, coeff 1). "
            "See metadata."
        )
    return {
        "description": description,
        "ordering": ordering,
        "realizations": realizations,
    }


def build_metadata_section(m: int, n: int) -> Dict[str, Any]:
    osp_p = 2 * m + 1
    osp_q = 2 * n
    return {
        "reference": (
            f"Frappat et al., Dictionary on Lie Algebras and Superalgebras (2000), "
            f"p.219-220. B({m},{n}) oscillator realization."
        ),
        "supplementary_fermion_convention": (
            "Frappat's supplementary fermion e (with e^2 = 1) is implemented as a_0 "
            "via e = sqrt(2) * a_0 (so that {a_0, a_0} = 1). "
            "Consequence: odd generators E_{±δ_k} = a_0 b_k^± in this file correspond "
            "to (1/sqrt(2)) * E_{±δ_k}^{Frappat}. "
            "Even generators and Cartan elements are unaffected."
        ),
        "generated_by": "experiments/generate_B_structure_json.py",
        "related_files": {
            "gamma_structure": f"gamma_structures/B_{m}_{n}_gamma.json",
        },
        "schema_version_history": (
            f"v5.0 generation for B({m},{n}) = osp({osp_p}|{osp_q}). "
            + (
                "Migrated from v3.0. "
                "Fermionic oscillator 'a' renamed to 'a_0'. "
                f"Bosonic labels b_0..b_{{2n-1}} renamed to b_i_p/m. "
                "Structure constants recomputed directly from Frappat formulae "
                "(not via v3.0 basis). H_n now includes constant term +1/2."
                if m == 0 else ""
            )
        ),
    }


def _cartan_standard_form_B0n(n: int) -> Dict[str, Any]:
    """
    Cartan generators for B(0,n). Case B algorithm:
      H_k = b_k^+ b_k^- - b_{k+1}^+ b_{k+1}^-   (1 <= k < n)
      H_n = b_n^+ b_n^- + 1/2                     (terminal, Frappat p.220)
    """
    cartans = {}
    for k in range(1, n + 1):
        if k < n:
            standard_form = [
                {"words": [f"b_{k}_p", f"b_{k}_m"], "coeff": "1"},
                {"words": [f"b_{k+1}_p", f"b_{k+1}_m"], "coeff": "-1"},
            ]
            frappat_form = f"b_{k}^+ b_{k}^- - b_{k+1}^+ b_{k+1}^-"
            note = (
                f"Cartan generator (diff-type). "
                f"Frappat p.220: H_k = b_k^+ b_k^- - b_{{k+1}}^+ b_{{k+1}}^-."
            )
        else:  # k == n: terminal, bosonic unified formula gives +1/2
            standard_form = [
                {"words": [f"b_{k}_p", f"b_{k}_m"], "coeff": "1"},
                {"words": [], "coeff": "1/2"},
            ]
            frappat_form = f"b_{k}^+ b_{k}^- + 1/2"
            note = (
                "Cartan generator (terminal, bosonic). "
                "Unified formula: (1/2)(b^+ b^- + b^- b^+) = b^+ b^- + 1/2. "
                "Frappat p.220."
            )
        cartans[f"H_{k}"] = {
            "standard_form": standard_form,
            "frappat_form": frappat_form,
            "parity": 0,
            "note": note,
        }
    return cartans


def _non_cartan_realization_B0n(n: int) -> Dict[str, Any]:
    """
    Non-Cartan generator realizations for B(0,n), built directly from
    Frappat notation (no dependency on v3.0 name map).

    Frappat (p.220):
      E_{±2δ_k}         = (b_k^±)^2
      E_{δ_k+δ_l}^±     = b_k^± b_l^±       (k < l)
      E_{δ_k-δ_l}       = b_k^+ b_l^-       (k != l)
      E_{±δ_k}          = a_0 b_k^±          [impl: (1/sqrt(2)) e b_k^±, e=sqrt(2)*a_0]
    """
    realizations: Dict[str, Any] = {}

    # --- Even: E_{±2δ_k} = (b_k^±)^2 ---
    for k in range(1, n + 1):
        for sign, suffix in [("+", "p"), ("-", "m")]:
            name = f"E_2del{k}_{suffix}"
            lk = f"b_{k}_{suffix}"
            realizations[name] = {
                "standard_form": [{"words": [lk, lk], "coeff": "1"}],
                "frappat_form": f"(b_{k}^{sign})^2",
                "parity": 0,
                "note": f"Long even root {sign}2δ_{k}. Frappat p.220: E_{{±2δ_k}} = (b_k^±)^2.",
            }

    # --- Even: E_{δ_k+δ_l}^± = b_k^± b_l^± (k < l) ---
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            for sign, suffix in [("+", "p"), ("-", "m")]:
                name = f"E_del{k}_del{l}_{suffix}{suffix}"
                lk = f"b_{k}_{suffix}"
                ll = f"b_{l}_{suffix}"
                realizations[name] = {
                    "standard_form": [{"words": [lk, ll], "coeff": "1"}],
                    "frappat_form": f"b_{k}^{sign} b_{l}^{sign}",
                    "parity": 0,
                    "note": (
                        f"Short even root {sign}(δ_{k}+δ_{l}). "
                        f"Frappat p.220: E_{{±δ_k±δ_l}} = b_k^± b_l^±."
                    ),
                }

    # --- Even: E_{δ_k-δ_l} = b_k^+ b_l^- (k != l) ---
    for k in range(1, n + 1):
        for l in range(k+1, n + 1):
            name = f"E_del{k}_del{l}_pm"
            realizations[name] = {
                "standard_form": [
                    {"words": [f"b_{k}_p", f"b_{l}_m"], "coeff": "1"}
                ],
                "frappat_form": f"b_{k}^+ b_{l}^-",
                "parity": 0,
                "note": (
                    f"Mixed even root δ_{k}-δ_{l}. "
                    f"Frappat p.220: E_{{δ_k-δ_l}} = b_k^+ b_l^-."
                ),
            }
            name = f"E_del{k}_del{l}_mp"
            realizations[name] = {
                "standard_form": [
                    {"words": [f"b_{k}_m", f"b_{l}_p"], "coeff": "1"}
                ],
                "frappat_form": f"b_{k}^- b_{l}^+",
                "parity": 0,
                "note": (
                    f"Mixed even root -δ_{k}+δ_{l}. "
                    f"Frappat p.220: E_{{-δ_k+δ_l}} = b_k^- b_l^+."
                ),
            }


    # --- Odd: E_{±δ_k} = a_0 b_k^± ---
    for k in range(1, n + 1):
        # Positive: E_{δ_k} = a_0 b_k^+
        name_p = f"E_del{k}_p"
        realizations[name_p] = {
            "standard_form": [{"words": ["a_0", f"b_{k}_p"], "coeff": "1"}],
            "frappat_form": f"b_{k}^+ e",
            "transformation": (
                f"b_{k}^+ e → a_0 b_{k}^+ "
                f"(commute [b^±, a_0]=0, PBW: a_0 first; no sign change)"
            ),
            "parity": 1,
            "note": (
                f"Odd root +δ_{k}. "
                f"Frappat p.220: E_{{δ_k}} = (1/√2) b_k^+ e. "
                f"Implementation: a_0 b_k^+ = (1/√2) E_{{δ_k}}^{{Frappat}} "
                f"(e = √2 a_0; see metadata)."
            ),
        }
        # Negative: E_{-δ_k} = a_0 b_k^-
        name_m = f"E_del{k}_m"
        realizations[name_m] = {
            "standard_form": [{"words": ["a_0", f"b_{k}_m"], "coeff": "1"}],
            "frappat_form": f"e b_{k}^-",
            "transformation": (
                f"e b_{k}^- → a_0 b_{k}^- (already in PBW order)"
            ),
            "parity": 1,
            "note": (
                f"Odd root -δ_{k}. "
                f"Frappat p.220: E_{{-δ_k}} = (1/√2) e b_k^-. "
                f"Implementation: a_0 b_k^- = (1/√2) E_{{-δ_k}}^{{Frappat}} "
                f"(e = √2 a_0; see metadata)."
            ),
        }

    return realizations

def _build_Bmn_realizations(m: int, n: int) -> Dict[str, Any]:
    """Build generator realization dict for B(m,n) with m >= 1."""
    r: Dict[str, Any] = {}

    # --- Cartan: bosonic diff-type H_k (k < n) ---
    for k in range(1, n):
        r[f"H_{k}"] = {
            "standard_form": [
                {"words": [f"b_{k}_p", f"b_{k}_m"], "coeff": "1"},
                {"words": [f"b_{k+1}_p", f"b_{k+1}_m"], "coeff": "-1"},
            ],
            "frappat_form": f"b_{k}^+ b_{k}^- - b_{k+1}^+ b_{k+1}^-",
            "parity": 0,
            "note": "Cartan (bosonic diff-type). Frappat p.219: H_k = b_k^+b_k^- - b_{k+1}^+b_{k+1}^-.",
        }

    # --- Cartan: boundary H_n = b_n^+b_n^- + a_1^+a_1^- ---
    r[f"H_{n}"] = {
        "standard_form": [
            {"words": [f"b_{n}_p", f"b_{n}_m"], "coeff": "1"},
            {"words": ["a_1_p", "a_1_m"], "coeff": "1"},
        ],
        "frappat_form": f"b_{n}^+ b_{n}^- + a_1^+ a_1^-",
        "parity": 0,
        "note": "Cartan (boundary, bosonic-fermionic). Frappat p.219: H_n = b_n^+b_n^- + a_1^+a_1^-.",
    }

    # --- Cartan: fermionic diff-type H_{n+i} (1 <= i < m) ---
    for i in range(1, m):
        r[f"H_{n+i}"] = {
            "standard_form": [
                {"words": [f"a_{i}_p", f"a_{i}_m"], "coeff": "1"},
                {"words": [f"a_{i+1}_p", f"a_{i+1}_m"], "coeff": "-1"},
            ],
            "frappat_form": f"a_{i}^+ a_{i}^- - a_{i+1}^+ a_{i+1}^-",
            "parity": 0,
            "note": "Cartan (fermionic diff-type). Frappat p.219: H_{n+i} = a_i^+a_i^- - a_{i+1}^+a_{i+1}^-.",
        }

    # --- Cartan: terminal fermionic H_{n+m} = 2a_m^+a_m^- - 1 ---
    r[f"H_{n+m}"] = {
        "standard_form": [
            {"words": [f"a_{m}_p", f"a_{m}_m"], "coeff": "2"},
            {"words": [], "coeff": "-1"},
        ],
        "frappat_form": f"2 a_{m}^+ a_{m}^- - 1",
        "parity": 0,
        "note": (
            "Cartan (terminal, fermionic). "
            "Unified formula: (a_m^+a_m^- - a_m^-a_m^+) = 2a_m^+a_m^- - 1. "
            "Frappat p.219."
        ),
    }

    # --- Even bosonic: E_{±2δ_k} = (b_k^±)^2 ---
    for k in range(1, n + 1):
        for sign, suf in [("+", "p"), ("-", "m")]:
            r[f"E_2del{k}_{suf}"] = {
                "standard_form": [{"words": [f"b_{k}_{suf}", f"b_{k}_{suf}"], "coeff": "1"}],
                "frappat_form": f"(b_{k}^{sign})^2",
                "parity": 0,
                "note": f"Long even root {sign}2δ_{k}. Frappat p.220.",
            }

    # --- Even bosonic short: E_{±(δ_k+δ_l)} = b_k^± b_l^± (k<l) ---
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            for sign, suf in [("+", "p"), ("-", "m")]:
                r[f"E_del{k}_del{l}_{suf}{suf}"] = {
                    "standard_form": [{"words": [f"b_{k}_{suf}", f"b_{l}_{suf}"], "coeff": "1"}],
                    "frappat_form": f"b_{k}^{sign} b_{l}^{sign}",
                    "parity": 0,
                    "note": f"Short even root {sign}(δ_{k}+δ_{l}). Frappat p.220.",
                }

    # --- Even fermionic long: E_{±2ε_i} = (a_i^±)^2 ---
    # Note: (a_i^±)^2 = 0 for standard CAR fermions. Included for completeness.
    for i in range(1, m + 1):
        for sign, suf in [("+", "p"), ("-", "m")]:
            r[f"E_2eps{i}_{suf}"] = {
                "standard_form": [{"words": [f"a_{i}_{suf}", f"a_{i}_{suf}"], "coeff": "1"}],
                "frappat_form": f"(a_{i}^{sign})^2",
                "parity": 0,
                "note": (
                    f"Long even root {sign}2ε_{i}. Frappat p.219. "
                    "NOTE: vanishes in standard CAR realization ((a_i^±)^2 = 0)."
                ),
            }

    # --- Even fermionic short: E_{±(ε_i+ε_j)} = a_i^± a_j^± (i<j) ---
    for i in range(1, m + 1):
        for j in range(i + 1, m + 1):
            for sign, suf in [("+", "p"), ("-", "m")]:
                r[f"E_eps{i}_eps{j}_{suf}{suf}"] = {
                    "standard_form": [{"words": [f"a_{i}_{suf}", f"a_{j}_{suf}"], "coeff": "1"}],
                    "frappat_form": f"a_{i}^{sign} a_{j}^{sign}",
                    "parity": 0,
                    "note": f"Short even root {sign}(ε_{i}+ε_{j}). Frappat p.219.",
                }

    # --- Even mixed: E_{±(ε_i+δ_k)} = a_i^± b_k^± ---
    for i in range(1, m + 1):
        for k in range(1, n + 1):
            for sign, suf in [("+", "p"), ("-", "m")]:
                r[f"E_eps{i}_del{k}_{suf}{suf}"] = {
                    "standard_form": [{"words": [f"a_{i}_{suf}", f"b_{k}_{suf}"], "coeff": "1"}],
                    "frappat_form": f"a_{i}^{sign} b_{k}^{sign}",
                    "parity": 1,
                    "note": f"Mixed even root {sign}(ε_{i}+δ_{k}). Frappat p.219.",
                }

    # --- Even bosonic mixed: E_{δ_k-δ_l} = b_k^+ b_l^- (k!=l) ---
    for k in range(1, n + 1):
        for l in range(k+1, n + 1):
            r[f"E_del{k}_del{l}_pm"] = {
                "standard_form": [{"words": [f"b_{k}_p", f"b_{l}_m"], "coeff": "1"}],
                "frappat_form": f"b_{k}^+ b_{l}^-",
                "parity": 0,
                "note": f"Mixed even root δ_{k}-δ_{l}. Frappat p.220.",
            }
            r[f"E_del{k}_del{l}_mp"] = {
                "standard_form": [{"words": [f"b_{k}_m", f"b_{l}_p"], "coeff": "1"}],
                "frappat_form": f"b_{k}^- b_{l}^+",
                "parity": 0,
                "note": f"Mixed even root -δ_{k}+δ_{l}. Frappat p.220.",
            }


    # --- Even fermionic mixed: E_{ε_i-ε_j} = a_i^+ a_j^- (i!=j) ---
    for i in range(1, m + 1):
        for j in range(i+1, m + 1):
            r[f"E_eps{i}_eps{j}_pm"] = {
                "standard_form": [{"words": [f"a_{i}_p", f"a_{j}_m"], "coeff": "1"}],
                "frappat_form": f"a_{i}^+ a_{j}^-",
                "parity": 0,
                "note": f"Mixed even root ε_{i}-ε_{j}. Frappat p.219.",
            }
            r[f"E_eps{i}_eps{j}_mp"] = {
                "standard_form": [{"words": [f"a_{i}_m", f"a_{j}_p"], "coeff": "1"}],
                "frappat_form": f"a_{i}^- a_{j}^+",
                "parity": 0,
                "note": f"Mixed even root -ε_{i}+ε_{j}. Frappat p.219.",
            }


    # --- Even boson-fermion mixed: E_{ε_i-δ_k} = a_i^+ b_k^-, E_{δ_k-ε_i} = b_k^+ a_i^- ---
    for i in range(1, m + 1):
        for k in range(1, n + 1):
            r[f"E_eps{i}_del{k}_pm"] = {
                "standard_form": [{"words": [f"a_{i}_p", f"b_{k}_m"], "coeff": "1"}],
                "frappat_form": f"a_{i}^+ b_{k}^-",
                "parity": 1,
                "note": f"Mixed even root ε_{i}-δ_{k}. Frappat p.219.",
            }
            r[f"E_eps{i}_del{k}_mp"] = {
                "standard_form": [{"words": [f"a_{i}_m", f"b_{k}_p"], "coeff": "1"}],
                "frappat_form": f"b_{k}^+ a_{i}^-",
                "parity": 1,
                "note": (
                    f"Mixed even root -δ_{k}+ε_{i}. Frappat p.219: b_k^+ a_i^-. "
                    "PBW: a_i^- b_k^+ -> reordered as a_i^- first (a before b)."
                ),
            }

    # --- Odd bosonic: E_{±δ_k} = a_0 b_k^± (B(0,n) convention, coeff 1) ---
    for k in range(1, n + 1):
        r[f"E_del{k}_p"] = {
            "standard_form": [{"words": ["a_0", f"b_{k}_p"], "coeff": "1"}],
            "frappat_form": f"b_{k}^+ e",
            "transformation": f"b_{k}^+ e -> a_0 b_{k}^+ ([b,a_0]=0, no sign change)",
            "parity": 1,
            "note": (
                f"Odd root +δ_{k}. B(0,n) convention: a_0 b_k^+. "
                "B(m,n) Frappat: sqrt(2)*a_0*b_k^+. See metadata."
            ),
        }
        r[f"E_del{k}_m"] = {
            "standard_form": [{"words": ["a_0", f"b_{k}_m"], "coeff": "1"}],
            "frappat_form": f"e b_{k}^-",
            "transformation": f"e b_{k}^- -> a_0 b_{k}^- (already PBW order)",
            "parity": 1,
            "note": (
                f"Odd root -δ_{k}. B(0,n) convention: a_0 b_k^-. "
                "B(m,n) Frappat: sqrt(2)*a_0*b_k^-. See metadata."
            ),
        }

    # --- Odd fermionic: E_{+ε_i} = -sqrt(2)*a_0*a_i^+, E_{-ε_i} = sqrt(2)*a_0*a_i^- ---
    for i in range(1, m + 1):
        r[f"E_eps{i}_p"] = {
            "standard_form": [{"words": ["a_0", f"a_{i}_p"], "coeff": "-sqrt(2)"}],
            "frappat_form": f"a_{i}^+ e",
            "transformation": (
                f"a_{i}^+ e = a_{i}^+ * sqrt(2)*a_0 = -sqrt(2)*a_0*a_i^+ "
                f"({{a_i^+, a_0}}=0 => a_i^+*a_0 = -a_0*a_i^+)"
            ),
            "parity": 1,
            "note": (
                f"Odd root +ε_{i}. Frappat p.219: E_{{ε_i}} = a_i^+ e. "
                "PBW reorder: a_i^+ e = -sqrt(2)*a_0*a_i^+. "
                "Coefficient is -sqrt(2) (sign from anticommutation with a_0)."
            ),
        }
        r[f"E_eps{i}_m"] = {
            "standard_form": [{"words": ["a_0", f"a_{i}_m"], "coeff": "sqrt(2)"}],
            "frappat_form": f"e a_{i}^-",
            "transformation": (
                f"e a_{i}^- = sqrt(2)*a_0*a_{i}^- (already PBW order)"
            ),
            "parity": 1,
            "note": (
                f"Odd root -ε_{i}. Frappat p.219: E_{{-ε_i}} = e a_i^-. "
                "PBW: already in order. Coefficient sqrt(2)."
            ),
        }

    return r
