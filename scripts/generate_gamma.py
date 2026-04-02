"""
generate_B_gamma_json.py

Generate JSON gamma structures for B(m,n) = osp(2m+1|2n) Lie superalgebras.

Schema version: 5.0
- Target algebra: osp(2m+1|2n) for general m >= 0, n >= 1
- Input: data/algebra_structures/B_{m}_{n}_structure.json (v5.0 schema)
- Output: data/gamma_structures/B_{m}_{n}_gamma.json

Key changes from v4 (generate_B_gamma_json_v4.py):
- gh parameter names: numeric index gh_i_j -> label-based gh_a0_b1p etc.
- Sign convention: [b_j, a_i] = -gh[i][j]·κ  (v5.0, κ leftmost in PBW)
- odd×odd negate removed: κ is PBW-leftmost, so κ·γ is already correct sign
- fermion_labels read in creation-first order (v5.0 canonical_pairs)
- algebra_reference: family "B", m, n (not "osp", 2m+1, 2n)
- inhomogeneous_deformation: v5.0 nested structure
- gamma_matrix wrapped in {"brackets": {...}}
- schema_version: "5.0"

Exchange relations (v5.0):
    [b_j, a_i] = -gh[i][j]·κ   (κ leftmost in PBW)
    [b_j, b_k] = J_{j,k}        (from symplectic_matrix in JSON)
    {a_i, a_j} = delta_{ij}/2   (canonical anticommutation)
    κ^2 = 0

Usage:
    python experiments/generate_B_gamma_json.py
    python experiments/generate_B_gamma_json.py --m 0 --n 1
    python experiments/generate_B_gamma_json.py --m 1 --n 1
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict, Counter

import sympy as sp

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))


# ---------------------------------------------------------------------------
# Label extraction from v5.0 JSON
# ---------------------------------------------------------------------------

def extract_oscillator_labels(
    structure_data: Dict[str, Any]
) -> Tuple[List[str], List[str], List[List[int]]]:
    """Extract fermion labels, boson labels, symplectic matrix from v5.0 JSON.

    fermion_labels: [a_0, a_1_p, a_1_m, ..., a_m_p, a_m_m]  (PBW order, creation-first)
    boson_labels:   [b_1_p, b_1_m, ..., b_n_p, b_n_m]        (PBW order, creation-first)
    """
    osc_gen = structure_data["oscillator_generators"]
    osc_rel = structure_data["oscillator_relations"]

    fermion_labels: List[str] = []

    # Supplementary fermion (always a_0)
    supp = osc_gen.get("supplementary_fermion", {})
    if supp.get("label"):
        fermion_labels.append(supp["label"])

    # Standard fermionic oscillators: v5.0 canonical_pairs is creation-first
    ferm_section = osc_gen.get("fermions", {})
    canonical_pairs = osc_rel["standard_fermion_anticommutators"].get("canonical_pairs", [])

    if canonical_pairs:
        for pair in canonical_pairs:
            for key in ["creation", "annihilation"]:  # v5.0: creation-first
                lbl = pair.get(key)
                if lbl and lbl not in fermion_labels:
                    fermion_labels.append(lbl)
    else:
        for lbl in ferm_section.get("labels", []):
            if lbl not in fermion_labels:
                fermion_labels.append(lbl)

    # Bosonic labels and symplectic matrix
    symp = osc_rel["bosonic_commutators"]["symplectic_matrix"]
    boson_labels: List[str] = symp["basis_order"]
    symplectic_matrix: List[List[int]] = symp["matrix"]

    return fermion_labels, boson_labels, symplectic_matrix


def _gh_label(fermion_lbl: str, boson_lbl: str) -> str:
    """Build gh parameter name from oscillator labels.

    e.g. ("a_0", "b_1_p") -> "gh_a0_b1p"
    """
    f = fermion_lbl.replace("_", "")  # a0, a1p, a1m
    b = boson_lbl.replace("_", "")    # b1p, b1m
    return f"gh_{f}_{b}"


# ---------------------------------------------------------------------------
# OscillatorRewriterV5
# ---------------------------------------------------------------------------

class OscillatorRewriterV5:
    """Rewrites non-commutative oscillator words into PBW standard form.

    PBW order (v5.0): fermion_labels (PBW order) < boson_labels (PBW order).
    κ is treated as leftmost in PBW — it is NOT included in the word alphabet;
    instead it is accumulated as a commutative scalar coefficient.

    Exchange relations (v5.0):
        [b_j, a_i] = -gh[i][j]·κ   (sign negated vs v4)
        [b_j, b_k] = J_{j,k}
        {a_i, a_j} = delta_{ij}/2
        κ^2 = 0
    """

    def __init__(
        self,
        fermion_labels: List[str],
        boson_labels: List[str],
        symplectic_matrix: List[List[int]],
    ):
        self.fermion_labels = fermion_labels
        self.boson_labels = boson_labels
        self.n_fermions = len(fermion_labels)
        self.n_bosons = len(boson_labels)

        self.fermion_order: Dict[str, int] = {lbl: i for i, lbl in enumerate(fermion_labels)}
        self.boson_order: Dict[str, int] = {lbl: j for j, lbl in enumerate(boson_labels)}

        self.pbw_rank: Dict[str, int] = {}
        for i, lbl in enumerate(fermion_labels):
            self.pbw_rank[lbl] = i
        for j, lbl in enumerate(boson_labels):
            self.pbw_rank[lbl] = self.n_fermions + j

        # Symplectic form as sparse dict
        self.J: Dict[Tuple[int, int], int] = {}
        for i, row in enumerate(symplectic_matrix):
            for j, val in enumerate(row):
                if val != 0:
                    self.J[(i, j)] = val

        self.kappa = sp.Symbol('kappa', commutative=True)

        # gh symbols: label-based names (v5.0)
        self.gh_syms: Dict[Tuple[int, int], sp.Symbol] = {
            (i, j): sp.Symbol(_gh_label(fermion_labels[i], boson_labels[j]), commutative=True)
            for i in range(self.n_fermions)
            for j in range(self.n_bosons)
        }

        # Standard commutative output symbols
        self.b_std: Dict[str, sp.Symbol] = {
            lbl: sp.Symbol(f"std_{lbl}", commutative=True) for lbl in boson_labels
        }
        self.a_std: Dict[str, sp.Symbol] = {
            lbl: sp.Symbol(f"std_{lbl}", commutative=True) for lbl in fermion_labels
        }

    def _kappa_reduce(self, expr: sp.Expr) -> sp.Expr:
        return sp.expand(expr.subs(self.kappa ** 2, 0))

    def _is_boson(self, lbl: str) -> bool:
        return lbl in self.boson_order

    def _is_fermion(self, lbl: str) -> bool:
        return lbl in self.fermion_order

    def rewrite_word(self, word: List[str]) -> sp.Expr:
        """Recursively rewrite oscillator word into PBW standard commutative form."""
        if not word:
            return sp.Integer(1)

        is_sorted = all(
            self.pbw_rank[word[i]] <= self.pbw_rank[word[i + 1]]
            for i in range(len(word) - 1)
        )

        if is_sorted:
            a_counts = Counter(w for w in word if self._is_fermion(w))
            b_part = [w for w in word if self._is_boson(w)]

            coeff = sp.Integer(1)
            remaining_a = []
            for a_lbl in sorted(a_counts.keys(), key=lambda x: self.fermion_order[x]):
                cnt = a_counts[a_lbl]
                coeff *= sp.Rational(1, 2) ** (cnt // 2)
                if cnt % 2 == 1:
                    remaining_a.append(a_lbl)

            expr = coeff
            for a_lbl in remaining_a:
                expr *= self.a_std[a_lbl]
            for b_lbl in b_part:
                expr *= self.b_std[b_lbl]
            return self._kappa_reduce(expr)

        for idx in range(len(word) - 1):
            left, right = word[idx], word[idx + 1]
            if self.pbw_rank[left] <= self.pbw_rank[right]:
                continue

            # Case 1: boson * fermion -> fermion * boson - gh[i][j]·κ  (v5.0 sign)
            if self._is_boson(left) and self._is_fermion(right):
                j_flat = self.boson_order[left]
                i_flat = self.fermion_order[right]

                swapped = word[:idx] + [right, left] + word[idx + 2:]
                term1 = self.rewrite_word(swapped)
                reduced = word[:idx] + word[idx + 2:]
                gh_val = self.gh_syms.get((i_flat, j_flat), sp.Integer(0))

                # κ は odd: before 中の odd 元を通過するたびに符号が変わる
                n_odd_before = sum(1 for tok in word[:idx] if tok in self.fermion_order)
                kappa_sign = (-1) ** n_odd_before

                term2 = kappa_sign * (-self.kappa) * gh_val * self.rewrite_word(reduced)
                return self._kappa_reduce(term1 + term2)

            # Case 2: boson * boson (wrong order)
            if self._is_boson(left) and self._is_boson(right):
                j_flat = self.boson_order[left]
                k_flat = self.boson_order[right]

                swapped = word[:idx] + [right, left] + word[idx + 2:]
                term1 = self.rewrite_word(swapped)
                J_val = self.J.get((j_flat, k_flat), 0)
                if J_val != 0:
                    reduced = word[:idx] + word[idx + 2:]
                    term2 = J_val * self.rewrite_word(reduced)
                    return self._kappa_reduce(term1 + term2)
                return self._kappa_reduce(term1)

            # Case 3: fermion * fermion (wrong order) -> anticommute
            if self._is_fermion(left) and self._is_fermion(right):
                swapped = word[:idx] + [right, left] + word[idx + 2:]
                return self._kappa_reduce(-self.rewrite_word(swapped))

        return sp.Integer(0)

    def rewrite_expr(self, expr: sp.Expr) -> sp.Expr:
        """Rewrite a SymPy NC expression into standard commutative form."""
        expanded = sp.expand(expr)
        result = sp.Integer(0)
        terms = expanded.args if expanded.is_Add else [expanded]

        for term in terms:
            if term.is_Mul:
                coeff = sp.Integer(1)
                nc_args = []
                for arg in term.args:
                    if arg.is_commutative:
                        coeff *= arg
                    else:
                        nc_args.append(arg)
                nc_part = (sp.Integer(1) if not nc_args
                           else nc_args[0] if len(nc_args) == 1
                           else sp.Mul(*nc_args, evaluate=False))
            else:
                coeff = term if term.is_commutative else sp.Integer(1)
                nc_part = sp.Integer(1) if term.is_commutative else term

            if nc_part == 1:
                result += coeff
                continue

            word = []
            parts = nc_part.args if getattr(nc_part, 'is_Mul', False) else [nc_part]
            for s in parts:
                if getattr(s, 'is_Symbol', False):
                    word.append(str(s))
                elif getattr(s, 'is_Pow', False):
                    base, exp = s.args
                    word.extend([str(base)] * int(exp))
                else:
                    raise ValueError(f"Unexpected NC part: {s}")

            result += coeff * self.rewrite_word(word)

        return self._kappa_reduce(result)


# ---------------------------------------------------------------------------
# Realization map builder (v5.0 JSON format)
# ---------------------------------------------------------------------------

def build_realization_map_v5(
    rewriter: OscillatorRewriterV5,
    gen_realization_section: Dict[str, Any],
) -> Dict[str, sp.Expr]:
    """Parse generator_realization.realizations section of v5.0 JSON."""
    all_labels = rewriter.fermion_labels + rewriter.boson_labels
    nc_syms: Dict[str, sp.Symbol] = {
        lbl: sp.Symbol(lbl, commutative=False) for lbl in all_labels
    }

    realizations_raw = gen_realization_section.get("realizations", {})
    result: Dict[str, sp.Expr] = {}

    for gen_name, gen_data in realizations_raw.items():
        standard_form = gen_data.get("standard_form", [])
        expr = sp.Integer(0)

        for term_dict in standard_form:
            words: List[str] = term_dict["words"]
            coeff = sp.sympify(term_dict["coeff"])

            if not words:
                continue  # constant term (identity): skip

            # Skip terms referencing oscillator labels absent from this algebra
            # (e.g. H_0 in B(m,0) formally references b_0± which don't exist).
            unknown = [lbl for lbl in words if lbl not in nc_syms]
            if unknown:
                print(
                    f"  NOTE: skipping term {words} in realization of '{gen_name}' "
                    f"(unknown labels {unknown} for this algebra)"
                )
                continue

            nc_word = sp.Integer(1)
            for lbl in words:
                nc_word = sp.Mul(nc_word, nc_syms[lbl], evaluate=False)

            expr += coeff * nc_word

        result[gen_name] = expr

    return result


# ---------------------------------------------------------------------------
# Basis recovery: std monomial -> algebra generator
# ---------------------------------------------------------------------------

def build_std_to_gen_map(
    rewriter: OscillatorRewriterV5,
    realization_exprs: Dict[str, sp.Expr],
) -> Dict[sp.Expr, Tuple[str, sp.Expr]]:
    all_std_syms = set(rewriter.a_std.values()) | set(rewriter.b_std.values())
    std_to_gen: Dict[sp.Expr, Tuple[str, sp.Expr]] = {}

    for gen_name, nc_expr in realization_exprs.items():
        std = rewriter.rewrite_expr(nc_expr)
        std_no_kappa = sp.expand(std.subs(rewriter.kappa, 0))

        if std_no_kappa == 0:
            continue

        terms = std_no_kappa.args if std_no_kappa.is_Add else [std_no_kappa]

        for term in terms:
            if term.is_Mul:
                num = sp.Integer(1)
                monomial = sp.Integer(1)
                for arg in term.args:
                    if arg.free_symbols & all_std_syms:
                        monomial *= arg
                    else:
                        num *= arg
            elif term in all_std_syms:
                num = sp.Integer(1)
                monomial = term
            elif term.is_Pow and term.args[0] in all_std_syms:  # ← 追加
                num = sp.Integer(1)
                monomial = term
            else:
                continue

            if monomial != 1:
                std_to_gen[monomial] = (gen_name, sp.Integer(1) / num)


    return std_to_gen


# ---------------------------------------------------------------------------
# Gamma coefficient extraction
# ---------------------------------------------------------------------------

def extract_gamma_coefficients_v5(
    rewriter: OscillatorRewriterV5,
    bracket_std: sp.Expr,
    std_to_gen: Dict[sp.Expr, Tuple[str, sp.Expr]],
) -> Dict[str, Dict[str, str]]:
    """Extract gamma coefficients from a standardized bracket expression.

    Picks out κ^1 terms and decomposes by gh parameter monomials.
    No sign negation needed: κ is PBW-leftmost, so [X,Y]_L = [X,Y] + κ·γ(X,Y)
    is directly read from the κ^1 coefficient.

    Returns:
        {gen_name: {gh_param_str: coeff_str}}
    """
    #print(bracket_std)

    expanded = sp.expand(bracket_std.subs(rewriter.kappa ** 2, 0))
    terms = expanded.args if expanded.is_Add else [expanded]

    gamma_map: Dict[str, Dict[str, sp.Expr]] = {}
    all_gh_syms = set(rewriter.gh_syms.values())

    for term in terms:
        kappa_power = 0
        osc_monomial = sp.Integer(1)
        gh_part = sp.Integer(1)
        numeric = sp.Integer(1)

        parts = term.args if term.is_Mul else [term]
        for arg in parts:
            if arg == rewriter.kappa:
                kappa_power += 1
            elif arg.is_Pow and arg.args[0] == rewriter.kappa:
                kappa_power += int(arg.args[1])
            elif (arg in rewriter.a_std.values()
                  or arg in rewriter.b_std.values()):
                osc_monomial *= arg
            elif arg.is_Pow and arg.args[0] in rewriter.b_std.values():
                osc_monomial *= arg
            elif arg in all_gh_syms:
                gh_part *= arg
            else:
                numeric *= arg

        if kappa_power != 1 or osc_monomial == 1:
            continue

        if osc_monomial not in std_to_gen:
            continue
        gen_name, inv_coeff = std_to_gen[osc_monomial]
        full_coeff = sp.expand(numeric * gh_part * inv_coeff)

        coeff_terms = full_coeff.args if full_coeff.is_Add else [full_coeff]
        for ct in coeff_terms:
            num_part = sp.Integer(1)
            gh_vars = []

            ct_parts = ct.args if ct.is_Mul else [ct]
            for arg in ct_parts:
                if arg in all_gh_syms:
                    gh_vars.append(arg)
                else:
                    num_part *= arg

            gh_key = "*".join(sorted(str(g) for g in gh_vars)) if gh_vars else "1"

            if gen_name not in gamma_map:
                gamma_map[gen_name] = {}
            if gh_key in gamma_map[gen_name]:
                gamma_map[gen_name][gh_key] = sp.expand(
                    gamma_map[gen_name][gh_key] + num_part
                )
            else:
                gamma_map[gen_name][gh_key] = num_part

    return {
        gen_name: {k: str(v) for k, v in gh_dict.items() if sp.sympify(v) != 0}
        for gen_name, gh_dict in gamma_map.items()
        if any(sp.sympify(v) != 0 for v in gh_dict.values())
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_algebra(m: int, n: int, structure_json_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Processing B({m},{n}) = osp({2*m+1}|{2*n})")
    print(f"{'='*60}")

    with open(structure_json_path, 'r', encoding='utf-8') as f:
        structure_data = json.load(f)

    schema_ver = structure_data.get("schema_version", "?")
    if schema_ver != "5.0":
        print(f"  WARNING: structure JSON schema_version is '{schema_ver}', expected '5.0'")

    basis_even: List[str] = structure_data["basis"]["even"]
    basis_odd: List[str] = structure_data["basis"]["odd"]
    gen_realization_section = structure_data["generator_realization"]

    fermion_labels, boson_labels, symplectic_matrix = extract_oscillator_labels(
        structure_data
    )
    print(f"  Fermion labels: {fermion_labels}")
    print(f"  Boson labels:   {boson_labels}")

    rewriter = OscillatorRewriterV5(fermion_labels, boson_labels, symplectic_matrix)
    realization_exprs = build_realization_map_v5(rewriter, gen_realization_section)
    std_to_gen = build_std_to_gen_map(rewriter, realization_exprs)

    missing = [g for g in basis_even + basis_odd if g not in realization_exprs]
    if missing:
        print(f"  WARNING: in basis but missing in realizations: {missing}")

    print(f"  Basis even: {basis_even}")
    print(f"  Basis odd:  {basis_odd}")
    print(f"  std->gen map entries: {len(std_to_gen)}")

    def get_expr(gen_name: str) -> Optional[sp.Expr]:
        if gen_name not in realization_exprs:
            print(f"  SKIP: no realization for '{gen_name}'")
            return None
        return realization_exprs[gen_name]

    all_gamma = []

    # (1) Odd x Odd (anticommutator)
    print("\n--- Odd x Odd ---")
    for i, q_i in enumerate(basis_odd):
        for j in range(i, len(basis_odd)):
            q_j = basis_odd[j]
            e_i, e_j = get_expr(q_i), get_expr(q_j)
            if e_i is None or e_j is None:
                continue
            anticomm = sp.expand(e_i * e_j + e_j * e_i)
            std = rewriter.rewrite_expr(anticomm)
            # v5.0: no negate — κ is leftmost, γ coefficient is read directly
            gamma = extract_gamma_coefficients_v5(rewriter, std, std_to_gen)
            key = f"{q_i},{q_j}"
            if gamma:
                all_gamma.append((key, gamma, 'odd_odd'))
                print(f"  {{{q_i}, {q_j}}} -> {gamma}")
            else:
                print(f"  {{{q_i}, {q_j}}} -> (no κ term)")

    # (2) Even x Even (commutator)
    print("\n--- Even x Even ---")
    for i, m_i in enumerate(basis_even):
        for j in range(i + 1, len(basis_even)):
            m_j = basis_even[j]
            e_i, e_j = get_expr(m_i), get_expr(m_j)
            if e_i is None or e_j is None:
                continue
            comm = sp.expand(e_i * e_j - e_j * e_i)
            std = rewriter.rewrite_expr(comm)
            gamma = extract_gamma_coefficients_v5(rewriter, std, std_to_gen)
            key = f"{m_i},{m_j}"
            if gamma:
                all_gamma.append((key, gamma, 'even_even'))
                print(f"  [{m_i}, {m_j}] -> {gamma}")
            else:
                print(f"  [{m_i}, {m_j}] -> (no κ term)")

    # (3) Even x Odd (commutator)
    print("\n--- Even x Odd ---")
    for m_i in basis_even:
        for q_j in basis_odd:
            e_i, e_j = get_expr(m_i), get_expr(q_j)
            if e_i is None or e_j is None:
                continue
            comm = sp.expand(e_i * e_j - e_j * e_i)
            std = rewriter.rewrite_expr(comm)
            gamma = extract_gamma_coefficients_v5(rewriter, std, std_to_gen)
            key = f"{m_i},{q_j}"
            if gamma:
                all_gamma.append((key, gamma, 'even_odd'))
                print(f"  [{m_i}, {q_j}] -> {gamma}")

    # Build gamma_matrix
    sort_order = {'odd_odd': 0, 'even_even': 1, 'even_odd': 2}
    gamma_matrix = OrderedDict(
        (k, v)
        for k, v, _ in sorted(all_gamma, key=lambda x: (sort_order[x[2]], x[0]))
    )

    n_ferm = len(fermion_labels)
    n_bos = len(boson_labels)

    # gh parameter matrix: label-based names (v5.0)
    gh_matrix = [
        [_gh_label(fermion_labels[i], boson_labels[j]) for j in range(n_bos)]
        for i in range(n_ferm)
    ]

    out_dict = {
        "schema_version": "5.0",
        "algebra_reference": {
            "family": structure_data["algebra"]["family"],
            "m": m,
            "n": n,
            "cartan_type": structure_data["algebra"]["cartan_type"],
            "structure_file": f"../algebra_structures/B_{m}_{n}_structure.json",
            "structure_schema_version": "5.0",
        },
        "inhomogeneous_deformation": {
            "exchange_relation": "[b_j, a_i] = G[b_j][a_i]·κ = -gh[i][j]·κ",
            "gram_matrix_convention": {
                "description": (
                    "Full Gram matrix G of the skew-supersymmetric bilinear form, "
                    "in PBW basis order (fermion-first). "
                    "Entry G[x][y] = (x|y)."
                ),
                "basis_order": (
                    "[ a_0, a_1_p, a_1_m, ..., a_m_p, a_m_m, "
                    "b_1_p, b_1_m, ..., b_n_p, b_n_m ]"
                ),
                "gh_submatrix": (
                    "gh[a_i][b_j] = (a_i | b_j). "
                    "Relation to paper B matrix: gh = -B_paper "
                    "(sign reversal due to basis reordering)."
                ),
            },
            "inhomogeneous_matrix": {
                "description": (
                    "gh submatrix (fermion rows × boson columns). "
                    "gh[i][j] = (f_i | e_j). "
                    "Exchange relation: [e_j, f_i] = -gh[i][j]·κ."
                ),
                "fermion_basis_order": fermion_labels,
                "boson_basis_order": boson_labels,
                "shape": [n_ferm, n_bos],
                "matrix": gh_matrix,
                "sympy_repr": gh_matrix,
            },
        },
        "gamma_matrix": {
            "description": (
                "γ(X,Y) coefficients: [X,Y]_L = [X,Y]_0 + κ·γ(X,Y). "
                "Only non-zero entries stored."
            ),
            "brackets": dict(gamma_matrix),
        },
        "computation_metadata": {
            "method": "standard_form_expansion_v5",
            "input_basis": f"Generator realizations from B_{m}_{n}_structure.json (v5.0)",
            "generated_by": "experiments/generate_B_gamma_json.py",
            "bracket_types": {
                "odd_odd": "anticommutator",
                "even_even": "commutator",
                "even_odd": "commutator",
            },
            "computed_combinations": {
                "odd_odd": "all pairs (i, j) with i <= j",
                "even_even": "all pairs (i, j) with i < j",
                "even_odd": "all pairs (i, j)",
            },
            "kappa_order": 1,
            "kappa_squared": 0,
            "description": (
                "γ(a,b) gives the κ-linear coefficient in [a,b]_L = [a,b]_0 + κ·γ(a,b). "
                "κ is PBW-leftmost (rank 0); no sign negation needed for odd×odd. "
                "Only non-zero κ terms stored."
            ),
        },
    }

    out_dir = root / "data" / "gamma_structures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"B_{m}_{n}_gamma.json"

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, indent=2, ensure_ascii=False)

    print(f"\n  -> Saved: {out_path}")
    print(f"     Non-zero gamma entries: {len(gamma_matrix)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_CASES = [(0, 1), (0, 2), (1, 1),(0, 3), (1, 2), (2, 1)]


def main():
    parser = argparse.ArgumentParser(
        description="Generate B(m,n) gamma structure JSON (v5.0 schema)"
    )
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    cases = [(args.m, args.n)] if (args.m is not None and args.n is not None) \
        else DEFAULT_CASES

    for m, n in cases:
        structure_path = (
            root / "data" / "algebra_structures" / f"B_{m}_{n}_structure.json"
        )
        if not structure_path.exists():
            print(f"\nSkipping B({m},{n}): {structure_path} not found")
            continue
        try:
            process_algebra(m, n, structure_path)
        except Exception as e:
            import traceback
            print(f"\nERROR in B({m},{n}): {e}")
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()