"""
oscillator_algebra.py

Oscillator algebra engine for B(m,n) = osp(2m+1|2n).

Computes Lie superalgebra structure constants by:
  1. Expressing each generator as a polynomial in oscillator symbols
  2. Computing super-brackets via product + CCR/CAR rewriting
  3. Decomposing the result onto the Frappat basis

PBW order (left to right):
  a0, a1_p, a1_m, a2_p, a2_m, ..., am_p, am_m,
  b1_p, b1_m, b2_p, b2_m, ..., bn_p, bn_m

Exchange relations:
  [bk_m, bl_p] = delta_{kl}          (bosonic CCR)
  {ai_m, aj_p} = delta_{ij}          (fermionic CAR)
  {a0,   a0  } = 1  =>  a0^2 = 1/2
  {ai_s, a0  } = 0  =>  ai_s * a0 = -a0 * ai_s
  [bk_s, a0  ] = 0
  [bk_s, ai_t] = 0

Coefficient ring: sympy exact arithmetic (Rational + sqrt(2)).

Conventions follow B_generators_derivation.md and B_generators.py (v5.0).
"""

import sympy as sp
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Token names (PBW order)
# ---------------------------------------------------------------------------

def _a0_tok() -> str:
    return "a0"

def _ai_tok(i: int, sign: int) -> str:
    return f"a{i}_{'p' if sign > 0 else 'm'}"

def _bk_tok(k: int, sign: int) -> str:
    return f"b{k}_{'p' if sign > 0 else 'm'}"

def _tok_rank(tok: str, m: int, n: int) -> int:
    """Assign an integer rank for PBW ordering."""
    if tok == "a0":
        return 0
    if tok.startswith("a") and "_" in tok:
        # ai_p or ai_m
        i = int(tok[1:tok.index("_")])
        s = 0 if tok.endswith("_p") else 1
        return 1 + 2 * (i - 1) + s   # 1..2m
    if tok.startswith("b") and "_" in tok:
        k = int(tok[1:tok.index("_")])
        s = 0 if tok.endswith("_p") else 1
        return 1 + 2 * m + 2 * (k - 1) + s  # 1+2m..1+2m+2n-1
    raise ValueError(f"Unknown token: {tok}")


# ---------------------------------------------------------------------------
# OscillatorRewriter  (cf. generate_gamma_structure_json.py)
# ---------------------------------------------------------------------------

class OscillatorRewriter:
    """
    Rewrites non-commutative oscillator words into standard (PBW) form
    using sympy commutative symbols.

    Each oscillator token maps to a unique commutative sympy Symbol.
    The rewriter applies exchange relations recursively, exactly as in
    generate_gamma_structure_json.OscillatorRewriter, but generalised
    for B(m,n).
    """

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n

        # Commutative symbol for each oscillator token
        self.syms: Dict[str, sp.Symbol] = {}
        self.syms["a0"] = sp.Symbol("a0", commutative=True)
        for i in range(1, m + 1):
            self.syms[_ai_tok(i, +1)] = sp.Symbol(f"a{i}_p", commutative=True)
            self.syms[_ai_tok(i, -1)] = sp.Symbol(f"a{i}_m", commutative=True)
        for k in range(1, n + 1):
            self.syms[_bk_tok(k, +1)] = sp.Symbol(f"b{k}_p", commutative=True)
            self.syms[_bk_tok(k, -1)] = sp.Symbol(f"b{k}_m", commutative=True)

        # PBW-ordered list of tokens
        self.pbw_order: List[str] = self._build_pbw_order()

        # Rank map
        self.rank: Dict[str, int] = {
            tok: self._rank(tok) for tok in self.pbw_order
        }

    def _build_pbw_order(self) -> List[str]:
        order = ["a0"]
        for i in range(1, self.m + 1):
            order += [_ai_tok(i, +1), _ai_tok(i, -1)]
        for k in range(1, self.n + 1):
            order += [_bk_tok(k, +1), _bk_tok(k, -1)]
        return order

    def _rank(self, tok: str) -> int:
        return self.pbw_order.index(tok)

    def _is_fermionic(self, tok: str) -> bool:
        """True for a0 and ai_p/ai_m tokens."""
        return tok == "a0" or (tok.startswith("a") and "_" in tok)

    def _is_bosonic(self, tok: str) -> bool:
        return tok.startswith("b") and "_" in tok

    # ------------------------------------------------------------------
    # Core rewriter: word -> sympy expression (standard form)
    # ------------------------------------------------------------------

    def rewrite_word(self, word: List[str]) -> sp.Expr:
        """
        Recursively rewrite a word (list of token strings) to a sympy
        expression in PBW standard form.

        Returns a commutative sympy expression.
        """
        if not word:
            return sp.Integer(1)

        # Check if already in PBW order
        if self._is_pbw(word):
            return self._eval_sorted_word(word)

        # Find first inversion and resolve
        for i in range(len(word) - 1):
            left, right = word[i], word[i + 1]
            rl, rr = self.rank[left], self.rank[right]
            if rl <= rr:
                continue

            # Out of PBW order: apply exchange relation
            return self._swap(word, i, left, right)

        return sp.Integer(0)  # should not reach

    def _is_pbw(self, word: List[str]) -> bool:
        for i in range(len(word) - 1):
            if self.rank[word[i]] > self.rank[word[i + 1]]:
                return False
        return True

    def _eval_sorted_word(self, word: List[str]) -> sp.Expr:
        """
        Evaluate a PBW-sorted word to a sympy expression,
        applying a0^2 = 1/2 and (ai_s)^2 = 0 reductions.
        """
        # Count a0 occurrences
        a0_count = word.count("a0")
        # Check fermionic nilpotency: (ai_s)^2 = 0
        from collections import Counter
        cnt = Counter(word)
        for tok, c in cnt.items():
            if self._is_fermionic(tok) and tok != "a0" and c > 1:
                return sp.Integer(0)

        # a0^k: a0^(2q) = (1/2)^q, a0^(2q+1) = (1/2)^q * a0_sym
        coeff = sp.Rational(1, 2) ** (a0_count // 2)
        has_a0 = (a0_count % 2 == 1)

        expr = coeff
        if has_a0:
            expr *= self.syms["a0"]
        for tok in word:
            if tok == "a0":
                continue
            expr *= self.syms[tok]
        return expr

    def _swap(self, word: List[str], i: int, left: str, right: str) -> sp.Expr:
        """
        Apply exchange relation for the pair (left, right) at position i
        and return the resulting expression.
        """
        before = word[:i]
        after = word[i + 2:]

        # Term 1: swapped pair (right * left) + possible sign
        # Term 2: commutator/anticommutator residual

        # Determine sign and residual
        sign, residual_word, residual_coeff = self._exchange(left, right)

        # Term 1: sign * [before] * right * left * [after]
        term1 = sp.Integer(0)
        if sign != 0:
            t1_word = before + [right, left] + after
            term1 = sign * self.rewrite_word(t1_word)

        # Term 2: residual (a scalar or shorter word)
        term2 = sp.Integer(0)
        if residual_coeff != 0:
            t2_word = before + residual_word + after
            term2 = residual_coeff * self.rewrite_word(t2_word)

        return sp.expand(term1 + term2)

    def _exchange(
        self, left: str, right: str
    ) -> Tuple[int, List[str], sp.Expr]:
        """
        Return (sign, residual_word, residual_coeff) for the exchange
        of (left, right) which are out of PBW order.

        Relation: left * right = sign * (right * left) + residual_coeff * residual_word

        Sign conventions:
          - Bosons commute:      bk_s * bl_t = bl_t * bk_s  + delta (if k==l, s=m, t=p)
          - Fermions anti-commute: ai_s * aj_t = -aj_t * ai_s + delta_{ij}*delta_{st reversed}
          - a0 anti-commutes with ai_s: a0 * ai_s = -ai_s * a0
          - a0 commutes with bk_s: a0 * bk_s = bk_s * a0
          - ai_s commutes with bk_t: ai_s * bk_t = bk_t * ai_s
          - a0 * a0 = 1/2 (handled via _eval_sorted_word, but here left=right=a0 won't occur
            since a0 only appears once in PBW sorted word; however in intermediate words
            we may have a0*a0 adjacent)
        """
        lf = self._is_fermionic(left)
        rf = self._is_fermionic(right)

        # a0 * a0  -> 1/2 * []
        if left == "a0" and right == "a0":
            return 0, [], sp.Rational(1, 2)

        # ai_s * a0  -> -a0 * ai_s  (anti-commute, no residual)
        if left != "a0" and lf and right == "a0":
            return -1, [], sp.Integer(0)

        # ai_s * bk_t  -> bk_t * ai_s  (commute)
        if lf and self._is_bosonic(right):
            return 1, [], sp.Integer(0)

        # bk_s * a0  -> a0 * bk_s  (commute)
        if self._is_bosonic(left) and right == "a0":
            return 1, [], sp.Integer(0)

        # bk_s * ai_t  -> ai_t * bk_s  (commute)
        if self._is_bosonic(left) and right != "a0" and rf:
            return 1, [], sp.Integer(0)

        # ai_s * aj_t  (fermionic, both non-a0)
        if left != "a0" and lf and right != "a0" and rf:
            i = int(left[1:left.index("_")])
            si = +1 if left.endswith("_p") else -1
            j = int(right[1:right.index("_")])
            sj = +1 if right.endswith("_p") else -1
            # Anti-commute: ai_s * aj_t = -aj_t * ai_s + {ai_s, aj_t}
            # {ai_m, aj_p} = delta_{ij}  (only m,p pair gives delta)
            if i == j and si == -1 and sj == +1:
                # left=ai_m, right=ai_p (but rank(ai_m) > rank(ai_p) so out of order)
                return -1, [], sp.Integer(1)
            else:
                return -1, [], sp.Integer(0)

        # bk_s * bl_t  (both bosonic)
        if self._is_bosonic(left) and self._is_bosonic(right):
            k = int(left[1:left.index("_")])
            sk = +1 if left.endswith("_p") else -1
            l = int(right[1:right.index("_")])
            sl = +1 if right.endswith("_p") else -1
            # Commute: bk_s * bl_t = bl_t * bk_s + [bk_s, bl_t]
            # [bk_m, bl_p] = delta_{kl}
            if k == l and sk == -1 and sl == +1:
                return 1, [], sp.Integer(1)
            else:
                return 1, [], sp.Integer(0)

        raise ValueError(f"Unhandled exchange: {left}, {right}")

    def rewrite_expr(self, expr: sp.Expr) -> sp.Expr:
        """
        Rewrite a sympy non-commutative expression to standard form.
        Wraps rewrite_word for multi-term expressions.
        (Same interface as generate_gamma_structure_json.OscillatorRewriter)
        """
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
                if not nc_args:
                    result += coeff
                    continue
                nc_part = sp.Mul(*nc_args, evaluate=False) if len(nc_args) > 1 else nc_args[0]
            else:
                if term.is_commutative:
                    result += term
                    continue
                coeff = sp.Integer(1)
                nc_part = term

            word = self._extract_word(nc_part)
            std = self.rewrite_word(word)
            result += coeff * std

        return sp.expand(result)

    def _extract_word(self, nc_expr: sp.Expr) -> List[str]:
        """Extract ordered token list from a non-commutative product."""
        word = []
        if getattr(nc_expr, "is_Mul", False):
            args = nc_expr.args
        else:
            args = [nc_expr]
        for s in args:
            if getattr(s, "is_Symbol", False):
                word.append(str(s))
            elif getattr(s, "is_Pow", False):
                base, exp = s.args
                word.extend([str(base)] * int(exp))
            else:
                raise ValueError(f"Unexpected non-commutative part: {s}")
        return word


# ---------------------------------------------------------------------------
# BmnAlgebra: generator realizations + structure constant computation
# ---------------------------------------------------------------------------

class BmnAlgebra:
    """
    Frappat-basis generator realizations and structure constant computation
    for B(m,n) = osp(2m+1|2n).

    Generator realizations follow B_generators_derivation.md §5.
    Structure constants are computed by:
      1. Evaluate [X, Y] = XY ± YX in oscillator algebra
      2. Rewrite to PBW standard form via OscillatorRewriter
      3. Decompose onto Frappat basis via _std_to_basis()
    """

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.rw = OscillatorRewriter(m, n)
        syms = self.rw.syms

        # Shorthand accessors
        def a0():
            return syms["a0"]
        def aip(i):
            return syms[_ai_tok(i, +1)]
        def aim(i):
            return syms[_ai_tok(i, -1)]
        def bkp(k):
            return syms[_bk_tok(k, +1)]
        def bkm(k):
            return syms[_bk_tok(k, -1)]

        # ------------------------------------------------------------------
        # Generator realizations: name -> (sympy_expr, parity)
        # All expressions are in commutative sympy symbols (standard form)
        # because generators are *defined* already in normal order.
        # ------------------------------------------------------------------
        self._gens: Dict[str, Tuple[sp.Expr, int]] = {}

        def reg(name: str, expr: sp.Expr, parity: int):
            self._gens[name] = (sp.expand(expr), parity)

        # --- Cartan ---
        for k in range(1, n + 1):
            if k < n:
                expr = bkp(k) * bkm(k) - bkp(k + 1) * bkm(k + 1)
            else:  # k == n
                if m >= 1:
                    expr = bkp(n) * bkm(n) + aip(1) * aim(1)
                else:
                    # B(0,n): H_n = b_n^+ b_n^- + 1/2
                    expr = bkp(n) * bkm(n) + sp.Rational(1, 2)
            reg(f"H_{k}", expr, 0)

        for i in range(1, m + 1):
            ni = n + i
            if i < m:
                expr = aip(i) * aim(i) - aip(i + 1) * aim(i + 1)
            else:  # i == m
                expr = 2 * aip(m) * aim(m) - 1
            reg(f"H_{ni}", expr, 0)

        # --- Even bosonic non-Cartan ---
        for k in range(1, n + 1):
            reg(f"E_2del{k}_p", bkp(k) ** 2, 0)
            reg(f"E_2del{k}_m", bkm(k) ** 2, 0)
        for k in range(1, n + 1):
            for l in range(k + 1, n + 1):
                reg(f"E_del{k}_del{l}_pp", bkp(k) * bkp(l), 0)
                reg(f"E_del{k}_del{l}_mm", bkm(k) * bkm(l), 0)
                reg(f"E_del{k}_del{l}_pm", bkp(k) * bkm(l), 0)
                reg(f"E_del{k}_del{l}_mp", bkm(k) * bkp(l), 0)

        # --- Even fermionic non-Cartan (m >= 2) ---
        for i in range(1, m + 1):
            for j in range(i + 1, m + 1):
                reg(f"E_eps{i}_eps{j}_pp", aip(i) * aip(j), 0)
                reg(f"E_eps{i}_eps{j}_mm", aim(i) * aim(j), 0)
                reg(f"E_eps{i}_eps{j}_pm", aip(i) * aim(j), 0)
                reg(f"E_eps{i}_eps{j}_mp", aim(i) * aip(j), 0)

        # --- Even: E_{±ε_i} = ∓sqrt(2) * a0 * ai^±  (§5.1 注1) ---
        for i in range(1, m + 1):
            reg(f"E_eps{i}_p", -sp.sqrt(2) * a0() * aip(i), 0)
            reg(f"E_eps{i}_m", +sp.sqrt(2) * a0() * aim(i), 0)

        # --- Odd: E_{±δ_k} = a0 * bk^± ---
        for k in range(1, n + 1):
            reg(f"E_del{k}_p", a0() * bkp(k), 1)
            reg(f"E_del{k}_m", a0() * bkm(k), 1)

        # --- Odd: E_{±ε_i ± δ_k} = ai^± bk^± ---
        for i in range(1, m + 1):
            for k in range(1, n + 1):
                reg(f"E_eps{i}_del{k}_pp", aip(i) * bkp(k), 1)
                reg(f"E_eps{i}_del{k}_mm", aim(i) * bkm(k), 1)
                reg(f"E_eps{i}_del{k}_pm", aip(i) * bkm(k), 1)
                reg(f"E_eps{i}_del{k}_mp", aim(i) * bkp(k), 1)

        # ------------------------------------------------------------------
        # Build basis for decomposition
        # Exclude Cartan from the direct monomial map (handled separately)
        # ------------------------------------------------------------------
        self._basis_map = self._build_basis_map()

    def _build_basis_map(self) -> Dict[sp.Expr, List[Tuple[str, sp.Expr]]]:
        """
        Build a map from canonical monomial -> [(gen_name, coeff_in_realization)].
        Used to invert the decomposition in _std_to_basis().
        Only non-Cartan generators are included here; Cartan is handled by
        matching number-operator patterns.
        """
        bmap: Dict[sp.Expr, List[Tuple[str, sp.Expr]]] = {}
        for name, (expr, _) in self._gens.items():
            if name.startswith("H_"):
                continue
            # expr may be a single monomial or a sum (E_{±ε_i} has sqrt(2)*a0*ai)
            # We store the canonical monomial form (as sympy key)
            terms = expr.args if expr.is_Add else [expr]
            for term in terms:
                key = sp.expand(term)
                if key not in bmap:
                    bmap[key] = []
                bmap[key].append((name, sp.Integer(1)))
        return bmap

    # ------------------------------------------------------------------
    # Non-commutative product (before rewriting)
    # ------------------------------------------------------------------

    def _nc_product(self, expr1: sp.Expr, expr2: sp.Expr) -> sp.Expr:
        """
        Form the non-commutative product expr1 * expr2 by creating
        non-commutative symbols for each token and multiplying.
        """
        return sp.expand(expr1 * expr2)

    # ------------------------------------------------------------------
    # Super-bracket computation
    # ------------------------------------------------------------------

    def super_bracket_expr(self, name_x: str, name_y: str) -> sp.Expr:
        """
        Compute the super-bracket [X, Y] as a sympy expression in standard form.

        Evaluates XY ± YX where the sign follows the Z_2-grading:
          - odd * odd  -> anticommutator {X, Y} = XY + YX
          - otherwise  -> commutator [X, Y] = XY - YX
        """
        expr_x, px = self._gens[name_x]
        expr_y, py = self._gens[name_y]

        # Form non-commutative products using non-commutative sympy symbols
        nc_x = self._to_nc(name_x)
        nc_y = self._to_nc(name_y)

        if px == 1 and py == 1:
            bracket_nc = sp.expand(nc_x * nc_y + nc_y * nc_x)
        else:
            bracket_nc = sp.expand(nc_x * nc_y - nc_y * nc_x)

        return self.rw.rewrite_expr(bracket_nc)

    def _to_nc(self, name: str) -> sp.Expr:
        expr, _ = self._gens[name]
        nc_syms = {
            tok: sp.Symbol(tok, commutative=False)
            for tok in self.rw.pbw_order
        }
        subs_map = {self.rw.syms[tok]: nc_syms[tok] for tok in self.rw.pbw_order}

        # sp.expand を使わず、Add の各項を個別に処理して順序を保持する
        expanded = sp.expand(expr)  # 可換式の展開は問題なし
        terms = expanded.args if expanded.is_Add else [expanded]

        nc_result = sp.Integer(0)
        for term in terms:
            if term.is_Mul:
                # 数値係数と振動子シンボルを分離
                coeff = sp.Integer(1)
                # PBW 順に従って振動子トークンを収集
                tok_list = []
                for arg in term.args:
                    if arg.is_number or (arg.is_Pow and arg.args[0].is_number):
                        coeff *= arg
                    else:
                        # 可換シンボルのみが来るはず
                        name_str = str(arg) if arg.is_Symbol else str(arg.args[0])
                        count = 1 if arg.is_Symbol else int(arg.args[1])
                        tok_list.extend([name_str] * count)
                # PBW 順にソート（可換式なので順序は保証されている）
                tok_list.sort(key=lambda t: self.rw.rank[t])
                # 非可換積を順序通りに構築
                nc_term = coeff
                for tok in tok_list:
                    nc_term = sp.Mul(nc_term, nc_syms[tok], evaluate=False)
                nc_result = nc_result + nc_term
            elif term.is_number:
                nc_result = nc_result + term
            elif term in self.rw.syms.values():
                nc_result = nc_result + nc_syms[str(term)]
            else:
                nc_result = nc_result + term.subs(subs_map)

        return nc_result


    def a_to_nc(self, name: str) -> sp.Expr:
        """
        Return a non-commutative sympy expression for generator `name`.
        Each oscillator token becomes a non-commutative Symbol.
        """
        expr, _ = self._gens[name]
        # expr is in commutative symbols; we re-express it using NC symbols
        nc_syms = {
            tok: sp.Symbol(tok, commutative=False)
            for tok in self.rw.pbw_order
        }
        # Map commutative -> non-commutative
        subs_map = {self.rw.syms[tok]: nc_syms[tok] for tok in self.rw.pbw_order}
        nc_expr = expr.subs(subs_map)
        return sp.expand(nc_expr)

    # ------------------------------------------------------------------
    # Basis decomposition
    # ------------------------------------------------------------------

    def std_to_basis(self, std_expr: sp.Expr) -> Dict[str, sp.Expr]:
        """
        Decompose a standard-form sympy expression into Frappat basis elements.

        Returns {gen_name: coefficient} where coefficient is a sympy expression
        (possibly involving sqrt(2)).

        Strategy:
          - Each non-Cartan generator has a unique monomial realization.
          - Cartan generators are identified by matching number-operator patterns.
          - Constant terms are discarded (not in the algebra basis, but recorded
            as a warning if non-zero).
        """
        expanded = sp.expand(std_expr)
        terms = expanded.args if expanded.is_Add else [expanded]

        result: Dict[str, sp.Expr] = {}

        def add_to(name: str, val: sp.Expr):
            result[name] = sp.expand(result.get(name, sp.Integer(0)) + val)

        syms = self.rw.syms
        m, n = self.m, self.n

        for term in terms:
            # Separate numeric coefficient from oscillator variables
            coeff, osc_monomial = self._split_coeff(term)
            if coeff == 0:
                continue

            # --- Constant term (no oscillators) ---
            if osc_monomial == sp.Integer(1):
                # Constants arise from commutators of number operators (e.g. H_n = b^+b^- + 1/2)
                # They should cancel in well-formed brackets; discard silently
                continue

            # --- Try to match against non-Cartan generators ---
            matched = False
            for name, (gen_expr, _) in self._gens.items():
                if name.startswith("H_"):
                    continue
                # gen_expr = c_gen * osc_monomial_gen
                c_gen, gen_mono = self._split_coeff(gen_expr)
                if c_gen == 0:
                    continue
                if sp.expand(gen_mono - osc_monomial) == 0:
                    add_to(name, sp.Rational(coeff) / c_gen
                           if isinstance(coeff, int) else coeff / c_gen)
                    matched = True
                    break

            if matched:
                continue

            # --- Try Cartan: match number-operator pairs ---
            cartan_coeff = self._match_cartan(osc_monomial, coeff)
            if cartan_coeff:
                for hname, hval in cartan_coeff.items():
                    add_to(hname, hval)
                continue

            # Unmatched term — should not occur for valid bracket output
            import warnings
            warnings.warn(
                f"std_to_basis: unmatched term {coeff} * {osc_monomial}",
                RuntimeWarning
            )

        # Clean up zeros
        return {k: v for k, v in result.items() if sp.expand(v) != 0}

    def _split_coeff(self, term: sp.Expr) -> Tuple[sp.Expr, sp.Expr]:
        """
        Split a sympy term into (numeric_coeff, oscillator_monomial).
        """
        if term.is_Mul:
            coeff = sp.Integer(1)
            osc = sp.Integer(1)
            for arg in term.args:
                if arg.is_number:
                    coeff *= arg
                elif arg in self.rw.syms.values():
                    osc *= arg
                elif arg.is_Pow and arg.args[0] in self.rw.syms.values():
                    osc *= arg
                else:
                    # Could be sqrt(2) etc. — treat as numeric
                    coeff *= arg
            return coeff, osc
        elif term.is_number:
            return term, sp.Integer(1)
        elif term in self.rw.syms.values():
            return sp.Integer(1), term
        elif term.is_Pow and term.args[0] in self.rw.syms.values():
            return sp.Integer(1), term
        else:
            return term, sp.Integer(1)

    def _match_cartan(
        self, osc_mono: sp.Expr, coeff: sp.Expr
    ) -> Optional[Dict[str, sp.Expr]]:
        """
        Match osc_mono against number-operator products b_k^+ b_k^- or a_i^+ a_i^-.
        Returns {H_name: coefficient} if matched, else None.

        Number operators and their H-expansions (from §4.3 of derivation doc):
          b_k^+ b_k^- = sum_{j=k}^{n} H_j - a_1^+ a_1^-
                      (after further expansion of a_1^+ a_1^-)
          For B(0,n): b_k^+ b_k^- = sum_{j=k}^{n} H_j - 1/2

        We use the direct H-expressions stored from the Cartan definitions.
        """
        syms = self.rw.syms
        m, n = self.m, self.n

        # Build number-operator -> H-expansion table
        # bose: b_k^+ b_k^-
        for k in range(1, n + 1):
            target = syms[_bk_tok(k, +1)] * syms[_bk_tok(k, -1)]
            if sp.expand(osc_mono - target) == 0:
                return self._bose_num_to_H(k, coeff)

        # fermi: a_i^+ a_i^-
        for i in range(1, m + 1):
            target = syms[_ai_tok(i, +1)] * syms[_ai_tok(i, -1)]
            if sp.expand(osc_mono - target) == 0:
                return self._fermi_num_to_H(i, coeff)

        return None

    def _bose_num_to_H(self, k: int, coeff: sp.Expr) -> Dict[str, sp.Expr]:
        """
        Express coeff * b_k^+ b_k^- in terms of H generators.
        b_k^+ b_k^- = sum_{j=k}^{n} H_j - a_1^+ a_1^-   (m >= 1)
                    = sum_{j=k}^{n} H_j - 1/2              (m == 0)
        The a_1^+ a_1^- term is further expanded via _fermi_num_to_H.
        """
        result = {}
        for j in range(k, self.n + 1):
            hname = f"H_{j}"
            result[hname] = sp.expand(result.get(hname, sp.Integer(0)) + coeff)
        if self.m >= 1:
            # Subtract a_1^+ a_1^- contribution
            sub = self._fermi_num_to_H(1, coeff)
            for hname, val in sub.items():
                result[hname] = sp.expand(result.get(hname, sp.Integer(0)) - val)
        else:
            # B(0,n): subtract constant 1/2 * coeff (discarded as constant)
            pass
        return {k: v for k, v in result.items() if sp.expand(v) != 0}

    def _fermi_num_to_H(self, i: int, coeff: sp.Expr) -> Dict[str, sp.Expr]:
        """
        Express coeff * a_i^+ a_i^- in terms of H generators.
        a_i^+ a_i^- = sum_{l=i}^{m-1} H_{n+l} + (H_{n+m} + 1) / 2
        The constant +1/2 is discarded.
        """
        result = {}
        n = self.n
        for l in range(i, self.m):
            hname = f"H_{n+l}"
            result[hname] = sp.expand(result.get(hname, sp.Integer(0)) + coeff)
        # Last term: (H_{n+m} + 1)/2 -> H_{n+m}/2 (constant dropped)
        hname = f"H_{n+self.m}"
        result[hname] = sp.expand(
            result.get(hname, sp.Integer(0)) + coeff / 2
        )
        return {k: v for k, v in result.items() if sp.expand(v) != 0}

    # ------------------------------------------------------------------
    # Public API: structure constants
    # ------------------------------------------------------------------

    def compute_structure_constants(
        self,
    ) -> Dict[Tuple[str, str], Dict[str, str]]:
        """
        Compute all structure constants for B(m,n).

        Iterates over all ordered pairs (X, Y) of basis generators,
        computes [X, Y], and decomposes the result onto the Frappat basis.

        Returns:
            {(gen1, gen2): {result_gen: coeff_str}}
        where coeff_str follows the convention in B_generators.py:
            "p"           for rational p
            "q*sqrt(2)"   for 0 + q*sqrt(2)
            "p+q*sqrt(2)" for p + q*sqrt(2)  (q > 0)
            "p-|q|*sqrt(2)" for p - |q|*sqrt(2)  (q < 0)
        """
        from itertools import product as iproduct

        gen_names = self.gen_names()
        br: Dict[Tuple[str, str], Dict[str, str]] = {}

        for name_x, name_y in iproduct(gen_names, gen_names):
            if name_x == name_y and self.parity(name_x) == 0:
                continue  # [even, even] with same generator is zero by antisymmetry

            std = self.super_bracket_expr(name_x, name_y)
            decomp = self.std_to_basis(std)

            if decomp:
                br[(name_x, name_y)] = {
                    rname: _coeff_to_str(val)
                    for rname, val in decomp.items()
                }

        # Remove empty entries
        return {k: v for k, v in br.items() if v}

    def gen_names(self) -> List[str]:
        return list(self._gens.keys())

    def parity(self, name: str) -> int:
        return self._gens[name][1]


# ---------------------------------------------------------------------------
# Coefficient string formatting  (matches B_generators.py convention)
# ---------------------------------------------------------------------------

def _coeff_to_str(val: sp.Expr) -> str:
    """
    Convert a sympy expression to a coefficient string.
    Supported forms: rational, rational*sqrt(2), rational ± rational*sqrt(2).
    """
    val = sp.nsimplify(sp.expand(val), [sp.sqrt(2)], rational=True)
    val = sp.radsimp(val)

    # Decompose: val = p + q*sqrt(2)
    p = val.subs(sp.sqrt(2), 0)
    q_sqrt2 = sp.expand(val - p)
    q = sp.simplify(q_sqrt2 / sp.sqrt(2))

    # Verify decomposition
    if sp.expand(p + q * sp.sqrt(2) - val) != 0:
        # Fallback: return sympy string
        return str(val)

    p = sp.nsimplify(p, rational=True)
    q = sp.nsimplify(q, rational=True)

    if q == 0:
        return str(p)
    elif p == 0:
        return f"{q}*sqrt(2)"
    else:
        sign = "+" if q > 0 else ""
        return f"{p}{sign}{q}*sqrt(2)"
