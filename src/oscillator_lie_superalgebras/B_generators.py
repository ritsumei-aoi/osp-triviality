"""
B_generators.py

Frappat-notation basis and structure constants for B(0,n) = osp(1|2n).
v4.1: Frappat naming convention (H_k, E_2delk_p, E_eps{i}_del{k}_pm, etc.)

Conventions:
  - Supplementary fermion: a_0 corresponds to Frappat's e via e = sqrt(2) * a_0.
    This relation is noted once in metadata; individual generators do not repeat it.
  - For B(0,n): E_{±δ_k} = a_0 b_k^± (implementation), = (1/sqrt(2)) e b_k^± (Frappat).
  - Structure constants are computed directly from Frappat formulae (not via v3.0 basis).

Phase 1: m=0 only (B(0,n) family).
Phase 2: m>=1 will be added in a future session.
"""

from typing import Dict, List, Tuple
from fractions import Fraction


# ---------------------------------------------------------------------------
# Basis ordering
# ---------------------------------------------------------------------------

def build_B0n_basis(n: int) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Build Frappat-notation basis for B(0,n) in physical ordering.

    Ordering convention (even):
      Cartan: H_1, ..., H_n
      Positive even roots: E_2del{k}_p (k=1..n), E_del{k}_del{l}_pp (k<l)
      Negative even roots: E_2del{k}_m (k=1..n), E_del{k}_del{l}_mm (k<l)
      Other even roots: E_del{k}_del{l}_pm, E_del{k}_del{l}_mp (k<l)
    Ordering convention (odd):
      Positive odd roots: E_del{k}_p (k=1..n)
      Negative odd roots: E_del{k}_m (k=1..n)

    Returns:
        even_basis, odd_basis, parity dict
    """
    even_basis: List[str] = []
    odd_basis: List[str] = []
    parity: Dict[str, int] = {}

    # Cartan
    for k in range(1, n + 1):
        name = f"H_{k}"
        even_basis.append(name)
        parity[name] = 0

    # Positive even roots
    for k in range(1, n + 1):
        name = f"E_2del{k}_p"
        even_basis.append(name)
        parity[name] = 0
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            name = f"E_del{k}_del{l}_pp"
            even_basis.append(name)
            parity[name] = 0

    # Negative even roots
    for k in range(1, n + 1):
        name = f"E_2del{k}_m"
        even_basis.append(name)
        parity[name] = 0
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            name = f"E_del{k}_del{l}_mm"
            even_basis.append(name)
            parity[name] = 0

    # Other even roots (not Cartan, not purely ±2δ_k, not purely ±(δ_k+δ_l)):
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            name_pm = f"E_del{k}_del{l}_pm"
            name_mp = f"E_del{k}_del{l}_mp"
            even_basis.append(name_pm)
            even_basis.append(name_mp)
            parity[name_pm] = 0
            parity[name_mp] = 0

    # Positive odd roots
    for k in range(1, n + 1):
        name = f"E_del{k}_p"
        odd_basis.append(name)
        parity[name] = 1

    # Negative odd roots
    for k in range(1, n + 1):
        name = f"E_del{k}_m"
        odd_basis.append(name)
        parity[name] = 1

    return even_basis, odd_basis, parity


# ---------------------------------------------------------------------------
# Structure constants: direct Frappat implementation for B(0,n)
# ---------------------------------------------------------------------------

def build_B0n_structure_constants(
    n: int,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Compute structure constants for B(0,n) directly from Frappat formulae.

    Generator realizations (implementation convention):
      H_k   = b_k^+ b_k^- - b_{k+1}^+ b_{k+1}^-   (1 <= k < n)
      H_n   = b_n^+ b_n^- + 1/2
      E_{2δ_k}^±      = (b_k^±)^2
      E_{±δ_k±δ_l}^±   = b_k^± b_l^±               (k < l)
      E_{±δ_k}        = a_0 b_k^±   (B(0,n) convention: e -> sqrt(2)*a_0,
                                      so impl = (1/sqrt(2)) * Frappat)

    Structure constants are given as exact rational strings.
    Brackets are indexed by (gen1, gen2) with gen names in Frappat notation.
    For odd-odd pairs the bracket is the anticommutator; otherwise commutator.

    Derivation basis:
      [b_k^-, b_l^+] = delta_{kl}
      {a_0, a_0} = 1  (a_0^2 = 1/2)
      [a_0, b_k^±] = 0
    """
    br: Dict[Tuple[str, str], Dict[str, str]] = {}

    def _parse_coeff(s: str) -> tuple:
        """Parse coefficient string to (p, q) where value = p + q*sqrt(2)."""
        if s == "0":
            return (Fraction(0), Fraction(0))
        if "*sqrt(2)" in s:
            q_str = s.replace("*sqrt(2)", "")
            return (Fraction(0), Fraction(q_str))
        if "+sqrt(2)" in s or "-sqrt(2)" in s:
            # e.g. "1/2+1/2*sqrt(2)"
            for sep in ["+", "-"]:
                idx = s.rfind(sep)
                if idx > 0:
                    p_str = s[:idx]
                    q_str = s[idx:].replace("*sqrt(2)", "")
                    return (Fraction(p_str), Fraction(q_str))
        return (Fraction(s), Fraction(0))

    def _format_coeff(p: Fraction, q: Fraction) -> str:
        """Format (p, q) to coefficient string."""
        if q == 0:
            return str(p)
        elif p == 0:
            return f"{q}*sqrt(2)"
        else:
            sign = "+" if q > 0 else ""
            return f"{p}{sign}{q}*sqrt(2)"

    def add(g1: str, g2: str, result: str, coeff: str) -> None:
        key = (g1, g2)
        if key not in br:
            br[key] = {}
        ep, eq = _parse_coeff(br[key].get(result, "0"))
        np_, nq = _parse_coeff(coeff)
        rp, rq = ep + np_, eq + nq
        if rp == 0 and rq == 0:
            br[key].pop(result, None)
        else:
            br[key][result] = _format_coeff(rp, rq)

    # Helper: Cartan eigenvalue <H_k, root>
    # H_k acts on b_l^± by ±delta_{kl} - ±delta_{k+1,l} for k<n,
    # and H_n acts on b_n^± by ±1 (the 1/2 constant commutes as 0).
    # More precisely: [H_k, b_l^±] = ±(δ_{kl} - δ_{k+1,l}) b_l^±  for k<n
    #                 [H_n, b_n^±] = ±b_n^±

    def cartan_bose_eigenvalue(k: int, l: int, sign: int) -> int:
        """
        Eigenvalue of H_k acting on b_l^{sign} where sign=+1 or -1.
        [H_k, b_l^+] = ev * b_l^+,  [H_k, b_l^-] = -ev * b_l^-
        """
        if k < n:
            ev = (1 if l == k else 0) - (1 if l == k + 1 else 0)
        else:  # k == n
            ev = (1 if l == n else 0)
        return sign * ev

    # -----------------------------------------------------------------------
    # 1. [H_k, E_{±2δ_l}] = ±2*(δ_{kl} - δ_{k+1,l}) E_{2δ_l}^±
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(1, n + 1):
            hk = f"H_{k}"
            ep = f"E_2del{l}_p"
            em = f"E_2del{l}_m"
            # [H_k, (b_l^+)^2]: each b_l^+ contributes eigenvalue
            ev = cartan_bose_eigenvalue(k, l, +1)
            coeff_p = 2 * ev  # two b_l^+ factors
            if coeff_p != 0:
                add(hk, ep, ep, str(coeff_p))
                add(ep, hk, ep, str(-coeff_p))
            ev_m = cartan_bose_eigenvalue(k, l, -1)
            coeff_m = 2 * ev_m
            if coeff_m != 0:
                add(hk, em, em, str(coeff_m))
                add(em, hk, em, str(-coeff_m))

    # -----------------------------------------------------------------------
    # 2. [H_k, E_{±(δ_l+δ_p)}] = ±(ev_l + ev_p) E_{±(δ_l+δ_p)}  (l < p)
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(1, n + 1):
            for p in range(l + 1, n + 1):
                hk = f"H_{k}"
                ep = f"E_del{l}_del{p}_pp"
                em = f"E_del{l}_del{p}_mm"
                ev = cartan_bose_eigenvalue(k, l, +1) + cartan_bose_eigenvalue(k, p, +1)
                if ev != 0:
                    add(hk, ep, ep, str(ev))
                    add(ep, hk, ep, str(-ev))
                ev_m = cartan_bose_eigenvalue(k, l, -1) + cartan_bose_eigenvalue(k, p, -1)
                if ev_m != 0:
                    add(hk, em, em, str(ev_m))
                    add(em, hk, em, str(-ev_m))

    # -----------------------------------------------------------------------
    # 3. [H_k, E_{δ_l-δ_p}] (l < p)
    #     E_{δ_l-δ_p} = b_l^+ b_p^-  → 固有値 = ev(l,+1) + ev(p,-1)
    #     E_{-δ_l+δ_p} = b_l^- b_p^+ → 固有値 = ev(l,-1) + ev(p,+1)
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(1, n + 1):
            for p in range(l + 1, n + 1):
                hk = f"H_{k}"
                pm = f"E_del{l}_del{p}_pm"
                mp = f"E_del{l}_del{p}_mp"
                ev_pm = cartan_bose_eigenvalue(k, l, +1) + cartan_bose_eigenvalue(k, p, -1)
                ev_mp = cartan_bose_eigenvalue(k, l, -1) + cartan_bose_eigenvalue(k, p, +1)
                if ev_pm != 0:
                    add(hk, pm, pm, str(ev_pm))
                    add(pm, hk, pm, str(-ev_pm))
                if ev_mp != 0:
                    add(hk, mp, mp, str(ev_mp))
                    add(mp, hk, mp, str(-ev_mp))
  
    
    # -----------------------------------------------------------------------
    # 4. [H_k, E_{±δ_l}] = ±ev_l * E_{±δ_l}
    #    E_{±δ_l} = a_0 b_l^±; H_k commutes with a_0, acts on b_l^±
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(1, n + 1):
            hk = f"H_{k}"
            ep = f"E_del{l}_p"
            em = f"E_del{l}_m"
            ev_p = cartan_bose_eigenvalue(k, l, +1)
            ev_m = cartan_bose_eigenvalue(k, l, -1)
            if ev_p != 0:
                add(hk, ep, ep, str(ev_p))
                add(ep, hk, ep, str(-ev_p))
            if ev_m != 0:
                add(hk, em, em, str(ev_m))
                add(em, hk, em, str(-ev_m))

    # -----------------------------------------------------------------------
    # 5. [E_{2δ_k}, E_{-2δ_k}] = [(b_k^+)^2, (b_k^-)^2] = -4b_k^+b_k^- - 2
    #
    #    Derivation:
    #      (b_k^+)^2(b_k^-)^2 = (b_k^+b_k^-)^2 - b_k^+b_k^-
    #      (b_k^-)^2(b_k^+)^2 = (b_k^+b_k^-)^2 + 3b_k^+b_k^- + 2
    #      difference = -4b_k^+b_k^- - 2
    #
    #    Express in terms of H generators via telescoping sum:
    #      b_k^+b_k^- = sum_{j=k}^{n} H_j - 1/2
    #      (using H_j = b_j^+b_j^- - b_{j+1}^+b_{j+1}^- for j<n,
    #             H_n = b_n^+b_n^- + 1/2)
    #
    #    Therefore:
    #      -4b_k^+b_k^- - 2 = -4*(sum_{j=k}^n H_j - 1/2) - 2
    #                       = -4 * sum_{j=k}^n H_j
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # 5. [E_{2δ_k}, E_{-2δ_k}] = [(b_k^+)^2, (b_k^-)^2] = -4b_k^+b_k^- - 2
    #
    #    Derivation:
    #      (b_k^+)^2(b_k^-)^2 = (b_k^+b_k^-)^2 - b_k^+b_k^-
    #      (b_k^-)^2(b_k^+)^2 = (b_k^+b_k^-)^2 + 3b_k^+b_k^- + 2
    #      difference = -4b_k^+b_k^- - 2
    #
    #    Express in terms of H generators via telescoping sum:
    #      b_k^+b_k^- = sum_{j=k}^{n} H_j - 1/2
    #      (using H_j = b_j^+b_j^- - b_{j+1}^+b_{j+1}^- for j<n,
    #             H_n = b_n^+b_n^- + 1/2)
    #
    #    Therefore:
    #      -4b_k^+b_k^- - 2 = -4*(sum_{j=k}^n H_j - 1/2) - 2
    #                       = -4 * sum_{j=k}^n H_j
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        ep = f"E_2del{k}_p"
        em = f"E_2del{k}_m"
        # b_k^+b_k^- = sum_{j=k}^n H_j - 1/2  (via bose_number_in_H analog)
        # -4*b_k^+b_k^- - 2 = -4*(sum_{j=k}^n H_j - 1/2) - 2 = -4*sum_{j=k}^n H_j
        for j in range(k, n + 1):
            add(ep, em, f"H_{j}", "-4")
        # antisymmetry
        for j in range(k, n + 1):
            add(em, ep, f"H_{j}", "4")

    # -----------------------------------------------------------------------
    # 6. [E_{2δ_k}, E_{2δ_l}] = 0 (same sign, commute since b_k b_l = b_l b_k)
    #    [E_{2δ_k}, E_{2δ_l}] = 0  (same)
    #    Already zero by default.

    # -----------------------------------------------------------------------
    # 7. [E_{2δ_k}^+, E_{-(δ_l+δ_p)}]  (k<=l<p)
    #    [(b_k^+)^2, b_l^- b_p^-] = -2δ_{kl} b_k^+ b_p^- - 2δ_{kp} b_k^+ b_l^-
    #    [E_{2δ_k}^-, E_{+(δ_l+δ_p)}]
    #    [(b_k^-)^2, b_l^+ b_p^+] = +2δ_{kl} b_k^- b_p^+ + 2δ_{kp} b_k^- b_l^+
    #
    #    Result naming: b_r^+ b_s^- with r<s → E_del{r}_del{s}_pm
    #                   b_r^- b_s^+ with r<s → E_del{r}_del{s}_mp
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(1, n + 1):
            for p in range(l + 1, n + 1):
                ep_k  = f"E_2del{k}_p"
                em_k  = f"E_2del{k}_m"
                em_lp = f"E_del{l}_del{p}_mm"
                ep_lp = f"E_del{l}_del{p}_pp"

                # [(b_k^+)^2, b_l^- b_p^-]:
                # -2δ_{kl} * b_k^+ b_p^-  (k=l < p なので k < p: E_del{k}_del{p}_pm)
                if k == l:
                    res = f"E_del{k}_del{p}_pm"
                    add(ep_k, em_lp, res, "-2")
                    add(em_lp, ep_k, res, "2")
                # -2δ_{kp} * b_k^+ b_l^-  (l < p=k なので l < k: E_del{l}_del{k}_pm)
                if k == p:
                    res = f"E_del{l}_del{k}_mp"
                    add(ep_k, em_lp, res, "-2")
                    add(em_lp, ep_k, res, "2")

                # [(b_k^-)^2, b_l^+ b_p^+]:
                # +2δ_{kl} * b_k^- b_p^+  (k=l < p: E_del{k}_del{p}_mp)
                if k == l:
                    res = f"E_del{k}_del{p}_mp"
                    add(em_k, ep_lp, res, "2")
                    add(ep_lp, em_k, res, "-2")
                # +2δ_{kp} * b_k^- b_l^+  (l < k=p: E_del{l}_del{k}_mp)
                if k == p:
                    res = f"E_del{l}_del{k}_pm"
                    add(em_k, ep_lp, res, "2")
                    add(ep_lp, em_k, res, "-2")

    # -----------------------------------------------------------------------
    # 7b. [E_{2δ_k}^+, E_{±(δ_l-δ_p)}]  and  [E_{2δ_k}^-, E_{±(δ_l-δ_p)}]
    #     [(b_k^+)^2, b_l^-b_p^+] = -2δ_{kl} b_k^+b_p^+   (k=l)
    #     [(b_k^+)^2, b_l^+b_p^-] = -2δ_{kp} b_k^+b_l^+   (k=p)
    #     [(b_k^-)^2, b_l^-b_p^+] = +2δ_{kp} b_k^-b_l^-   (k=p)
    #     [(b_k^-)^2, b_l^+b_p^-] = +2δ_{kl} b_k^-b_p^-   (k=l)
    #
    #     Result naming (r < s): b_r^+b_s^+ → E_del{r}_del{s}_pp
    #                            b_r^-b_s^- → E_del{r}_del{s}_mm
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(1, n + 1):
            for p in range(l + 1, n + 1):  # l < p に限定（l==p は除外）
                ep_k = f"E_2del{k}_p"
                em_k = f"E_2del{k}_m"
                gen_mp = f"E_del{l}_del{p}_mp"  # b_l^- b_p^+
                gen_pm = f"E_del{l}_del{p}_pm"  # b_l^+ b_p^-

                # [(b_k^+)^2, b_l^-b_p^+] = -2δ_{kl} b_k^+b_p^+
                if k == l:
                    r, s = (k, p) if k < p else (p, k)
                    res = f"E_del{r}_del{s}_pp"
                    add(ep_k, gen_mp, res, "-2")
                    add(gen_mp, ep_k, res, "2")

                # [(b_k^+)^2, b_l^+b_p^-] = -2δ_{kp} b_k^+b_l^+
                if k == p:
                    r, s = (k, l) if k < l else (l, k)
                    res = f"E_del{r}_del{s}_pp"
                    add(ep_k, gen_pm, res, "-2")
                    add(gen_pm, ep_k, res, "2")

                # [(b_k^-)^2, b_l^-b_p^+] = +2δ_{kp} b_k^-b_l^-
                if k == p:
                    r, s = (k, l) if k < l else (l, k)
                    res = f"E_del{r}_del{s}_mm"
                    add(em_k, gen_mp, res, "2")
                    add(gen_mp, em_k, res, "-2")

                # [(b_k^-)^2, b_l^+b_p^-] = +2δ_{kl} b_k^-b_p^-
                if k == l:
                    r, s = (k, p) if k < p else (p, k)
                    res = f"E_del{r}_del{s}_mm"
                    add(em_k, gen_pm, res, "2")
                    add(gen_pm, em_k, res, "-2")

    # -----------------------------------------------------------------------
    # 8. [E_{2δ_k}^±, E_{∓δ_l}]  (even-odd)
    #    [(b_k^+)^2, a_0 b_l^-] = a_0 [(b_k^+)^2, b_l^-] = a_0 * (-2*delta_{kl}*b_k^+)
    #                            = -2*delta_{kl} * a_0 b_k^+ = -2*delta_{kl} * E_{δ_k}^+
    #    [(b_k^-)^2, a_0 b_l^+] = a_0 [(b_k^-)^2, b_l^+] = a_0 * 2*delta_{kl}*b_k^-
    #                            = 2*delta_{kl} * E_{δ_k}^-   ... sign check:
    #    [b_k^-, b_l^+] = delta_{kl}, so [(b_k^-)^2, b_l^+] = 2*delta_{kl}*b_k^-
    #    => [(b_k^-)^2, a_0 b_l^+] = 2*delta_{kl}*a_0*b_k^- = 2*delta_{kl}*E_{δ_k}^-
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(1, n + 1):
            if k != l:
                continue
            ep_k = f"E_2del{k}_p"
            em_k = f"E_2del{k}_m"
            e_odd_p = f"E_del{k}_p"
            e_odd_m = f"E_del{k}_m"
            # [E_{2δ_k}^+, E_{-δ_k}] = -2 E_{δ_k}^+
            add(ep_k, e_odd_m, e_odd_p, "-2")
            add(e_odd_m, ep_k, e_odd_p, "2")
            # [E_{2δ_k}^-, E_{+δ_k}] = 2 E_{δ_k}^-
            add(em_k, e_odd_p, e_odd_m, "2")
            add(e_odd_p, em_k, e_odd_m, "-2")

    # -----------------------------------------------------------------------
    # 8c. Same-index cross brackets: [E_{δk+δl}, E_{δk-δl}] etc.
    #     [b_k^+b_l^+, b_k^+b_l^-] = b_k^+[b_l^+,b_l^-]b_k^+ ... 
    #
    #     [E_del{k}_del{l}_pp, E_del{k}_del{l}_pm] = -E_2del{k}_p
    #     [E_del{k}_del{l}_pp, E_del{k}_del{l}_mp] = -E_2del{l}_p
    #     [E_del{k}_del{l}_mm, E_del{k}_del{l}_pm] = +E_2del{l}_m
    #     [E_del{k}_del{l}_mm, E_del{k}_del{l}_mp] = +E_2del{k}_m
    #     [E_del{k}_del{l}_pm, E_del{k}_del{l}_mp] = ?
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            pp = f"E_del{k}_del{l}_pp"
            mm = f"E_del{k}_del{l}_mm"
            pm = f"E_del{k}_del{l}_pm"
            mp = f"E_del{k}_del{l}_mp"
            e2k_p = f"E_2del{k}_p"
            e2k_m = f"E_2del{k}_m"
            e2l_p = f"E_2del{l}_p"
            e2l_m = f"E_2del{l}_m"

            # [b_k^+b_l^+, b_k^+b_l^-] = b_k^+[b_l^+,b_l^-] b_... 
            # = b_k^+(b_l^+b_l^- - b_l^-b_l^+)... 直接計算:
            # [b_k^+b_l^+, b_k^+b_l^-] = b_k^+[b_l^+,b_k^+b_l^-] + [b_k^+,b_k^+b_l^-]b_l^+
            # = b_k^+(b_k^+[b_l^+,b_l^-]) + 0 = b_k^+*b_k^+*(-1) = -(b_k^+)^2
            add(pp, pm, e2k_p, "-1")
            add(pm, pp, e2k_p, "1")

            # [b_k^+b_l^+, b_k^-b_l^+] = [b_k^+,b_k^-](b_l^+)^2 ... 
            # = b_l^+[b_k^+,b_k^-b_l^+] + [b_k^+,b_k^-b_l^+]b_l^+... 直接:
            # [b_k^+b_l^+, b_k^-b_l^+] = [b_k^+,b_k^-]b_l^+b_l^+ = (-1)(b_l^+)^2
            add(pp, mp, e2l_p, "-1")
            add(mp, pp, e2l_p, "1")

            # [b_k^-b_l^-, b_k^+b_l^-] = [b_k^-,b_k^+]b_l^-b_l^- = (-1)(b_l^-)^2... 
            # 符号: [b_k^-, b_k^+] = -1 → -(b_l^-)^2 = -E_2del{l}_m? 
            # Wait: E_2del{l}_m = (b_l^-)^2 なので result = -E_2del{l}_m
            # しかし [mm, pm]: [b_k^-b_l^-, b_k^+b_l^-]
            # = b_l^-[b_k^-, b_k^+b_l^-] + [b_k^-, b_k^+b_l^-]... 
            # [b_k^-, b_k^+]b_l^- → (-1)b_l^-... × b_l^- = (-1)(b_l^-)^2
            # hmm: 全体として [b_k^-b_l^-, b_k^+b_l^-] = [b_k^-, b_k^+](b_l^-)^2 = -(b_l^-)^2
            add(mm, pm, e2l_m, "1")
            add(pm, mm, e2l_m, "-1")

            # [b_k^-b_l^-, b_k^-b_l^+] = (b_k^-)^2[b_l^-,b_l^+] = (b_k^-)^2*(-1) ... 
            # [b_k^-b_l^-, b_k^-b_l^+] = b_k^-[b_l^-,b_k^-b_l^+] + [b_k^-,b_k^-b_l^+]b_l^-
            # = b_k^-(b_k^-[b_l^-,b_l^+]) + 0 = b_k^-*b_k^-*(-1) = -(b_k^-)^2
            add(mm, mp, e2k_m, "1")    # [b_k^-b_l^-, b_k^-b_l^+] = +(b_k^-)^2
            add(mp, mm, e2k_m, "-1")


            # [b_k^-b_l^+, b_k^+b_l^-] = -sum_{j=k}^{l-1} H_j  (k < l)
            for j in range(k, l):
                add(mp, pm, f"H_{j}", "-1")
                add(pm, mp, f"H_{j}", "1")

            # # [b_k^-b_l^-, b_k^+b_l^+] = sum_{j=k}^n H_j + sum_{j=l}^n H_j
            # for j in range(k, n + 1):
            #     add(mm, pp, f"H_{j}", "1")
            #     add(pp, mm, f"H_{j}", "-1")
            # for j in range(l, n + 1):
            #     add(mm, pp, f"H_{j}", "1")
            #     add(pp, mm, f"H_{j}", "-1")


    # -----------------------------------------------------------------------
    # 9. [E_{δ_k+δ_l}^+, E_{∓δ_p}]  (even-odd, n>=2)
    #    [b_k^+ b_l^+, a_0 b_p^-] = a_0 [b_k^+ b_l^+, b_p^-]
    #    [b_k^+ b_l^+, b_p^-] = delta_{kp}*b_l^+ + delta_{lp}*b_k^+  (by Leibniz, [b^+,b^-]=1)
    #    wait: [b_k^+, b_p^-] = delta_{kp} (from [b^-,b^+]=1 => [b^+,b^-]=-[b^-,b^+]=-delta)
    #    Actually: [b_p^-, b_k^+] = delta_{pk} => [b_k^+, b_p^-] = delta_{kp}... 
    #    Let's be careful: {b^-, b^+}? No, bosons: [b^-, b^+] = 1 => [b^+, b^-] = -1... 
    #    Standard: [b_k^-, b_l^+] = delta_{kl}, so [b_l^+, b_k^-] = -delta_{kl}
    #    => [b_k^+ b_l^+, b_p^-] = b_k^+[b_l^+,b_p^-] + [b_k^+,b_p^-]b_l^+
    #                             = -delta_{lp}*b_k^+ - delta_{kp}*b_l^+
    #    => [E_{δ_k+δ_l}^+, E_{-δ_p}] = a_0 * (-delta_{lp}*b_k^+ - delta_{kp}*b_l^+)
    #                                  = -delta_{lp}*E_{δ_k}^+ - delta_{kp}*E_{δ_l}^+
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            for p in range(1, n + 1):
                ep_kl = f"E_del{k}_del{l}_pp"
                em_kl = f"E_del{k}_del{l}_mm"
                e_odd_m_p = f"E_del{p}_m"
                e_odd_p_p = f"E_del{p}_p"
                # [E_{δ_k+δ_l}^+, E_{-δ_p}]
                if l == p:
                    add(ep_kl, e_odd_m_p, f"E_del{k}_p", "-1")
                    add(e_odd_m_p, ep_kl, f"E_del{k}_p", "1")
                if k == p:
                    add(ep_kl, e_odd_m_p, f"E_del{l}_p", "-1")
                    add(e_odd_m_p, ep_kl, f"E_del{l}_p", "1")
                # [E_{δ_k+δ_l}^-, E_{+δ_p}]
                # [b_k^- b_l^-, a_0 b_p^+] = a_0[b_k^- b_l^-, b_p^+]
                # [b_k^- b_l^-, b_p^+] = b_k^-[b_l^-,b_p^+] + [b_k^-,b_p^+]b_l^-
                #   = delta_{lp}*b_k^- + delta_{kp}*b_l^-
                if l == p:
                    add(em_kl, e_odd_p_p, f"E_del{k}_m", "1")
                    add(e_odd_p_p, em_kl, f"E_del{k}_m", "-1")
                if k == p:
                    add(em_kl, e_odd_p_p, f"E_del{l}_m", "1")
                    add(e_odd_p_p, em_kl, f"E_del{l}_m", "-1")

    # -----------------------------------------------------------------------
    # 9b. [E_{δk-δl}_pm, E_{±δp}]  and  [E_{-δk+δl}_mp, E_{±δp}]
    #     [b_k^+b_l^-, a_0 b_p^+] = a_0 delta_{lp} b_k^+ = delta_{lp} E_{δk}^+
    #     [b_k^+b_l^-, a_0 b_p^-] = a_0(-delta_{kp}) b_l^- = -delta_{kp} E_{δl}^-
    #     [b_k^-b_l^+, a_0 b_p^+] = -a_0 delta_{kp} b_l^+ = -delta_{kp} E_{δl}^+
    #     [b_k^-b_l^+, a_0 b_p^-] = a_0 delta_{lp} b_k^- = delta_{lp} E_{δk}^-
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            pm = f"E_del{k}_del{l}_pm"   # b_k^+ b_l^-
            mp = f"E_del{k}_del{l}_mp"   # b_k^- b_l^+  ← 注: mp は E_del{k}_del{l}_mp
            # ↑ 修正: mp = f"E_del{k}_del{l}_mp"
            for p in range(1, n + 1):
                e_odd_p = f"E_del{p}_p"
                e_odd_m = f"E_del{p}_m"
                # [pm, E_δp^+]: delta_{lp} * E_{δk}^+
                if l == p:
                    add(pm, e_odd_p, f"E_del{k}_p", "1")
                    add(e_odd_p, pm, f"E_del{k}_p", "-1")
                # [pm, E_δp^-]: -delta_{kp} * E_{δl}^-
                if k == p:
                    add(pm, e_odd_m, f"E_del{l}_m", "-1")
                    add(e_odd_m, pm, f"E_del{l}_m", "1")

                # [b_k^-b_l^+, a_0 b_p^+] = a_0 [b_k^-, b_p^+] b_l^+ = a_0 delta_{kp} b_l^+
                #                           = +delta_{kp} * E_{δl}^+   ← 符号 +1
                if k == p:
                    add(mp, e_odd_p, f"E_del{l}_p", "1")   # "-1" → "1"
                    add(e_odd_p, mp, f"E_del{l}_p", "-1")  # "1" → "-1"

                # [mp, E_δp^-]: -delta_{lp} * E_{δk}^-
                if l == p:
                    add(mp, e_odd_m, f"E_del{k}_m", "-1")  # "1" → "-1"
                    add(e_odd_m, mp, f"E_del{k}_m", "1")   # "-1" → "1"
   

    # -----------------------------------------------------------------------
    # 10. {E_{δ_k}, E_{δ_l}} = {a_0 b_k^+, a_0 b_l^+}
    #     = a_0^2 b_k^+ b_l^+ + a_0^2 b_l^+ b_k^+ = (1/2)(b_k^+ b_l^+ + b_l^+ b_k^+)
    #     Bosons commute: = b_k^+ b_l^+ = E_{δ_k+δ_l}^+  (k<l) or E_{2δ_k}^+ (k=l)
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(k, n + 1):
            ek = f"E_del{k}_p"
            el = f"E_del{l}_p"
            if k == l:
                add(ek, el, f"E_2del{k}_p", "1")
            else:
                add(ek, el, f"E_del{k}_del{l}_pp", "1")
                add(el, ek, f"E_del{k}_del{l}_pp", "1")

    # -----------------------------------------------------------------------
    # 11. {E_{δ_k}^-, E_{δ_l}^-} = b_k^- b_l^- = E_{-δ_k-δ_l} or E_{2δ_k}^-
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(k, n + 1):
            ek = f"E_del{k}_m"
            el = f"E_del{l}_m"
            if k == l:
                add(ek, el, f"E_2del{k}_m", "1")
            else:
                add(ek, el, f"E_del{k}_del{l}_mm", "1")
                add(el, ek, f"E_del{k}_del{l}_mm", "1")

    # -----------------------------------------------------------------------
    # 12. {E_{δ_k}, E_{-δ_l}} = a_0^2(b_k^+ b_l^- + b_l^- b_k^+)
    #                              = (1/2)(b_k^+ b_l^- + b_l^- b_k^+)
    #     k == l: = b_k^+ b_k^- + 1/2 = H_k  (B(0,n) Cartan formula with constant)
    #             expressed as sum of H_j: b_k^+b_k^- = sum_{j=k}^n H_j - 1/2
    #             so {E_{δ_k}^+, E_{δ_k}^-} = sum_{j=k}^n H_j - 1/2 + 1/2 = sum_{j=k}^n H_j
    #             Wait: H_k = b_k^+b_k^- + 1/2 (for k=n), so {E_{δ_n}^+,E_{δ_n}^-} = H_n. ✓
    #             For k < n: b_k^+b_k^- + 1/2 = H_k + b_{k+1}^+b_{k+1}^- + 1/2
    #             = H_k + H_{k+1} + ... + H_n  (by telescoping)
    #     k != l: = (1/2)(b_k^+ b_l^- + b_l^- b_k^+) = b_k^+ b_l^- = E_{δ_k-δ_l} (bosons commute)
    # -----------------------------------------------------------------------
    for k in range(1, n + 1):
        for l in range(1, n + 1):
            ek_p = f"E_del{k}_p"
            el_m = f"E_del{l}_m"
            if k == l:
                # {E_{δ_k}^+, E_{δ_k}^-} = sum_{j=k}^n H_j
                for j in range(k, n + 1):
                    add(ek_p, el_m, f"H_{j}", "1")
                    add(el_m, ek_p, f"H_{j}", "1")
            else:
                # {E_{δ_k}^+, E_{-δ_l}^-} = E_{δ_k-δ_l}
                if k < l:
                    name = f"E_del{k}_del{l}_pm"
                elif k > l:
                    name = f"E_del{l}_del{k}_mp"  # インデックス順を正規化
                add(ek_p, el_m, name, "1")
                add(el_m, ek_p, name, "1")

    # -----------------------------------------------------------------------
    # 13. Even-even: [E_{δ_k+δ_l}, E_{-δ_p-δ_q}]  (l<p not needed; general)
    #     [b_k^+ b_l^+, b_p^- b_q^-] (k<l, p<q)
    #     = [b_k^+,b_p^-]b_l^+b_q^- + b_p^-[b_k^+,b_q^-]b_l^+ + b_k^+[b_l^+,b_p^-]b_q^- + b_k^+b_p^-[b_l^+,b_q^-]
    #     = delta_{kp}*b_l^+b_q^- ... (expand carefully using [b^+,b^-]=delta via -[b^-,b^+])
    #     [b_k^+, b_p^-] = delta_{kp} (from [b_p^-,b_k^+]=delta_{kp} => [b_k^+,b_p^-]=-... 
    #     Careful: [b_p^-, b_k^+] = delta_{pk} => [b_k^+, b_p^-] = -[b_p^-, b_k^+] = -delta_{kp}? 
    #     No: [A,B] = -[B,A], so [b_k^+, b_p^-] = -[b_p^-, b_k^+] = -delta_{kp}.
    #     Let's recheck: bosonic CCR: [b_i, b_j^†] = delta_{ij}  i.e. [b^-, b^+] = 1
    #     So [b_k^-, b_l^+] = delta_{kl}, [b_l^+, b_k^-] = -delta_{kl}
    #     [b_k^+, b_l^-] = -[b_l^-, b_k^+] = -delta_{lk} = -delta_{kl}
    # -----------------------------------------------------------------------
    # This is already correctly handled: [b_k^+, b_p^-] = -delta_{kp}
    # used consistently above in section 9.
    # [E_{δ_k+δ_l}, E_{-δ_p-δ_q}] for k<l, p<q:
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            for p in range(1, n + 1):
                for q in range(p + 1, n + 1):
                    ep_kl = f"E_del{k}_del{l}_pp"
                    em_pq = f"E_del{p}_del{q}_mm"
                    # [b_k^+b_l^+, b_p^-b_q^-]
                    # = b_k^+[b_l^+,b_p^-]b_q^- + b_k^+b_p^-[b_l^+,b_q^-]
                    #   + [b_k^+,b_p^-]b_l^+b_q^- + b_p^-[b_k^+,b_q^-]b_l^+
                    # [b_l^+,b_p^-] = -delta_{lp}
                    # Result terms: b_k^+ b_q^- (from -delta_{lp}), b_k^+ b_p^- ... wait
                    # Let me be systematic:
                    # = -delta_{lp}*b_k^+*b_q^- - delta_{lq}*b_k^+*b_p^-
                    #   - delta_{kp}*b_l^+*b_q^- - delta_{kq}*b_p^-*b_l^+... 
                    # Actually more carefully: operators all commute except k^+ with k^-
                    # [b_k^+b_l^+, b_p^-b_q^-] expand:
                    # = b_k^+([b_l^+,b_p^-]b_q^- + b_p^-[b_l^+,b_q^-]) + [b_k^+,b_p^-]b_l^+b_q^- + b_p^-[b_k^+,b_q^-]b_l^+
                    # But b_p^-b_l^+ = b_l^+b_p^- + [b_p^-,b_l^+]... no: [b_p^-,b_l^+]=delta_{pl}
                    # This gets complex. For n>=2 let's use the cleaner form:
                    # [b_k^+b_l^+, b_p^-b_q^-] = sum over all pairings of contractions
                    # = -delta_{kp}(-delta_{lq}) ... use Wick-like:
                    # = delta_{kq}delta_{lp} - delta_{kp}delta_{lq}   (Wick for bosons, sign from ordering)
                    # Actually for number-conserving: result is in terms of b^+b^-
                    # Standard result: [b_k^+b_l^+, b_p^-b_q^-] 
                    #   = delta_{kp}b_l^+b_q^- - delta_{kq}... let me just do it:
                    # Step1: b_k^+(b_l^+b_p^-b_q^- - b_p^-b_q^-b_l^+ ... ) + ...
                    # Use [b_l^+, b_p^-] = -delta_{lp}:
                    # b_l^+b_p^- = b_p^-b_l^+ - delta_{lp}
                    # [b_k^+b_l^+, b_p^-b_q^-]:
                    # = b_k^+b_l^+b_p^-b_q^- - b_p^-b_q^-b_k^+b_l^+
                    # Move b_p^- past b_l^+: b_l^+b_p^- = b_p^-b_l^+ - delta_{lp}
                    # Move b_q^- past b_l^+: b_l^+b_q^- = b_q^-b_l^+ - delta_{lq}
                    # ... this is getting lengthy. For correctness, compute case-by-case
                    # for small n. For the JSON generator, implement symbolically.
                    # Here we handle all delta combinations:
                    res = {}
                    # -delta_{lp}*b_k^+*b_q^- term
                    if l == p:
                        # b_k^+ b_q^-: if k==q -> Cartan-like, else E_{δ_k-δ_q}
                        if k == q:
                            for j in range(k, n + 1):
                                key2 = f"H_{j}"
                                res[key2] = res.get(key2, 0) - 1
                    # -delta_{lq}*b_k^+*b_p^- term
                    if l == q:
                        if k == p:
                            for j in range(k, n + 1):
                                key2 = f"H_{j}"
                                res[key2] = res.get(key2, 0) - 1
                    # -delta_{kp}*b_l^+*b_q^- term
                    if k == p:
                        if l == q:
                            for j in range(l, n + 1):
                                key2 = f"H_{j}"
                                res[key2] = res.get(key2, 0) - 1
                    # -delta_{kq}*b_l^+*b_p^- term (sign from reordering)
                    if k == q:
                        if l == p:
                            for j in range(l, n + 1):
                                key2 = f"H_{j}"
                                res[key2] = res.get(key2, 0) + 1  # opposite sign
                    for result_gen, c in res.items():
                        if c != 0:
                            add(ep_kl, em_pq, result_gen, str(c))
                            add(em_pq, ep_kl, result_gen, str(-c))

    # Remove empty entries
    br = {k: v for k, v in br.items() if v}
    return br
# ===========================================================================
# B(m, n) with m >= 1
# ===========================================================================

def build_Bmn_basis(
    m: int, n: int
) -> tuple:
    """
    Build Frappat-notation basis for B(m,n) in physical ordering.

    For m == 0, delegates to build_B0n_basis(n).

    Root ordering follows §3.1 of B_generators_derivation.md.

    Even basis ordering:
      (1) Cartan: H_1,...,H_{n+m}
      (2) ±ε_i±ε_j (i<j): E_eps{i}_eps{j}_{pp/pm/mp/mm}
      (3) ±ε_i:              E_eps{i}_{p/m}   -- a_0 a_i^± is even
      (4) ±δ_k±δ_l (k<l): E_del{k}_del{l}_{pp/pm/mp/mm}
      (5) ±2δ_k:             E_2del{k}_{p/m}

    Odd basis ordering:
      (6)  ±ε_i±δ_k: E_eps{i}_del{k}_{pp/pm/mp/mm}
      (7) +δ_k:        E_del{k}_p
      (8) -δ_k:        E_del{k}_m

    Returns:
        even_basis, odd_basis, parity dict
    """
    if m == 0:
        return build_B0n_basis(n)

    even_basis: List[str] = []
    odd_basis: List[str] = []
    parity: Dict[str, int] = {}

    def reg(name: str, p: int, lst: List[str]) -> None:
        lst.append(name)
        parity[name] = p

    # --- Cartan: H_1..H_n (bosonic), H_{n+1}..H_{n+m} (fermionic) ---
    for k in range(1, n + 1):
        reg(f"H_{k}", 0, even_basis)
    for i in range(1, m + 1):
        reg(f"H_{n+i}", 0, even_basis)

    # --- Positive even: fermionic short ---
    for i in range(1, m + 1):
        for j in range(i + 1, m + 1):
            reg(f"E_eps{i}_eps{j}_pp", 0, even_basis)

    # --- Negative even: fermionic short ---
    for i in range(1, m + 1):
        for j in range(i + 1, m + 1):
            reg(f"E_eps{i}_eps{j}_mm", 0, even_basis)

    # --- Mixed even: fermionic ε_i - ε_j ---
    for i in range(1, m + 1):
        for j in range(i + 1, m + 1):
            reg(f"E_eps{i}_eps{j}_pm", 0, even_basis)
            reg(f"E_eps{i}_eps{j}_mp", 0, even_basis)

    # --- Positive even: fermionic ε_i ---
    for i in range(1, m + 1):
        reg(f"E_eps{i}_p", 0, even_basis)

    # --- Negative even: fermionic ε_i ---
    for i in range(1, m + 1):
        reg(f"E_eps{i}_m", 0, even_basis)

    # --- Positive even: bosonic short ---
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            reg(f"E_del{k}_del{l}_pp", 0, even_basis)

    # --- Negative even: bosonic short ---
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            reg(f"E_del{k}_del{l}_mm", 0, even_basis)

    # --- Mixed even: bosonic δ_k - δ_l ---
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            reg(f"E_del{k}_del{l}_pm", 0, even_basis)
            reg(f"E_del{k}_del{l}_mp", 0, even_basis)


    # --- Positive even: bosonic long ---
    for k in range(1, n + 1):
        reg(f"E_2del{k}_p", 0, even_basis)

    # --- Negative even: bosonic long ---
    for k in range(1, n + 1):
        reg(f"E_2del{k}_m", 0, even_basis)

    # --- Positive odd: mixed (ε_i + δ_k) ---
    for i in range(1, m + 1):
        for k in range(1, n + 1):
            reg(f"E_eps{i}_del{k}_pp", 1, odd_basis)

    # --- Negative odd: mixed (ε_i + δ_k) ---
    for i in range(1, m + 1):
        for k in range(1, n + 1):
            reg(f"E_eps{i}_del{k}_mm", 1, odd_basis)

    # --- Odd: ε_i - δ_k (a_i^+ b_k^-) ---
    for i in range(1, m + 1):
        for k in range(1, n + 1):
            reg(f"E_eps{i}_del{k}_pm", 1, odd_basis)

    # --- Odd: -ε_i + δ_k (a_i^- b_k^+) ---
    for i in range(1, m + 1):
        for k in range(1, n + 1):
            reg(f"E_eps{i}_del{k}_mp", 1, odd_basis)

    # --- Odd: positive bosonic δ_k ---
    for k in range(1, n + 1):
        reg(f"E_del{k}_p", 1, odd_basis)

    # --- Odd: negative bosonic δ_k ---
    for k in range(1, n + 1):
        reg(f"E_del{k}_m", 1, odd_basis)


    return even_basis, odd_basis, parity


def build_Bmn_structure_constants(
    m: int, n: int
) -> "Dict[Tuple[str, str], Dict[str, str]]":
    """
        Compute structure constants for B(m,n) directly from oscillator formulas.

        For m == 0, delegates to build_B0n_structure_constants(n).
        Conventions and derivations are documented in
        docs/theory/derivations/B_generators_derivation.md.

        Notes:
            - Coefficients are represented as p + q*sqrt(2).
            - Odd delta generators follow the B(0,n)-compatible normalization
                used in this repository.
    """
    if m == 0:
        return build_B0n_structure_constants(n)


    br: "Dict[Tuple[str,str], Dict[str,str]]" = {}

    def _parse_coeff(s: str) -> tuple:
        """Parse coefficient string to (p, q) where value = p + q*sqrt(2)."""
        if s == "0":
            return (Fraction(0), Fraction(0))
        if "*sqrt(2)" in s:
            q_str = s.replace("*sqrt(2)", "")
            return (Fraction(0), Fraction(q_str))
        if "sqrt(2)" in s and "+" in s[1:]:
            idx = s.rfind("+")
            if idx > 0:
                return (Fraction(s[:idx]), Fraction(s[idx+1:].replace("*sqrt(2)", "")))
        if "sqrt(2)" in s and s.count("-") >= 2:
            idx = s.rfind("-")
            if idx > 0:
                return (Fraction(s[:idx]), Fraction(s[idx:].replace("*sqrt(2)", "")))
        return (Fraction(s), Fraction(0))

    def _format_coeff(p: Fraction, q: Fraction) -> str:
        """Format (p, q) to coefficient string."""
        if q == 0:
            return str(p)
        elif p == 0:
            return f"{q}*sqrt(2)"
        else:
            sign = "+" if q > 0 else ""
            return f"{p}{sign}{q}*sqrt(2)"

    def add(g1: str, g2: str, result: str, coeff: str) -> None:
        key = (g1, g2)
        if key not in br:
            br[key] = {}
        ep, eq = _parse_coeff(br[key].get(result, "0"))
        np_, nq = _parse_coeff(coeff)
        rp, rq = ep + np_, eq + nq
        if rp == 0 and rq == 0:
            br[key].pop(result, None)
        else:
            br[key][result] = _format_coeff(rp, rq)

    # ------------------------------------------------------------------
    # Eigenvalue helpers
    # ------------------------------------------------------------------

    def bose_ev(k: int, l: int, sign: int) -> int:
        """Eigenvalue of H_k acting on b_l^{sign}: [H_k, b_l^{sign}] = ev * b_l^{sign}"""
        if k < n:
            ev = (1 if l == k else 0) - (1 if l == k + 1 else 0)
        elif k == n:
            # H_n = b_n^+b_n^- + a_1^+a_1^-; acts on bosons: ev_bose = delta_{l,n}
            ev = 1 if l == n else 0
        else:
            # H_{n+i} is fermionic Cartan, doesn't act on bosons
            ev = 0
        return sign * ev

    def fermi_ev(k: int, j: int, sign: int) -> int:
        """Eigenvalue of H_k acting on a_j^{sign}: [H_k, a_j^{sign}] = ev * a_j^{sign}"""
        if k < n:
            # H_k = b_k^+b_k^- - b_{k+1}^+b_{k+1}^-: doesn't act on fermions
            ev = 0
        elif k == n:
            # H_n = b_n^+b_n^- + a_1^+a_1^-: acts on a_1 only
            ev = 1 if j == 1 else 0
        else:
            # H_{n+i} = a_i^+a_i^- - a_{i+1}^+a_{i+1}^- (i<m) or 2a_m^+a_m^- - 1 (i=m)
            i = k - n
            if i < m:
                ev = (1 if j == i else 0) - (1 if j == i + 1 else 0)
            else:  # i == m: H_{n+m} = 2a_m^+a_m^- - 1, [H_{n+m}, a_m^±] = ±2 a_m^±
                ev = 2 if j == m else 0
        return sign * ev

    # ------------------------------------------------------------------
    # Utility: add Cartan bracket [H_k, X] = ev * X for all Cartan k
    # ------------------------------------------------------------------
    def add_cartan_action(gen: str, bose_indices: list, bose_signs: list,
                          fermi_indices: list, fermi_signs: list) -> None:
        """
        Add [H_k, gen] = ev_k * gen for all k=1..n+m.
        bose_indices: list of bosonic oscillator indices contributing to gen
        bose_signs: corresponding signs (+1 for b^+, -1 for b^-)
        fermi_indices: list of fermionic oscillator indices
        fermi_signs: corresponding signs
        """
        for k in range(1, n + m + 1):
            hk = f"H_{k}"
            ev = sum(bose_ev(k, l, s) for l, s in zip(bose_indices, bose_signs))
            ev += sum(fermi_ev(k, j, s) for j, s in zip(fermi_indices, fermi_signs))
            if ev != 0:
                add(hk, gen, gen, str(ev))
                add(gen, hk, gen, str(-ev))

    # ==================================================================
    # Section A: Cartan actions on all non-Cartan generators
    # ==================================================================

    # A1. Even bosonic: E_{±2δ_k}, E_{±δ_k±δ_l}
    for k in range(1, n + 1):
        add_cartan_action(f"E_2del{k}_p",  [k, k], [+1, +1], [], [])
        add_cartan_action(f"E_2del{k}_m", [k, k], [-1, -1], [], [])
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            add_cartan_action(f"E_del{k}_del{l}_pp",  [k, l], [+1, +1], [], [])
            add_cartan_action(f"E_del{k}_del{l}_pm",  [k, l], [+1, -1], [], [])
            add_cartan_action(f"E_del{k}_del{l}_mp",  [k, l], [-1, +1], [], [])
            add_cartan_action(f"E_del{k}_del{l}_mm", [k, l], [-1, -1], [], [])

    # A2. Even fermionic: E_{±ε_i±ε_j}
    for i in range(1, m + 1):
        for j in range(i + 1, m + 1):
            add_cartan_action(f"E_eps{i}_eps{j}_pp",  [], [], [i, j], [+1, +1])
            add_cartan_action(f"E_eps{i}_eps{j}_pm", [], [], [i, j], [+1, -1])
            add_cartan_action(f"E_eps{i}_eps{j}_mp", [], [], [i, j], [-1, +1])
            add_cartan_action(f"E_eps{i}_eps{j}_mm", [], [], [i, j], [-1, -1])

    # A3. Odd mixed: E_{±ε_i±δ_k}(a_i^± b_k^±)
    for i in range(1, m + 1):
        for k in range(1, n + 1):
            add_cartan_action(f"E_eps{i}_del{k}_pp",  [k], [+1], [i], [+1])
            add_cartan_action(f"E_eps{i}_del{k}_pm", [k], [-1], [i], [+1])
            add_cartan_action(f"E_eps{i}_del{k}_mp", [k], [+1], [i], [-1])
            add_cartan_action(f"E_eps{i}_del{k}_mm", [k], [-1], [i], [-1])

    # A4. Odd: E_{±δ_k} (a_0 b_k^±, coeff 1), Even E_{±ε_i} (±sqrt(2) a_0 a_i^±)
    #     a_0 has no bosonic/fermionic index -> eigenvalue from oscillator only
    for k in range(1, n + 1):
        add_cartan_action(f"E_del{k}_p",  [k], [+1], [], [])
        add_cartan_action(f"E_del{k}_m", [k], [-1], [], [])
    for i in range(1, m + 1):
        add_cartan_action(f"E_eps{i}_p",  [], [], [i], [+1])
        add_cartan_action(f"E_eps{i}_m", [], [], [i], [-1])

    def bose_number_in_H(l: int) -> "Dict[str, Fraction]":
        """Express b_l^+b_l^- in terms of H generators as Fraction coefficients."""
        # b_l^+b_l^- = sum_{k=l}^{n} H_k - a_1^+a_1^-
        # a_1^+a_1^- = sum_{i=1}^{m-1} H_{n+i} + (H_{n+m} + 1) / 2
        result = {}
        for k in range(l, n + 1):
            result[f"H_{k}"] = result.get(f"H_{k}", Fraction(0)) + Fraction(1)
        if m == 1:
            # a_1^+a_1^- = (H_{n+1} + 1)/2
            result[f"H_{n+1}"] = result.get(f"H_{n+1}", Fraction(0)) - Fraction(1, 2)
            result["__const__"] = result.get("__const__", Fraction(0)) - Fraction(1, 2)
        else:
            for i in range(1, m):
                result[f"H_{n+i}"] = result.get(f"H_{n+i}", Fraction(0)) - Fraction(1)
            # a_m^+a_m^- = (H_{n+m} + 1)/2
            result[f"H_{n+m}"] = result.get(f"H_{n+m}", Fraction(0)) - Fraction(1, 2)
            result["__const__"] = result.get("__const__", Fraction(0)) - Fraction(1, 2)
        return result

    def fermi_number_in_H(j: int) -> "Dict[str, Fraction]":
        """Express a_j^+a_j^- in terms of H generators as Fraction coefficients."""
        # From Cartan definitions:
        # H_{n+i} = a_i^+a_i^- - a_{i+1}^+a_{i+1}^-  (i<m)
        # H_{n+m} = 2a_m^+a_m^- - 1
        # Telescoping from j to m:
        # a_j^+a_j^- = sum_{i=j}^{m-1} H_{n+i} + a_m^+a_m^-
        #            = sum_{i=j}^{m-1} H_{n+i} + (H_{n+m} + 1) / 2
        result = {}
        for i in range(j, m):
            result[f"H_{n+i}"] = result.get(f"H_{n+i}", Fraction(0)) + Fraction(1)
        result[f"H_{n+m}"] = result.get(f"H_{n+m}", Fraction(0)) + Fraction(1, 2)
        result["__const__"] = result.get("__const__", Fraction(0)) + Fraction(1, 2)
        return result

    def add_H_expansion(g1: str, g2: str, expansion: "Dict[str, Fraction]",
                        overall: Fraction = Fraction(1)) -> None:
        """Add bracket result from H expansion dict (skip __const__ key)."""
        for gen, c in expansion.items():
            if gen == "__const__":
                continue
            val = overall * c
            if val != 0:
                add(g1, g2, gen, str(val))
    # 【Section B: Even-even brackets】

    # B1. [(b_k^+)^2, (b_k^-)^2] = 4b_k^+b_k^- + 2
    for k in range(1, n + 1):
        ep = f"E_2del{k}_p"
        em = f"E_2del{k}_m"
        # 4b_k^+b_k^- + 2: in H basis = 4 * bose_number_in_H(k) + 2*(identity)
        # But identity is not a basis element; handle via expansion
        exp = bose_number_in_H(k)
        # 4*(b_k^+b_k^-) + 2 = 4*(sum H - a_1^+a_1^-) + 2
        # = 4*sum H_k' - 4*a_1^+a_1^- + 2
        # Using exp: 4*exp[H_j] + 2*(__const__ correction)
        # Actually: 4*b_k^+b_k^- + 2 = 4*(exp without const) + 4*exp[const] + 2
        const_part = exp.get("__const__", Fraction(0))
        total_const = 4 * const_part + 2  # should be 0 for consistency
        for gen, c in exp.items():
            if gen == "__const__":
                continue
            val = 4 * c
            if val != 0:
                add(ep, em, gen, str(val))
                add(em, ep, gen, str(-val))
    # B2. [E_{ε_i+ε_j}, E_{-ε_p-ε_q}] for i<j, p<q: result in terms of Cartan H generators
    for i in range(1, m + 1):
        for j in range(i + 1, m + 1):
            for p in range(1, m + 1):
                for q in range(p + 1, m + 1):
                    ep_ij = f"E_eps{i}_eps{j}_pp"
                    em_pq = f"E_eps{p}_eps{q}_mm"
                    # [a_i^+a_j^+, a_p^-a_q^-]
                    # Term 1: +delta_{jp} * a_i^+a_q^-
                    if j == p:
                        if i == q:
                            exp = fermi_number_in_H(i)
                            add_H_expansion(ep_ij, em_pq, exp, Fraction(1))
                            # also antisymmetric
                            add_H_expansion(em_pq, ep_ij, exp, Fraction(-1))
                    # Term 2: -delta_{jq} * a_i^+a_p^-
                    if j == q:
                        if i == p:
                            exp = fermi_number_in_H(i)
                            add_H_expansion(ep_ij, em_pq, exp, Fraction(-1))
                            add_H_expansion(em_pq, ep_ij, exp, Fraction(1))
                    # Term 3: -delta_{ip} * a_j^+a_q^-
                    if i == p:
                        if j == q:
                            exp = fermi_number_in_H(j)
                            add_H_expansion(ep_ij, em_pq, exp, Fraction(-1))
                            add_H_expansion(em_pq, ep_ij, exp, Fraction(1))
                    # Term 4: +delta_{iq} * a_j^+a_p^-
                    if i == q:
                        if j == p:
                            exp = fermi_number_in_H(j)
                            add_H_expansion(ep_ij, em_pq, exp, Fraction(1))
                            add_H_expansion(em_pq, ep_ij, exp, Fraction(-1))
    # B3 (see derivation doc §6): E_{±2δ} with E_{∓(δ+δ)}
    for k in range(1, n + 1):
        for l in range(1, n + 1):
            for p in range(l + 1, n + 1):
                ep_2k = f"E_2del{k}_p"
                em_lp = f"E_del{l}_del{p}_mm"
                ep_lp = f"E_del{l}_del{p}_pp"
                em_2k = f"E_2del{k}_m"

                if k == l:
                    res = f"E_del{k}_del{p}_pm"
                    add(ep_2k, em_lp, res, "-2")
                    add(em_lp, ep_2k, res, "2")
                if k == p:
                    res = f"E_del{l}_del{k}_pm"
                    add(ep_2k, em_lp, res, "-2")
                    add(em_lp, ep_2k, res, "2")

                if k == l:
                    res = f"E_del{k}_del{p}_mp"
                    add(em_2k, ep_lp, res, "2")
                    add(ep_lp, em_2k, res, "-2")
                if k == p:
                    res = f"E_del{l}_del{k}_mp"
                    add(em_2k, ep_lp, res, "2")
                    add(ep_lp, em_2k, res, "-2")

    # B4 (see derivation doc §6): [E_{+(δ+δ)}, E_{-(δ+δ)}]
    def _add_bose_bilinear(g1: str, g2: str, r: int, s: int, c: int) -> None:
        """Add c * b_r^+b_s^- to bracket (g1, g2), expanding diagonal via bose_number_in_H."""
        if c == 0:
            return
        if r == s:
            exp = bose_number_in_H(r)
            add_H_expansion(g1, g2, exp, Fraction(c))
        elif r < s:
            add(g1, g2, f"E_del{r}_del{s}_pm", str(c))
        else:  # r > s
            add(g1, g2, f"E_del{s}_del{r}_mp", str(c))

    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            for p in range(1, n + 1):
                for q in range(p + 1, n + 1):
                    ep_kl = f"E_del{k}_del{l}_pp"
                    em_pq = f"E_del{p}_del{q}_mm"
                    if l == p:
                        _add_bose_bilinear(ep_kl, em_pq, k, q, -1)
                        _add_bose_bilinear(em_pq, ep_kl, k, q, +1)
                    if l == q:
                        _add_bose_bilinear(ep_kl, em_pq, k, p, -1)
                        _add_bose_bilinear(em_pq, ep_kl, k, p, +1)
                    if k == p:
                        _add_bose_bilinear(ep_kl, em_pq, l, q, -1)
                        _add_bose_bilinear(em_pq, ep_kl, l, q, +1)
                    if k == q:
                        _add_bose_bilinear(ep_kl, em_pq, l, p, +1)
                        _add_bose_bilinear(em_pq, ep_kl, l, p, -1)


    # 【Section C: Even-odd brackets】
    # C1. [E_{+2δk}, E_{-ek}]= 2E_{ek},  [E_{-2δk}, E_{ek}] = 2E_{-ek}
    for k in range(1, n + 1):
        ep_k = f"E_2del{k}_p"
        em_k = f"E_2del{k}_m"
        e_odd_p = f"E_del{k}_p"
        e_odd_m = f"E_del{k}_m"
        add(ep_k, e_odd_m, e_odd_p, "-2")
        add(e_odd_m, ep_k, e_odd_p, "2")
        add(em_k, e_odd_p, e_odd_m, "2")
        add(e_odd_p, em_k, e_odd_m, "-2")

    # C2. [E_{±(δ_k+δ_l)}, E_{∓δ_p}]  (bosonic short × odd δ)
    for k in range(1, n + 1):
        for l in range(k + 1, n + 1):
            ep_kl = f"E_del{k}_del{l}_pp"
            em_kl = f"E_del{k}_del{l}_mm"
            for p in range(1, n + 1):
                e_odd_m_p = f"E_del{p}_m"
                e_odd_p_p = f"E_del{p}_p"
                if l == p:
                    add(ep_kl, e_odd_m_p, f"E_del{k}_p", "-1")
                    add(e_odd_m_p, ep_kl, f"E_del{k}_p", "1")
                if k == p:
                    add(ep_kl, e_odd_m_p, f"E_del{l}_p", "-1")
                    add(e_odd_m_p, ep_kl, f"E_del{l}_p", "1")
                if l == p:
                    add(em_kl, e_odd_p_p, f"E_del{k}_m", "1")
                    add(e_odd_p_p, em_kl, f"E_del{k}_m", "-1")
                if k == p:
                    add(em_kl, e_odd_p_p, f"E_del{l}_m", "1")
                    add(e_odd_p_p, em_kl, f"E_del{l}_m", "-1")

    # C2. {E_{δ_k}, E_{δ_l}} -- same as B(0,n)
    for k in range(1, n + 1):
        for l in range(k, n + 1):
            ek = f"E_del{k}_p"
            el = f"E_del{l}_p"
            if k == l:
                add(ek, el, f"E_2del{k}_p", "1")
            else:
                # {E_{δ_k}^+, E_{-δ_l}^-} = E_{δ_k-δ_l}
                add(ek, el, f"E_del{k}_del{l}_pm", "1")
                add(el, ek, f"E_del{k}_del{l}_pm", "1")
    

    # C3 (see derivation doc §6): fermionic short × mixed ε
    for i in range(1, m + 1):
        for j in range(i + 1, m + 1):
            for p in range(1, m + 1):
                for q in range(p + 1, m + 1):
                    epp_ij = f"E_eps{i}_eps{j}_pp"
                    emm_ij = f"E_eps{i}_eps{j}_mm"
                    epm_pq = f"E_eps{p}_eps{q}_pm"
                    emp_pq = f"E_eps{p}_eps{q}_mp"

                    if j == q and i != p:
                        r, s, sgn = (i, p, -1) if i < p else (p, i, +1)
                        add(epp_ij, epm_pq, f"E_eps{r}_eps{s}_pp", str(sgn))
                        add(epm_pq, epp_ij, f"E_eps{r}_eps{s}_pp", str(-sgn))
                    if i == q:
                        add(epp_ij, epm_pq, f"E_eps{p}_eps{j}_pp", "1")
                        add(epm_pq, epp_ij, f"E_eps{p}_eps{j}_pp", "-1")

                    if j == p:
                        add(epp_ij, emp_pq, f"E_eps{i}_eps{q}_pp", "1")
                        add(emp_pq, epp_ij, f"E_eps{i}_eps{q}_pp", "-1")
                    if i == p and j != q:
                        r, s, sgn = (j, q, -1) if j < q else (q, j, +1)
                        add(epp_ij, emp_pq, f"E_eps{r}_eps{s}_pp", str(sgn))
                        add(emp_pq, epp_ij, f"E_eps{r}_eps{s}_pp", str(-sgn))

                    if j == q and i != p:
                        r, s, sgn = (i, p, -1) if i < p else (p, i, +1)
                        add(emm_ij, emp_pq, f"E_eps{r}_eps{s}_mm", str(sgn))
                        add(emp_pq, emm_ij, f"E_eps{r}_eps{s}_mm", str(-sgn))
                    if i == q:
                        add(emm_ij, emp_pq, f"E_eps{p}_eps{j}_mm", "1")
                        add(emp_pq, emm_ij, f"E_eps{p}_eps{j}_mm", "-1")

                    if j == p:
                        add(emm_ij, epm_pq, f"E_eps{i}_eps{q}_mm", "1")
                        add(epm_pq, emm_ij, f"E_eps{i}_eps{q}_mm", "-1")
                    if i == p and j != q:
                        r, s, sgn = (j, q, -1) if j < q else (q, j, +1)
                        add(emm_ij, epm_pq, f"E_eps{r}_eps{s}_mm", str(sgn))
                        add(epm_pq, emm_ij, f"E_eps{r}_eps{s}_mm", str(-sgn))

    # C4 (see derivation doc §6): E_{±ε_i} × E_{∓(ε_j+ε_k)}
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            for k in range(j + 1, m + 1):
                ep_i   = f"E_eps{i}_p"
                em_i   = f"E_eps{i}_m"
                emm_jk = f"E_eps{j}_eps{k}_mm"
                epp_jk = f"E_eps{j}_eps{k}_pp"
                if i == j:
                    add(ep_i, emm_jk, f"E_eps{k}_m", "-1")
                    add(emm_jk, ep_i, f"E_eps{k}_m", "1")
                if i == k:
                    add(ep_i, emm_jk, f"E_eps{j}_m", "1")
                    add(emm_jk, ep_i, f"E_eps{j}_m", "-1")
                if i == j:
                    add(em_i, epp_jk, f"E_eps{k}_p", "1")
                    add(epp_jk, em_i, f"E_eps{k}_p", "-1")
                if i == k:
                    add(em_i, epp_jk, f"E_eps{j}_p", "-1")
                    add(epp_jk, em_i, f"E_eps{j}_p", "1")

    # C5 (see derivation doc §6): odd mixed anticommutator

    for i in range(1, m + 1):
        for k in range(1, n + 1):
            for j in range(1, m + 1):
                for l in range(1, n + 1):
                    ep_ik = f"E_eps{i}_del{k}_pp"
                    em_jl = f"E_eps{j}_del{l}_mm"
                    if i == j:
                        if k != l:
                            pass
                        else:
                            exp = bose_number_in_H(k)
                            add_H_expansion(ep_ik, em_jl, exp, Fraction(1))
                            add_H_expansion(em_jl, ep_ik, exp, Fraction(-1))
                    if k == l:
                        if i != j:
                            pass
                        else:
                            exp = fermi_number_in_H(i)
                            add_H_expansion(ep_ik, em_jl, exp, Fraction(-1))
                            add_H_expansion(em_jl, ep_ik, exp, Fraction(1))

    # 【Section D: Odd-odd brackets (bosonic δ only)】
    # D1. [E_{δk}, E_{δl}] = delta_{kl} E_{2δk} + (1-delta_{kl}) E_{δk+δl}
    for k in range(1, n + 1):
        for l in range(k, n + 1):
            ek = f"E_del{k}_p"
            el = f"E_del{l}_p"
            if k == l:
                add(ek, el, f"E_2del{k}_p", "1")
            else:
                add(ek, el, f"E_del{k}_del{l}_pp", "1")
                add(el, ek, f"E_del{k}_del{l}_pp", "1")

    # D2. {E_{+δ_k}, E_{-δ_l}}
    for k in range(1, n + 1):
        for l in range(1, n + 1):
            ek_p = f"E_del{k}_p"
            el_m = f"E_del{l}_m"
            if k == l:
                exp = bose_number_in_H(k)
                add_H_expansion(ek_p, el_m, exp, Fraction(1))
                add_H_expansion(el_m, ek_p, exp, Fraction(1))
            else:
                if k < l:
                    name = f"E_del{k}_del{l}_pm"
                elif k > l:
                    name = f"E_del{l}_del{k}_mp"
                add(ek_p, el_m, name, "1")
                add(el_m, ek_p, name, "1")

    # D3. {E_{-δ_k}, E_{-δ_l}}
    for k in range(1, n + 1):
        for l in range(k, n + 1):
            ek = f"E_del{k}_m"
            el = f"E_del{l}_m"
            if k == l:
                add(ek, el, f"E_2del{k}_m", "1")
            else:
                add(ek, el, f"E_del{k}_del{l}_mm", "1")
                add(el, ek, f"E_del{k}_del{l}_mm", "1")

    # ==================================================================
    # Section E: Odd-odd brackets (mixed εi±δk)
    # Ref: docs/theory/derivations/B_generators_derivation.md §6.4, §6.5
    # ==================================================================

    # E1-E4: mixed odd with E_{±δ} (coefficients follow §6.4/§6.5)
    for i in range(1, m + 1):
        for k in range(1, n + 1):
            for l in range(1, n + 1):
                if k != l:
                    continue
                ep_ik = f"E_eps{i}_del{k}_pp"
                e_del_m = f"E_del{k}_m"
                e_eps_p = f"E_eps{i}_p"
                add(ep_ik, e_del_m, e_eps_p, "1/2*sqrt(2)")
                add(e_del_m, ep_ik, e_eps_p, "1/2*sqrt(2)")

    for i in range(1, m + 1):
        for k in range(1, n + 1):
            for l in range(1, n + 1):
                if k != l:
                    continue
                em_ik = f"E_eps{i}_del{k}_pp"
                e_del_p = f"E_del{k}_p"
                e_eps_m = f"E_eps{i}_m"
                add(em_ik, e_del_p, e_eps_m, "-1/2*sqrt(2)")
                add(e_del_p, em_ik, e_eps_m, "-1/2*sqrt(2)")

    for i in range(1, m + 1):
        for k in range(1, n + 1):
            for l in range(1, n + 1):
                if k != l:
                    continue
                e_mdel = f"E_eps{i}_del{k}_pm"
                e_del_p = f"E_del{k}_p"
                e_eps_p = f"E_eps{i}_p"
                add(e_mdel, e_del_p, e_eps_p, "1/2*sqrt(2)")
                add(e_del_p, e_mdel, e_eps_p, "1/2*sqrt(2)")

    for i in range(1, m + 1):
        for k in range(1, n + 1):
            for l in range(1, n + 1):
                if k != l:
                    continue
                e_meps = f"E_eps{i}_del{k}_mp"
                e_del_m = f"E_del{k}_m"
                e_eps_m = f"E_eps{i}_m"
                add(e_meps, e_del_m, e_eps_m, "1/2*sqrt(2)")
                add(e_del_m, e_meps, e_eps_m, "1/2*sqrt(2)")

    # E5-E8: mixed odd anticommutators (see derivation doc §6.4/§6.5)
    def _add_fermi_bilinear(g1: str, g2: str, i: int, j: int, c: int) -> None:
        """Add c * a_i^+a_j^- to bracket (g1, g2)."""
        if c == 0:
            return
        if i == j:
            exp = fermi_number_in_H(i)
            add_H_expansion(g1, g2, exp, Fraction(c))
        elif i < j:
            add(g1, g2, f"E_eps{i}_eps{j}_pm", str(c))
        else:
            add(g1, g2, f"E_eps{j}_eps{i}_mp", str(c))

    for i in range(1, m + 1):
        for k in range(1, n + 1):
            for j in range(1, m + 1):
                for l in range(1, n + 1):
                    epp_ik = f"E_eps{i}_del{k}_pp"
                    emm_jl = f"E_eps{j}_del{l}_mm"
                    if i == j:
                        _add_bose_bilinear(epp_ik, emm_jl, k, l, +1)
                        _add_bose_bilinear(emm_jl, epp_ik, k, l, +1)
                    if k == l:
                        _add_fermi_bilinear(epp_ik, emm_jl, i, j, +1)
                        _add_fermi_bilinear(emm_jl, epp_ik, i, j, +1)
    for i in range(1, m + 1):
        for k in range(1, n + 1):
            for j in range(1, m + 1):
                for l in range(1, n + 1):
                    epm_ik = f"E_eps{i}_del{k}_pm"
                    emp_jl = f"E_eps{j}_del{l}_mp"
                    if i == j:
                        _add_bose_bilinear(epm_ik, emp_jl, l, k, -1)
                        _add_bose_bilinear(emp_jl, epm_ik, l, k, -1)
                    if k == l:
                        _add_fermi_bilinear(epm_ik, emp_jl, i, j, -1)
                        _add_fermi_bilinear(emp_jl, epm_ik, i, j, -1)

    # E7.
    for i in range(1, m + 1):
        for k in range(1, n + 1):
            for j in range(1, m + 1):
                for l in range(1, n + 1):
                    epp_ik = f"E_eps{i}_del{k}_pp"
                    emp_jl = f"E_eps{j}_del{l}_mp"
                    if i == j:
                        r, s = (k, l) if k <= l else (l, k)
                        if r == s:
                            add(epp_ik, emp_jl, f"E_2del{r}_p", "2")
                            add(emp_jl, epp_ik, f"E_2del{r}_p", "2")
                        else:
                            add(epp_ik, emp_jl, f"E_del{r}_del{s}_pp", "2")
                            add(emp_jl, epp_ik, f"E_del{r}_del{s}_pp", "2")

    # E8.
    for i in range(1, m + 1):
        for k in range(1, n + 1):
            for j in range(1, m + 1):
                for l in range(1, n + 1):
                    emm_ik = f"E_eps{i}_del{k}_mm"
                    epm_jl = f"E_eps{j}_del{l}_pm"
                    if i == j:
                        r, s = (k, l) if k <= l else (l, k)
                        if r == s:
                            add(emm_ik, epm_jl, f"E_2del{r}_m", "2")
                            add(epm_jl, emm_ik, f"E_2del{r}_m", "2")
                        else:
                            add(emm_ik, epm_jl, f"E_del{r}_del{s}_mm", "2")
                            add(epm_jl, emm_ik, f"E_del{r}_del{s}_mm", "2")

    br = {k: v for k, v in br.items() if v}
    return br
