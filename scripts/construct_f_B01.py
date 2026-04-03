"""
Explicit construction of the odd map f for B(0,1) — all gb are trivial.

For each basis direction gb=[1,0] and gb=[0,1], solve
    A · vec(f) = L[:, k]   (k = 0, 1)
over Q using SymPy exact arithmetic, then verify δf = γ(gb) for general gb.

Usage:
    python experiments/construct_f_B01.py
"""

import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oscillator_lie_superalgebras.trivial_subspace import (
    build_coboundary_matrix,
    build_gamma_linear_map,
)

M, N = 0, 1

# ── 1. Build A and L (float) ──────────────────────────────────────────────────
A_num, basis, parity = build_coboundary_matrix(M, N)
L_num, gamma_index, gb_vars, _, _ = build_gamma_linear_map(M, N)

print(f"basis : {basis}")
print(f"gb_vars : {gb_vars}")
print(f"A shape : {A_num.shape},  L shape : {L_num.shape}")
print(f"rank(A) = {np.linalg.matrix_rank(A_num)},  rank(L) = {np.linalg.matrix_rank(L_num)}")

# ── 2. Convert A and L to SymPy rational (exact) ─────────────────────────────
A_sym = sp.Matrix(A_num.real.tolist()).applyfunc(sp.nsimplify)
L_sym = sp.Matrix(L_num.real.tolist()).applyfunc(sp.nsimplify)

# ── 3. Solve A · vec(f^(k)) = L[:, k] for k=0,1 via gauss_jordan_solve ───────
# gauss_jordan_solve returns (particular_solution, free_variable_basis)
solutions = {}
for k, gb_name in enumerate(gb_vars):
    rhs = L_sym.col(k)
    try:
        sol, free_syms = A_sym.gauss_jordan_solve(rhs)
        # Use particular solution (set free variables to 0)
        particular = sol.subs({s: 0 for s in free_syms})
        solutions[gb_name] = particular
        residual = A_sym * particular - rhs
        print(f"\n--- gb direction: {gb_name} ---")
        print(f"  free variables: {len(free_syms)}")
        print(f"  residual norm  = {residual.norm()}")
    except Exception as e:
        print(f"\n--- gb direction: {gb_name} --- ERROR: {e}")
        solutions[gb_name] = sp.zeros(A_sym.shape[1], 1)

valid_f_pairs = [
    (src, dst)
    for src in basis
    for dst in basis
    if parity[src] != parity[dst]
]

# ── 3b. Display ker(A) basis (free variable directions) ──────────────────────
print("\n=== ker(A) basis ===")
# gauss_jordan_solve の第2引数として free_syms が返っている
# 各 free_sym に 1 を代入し他を 0 にすることで basis vector を得る
for k, gb_name in enumerate(gb_vars):
    rhs = L_sym.col(k)
    sol_full, free_syms = A_sym.gauss_jordan_solve(rhs)
    print(f"\n  [{gb_name}] free symbols: {free_syms}")
    for fs in free_syms:
        ker_vec = sol_full.subs({s: (1 if s == fs else 0) for s in free_syms})
        ker_vec = ker_vec.subs({s: 0 for s in free_syms})  # 念のため残りを0に
        print(f"  ker direction ({fs}=1):")
        for idx, (src, dst) in enumerate(valid_f_pairs):
            val = ker_vec[idx]
            if val != 0:
                print(f"    f({src}) += {val} * {dst}")

# ── 4. Enumerate valid f-variable pairs ──────────────────────────────────────

print("\n=== f^(0)  [gb=[1,0] direction, gb_a0_b1p] ===")
for idx, (src, dst) in enumerate(valid_f_pairs):
    val = solutions[gb_vars[0]][idx]
    if val != 0:
        print(f"  f({src}) += {val} * {dst}")

print("\n=== f^(1)  [gb=[0,1] direction, gb_a0_b1m] ===")
for idx, (src, dst) in enumerate(valid_f_pairs):
    val = solutions[gb_vars[1]][idx]
    if val != 0:
        print(f"  f({src}) += {val} * {dst}")

# ── 5. Verification: δ(g0*f0 + g1*f1) = γ(gb) for symbolic g0, g1 ───────────
g0, g1 = sp.symbols("g0 g1")
f_gen = g0 * solutions[gb_vars[0]] + g1 * solutions[gb_vars[1]]
gamma_gen = g0 * L_sym.col(0) + g1 * L_sym.col(1)

residual_gen = A_sym * f_gen - gamma_gen
residual_expanded = residual_gen.applyfunc(sp.expand)

print(f"\n=== Verification for general gb (g0, g1 symbolic) ===")
all_zero = all(c == 0 for c in residual_expanded)
if not all_zero:
    nonzero = [(i, c) for i, c in enumerate(residual_expanded) if c != 0]
    print(f"  Non-zero residuals: {nonzero[:5]}")
else:
    print("  ALL TRIVIAL: True — δ(g0·f⁰ + g1·f¹) = γ(gb)  ✓")

    # ── 6. 論文の f (gb=[[0,-1]]) を直接検証 ────────────────────────────────────
# 論文: fF+ = E+, fF- = H, others = 0
# PBW 変換: E_del1_p -> E_2del1_p, E_del1_m -> H_1
# (論文記法 → PBW: F+ = (1/2)E_del1_p, E+ = (1/2)E_2del1_p, F- = (1/2)E_del1_m, H = (1/2)H_1)
# fF+ = E+  => f(E_del1_p) = E_2del1_p  (係数 1: (1/2)f^PBW(2F+)=(1/2)E_2del1_p => f^PBW(E_del1_p)=E_2del1_p)
# fF- = H   => f(E_del1_m) = H_1

print("\n=== Direct verification: 論文の f (gb=[[0,-1]]) ===")
# valid_f_pairs のインデックスで vec(f_paper) を構成
f_paper_vec = sp.zeros(len(valid_f_pairs), 1)
for idx, (src, dst) in enumerate(valid_f_pairs):
    if src == 'E_del1_p' and dst == 'E_2del1_p':
        f_paper_vec[idx] = sp.Integer(1)
    elif src == 'E_del1_m' and dst == 'H_1':
        f_paper_vec[idx] = sp.Integer(1)

# γ(gb=[[0,-1]]) = L[:, 1] * (-1)
gamma_paper = -L_sym.col(1)

residual_paper = A_sym * f_paper_vec - gamma_paper
residual_paper_expanded = residual_paper.applyfunc(sp.expand)
all_zero_paper = all(c == 0 for c in residual_paper_expanded)

print("  f_paper vec (nonzero entries):")
for idx, (src, dst) in enumerate(valid_f_pairs):
    if f_paper_vec[idx] != 0:
        print(f"    f({src}) = {f_paper_vec[idx]} * {dst}")

if all_zero_paper:
    print("  δf_paper = γ([[0,-1]])  ✅")
else:
    nonzero = [(gamma_index[i], residual_paper_expanded[i])
               for i in range(len(residual_paper_expanded))
               if residual_paper_expanded[i] != 0]
    print(f"  MISMATCH — nonzero residuals: {nonzero[:5]}")