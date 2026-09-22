import InhomogeneousDeformations.SourceLifts
import Mathlib.LinearAlgebra.LinearIndependent.Lemmas
import Mathlib.LinearAlgebra.Dual.Lemmas

/-!
# I106 R2, W4 — the independence statement, written and assessed

State (against `SourceLifts.lean`'s own `L0`/`F0`) the manuscript's claim:

> the family `{L^0_{uv} : u ≤ v} ∪ {F^0_u}` is `ℚ`-linearly independent in `A_0`

(`IndependenceStatement`), and assess whether it is provable in the operator model.

## Result of the assessment

The statement **splits cleanly, and provably, into two independent sub-claims**
(`independence_of_L0_and_F0`), because `L0`- and `F0`-family elements live in *different*
`A0Grading` degrees (`0` and `1` respectively — `SourceLifts.lean`'s
`L0_mem_A0Grading_zero`/`F0_mem_A0Grading_one`), and `A0Grading`'s internal direct-sum
decomposition (`A0GradedAlgebra`, `SourceTensor.lean`) makes `A0Grading n 0` and `A0Grading n 1`
disjoint submodules. So the full family is independent as soon as each of

1. `{F^0_u : u}` is independent in `A_0`, and
2. `{L^0_{uv} : u ≤ v}` is independent in `A_0`

holds separately.

**(1) is proved outright** (`F0_linearIndependent`): `F^0_u = (1/2)(1 ᵍ⊗ₜ a)(B_u ᵍ⊗ₜ 1) =
(1/2)(B_u ᵍ⊗ₜ a)` (`tmul_coe_mul_zero_coe_tmul`, since the `W_n`-factor is always degree `0`),
so pairing against a linear functional on `C` dual to `a` (`Module.Projective.exists_dual_eq_one`,
the same device `SourceTensor.lean` already uses for `A3`'s nonzero-ness) reduces a vanishing
`ℚ`-combination of `F^0_u`'s to the same combination of `B_u`'s in `Module.End ℚ (WPoly n)`. That
in turn is proved by direct evaluation on test polynomials (`B_linearIndependent`) — exactly the
method the packet's own `W1` candidate suggested: apply a purported vanishing combination
`∑_u g_u B_u = 0` to the constant polynomial `1` (isolating the multiplication-type, odd-index
coefficients via a functional dual to each `X_j`, since `pderiv _ 1 = 0` kills every
differentiation-type term), and to each variable `X_k` (isolating the remaining
differentiation-type, even-index coefficients via `pderiv_i (X_k) = if i = k then 1 else 0`).

**(2) is genuinely open, and left as an assessed (not proved) claim**, exactly as the packet
allows ("You need not prove it; you must say whether R3+ can, and on what it would turn"). The
manuscript's own proof of `lem:base` does not re-derive it from scratch either — it invokes "the
classical one-pair Weyl-basis theorem" (PBW for `W_n`) plus a leading-symbol argument, not a
finite elementary computation. The evaluation method that closed (1) *does* extend in principle:
`L^0_{uv}` breaks into exactly four parity sectors —
both-even (`pderiv_i pderiv_j`, symmetrizing to `2·pderiv_i∘pderiv_j`),
both-odd (`2·mulLeft(X_i X_j)`), and
mixed (`2·mulLeft(X_j)∘pderiv_i + (if i = j then 1 else 0)•id`, via the one commutator relation
`A1` (`SourceWeyl.lean`'s `pderiv_mulLeft_comm`) already proves) —
and evaluating the full combination on the degree-`≤2` monomials `1`, `X_k`, `X_k X_l` should, by
the same coefficient-extraction method, isolate each sector's coefficients in turn. What was
**not** attempted here is formalizing that four-sector evaluation in full: it is a genuine,
bounded (not requiring a general PBW theorem — only this specific finite family's independence),
but nontrivial, additional piece of casework on top of what this round already built. **This is
the honest boundary of what R2 established**: not a wall, but unfinished work, named precisely
enough that a further round can pick it up without re-deriving the reduction above.

**No axiom is declared for (2).** Per the packet's axiom-versus-parameter rule, an unproved fact
must be threaded as a hypothesis parameter into anything that *uses* it, never a free-standing
`axiom` — and nothing in this round's own scope (W0-W4) consumes (2), so nothing is declared at
all: `IndependenceStatement` is stated as a plain `Prop` and left unproved overall; (2) is
recorded here as prose plus the `L0FamilyIndependent` *statement* (a `def : Prop`, deliberately
carrying no proof, so it cannot be mistaken for a completed claim). Any future round that needs
(2) must take it as an explicit hypothesis argument to whatever it proves
(`independence_of_L0_and_F0` below already shows exactly how: it takes independence of the `L0`
family as a hypothesis, not an axiom, and discharges the rest unconditionally).

**Update (I106 R3, `SourceQuadraticIndependence.lean`): (2) is no longer open.** The
four-sector evaluation left unattempted above was carried out there, proving the frozen
`L0FamilyIndependent` unconditionally (`L0FamilyIndependent_proved`) and, via
`independence_of_L0_and_F0`, discharging `IndependenceStatement` unconditionally as well
(`IndependenceStatement_proved`) — both on the standard three axioms, no `sorryAx`, no
hypothesis left unclosed. The assessment above is kept as the record of how the reduction was
found and why R2 itself left (2) open; nothing below this note should be read as still
describing the current state of `IndependenceStatement`.
-/

namespace InhomogeneousDeformations
namespace Source

open scoped TensorProduct

/-! ## The statement -/

/-- The index family: `{L^0_{uv} : u ≤ v} ∪ {F^0_u}`, as a single `Sum`-indexed function into
`A_0`, matching the packet's own phrasing of `W4`'s statement. -/
noncomputable def liftsFamily (n : ℕ) :
    {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} ⊕ Fin (2 * n) → A0 n :=
  Sum.elim (fun p => L0 n p.1.1 p.1.2) (F0 n)

/-- **W4's independence statement**, exactly as phrased: `{L^0_{uv} : u ≤ v} ∪ {F^0_u}` is
`ℚ`-linearly independent in `A_0`. -/
def IndependenceStatement (n : ℕ) : Prop :=
  LinearIndependent ℚ (liftsFamily n)

/-- Part (2) of the assessment, as a bare (deliberately unproved) `Prop`: `{L^0_{uv} : u ≤ v}`
independent in `A_0`. See the module docstring for why this is left open. -/
def L0FamilyIndependent (n : ℕ) : Prop :=
  LinearIndependent ℚ (fun p : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} => L0 n p.1.1 p.1.2)

/-! ## Part (1): `{B_u}`, hence `{F^0_u}`, independent — proved -/

/-- `{B_u}` is linearly independent in `Module.End ℚ (WPoly n)`, by evaluation on test
polynomials: a dual functional `φ_even j` (evaluate at `X_j`, read off the constant coefficient)
isolates `g (evenIdx j)`, and `φ_odd j` (evaluate at `1`, read off the coefficient of `X_j`)
isolates `g (oddIdx j)`, for every `j`. -/
theorem B_linearIndependent (n : ℕ) : LinearIndependent ℚ (B n) := by
  have even_ne_even : ∀ i j : Fin n, i ≠ j → Indexed.evenIdx n i ≠ Indexed.evenIdx n j := by
    intro i j hij h
    apply hij
    simp only [Fin.ext_iff, Indexed.evenIdx] at h ⊢
    omega
  have odd_ne_odd : ∀ i j : Fin n, i ≠ j → Indexed.oddIdx n i ≠ Indexed.oddIdx n j := by
    intro i j hij h
    apply hij
    simp only [Fin.ext_iff, Indexed.oddIdx] at h ⊢
    omega
  have even_ne_odd : ∀ i j : Fin n, Indexed.evenIdx n i ≠ Indexed.oddIdx n j := by
    intro i j h
    simp only [Fin.ext_iff, Indexed.evenIdx, Indexed.oddIdx] at h
    omega
  rw [Fintype.linearIndependent_iff]
  intro g hg u
  have hφeven : ∀ j : Fin n, ∀ v : Fin (2 * n),
      MvPolynomial.coeff 0 ((B n v) (MvPolynomial.X j)) = if v = Indexed.evenIdx n j then 1 else 0 := by
    intro j v
    by_cases hv : (v : ℕ) % 2 = 0
    · obtain ⟨i, rfl⟩ : ∃ i, v = Indexed.evenIdx n i := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
      rw [B_evenIdx, Derivation.coeFn_coe]
      by_cases hij : i = j
      · subst hij; rw [MvPolynomial.pderiv_X_self]; simp
      · rw [MvPolynomial.pderiv_X_of_ne (Ne.symm hij), if_neg (even_ne_even i j hij)]
        simp
    · obtain ⟨i, rfl⟩ : ∃ i, v = Indexed.oddIdx n i := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
      rw [B_oddIdx, LinearMap.mulLeft_apply, if_neg (Ne.symm (even_ne_odd j i)),
        ← MvPolynomial.constantCoeff_eq, map_mul, MvPolynomial.constantCoeff_X,
        MvPolynomial.constantCoeff_X, mul_zero]
  have hφodd : ∀ j : Fin n, ∀ v : Fin (2 * n),
      MvPolynomial.coeff (Finsupp.single j 1) ((B n v) (1 : WPoly n)) =
        if v = Indexed.oddIdx n j then 1 else 0 := by
    intro j v
    by_cases hv : (v : ℕ) % 2 = 0
    · obtain ⟨i, rfl⟩ : ∃ i, v = Indexed.evenIdx n i := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
      rw [B_evenIdx, Derivation.coeFn_coe, MvPolynomial.pderiv_one]
      rw [if_neg (even_ne_odd i j)]
      simp
    · obtain ⟨i, rfl⟩ : ∃ i, v = Indexed.oddIdx n i := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
      rw [B_oddIdx, LinearMap.mulLeft_apply, mul_one, MvPolynomial.coeff_X']
      simp only [Finsupp.single_left_inj (one_ne_zero (α := ℕ))]
      by_cases hij : i = j
      · subst hij; simp
      · rw [if_neg hij, if_neg (odd_ne_odd i j hij)]
  by_cases hu : (u : ℕ) % 2 = 0
  · obtain ⟨j, rfl⟩ : ∃ j, u = Indexed.evenIdx n j := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    have hsum : (∑ v, g v • B n v) (MvPolynomial.X j) = 0 := by rw [hg]; simp
    have hcoeff : MvPolynomial.coeff 0 ((∑ v, g v • B n v) (MvPolynomial.X j)) = 0 := by
      rw [hsum]; simp
    rw [LinearMap.sum_apply] at hcoeff
    simp only [LinearMap.smul_apply] at hcoeff
    rw [MvPolynomial.coeff_sum] at hcoeff
    simp only [MvPolynomial.coeff_smul, smul_eq_mul] at hcoeff
    rw [Finset.sum_congr rfl (fun v _ => by rw [hφeven j v, mul_boole])] at hcoeff
    rwa [Finset.sum_ite_eq' Finset.univ (Indexed.evenIdx n j) g, if_pos (Finset.mem_univ _)] at hcoeff
  · obtain ⟨j, rfl⟩ : ∃ j, u = Indexed.oddIdx n j := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    have hsum : (∑ v, g v • B n v) (1 : WPoly n) = 0 := by rw [hg]; simp
    have hcoeff : MvPolynomial.coeff (Finsupp.single j 1) ((∑ v, g v • B n v) (1 : WPoly n)) = 0 := by
      rw [hsum]; simp
    rw [LinearMap.sum_apply] at hcoeff
    simp only [LinearMap.smul_apply] at hcoeff
    rw [MvPolynomial.coeff_sum] at hcoeff
    simp only [MvPolynomial.coeff_smul, smul_eq_mul] at hcoeff
    rw [Finset.sum_congr rfl (fun v _ => by rw [hφodd j v, mul_boole])] at hcoeff
    rwa [Finset.sum_ite_eq' Finset.univ (Indexed.oddIdx n j) g, if_pos (Finset.mem_univ _)] at hcoeff

/-- `F^0_u = (1/2)(B_u ᵍ⊗ₜ a)`: the transcribed `F0 n u = (1/2) • (aA0 n * Bu0 n u)` collapsed
via the "no sign" formula, since the `W_n`-factor `1` is always degree `0`. -/
theorem F0_eq_smul_tmul (n : ℕ) (u : Fin (2 * n)) :
    F0 n u = (1 / 2 : ℚ) • ((B n u) ᵍ⊗ₜ[ℚ] (Source.a : C)) := by
  unfold F0 aA0 Bu0
  congr 1
  rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := WGrading n) (ℬ := CGrading)
    (1 : Module.End ℚ (WPoly n)) (⟨Source.a, Source.a_odd⟩ : CGrading 1)
    (⟨B n u, trivial⟩ : WGrading n 0) (1 : C)]
  simp

/-- **Part (1)**: `{F^0_u}` is linearly independent in `A_0`, via a functional on `C` dual to
`a` (mirroring `A3_independence`'s own nonzero-ness argument) reducing to `B_linearIndependent`. -/
theorem F0_linearIndependent (n : ℕ) : LinearIndependent ℚ (F0 n) := by
  rw [Fintype.linearIndependent_iff]
  intro g hg u
  obtain ⟨φ, hφ⟩ := Module.Projective.exists_dual_eq_one ℚ Source.a_ne_zero
  set Ψ : A0 n →ₗ[ℚ] Module.End ℚ (WPoly n) :=
    (TensorProduct.rid ℚ (Module.End ℚ (WPoly n))).toLinearMap ∘ₗ
      (TensorProduct.map (LinearMap.id : Module.End ℚ (WPoly n) →ₗ[ℚ] _) φ) ∘ₗ
      (GradedTensorProduct.of ℚ (WGrading n) CGrading).symm.toLinearMap with hΨdef
  have hΨB : ∀ v : Fin (2 * n), Ψ ((B n v) ᵍ⊗ₜ[ℚ] (Source.a : C)) = B n v := by
    intro v
    show Ψ (GradedTensorProduct.of ℚ (WGrading n) CGrading (B n v ⊗ₜ (Source.a : C))) = B n v
    rw [hΨdef]
    simp [TensorProduct.map_tmul, hφ]
  have key : Ψ (∑ v, g v • F0 n v) = (1 / 2 : ℚ) • (∑ v, g v • B n v) := by
    simp only [map_sum, F0_eq_smul_tmul, map_smul, hΨB]
    rw [Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro v _
    rw [smul_smul, smul_smul, mul_comm]
  rw [hg, map_zero] at key
  have key2 : ∑ v, g v • B n v = 0 := by
    have h1 := congrArg (fun x => (2 : ℚ) • x) key
    simp only [smul_zero, smul_smul] at h1
    norm_num at h1
    exact h1.symm
  exact (Fintype.linearIndependent_iff.mp (B_linearIndependent n)) g key2 u

/-! ## The splitting: `IndependenceStatement` follows from `L0FamilyIndependent` -/

theorem A0Grading_disjoint_zero_one (n : ℕ) : Disjoint (A0Grading n 0) (A0Grading n 1) :=
  (DirectSum.Decomposition.isInternal (A0Grading n)).submodule_iSupIndep.pairwiseDisjoint
    (by decide)

/-- **The splitting**: `IndependenceStatement` follows from `L0FamilyIndependent` (part (2),
open) together with `F0_linearIndependent` (part (1), proved above) — the only additional
ingredient is that `L0`- and `F0`-family elements live in different, disjoint `A0Grading`
degrees, so a vanishing combination of the whole family forces each half to vanish separately. -/
theorem independence_of_L0_and_F0 (n : ℕ) (hL0 : L0FamilyIndependent n) :
    IndependenceStatement n := by
  rw [IndependenceStatement, Fintype.linearIndependent_iff]
  intro g hg x
  have hsplit : (∑ p, g (Sum.inl p) • L0 n p.1.1 p.1.2) + (∑ u, g (Sum.inr u) • F0 n u) = 0 := by
    have hsplit0 := Fintype.sum_sum_type (fun y => g y • liftsFamily n y)
    rw [hg] at hsplit0
    simp only [liftsFamily, Sum.elim_inl, Sum.elim_inr] at hsplit0
    exact hsplit0.symm
  have hL : (∑ p, g (Sum.inl p) • L0 n p.1.1 p.1.2) ∈ A0Grading n 0 :=
    Submodule.sum_mem _ (fun p _ => Submodule.smul_mem _ _ (L0_mem_A0Grading_zero n p.1.1 p.1.2))
  have hF : (∑ u, g (Sum.inr u) • F0 n u) ∈ A0Grading n 1 :=
    Submodule.sum_mem _ (fun u _ => Submodule.smul_mem _ _ (F0_mem_A0Grading_one n u))
  have hL0eq : (∑ p, g (Sum.inl p) • L0 n p.1.1 p.1.2) = 0 ∧
      (∑ u, g (Sum.inr u) • F0 n u) = 0 := by
    have hmem : (∑ p, g (Sum.inl p) • L0 n p.1.1 p.1.2) ∈ A0Grading n 0 ⊓ A0Grading n 1 := by
      refine ⟨hL, ?_⟩
      have : (∑ p, g (Sum.inl p) • L0 n p.1.1 p.1.2) = -(∑ u, g (Sum.inr u) • F0 n u) :=
        eq_neg_of_add_eq_zero_left hsplit
      rw [this]
      exact neg_mem hF
    have hzero : (∑ p, g (Sum.inl p) • L0 n p.1.1 p.1.2) = 0 :=
      (Submodule.disjoint_def.mp (A0Grading_disjoint_zero_one n)) _ hmem.1 hmem.2
    refine ⟨hzero, ?_⟩
    rw [hzero, zero_add] at hsplit
    exact hsplit
  cases x with
  | inl p =>
    exact (Fintype.linearIndependent_iff.mp hL0) (fun p => g (Sum.inl p)) hL0eq.1 p
  | inr u =>
    exact (Fintype.linearIndependent_iff.mp (F0_linearIndependent n)) (fun u => g (Sum.inr u))
      hL0eq.2 u

end Source
end InhomogeneousDeformations
