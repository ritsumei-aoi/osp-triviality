import InhomogeneousDeformations.Indexed
import InhomogeneousDeformations.HomogeneousDegree
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.FinCases

/-!
# C1.2/C1.3/C2.2 — explicit n=1 correspondence and transported degree closure

`C1.2`: the coefficient identification `rho : Indexed.Pn 1 → Coeff` (these are
definitionally the same type, `MvPolynomial (Fin 2) ℚ`, since `2*1` reduces
to `2`; exploited directly, not postulated as an abstract isomorphism), and
the basis equivalence `E : Indexed.IndexedBasis 1 ≃ Basis5`, with parity
preservation.

`C1.3`: `Phi`, the induced module reindexing, its bijectivity/linearity, the
homogeneity equivalence, and the required bracket-compatibility theorem
`Phi (Indexed.bracketN 1 x y) = bracket (Phi x) (Phi y)`.

`C2.2`: transports `bracket_isHomog` (C2.1) to the indexed `n=1` module via
this correspondence.
-/

namespace InhomogeneousDeformations
namespace Indexed

open Basis5

/-- `C1.2`: `Indexed.Pn 1` and `Coeff` are definitionally the same ring
(`Fin (2*1)` reduces to `Fin 2`); the coefficient identification is the
identity map on this shared type, not an abstract isomorphism. -/
noncomputable def rho : Pn 1 → Coeff := id

@[simp] lemma rho_apply (p : Pn 1) : rho p = p := rfl

/-- The first indexed variable is identified with `beta1`, the second with
`beta2`, by direct computation (not evaluation/substitution). -/
lemma rho_beta1 : rho (MvPolynomial.X (0 : Fin (2 * 1))) = InhomogeneousDeformations.beta1 := rfl
lemma rho_beta2 : rho (MvPolynomial.X (1 : Fin (2 * 1))) = InhomogeneousDeformations.beta2 := rfl

/-- `rho` preserves the ring operations and rational constants (it is the
identity on a shared type). -/
lemma rho_add (p q : Pn 1) : rho (p + q) = rho p + rho q := rfl
lemma rho_mul (p q : Pn 1) : rho (p * q) = rho p * rho q := rfl
lemma rho_C (q : ℚ) : rho (cratN 1 q) = crat q := rfl

/-- `cratN` at rank `1` is definitionally `crat` (both are `MvPolynomial.C` on
the shared type `Pn 1 = Coeff`); bridges the two opaque embeddings directly,
without unfolding either to `MvPolynomial.C`. -/
@[simp] lemma cratN_one_eq_crat (q : ℚ) : cratN 1 q = crat q := rfl

/-- `basis_case`'s 4-term `bracketLLn` sums can produce a bare literal `4`
(not `crat _`) as a `Coeff` numeral; `N1Proofs.lean` (protected, byte-exact)
only supplies a `crat_two_mul`/`crat_mul_two` bridge, so bridge the literal
`4` case locally instead of editing that file. -/
@[simp] lemma crat_four_eq : (4 : Coeff) = crat (4 : ℚ) := by norm_cast
@[simp] lemma crat_mul_four (a : ℚ) : crat a * (4 : Coeff) = crat (a * 4) := by
  rw [crat_four_eq, crat_mul]
@[simp] lemma crat_four_mul (a : ℚ) : (4 : Coeff) * crat a = crat (4 * a) := by
  rw [crat_four_eq, crat_mul]

/-- `C1.2`: the basis equivalence, `n=1` even pairs to `L11,L12,L22` and odd
indices to `F1,F2`. Deliberately written via `Fin.val` comparisons (not
literal `Fin`-pattern matches) so the equation compiler does not synthesize
an auxiliary `NeZero`/`Fin.instOfNat` proof term for the `2*1`-indexed
modulus; that auxiliary instance was found to block `simp`/`decide` from
recognizing the resulting numerals as equal to the standard-instance
numerals used elsewhere (`Ebackward`, `oracleFull`, ...), leaving several
`basis_case` goals stuck as unresolved decidable (dis)equalities. -/
def Eforward : IndexedBasis 1 → Basis5
  | .inl ⟨(u, v), _⟩ =>
      if v.val = 0 then .L11 else if u.val = 0 then .L12 else .L22
  | .inr u => if u.val = 0 then .F1 else .F2

def Ebackward : Basis5 → IndexedBasis 1
  | .L11 => .inl ⟨(0, 0), le_refl _⟩
  | .L12 => .inl ⟨(0, 1), by decide⟩
  | .L22 => .inl ⟨(1, 1), le_refl _⟩
  | .F1 => .inr 0
  | .F2 => .inr 1

lemma Eforward_backward : ∀ b : Basis5, Eforward (Ebackward b) = b := by
  intro b; cases b <;> simp [Eforward, Ebackward]

lemma Ebackward_forward : ∀ p : IndexedBasis 1, Ebackward (Eforward p) = p := by
  intro p
  match p with
  | .inl ⟨(u, v), h⟩ =>
    fin_cases u <;> fin_cases v <;>
      first
      | omega
      | exact absurd h (by decide)
      | (simp_all [Eforward, Ebackward])
  | .inr u =>
    fin_cases u <;> simp [Eforward, Ebackward]

/-- `C1.2`: the basis equivalence `E : IndexedBasis 1 ≃ Basis5`. -/
def E : IndexedBasis 1 ≃ Basis5 where
  toFun := Eforward
  invFun := Ebackward
  left_inv := Ebackward_forward
  right_inv := Eforward_backward

/-- `E` preserves parity. -/
theorem E_parity (p : IndexedBasis 1) : Basis5.parity (E p) = Indexed.parity p := by
  match p with
  | .inl ⟨(u, v), h⟩ =>
    fin_cases u <;> fin_cases v <;>
      first | (simp_all [E, Eforward, Indexed.parity, Basis5.parity]) | omega
  | .inr u =>
    fin_cases u <;> simp [E, Eforward, Indexed.parity, Basis5.parity]

/-- `C1` (correction002) rank-one compatibility: `Basis5`'s canonical
position, `L11,L12,L22,F1,F2` at ranks `0,1,2,3,4`. Local to this file (not
added to the protected `Basis.lean`), used only to state that `E` preserves
the canonical indexed order. -/
def basis5Rank : Basis5 → ℕ
  | .L11 => 0
  | .L12 => 1
  | .L22 => 2
  | .F1 => 3
  | .F2 => 4

/-- `C1`: the canonical `n=1` indexed order, via `Ebackward`, is exactly
`L11,L12,L22,F1,F2` in that increasing order. -/
theorem canonical_order_n1 :
    Ebackward Basis5.L11 < Ebackward Basis5.L12 ∧
    Ebackward Basis5.L12 < Ebackward Basis5.L22 ∧
    Ebackward Basis5.L22 < Ebackward Basis5.F1 ∧
    Ebackward Basis5.F1 < Ebackward Basis5.F2 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    · show Indexed.orderKey _ < Indexed.orderKey _
      simp [Ebackward, Indexed.orderKey, Prod.Lex.toLex_lt_toLex]

/-- `C1`: `E` preserves the canonical indexed order -- comparing
`p < q : IndexedBasis 1` agrees exactly with comparing `basis5Rank (E p) <
basis5Rank (E q)`, whose values are `L11,L12,L22,F1,F2 ↦ 0,1,2,3,4` by
construction, so this identifies `E`'s action with the canonical position
sequence proved above. -/
theorem E_preserves_order (p q : IndexedBasis 1) :
    p < q ↔ basis5Rank (E p) < basis5Rank (E q) := by
  show Indexed.orderKey p < Indexed.orderKey q ↔ basis5Rank (E p) < basis5Rank (E q)
  have hp : p = Ebackward (E p) := (E.symm_apply_apply p).symm
  have hq : q = Ebackward (E q) := (E.symm_apply_apply q).symm
  conv_lhs => rw [hp, hq]
  cases hEp : E p <;> cases hEq : E q <;>
    simp [Ebackward, basis5Rank, Indexed.orderKey, Prod.Lex.toLex_lt_toLex]

/-- `C1.3`: the induced module map, reindexing coefficients through `E` and `rho`. -/
noncomputable def Phi (x : IndexedMod 1) : Mod := fun b => rho (x (E.symm b))

/-- `C1.3`: `Phi` is bijective. -/
theorem Phi_bijective : Function.Bijective Phi := by
  constructor
  · intro x y hxy
    funext p
    have := congrFun hxy (E p)
    simpa [Phi, rho, E.symm_apply_apply] using this
  · intro y
    refine ⟨fun p => y (E p), ?_⟩
    funext b
    simp [Phi, rho]

/-- `C1.3`: `Phi` is additive and `Coeff`-linear (via `rho`'s identification). -/
theorem Phi_add (x y : IndexedMod 1) : Phi (x + y) = Phi x + Phi y := by
  funext b; simp [Phi, rho]

theorem Phi_smul (c : Pn 1) (x : IndexedMod 1) : Phi (c • x) = rho c • Phi x := by
  funext b; simp [Phi, rho, smul_eq_mul]

/-- `C1.3`: indexed homogeneity corresponds to old `IsHomog` under `Phi`, in
both directions. -/
theorem IsHomog_Phi_iff (x : IndexedMod 1) (d : ZMod 2) :
    (∀ p : IndexedBasis 1, x p ≠ 0 → Indexed.parity p = d) ↔ IsHomog (Phi x) d := by
  constructor
  · intro h b hb
    have hx : x (E.symm b) ≠ 0 := by
      intro h0; apply hb; simp [Phi, h0, rho]
    have hp : Indexed.parity (E.symm b) = d := h (E.symm b) hx
    have hb2 : Basis5.parity (E (E.symm b)) = Indexed.parity (E.symm b) := E_parity (E.symm b)
    rw [E.apply_symm_apply] at hb2
    rw [hb2, hp]
  · intro h p hp
    have hb : Phi x (E p) ≠ 0 := by
      simp [Phi, rho, E.symm_apply_apply]; exact hp
    have := h (E p) hb
    rwa [E_parity] at this

@[simp] lemma Phi_zero : Phi (0 : IndexedMod 1) = 0 := by funext b; simp [Phi, rho]

lemma rho_neg (p : Pn 1) : rho (-p) = -(rho p) := rfl

theorem Phi_neg (x : IndexedMod 1) : Phi (-x) = -(Phi x) := by
  funext b; simp [Phi, rho_neg, indexedMod_neg_apply]

/-- `Phi` distributes over an arbitrary finite sum. -/
lemma Phi_sum {ι : Type*} [DecidableEq ι] (s : Finset ι) (f : ι → IndexedMod 1) :
    Phi (∑ i ∈ s, f i) = ∑ i ∈ s, Phi (f i) := by
  induction s using Finset.induction with
  | empty => simp
  | @insert a s ha ih => rw [Finset.sum_insert ha, Phi_add, ih, Finset.sum_insert ha]

/-! ## C2 (correction002) bridge lemmas, per Agent1's recommended route:
small, separately-audited facts relating the indexed basis machinery to the
native (old) `n=1` model through `Phi`/`E`, proved BEFORE the finite
compatibility table so the table itself carries no proof-term equality or
opaque-constant arithmetic. -/

/-- `Phi` of an indexed basis indicator is the corresponding native basis
indicator. -/
theorem Phi_eN (p : IndexedBasis 1) : Phi (eN p) = e (E p) := by
  funext b
  simp only [Phi, eN, e, rho_apply]
  by_cases h : b = E p
  · subst h; simp [E.symm_apply_apply]
  · have h' : E.symm b ≠ p := by
      intro hc; exact h (by rw [← hc, E.apply_symm_apply])
    simp [h, h']

/-- Rank-one `Jn` is the native `Jmat`, through `rho`. -/
theorem rho_Jn (u v : Fin 2) : rho (Jn 1 u v) = InhomogeneousDeformations.Jmat u v := by
  fin_cases u <;> fin_cases v <;> simp [Jn, InhomogeneousDeformations.Jmat, rho, cratN_one_eq_crat]

/-- Rank-one `Lof` (indexed), through `E`, is the native `Lof`. -/
theorem E_Lof (u v : Fin 2) : E (Lof u v) = InhomogeneousDeformations.Lof u v := by
  fin_cases u <;> fin_cases v <;> simp [Lof, E, Eforward, InhomogeneousDeformations.Lof]

/-- Rank-one `Fof` (indexed), through `E`, is the native `Fof`. -/
theorem E_Fof (u : Fin 2) : E (Fof u) = InhomogeneousDeformations.Fof u := by
  fin_cases u <;> simp [Fof, E, Eforward, InhomogeneousDeformations.Fof]

/-- General (not case-split on which pair) bridge: `Phi` of the indexed
`F,F` bracket formula is the native `F,F` bracket formula, for every
`u v : Fin 2`. -/
theorem Phi_bracketFFn (u v : Fin 2) :
    Phi (bracketFFn 1 u v) = InhomogeneousDeformations.bracketFF u v := by
  simp only [bracketFFn, InhomogeneousDeformations.bracketFF, Phi_smul, rho_C, Phi_eN, E_Lof]

/-- General bridge: `Phi` of the indexed `L,F` bracket formula is the native
`L,F` bracket formula, for every `u v w : Fin 2`. -/
theorem Phi_bracketLFn (u v w : Fin 2) :
    Phi (bracketLFn 1 u v w) = InhomogeneousDeformations.bracketLF u v w := by
  simp only [bracketLFn, InhomogeneousDeformations.bracketLF, Phi_add, Phi_smul, rho_mul, rho_C, rho_Jn,
    Phi_eN, E_Fof]

/-- General bridge: `Phi` of the indexed `L,L` bracket formula is the native
`L,L` bracket formula, for every `u v w z : Fin 2`. -/
theorem Phi_bracketLLn (u v w z : Fin 2) :
    Phi (bracketLLn 1 u v w z) = InhomogeneousDeformations.bracketLL u v w z := by
  simp only [bracketLLn, InhomogeneousDeformations.bracketLL, Phi_add, Phi_smul, rho_mul, rho_C, rho_Jn,
    Phi_eN, E_Lof]

/-- `C2` basis-pair compatibility table: 25 separately named lemmas, one per
`(E p, E q)` branch, each closed by a single bridge-lemma application (or a
`Phi_neg`/bridge composition for the `F,L` rows) rather than a shared tactic
chain. -/
theorem Phi_bracket_basis_L11_L11 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L11) (Ebackward Basis5.L11))
      = bracket (e Basis5.L11) (e Basis5.L11) := by
  rw [bracket_e_e]; exact Phi_bracketLLn 0 0 0 0

theorem Phi_bracket_basis_L11_L12 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L11) (Ebackward Basis5.L12))
      = bracket (e Basis5.L11) (e Basis5.L12) := by
  rw [bracket_e_e]; exact Phi_bracketLLn 0 0 0 1

theorem Phi_bracket_basis_L11_L22 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L11) (Ebackward Basis5.L22))
      = bracket (e Basis5.L11) (e Basis5.L22) := by
  rw [bracket_e_e]; exact Phi_bracketLLn 0 0 1 1

theorem Phi_bracket_basis_L12_L11 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L12) (Ebackward Basis5.L11))
      = bracket (e Basis5.L12) (e Basis5.L11) := by
  rw [bracket_e_e]; exact Phi_bracketLLn 0 1 0 0

theorem Phi_bracket_basis_L12_L12 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L12) (Ebackward Basis5.L12))
      = bracket (e Basis5.L12) (e Basis5.L12) := by
  rw [bracket_e_e]; exact Phi_bracketLLn 0 1 0 1

theorem Phi_bracket_basis_L12_L22 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L12) (Ebackward Basis5.L22))
      = bracket (e Basis5.L12) (e Basis5.L22) := by
  rw [bracket_e_e]; exact Phi_bracketLLn 0 1 1 1

theorem Phi_bracket_basis_L22_L11 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L22) (Ebackward Basis5.L11))
      = bracket (e Basis5.L22) (e Basis5.L11) := by
  rw [bracket_e_e]; exact Phi_bracketLLn 1 1 0 0

theorem Phi_bracket_basis_L22_L12 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L22) (Ebackward Basis5.L12))
      = bracket (e Basis5.L22) (e Basis5.L12) := by
  rw [bracket_e_e]; exact Phi_bracketLLn 1 1 0 1

theorem Phi_bracket_basis_L22_L22 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L22) (Ebackward Basis5.L22))
      = bracket (e Basis5.L22) (e Basis5.L22) := by
  rw [bracket_e_e]; exact Phi_bracketLLn 1 1 1 1

theorem Phi_bracket_basis_L11_F1 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L11) (Ebackward Basis5.F1))
      = bracket (e Basis5.L11) (e Basis5.F1) := by
  rw [bracket_e_e]; exact Phi_bracketLFn 0 0 0

theorem Phi_bracket_basis_L11_F2 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L11) (Ebackward Basis5.F2))
      = bracket (e Basis5.L11) (e Basis5.F2) := by
  rw [bracket_e_e]; exact Phi_bracketLFn 0 0 1

theorem Phi_bracket_basis_L12_F1 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L12) (Ebackward Basis5.F1))
      = bracket (e Basis5.L12) (e Basis5.F1) := by
  rw [bracket_e_e]; exact Phi_bracketLFn 0 1 0

theorem Phi_bracket_basis_L12_F2 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L12) (Ebackward Basis5.F2))
      = bracket (e Basis5.L12) (e Basis5.F2) := by
  rw [bracket_e_e]; exact Phi_bracketLFn 0 1 1

theorem Phi_bracket_basis_L22_F1 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L22) (Ebackward Basis5.F1))
      = bracket (e Basis5.L22) (e Basis5.F1) := by
  rw [bracket_e_e]; exact Phi_bracketLFn 1 1 0

theorem Phi_bracket_basis_L22_F2 :
    Phi (bracketBasisN 1 (Ebackward Basis5.L22) (Ebackward Basis5.F2))
      = bracket (e Basis5.L22) (e Basis5.F2) := by
  rw [bracket_e_e]; exact Phi_bracketLFn 1 1 1

theorem Phi_bracket_basis_F1_L11 :
    Phi (bracketBasisN 1 (Ebackward Basis5.F1) (Ebackward Basis5.L11))
      = bracket (e Basis5.F1) (e Basis5.L11) := by
  rw [bracket_e_e]
  exact (Phi_neg (bracketLFn 1 0 0 0)).trans (congrArg Neg.neg (Phi_bracketLFn 0 0 0))

theorem Phi_bracket_basis_F1_L12 :
    Phi (bracketBasisN 1 (Ebackward Basis5.F1) (Ebackward Basis5.L12))
      = bracket (e Basis5.F1) (e Basis5.L12) := by
  rw [bracket_e_e]
  exact (Phi_neg (bracketLFn 1 0 1 0)).trans (congrArg Neg.neg (Phi_bracketLFn 0 1 0))

theorem Phi_bracket_basis_F1_L22 :
    Phi (bracketBasisN 1 (Ebackward Basis5.F1) (Ebackward Basis5.L22))
      = bracket (e Basis5.F1) (e Basis5.L22) := by
  rw [bracket_e_e]
  exact (Phi_neg (bracketLFn 1 1 1 0)).trans (congrArg Neg.neg (Phi_bracketLFn 1 1 0))

theorem Phi_bracket_basis_F2_L11 :
    Phi (bracketBasisN 1 (Ebackward Basis5.F2) (Ebackward Basis5.L11))
      = bracket (e Basis5.F2) (e Basis5.L11) := by
  rw [bracket_e_e]
  exact (Phi_neg (bracketLFn 1 0 0 1)).trans (congrArg Neg.neg (Phi_bracketLFn 0 0 1))

theorem Phi_bracket_basis_F2_L12 :
    Phi (bracketBasisN 1 (Ebackward Basis5.F2) (Ebackward Basis5.L12))
      = bracket (e Basis5.F2) (e Basis5.L12) := by
  rw [bracket_e_e]
  exact (Phi_neg (bracketLFn 1 0 1 1)).trans (congrArg Neg.neg (Phi_bracketLFn 0 1 1))

theorem Phi_bracket_basis_F2_L22 :
    Phi (bracketBasisN 1 (Ebackward Basis5.F2) (Ebackward Basis5.L22))
      = bracket (e Basis5.F2) (e Basis5.L22) := by
  rw [bracket_e_e]
  exact (Phi_neg (bracketLFn 1 1 1 1)).trans (congrArg Neg.neg (Phi_bracketLFn 1 1 1))

theorem Phi_bracket_basis_F1_F1 :
    Phi (bracketBasisN 1 (Ebackward Basis5.F1) (Ebackward Basis5.F1))
      = bracket (e Basis5.F1) (e Basis5.F1) := by
  rw [bracket_e_e]; exact Phi_bracketFFn 0 0

theorem Phi_bracket_basis_F1_F2 :
    Phi (bracketBasisN 1 (Ebackward Basis5.F1) (Ebackward Basis5.F2))
      = bracket (e Basis5.F1) (e Basis5.F2) := by
  rw [bracket_e_e]; exact Phi_bracketFFn 0 1

theorem Phi_bracket_basis_F2_F1 :
    Phi (bracketBasisN 1 (Ebackward Basis5.F2) (Ebackward Basis5.F1))
      = bracket (e Basis5.F2) (e Basis5.F1) := by
  rw [bracket_e_e]; exact Phi_bracketFFn 1 0

theorem Phi_bracket_basis_F2_F2 :
    Phi (bracketBasisN 1 (Ebackward Basis5.F2) (Ebackward Basis5.F2))
      = bracket (e Basis5.F2) (e Basis5.F2) := by
  rw [bracket_e_e]; exact Phi_bracketFFn 1 1

set_option maxHeartbeats 4000000 in
/-- `C2`: the general basis-pair compatibility lemma, for ALL `p q :
IndexedBasis 1` (not just the 25 concretely-named ones above), by rewriting
`p`/`q` to `E.symm (E p)`/`E.symm (E q)` (defeq to `Ebackward (E p)`/
`Ebackward (E q)`) and dispatching on the concrete value of `E p`, `E q` to
the matching one of the 25 lemmas above. -/
theorem basis_case (p q : IndexedBasis 1) :
    Phi (bracketBasisN 1 p q) = bracket (e (E p)) (e (E q)) := by
  conv_lhs => rw [show p = E.symm (E p) from (E.symm_apply_apply p).symm,
                  show q = E.symm (E q) from (E.symm_apply_apply q).symm]
  cases hEp : E p <;> cases hEq : E q <;>
    first
    | exact Phi_bracket_basis_L11_L11 | exact Phi_bracket_basis_L11_L12
    | exact Phi_bracket_basis_L11_L22 | exact Phi_bracket_basis_L11_F1
    | exact Phi_bracket_basis_L11_F2 | exact Phi_bracket_basis_L12_L11
    | exact Phi_bracket_basis_L12_L12 | exact Phi_bracket_basis_L12_L22
    | exact Phi_bracket_basis_L12_F1 | exact Phi_bracket_basis_L12_F2
    | exact Phi_bracket_basis_L22_L11 | exact Phi_bracket_basis_L22_L12
    | exact Phi_bracket_basis_L22_L22 | exact Phi_bracket_basis_L22_F1
    | exact Phi_bracket_basis_L22_F2 | exact Phi_bracket_basis_F1_L11
    | exact Phi_bracket_basis_F1_L12 | exact Phi_bracket_basis_F1_L22
    | exact Phi_bracket_basis_F1_F1 | exact Phi_bracket_basis_F1_F2
    | exact Phi_bracket_basis_F2_L11 | exact Phi_bracket_basis_F2_L12
    | exact Phi_bracket_basis_F2_L22 | exact Phi_bracket_basis_F2_F1
    | exact Phi_bracket_basis_F2_F2

/-- `C1.3`/`C2`: `Phi` intertwines the indexed and native brackets on all of
`IndexedMod 1`. Reduces to `basis_case` (the 25-basis-pair compatibility
table above) plus the finite-sum bilinear-extension argument. -/
theorem Phi_bracket (x y : IndexedMod 1) :
    Phi (bracketN 1 x y) = bracket (Phi x) (Phi y) := by
  have expand : bracketN 1 x y
      = ∑ p : IndexedBasis 1, ∑ q : IndexedBasis 1, (x p * y q) • bracketBasisN 1 p q := rfl
  have step : ∀ p : IndexedBasis 1,
      Phi (∑ q : IndexedBasis 1, (x p * y q) • bracketBasisN 1 p q)
        = ∑ q : IndexedBasis 1, (rho (x p) * rho (y q)) • bracket (e (E p)) (e (E q)) := by
    intro p
    rw [Phi_sum]
    apply Finset.sum_congr rfl; intro q _
    rw [Phi_smul, basis_case, rho_mul]
  have main : Phi (bracketN 1 x y)
      = ∑ p : IndexedBasis 1, ∑ q : IndexedBasis 1,
          (rho (x p) * rho (y q)) • bracket (e (E p)) (e (E q)) := by
    rw [expand, Phi_sum]
    exact Finset.sum_congr rfl (fun p _ => step p)
  rw [main, bracket_bilinear_expand (Phi x) (Phi y)]
  have inner : ∀ p : IndexedBasis 1, (∑ q : IndexedBasis 1,
        (rho (x p) * rho (y q)) • bracket (e (E p)) (e (E q)))
      = ∑ j : Basis5, (rho (x p) * Phi y j) • bracket (e (E p)) (e j) := by
    intro p
    apply Fintype.sum_equiv E
    intro q
    show (rho (x p) * rho (y q)) • bracket (e (E p)) (e (E q))
      = (rho (x p) * Phi y (E q)) • bracket (e (E p)) (e (E q))
    congr 2
    show rho (y q) = rho (y (E.symm (E q)))
    rw [E.symm_apply_apply]
  have outer : (∑ p : IndexedBasis 1, ∑ q : IndexedBasis 1,
        (rho (x p) * rho (y q)) • bracket (e (E p)) (e (E q)))
      = ∑ i : Basis5, ∑ j : Basis5, (Phi x i * Phi y j) • bracket (e i) (e j) := by
    simp_rw [inner]
    apply Fintype.sum_equiv E
    intro p
    show (∑ j : Basis5, (rho (x p) * Phi y j) • bracket (e (E p)) (e j))
      = ∑ j : Basis5, (Phi x (E p) * Phi y j) • bracket (e (E p)) (e j)
    apply Finset.sum_congr rfl; intro j _
    congr 2
    show rho (x p) = rho (x (E.symm (E p)))
    rw [E.symm_apply_apply]
  rw [outer]

/-- `C2.2`: the general homogeneous degree-closure law (`C2.1`), transported
to the indexed `n=1` specialization via `Phi`. -/
theorem bracketN_isHomog (x y : IndexedMod 1) (dx dy : ZMod 2)
    (hx : ∀ p : IndexedBasis 1, x p ≠ 0 → Indexed.parity p = dx)
    (hy : ∀ p : IndexedBasis 1, y p ≠ 0 → Indexed.parity p = dy) :
    ∀ p : IndexedBasis 1, bracketN 1 x y p ≠ 0 → Indexed.parity p = dx + dy := by
  have hPhix : IsHomog (Phi x) dx := (IsHomog_Phi_iff x dx).mp hx
  have hPhiy : IsHomog (Phi y) dy := (IsHomog_Phi_iff y dy).mp hy
  have hPhiBracket : IsHomog (Phi (bracketN 1 x y)) (dx + dy) := by
    rw [Phi_bracket]; exact bracket_isHomog (Phi x) (Phi y) dx dy hPhix hPhiy
  have := (IsHomog_Phi_iff (bracketN 1 x y) (dx + dy)).mpr hPhiBracket
  exact this

end Indexed
end InhomogeneousDeformations
