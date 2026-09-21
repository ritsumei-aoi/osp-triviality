import InhomogeneousDeformations.SourceRecoveryLifts
import InhomogeneousDeformations.SourceRecoveryBase

/-!
# I106 R5, Y4 stage 1 — the bridge `iota0` from the abstract coordinate model to `A_B`

This file is new (not frozen); it never reopens `Indexed.lean` through `SourceIsomorphism.lean`,
`SourceRecoveryLifts.lean`, or `SourceRecoveryBase.lean`, only reads their already-accepted
definitions/theorems.

**Scope (stage 1 of Y4 only)**: build `iota0 : IndexedMod n → AB n`, the `Pn n`-linear extension
of the basis map `Lof u v ↦ L0 n u v`, `Fof u ↦ F0 n u`, and show it intertwines the *undeformed*
abstract bracket `bracketN` with the concrete (super-)commutator structure of `A_0 n`/`AB n`, using
only `SourceRecoveryBase.lean`'s three already-proved facts (`L0hat_comm`, `L0_comm_F0`, `F0_comm_F0`).
The *deformed* bracket / `GammaBetaN` connection (stage 2) is explicitly out of scope here.
-/

namespace InhomogeneousDeformations
namespace Source

open scoped TensorProduct DirectSum

/-! ## The basis-level embedding -/

/-- The coordinate basis vectors' images in `A_0 n`: `L_{uv} ↦ L^0_{uv}`, `F_u ↦ F^0_u`. -/
noncomputable def iota0Basis (n : ℕ) : Indexed.IndexedBasis n → A0 n
  | .inl p => L0 n p.1.1 p.1.2
  | .inr u => F0 n u

/-! ## `Pn n`'s algebra map into `RRing n` lands entirely in degree `0`

`RRing n = CliffordAlgebra (Qzero n)` is a Clifford algebra *over* `Indexed.Pn n` itself, so its
own `algebraMap` always lands in `RGrading n 0` (`algebraMap c = c • 1` and `1` is degree `0`) --
for *every* `c : Pn n`, not just `betaN`'s values (`SourceIsomorphism.lean` only ever needed the
`betaN`-specific instance of this fact). -/

theorem algebraMap_mem_RGradingQ_zero (n : ℕ) (c : Indexed.Pn n) :
    algebraMap (Indexed.Pn n) (RRing n) c ∈ RGradingQ n 0 := by
  rw [Algebra.algebraMap_eq_smul_one]
  exact Submodule.smul_mem _ c (SetLike.one_mem_graded (RGrading n))

/-- `Jn n u v` is always a constant polynomial (`Indexed.cratN n q = C q` for `q ∈ {1,-1,0}`), so
it equals `C` of its own constant coefficient `JnQ n u v`. -/
theorem Jn_eq_C_JnQ (n : ℕ) (u v : Fin (2 * n)) :
    Indexed.Jn n u v = MvPolynomial.C (JnQ n u v) := by
  unfold JnQ Indexed.Jn
  split_ifs with h1 h2 <;> simp [Indexed.cratN]

theorem algebraMap_Jn (n : ℕ) (u v : Fin (2 * n)) :
    algebraMap (Indexed.Pn n) (RRing n) (Indexed.Jn n u v) = (JnQ n u v) • (1 : RRing n) := by
  rw [Jn_eq_C_JnQ, ← MvPolynomial.algebraMap_eq, ← IsScalarTower.algebraMap_apply ℚ (Indexed.Pn n) (RRing n),
    Algebra.algebraMap_eq_smul_one]

theorem algebraMap_cratN (n : ℕ) (q : ℚ) :
    algebraMap (Indexed.Pn n) (RRing n) (Indexed.cratN n q) = q • (1 : RRing n) := by
  unfold Indexed.cratN
  rw [← MvPolynomial.algebraMap_eq, ← IsScalarTower.algebraMap_apply ℚ (Indexed.Pn n) (RRing n),
    Algebra.algebraMap_eq_smul_one]

/-! ## `iota0` : `Pn n`-linear extension of `iota0Basis`, landing in `A_B` -/

/-- The bridge map itself: `x ↦ ∑_b algebraMap(x b) ᵍ⊗ₜ (iota0Basis b)`. Targets `AB n`, not
`A0 n` -- `A0 n` carries no `Pn n`-module structure, so a general (non-constant-coefficient)
`x : IndexedMod n` cannot scale an `A0 n`-element directly; `Pn n` embeds into `RRing n` (`R`'s
own base ring, `SourceTensor.lean`'s `RRing_algebra_rat`/`RRing_isScalarTower`), and `AB n = R
ᵍ⊗[ℚ] A0 n` is exactly where that embedding can act. -/
noncomputable def iota0 (n : ℕ) (x : Indexed.IndexedMod n) : AB n :=
  ∑ b : Indexed.IndexedBasis n,
    (algebraMap (Indexed.Pn n) (RRing n) (x b)) ᵍ⊗ₜ[ℚ] (iota0Basis n b)

/-- The `AB n`-image of a single basis vector, i.e. `iota0 n (eN b)` (proved as `iota0_eN`
below) -- the natural target for the basis-pair intertwining theorem. -/
noncomputable def iota0AB (n : ℕ) (b : Indexed.IndexedBasis n) : AB n :=
  (1 : RRing n) ᵍ⊗ₜ[ℚ] (iota0Basis n b)

theorem iota0_add (n : ℕ) (x y : Indexed.IndexedMod n) :
    iota0 n (x + y) = iota0 n x + iota0 n y := by
  unfold iota0
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro b _
  rw [Indexed.indexedMod_add_apply, map_add]
  show GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
      ((algebraMap (Indexed.Pn n) (RRing n) (x b) + algebraMap (Indexed.Pn n) (RRing n) (y b))
        ⊗ₜ[ℚ] (iota0Basis n b))
    = GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
        ((algebraMap (Indexed.Pn n) (RRing n) (x b)) ⊗ₜ[ℚ] (iota0Basis n b))
      + GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
        ((algebraMap (Indexed.Pn n) (RRing n) (y b)) ⊗ₜ[ℚ] (iota0Basis n b))
  rw [TensorProduct.add_tmul, map_add]

theorem iota0_smul (n : ℕ) (c : Indexed.Pn n) (x : Indexed.IndexedMod n) :
    iota0 n (c • x) =
      ((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * iota0 n x := by
  unfold iota0
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro b _
  rw [Indexed.indexedMod_smul_apply, map_mul]
  exact (GradedTensorProduct.tmul_one_mul_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
    (algebraMap (Indexed.Pn n) (RRing n) c)
    (⟨algebraMap (Indexed.Pn n) (RRing n) (x b), algebraMap_mem_RGradingQ_zero n (x b)⟩ :
      RGradingQ n 0)
    (iota0Basis n b)).symm

theorem iota0_eN (n : ℕ) (b : Indexed.IndexedBasis n) :
    iota0 n (Indexed.eN b) = iota0AB n b := by
  unfold iota0 iota0AB
  rw [Finset.sum_eq_single b]
  · unfold Indexed.eN; simp
  · intro c _ hc
    unfold Indexed.eN
    rw [if_neg hc]
    simp
  · intro h; exact absurd (Finset.mem_univ b) h

theorem AB_zero_tmul (n : ℕ) (x : A0 n) : ((0 : RRing n) ᵍ⊗ₜ[ℚ] x : AB n) = 0 := by
  show GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) ((0 : RRing n) ⊗ₜ[ℚ] x) = 0
  rw [TensorProduct.zero_tmul, map_zero]

theorem iota0_zero (n : ℕ) : iota0 n (0 : Indexed.IndexedMod n) = 0 := by
  unfold iota0
  simp only [Indexed.indexedMod_zero_apply, map_zero, AB_zero_tmul, Finset.sum_const_zero]

theorem iota0_neg (n : ℕ) (x : Indexed.IndexedMod n) : iota0 n (-x) = -iota0 n x := by
  have h : iota0 n (-x) + iota0 n x = 0 := by
    rw [← iota0_add, neg_add_cancel, iota0_zero]
  exact eq_neg_of_add_eq_zero_left h

/-- `iota0` applied to a single `c`-scaled basis vector: general form used for every sector's
"sum form" below (`c` need not be a constant -- `Pn n`'s full generality is used here). -/
theorem iota0AB_scale (n : ℕ) (c : Indexed.Pn n) (b : Indexed.IndexedBasis n) :
    (algebraMap (Indexed.Pn n) (RRing n) c ᵍ⊗ₜ[ℚ] (1 : A0 n) : AB n) * iota0AB n b
      = algebraMap (Indexed.Pn n) (RRing n) c ᵍ⊗ₜ[ℚ] (iota0Basis n b) := by
  unfold iota0AB
  rw [GradedTensorProduct.tmul_one_mul_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
    (algebraMap (Indexed.Pn n) (RRing n) c)
    (⟨1, SetLike.one_mem_graded (RGradingQ n)⟩ : RGradingQ n 0) (iota0Basis n b)]
  norm_num

theorem iota0_smul_eN (n : ℕ) (c : Indexed.Pn n) (b : Indexed.IndexedBasis n) :
    iota0 n (c • Indexed.eN b) = algebraMap (Indexed.Pn n) (RRing n) c ᵍ⊗ₜ[ℚ] (iota0Basis n b) := by
  rw [iota0_smul, iota0_eN, iota0AB_scale]

theorem AB_tmul_smul_left (n : ℕ) (c : ℚ) (r : RRing n) (x : A0 n) :
    ((c • r : RRing n) ᵍ⊗ₜ[ℚ] x : AB n) = c • (r ᵍ⊗ₜ[ℚ] x) := by
  show GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) ((c • r) ⊗ₜ[ℚ] x)
      = c • GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) (r ⊗ₜ[ℚ] x)
  rw [← map_smul, TensorProduct.smul_tmul']

theorem algebraMap_cratN_half_mul_Jn (n : ℕ) (u v : Fin (2 * n)) :
    algebraMap (Indexed.Pn n) (RRing n) (Indexed.cratN n (1/2) * Indexed.Jn n u v)
      = ((1/2 : ℚ) * JnQ n u v) • (1 : RRing n) := by
  rw [map_mul, algebraMap_cratN, algebraMap_Jn, smul_mul_smul_comm, mul_one]

/-! ## Lifting `L0hat_comm` to `A0 n`

`SourceRecoveryBase.lean` proves `L0hat_comm` at the `Module.End ℚ (WPoly n)` level only, and
`L0_comm_F0`/`F0_comm_F0` already at the `A0 n` level; `L0_comm` (the `A0 n`-level `L0`-`L0`
commutator) is the one missing piece, lifted through `L0_eq_tmul` exactly as
`L0_comm_F0`/`F0_comm_F0`'s own proofs lift their `Module.End`/`aA0`-level identities. -/

/-- Two `Module.End`-level elements tensored against `1 : C` multiply exactly as their own
product (both sit in `C`-degree `0`, so no Koszul sign) -- the general form of `Bu0_mul'`. -/
theorem A0_tmul_add (n : ℕ) (x y : Module.End ℚ (WPoly n)) (b : C) :
    ((x + y : Module.End ℚ (WPoly n)) ᵍ⊗ₜ[ℚ] b : A0 n) = x ᵍ⊗ₜ[ℚ] b + y ᵍ⊗ₜ[ℚ] b := by
  show GradedTensorProduct.of ℚ (WGrading n) CGrading ((x + y) ⊗ₜ[ℚ] b)
      = GradedTensorProduct.of ℚ (WGrading n) CGrading (x ⊗ₜ[ℚ] b)
        + GradedTensorProduct.of ℚ (WGrading n) CGrading (y ⊗ₜ[ℚ] b)
  rw [← map_add, TensorProduct.add_tmul]

theorem A0_tmul_smul (n : ℕ) (c : ℚ) (x : Module.End ℚ (WPoly n)) (b : C) :
    ((c • x : Module.End ℚ (WPoly n)) ᵍ⊗ₜ[ℚ] b : A0 n) = c • (x ᵍ⊗ₜ[ℚ] b) := by
  show GradedTensorProduct.of ℚ (WGrading n) CGrading ((c • x) ⊗ₜ[ℚ] b)
      = c • GradedTensorProduct.of ℚ (WGrading n) CGrading (x ⊗ₜ[ℚ] b)
  rw [← map_smul, TensorProduct.smul_tmul']

theorem tmul_one_mul_tmul_one (n : ℕ) (p q : Module.End ℚ (WPoly n)) :
    (p ᵍ⊗ₜ[ℚ] (1 : C) : A0 n) * (q ᵍ⊗ₜ[ℚ] (1 : C) : A0 n) = (p * q) ᵍ⊗ₜ[ℚ] (1 : C) := by
  rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := WGrading n) (ℬ := CGrading)
    p (⟨1, SetLike.one_mem_graded CGrading⟩ : CGrading 0) (⟨q, trivial⟩ : WGrading n 0) (1 : C)]
  norm_num

theorem L0_comm (n : ℕ) (u v w z : Fin (2 * n)) :
    L0 n u v * L0 n w z - L0 n w z * L0 n u v
      = (1/2 : ℚ) • (JnQ n v w • L0 n u z + JnQ n u w • L0 n v z
          + JnQ n v z • L0 n u w + JnQ n u z • L0 n v w) := by
  rw [L0_eq_tmul, L0_eq_tmul, L0_eq_tmul, L0_eq_tmul, L0_eq_tmul, L0_eq_tmul]
  rw [tmul_one_mul_tmul_one, tmul_one_mul_tmul_one]
  rw [show (L0hat n u v * L0hat n w z : Module.End ℚ (WPoly n)) ᵍ⊗ₜ[ℚ] (1:C)
        - (L0hat n w z * L0hat n u v : Module.End ℚ (WPoly n)) ᵍ⊗ₜ[ℚ] (1:C)
      = (L0hat n u v * L0hat n w z - L0hat n w z * L0hat n u v) ᵍ⊗ₜ[ℚ] (1:C) from by
    show GradedTensorProduct.of ℚ (WGrading n) CGrading
        ((L0hat n u v * L0hat n w z : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1:C))
      - GradedTensorProduct.of ℚ (WGrading n) CGrading
        ((L0hat n w z * L0hat n u v : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1:C))
      = GradedTensorProduct.of ℚ (WGrading n) CGrading
        ((L0hat n u v * L0hat n w z - L0hat n w z * L0hat n u v : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (1:C))
    rw [← map_sub, TensorProduct.sub_tmul]]
  rw [L0hat_comm, A0_tmul_smul, A0_tmul_add, A0_tmul_add, A0_tmul_add,
    A0_tmul_smul, A0_tmul_smul, A0_tmul_smul, A0_tmul_smul]

/-! ## Basis-pair intertwining, case by case -/

theorem iota0Basis_Lof (n : ℕ) (u v : Fin (2 * n)) : iota0Basis n (Indexed.Lof u v) = L0 n u v := by
  unfold Indexed.Lof
  split
  · rfl
  · exact (L0_symm n u v).symm

theorem iota0Basis_Fof (n : ℕ) (u : Fin (2 * n)) : iota0Basis n (Indexed.Fof u) = F0 n u := rfl

theorem AB_tmul_add (n : ℕ) (r : RRing n) (x y : A0 n) :
    (r ᵍ⊗ₜ[ℚ] (x + y) : AB n) = r ᵍ⊗ₜ[ℚ] x + r ᵍ⊗ₜ[ℚ] y := by
  show GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) (r ⊗ₜ[ℚ] (x + y))
      = GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) (r ⊗ₜ[ℚ] x)
        + GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) (r ⊗ₜ[ℚ] y)
  rw [← map_add, TensorProduct.tmul_add]

theorem AB_tmul_smul (n : ℕ) (c : ℚ) (r : RRing n) (x : A0 n) :
    (r ᵍ⊗ₜ[ℚ] (c • x) : AB n) = c • (r ᵍ⊗ₜ[ℚ] x) := by
  show GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) (r ⊗ₜ[ℚ] (c • x))
      = c • GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) (r ⊗ₜ[ℚ] x)
  rw [← map_smul, TensorProduct.tmul_smul]

theorem AB_tmul_sub (n : ℕ) (r : RRing n) (x y : A0 n) :
    (r ᵍ⊗ₜ[ℚ] (x - y) : AB n) = r ᵍ⊗ₜ[ℚ] x - r ᵍ⊗ₜ[ℚ] y := by
  show GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) (r ⊗ₜ[ℚ] (x - y))
      = GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) (r ⊗ₜ[ℚ] x)
        - GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) (r ⊗ₜ[ℚ] y)
  rw [← map_sub, TensorProduct.tmul_sub]

/-- Two `A0 n`-elements tensored against `1 : RRing n` (on the left) multiply exactly as their
own product, provided the *first* one has a definite `A0`-degree `j` -- only `a2`'s `RRing n`
side needs degree `0` (always true of `1`), matching `tmul_coe_mul_zero_coe_tmul`'s own shape. -/
theorem AB_one_tmul_mul_one_tmul (n : ℕ) {j : ZMod 2} (a b : A0 n) (ha : a ∈ A0Grading n j) :
    ((1 : RRing n) ᵍ⊗ₜ[ℚ] a : AB n) * ((1 : RRing n) ᵍ⊗ₜ[ℚ] b) = (1 : RRing n) ᵍ⊗ₜ[ℚ] (a * b) := by
  rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
    (1 : RRing n) (⟨a, ha⟩ : A0Grading n j)
    (⟨1, SetLike.one_mem_graded (RGradingQ n)⟩ : RGradingQ n 0) b]
  norm_num

/-- **LL case**, citing `L0_comm` (lifted from `L0hat_comm`). -/
theorem iota0AB_comm_LL (n : ℕ) (u v w z : Fin (2 * n)) :
    iota0AB n (Indexed.Lof u v) * iota0AB n (Indexed.Lof w z)
        - iota0AB n (Indexed.Lof w z) * iota0AB n (Indexed.Lof u v)
      = (1/2 : ℚ) • (JnQ n v w • iota0AB n (Indexed.Lof u z) + JnQ n u w • iota0AB n (Indexed.Lof v z)
          + JnQ n v z • iota0AB n (Indexed.Lof u w) + JnQ n u z • iota0AB n (Indexed.Lof v w)) := by
  unfold iota0AB
  rw [iota0Basis_Lof, iota0Basis_Lof, iota0Basis_Lof, iota0Basis_Lof, iota0Basis_Lof, iota0Basis_Lof]
  rw [AB_one_tmul_mul_one_tmul (ha := L0_mem_A0Grading_zero n u v),
    AB_one_tmul_mul_one_tmul (ha := L0_mem_A0Grading_zero n w z), ← AB_tmul_sub, L0_comm,
    AB_tmul_smul, AB_tmul_add, AB_tmul_add, AB_tmul_add,
    AB_tmul_smul, AB_tmul_smul, AB_tmul_smul, AB_tmul_smul]

/-- **LF case**, citing `L0_comm_F0` directly (already at the `A0 n` level). -/
theorem iota0AB_comm_LF (n : ℕ) (u v w : Fin (2 * n)) :
    iota0AB n (Indexed.Lof u v) * iota0AB n (Indexed.Fof w)
        - iota0AB n (Indexed.Fof w) * iota0AB n (Indexed.Lof u v)
      = (1/2 : ℚ) • (JnQ n v w • iota0AB n (Indexed.Fof u) + JnQ n u w • iota0AB n (Indexed.Fof v)) := by
  unfold iota0AB
  rw [iota0Basis_Lof, iota0Basis_Fof, iota0Basis_Fof, iota0Basis_Fof]
  rw [AB_one_tmul_mul_one_tmul (ha := L0_mem_A0Grading_zero n u v),
    AB_one_tmul_mul_one_tmul (ha := F0_mem_A0Grading_one n w), ← AB_tmul_sub, L0_comm_F0,
    AB_tmul_smul, AB_tmul_add, AB_tmul_smul, AB_tmul_smul]

/-- **FL case**, by negation of the LF case (not a fresh computation). -/
theorem iota0AB_comm_FL (n : ℕ) (u v w : Fin (2 * n)) :
    iota0AB n (Indexed.Fof w) * iota0AB n (Indexed.Lof u v)
        - iota0AB n (Indexed.Lof u v) * iota0AB n (Indexed.Fof w)
      = -((1/2 : ℚ) • (JnQ n v w • iota0AB n (Indexed.Fof u) + JnQ n u w • iota0AB n (Indexed.Fof v))) := by
  rw [← iota0AB_comm_LF]; abel

/-- **FF case**, citing `F0_comm_F0` directly (already at the `A0 n` level); this is the one
sector where the two `iota0AB`-images *anticommute*, not commute. -/
theorem iota0AB_anticomm_FF (n : ℕ) (u v : Fin (2 * n)) :
    iota0AB n (Indexed.Fof u) * iota0AB n (Indexed.Fof v)
        + iota0AB n (Indexed.Fof v) * iota0AB n (Indexed.Fof u)
      = (1/2 : ℚ) • iota0AB n (Indexed.Lof u v) := by
  unfold iota0AB
  rw [iota0Basis_Fof, iota0Basis_Fof, iota0Basis_Lof]
  rw [AB_one_tmul_mul_one_tmul (ha := F0_mem_A0Grading_one n u),
    AB_one_tmul_mul_one_tmul (ha := F0_mem_A0Grading_one n v), ← AB_tmul_add, F0_comm_F0,
    AB_tmul_smul]

/-! ## `iota0` applied directly to `bracketLLn`/`bracketLFn`/`bracketFFn` themselves

Each sector's "sum form" (`iota0` pushed through the defining sum of `c • eN` terms) followed by
matching it, via `module`, against the corresponding `iota0AB_comm_*`/`iota0AB_anticomm_FF`
identity above -- these are **the basis-pair intertwining theorems** the round asked for, stated
literally as `iota0 n (Indexed.bracketXXn ...) = ⟨(anti)commutator of iota0AB-embedded basis
vectors⟩`, one per branch of `Indexed.bracketBasisN`'s own match. -/

theorem iota0_bracketLLn_sum (n : ℕ) (u v w z : Fin (2 * n)) :
    iota0 n (Indexed.bracketLLn n u v w z)
      = ((1/2 : ℚ) * JnQ n v w) • iota0AB n (Indexed.Lof u z)
        + ((1/2 : ℚ) * JnQ n u w) • iota0AB n (Indexed.Lof v z)
        + ((1/2 : ℚ) * JnQ n v z) • iota0AB n (Indexed.Lof u w)
        + ((1/2 : ℚ) * JnQ n u z) • iota0AB n (Indexed.Lof v w) := by
  unfold Indexed.bracketLLn
  rw [iota0_add, iota0_add, iota0_add,
    iota0_smul_eN, iota0_smul_eN, iota0_smul_eN, iota0_smul_eN,
    algebraMap_cratN_half_mul_Jn, algebraMap_cratN_half_mul_Jn,
    algebraMap_cratN_half_mul_Jn, algebraMap_cratN_half_mul_Jn,
    AB_tmul_smul_left, AB_tmul_smul_left, AB_tmul_smul_left, AB_tmul_smul_left]
  rfl

/-- **LL case.** -/
theorem iota0_bracketLLn (n : ℕ) (u v w z : Fin (2 * n)) :
    iota0 n (Indexed.bracketLLn n u v w z)
      = iota0AB n (Indexed.Lof u v) * iota0AB n (Indexed.Lof w z)
        - iota0AB n (Indexed.Lof w z) * iota0AB n (Indexed.Lof u v) := by
  rw [iota0_bracketLLn_sum, iota0AB_comm_LL]; module

theorem iota0_bracketLFn_sum (n : ℕ) (u v w : Fin (2 * n)) :
    iota0 n (Indexed.bracketLFn n u v w)
      = ((1/2 : ℚ) * JnQ n v w) • iota0AB n (Indexed.Fof u)
        + ((1/2 : ℚ) * JnQ n u w) • iota0AB n (Indexed.Fof v) := by
  unfold Indexed.bracketLFn
  rw [iota0_add, iota0_smul_eN, iota0_smul_eN,
    algebraMap_cratN_half_mul_Jn, algebraMap_cratN_half_mul_Jn,
    AB_tmul_smul_left, AB_tmul_smul_left]
  rfl

/-- **LF case.** -/
theorem iota0_bracketLFn (n : ℕ) (u v w : Fin (2 * n)) :
    iota0 n (Indexed.bracketLFn n u v w)
      = iota0AB n (Indexed.Lof u v) * iota0AB n (Indexed.Fof w)
        - iota0AB n (Indexed.Fof w) * iota0AB n (Indexed.Lof u v) := by
  rw [iota0_bracketLFn_sum, iota0AB_comm_LF]; module

/-- **FL case, by negation of the LF case** (not a fresh computation), matching
`Indexed.bracketBasisN_Fof_Lof : bracketBasisN n (Fof w) (Lof u v) = -bracketLFn n u v w`. -/
theorem iota0_bracketFLn (n : ℕ) (u v w : Fin (2 * n)) :
    iota0 n (-Indexed.bracketLFn n u v w)
      = iota0AB n (Indexed.Fof w) * iota0AB n (Indexed.Lof u v)
        - iota0AB n (Indexed.Lof u v) * iota0AB n (Indexed.Fof w) := by
  rw [iota0_neg, iota0_bracketLFn]; abel

theorem iota0_bracketFFn_sum (n : ℕ) (u v : Fin (2 * n)) :
    iota0 n (Indexed.bracketFFn n u v) = (1/2 : ℚ) • iota0AB n (Indexed.Lof u v) := by
  unfold Indexed.bracketFFn
  rw [iota0_smul_eN, algebraMap_cratN, AB_tmul_smul_left]
  rfl

/-- **FF case** -- the one sector where the two `iota0AB`-images *anticommute*, not commute. -/
theorem iota0_bracketFFn (n : ℕ) (u v : Fin (2 * n)) :
    iota0 n (Indexed.bracketFFn n u v)
      = iota0AB n (Indexed.Fof u) * iota0AB n (Indexed.Fof v)
        + iota0AB n (Indexed.Fof v) * iota0AB n (Indexed.Fof u) := by
  rw [iota0_bracketFFn_sum, iota0AB_anticomm_FF]

/-! ## The combined statement, `Indexed.bracketBasisN` itself

Restated on every basis pair via `Lof`/`Fof` (which cover `IndexedBasis n` exhaustively), citing
the four theorems above -- one per branch of `Indexed.bracketBasisN`'s own match, mirroring it
exactly (`Indexed.bracketBasisN_Lof_Lof`/`_Lof_Fof`/`_Fof_Lof`/`_Fof_Fof`, from the frozen
`IndexedJacobi.lean`, restate `bracketBasisN` on `Lof`/`Fof` inputs). -/

theorem iota0_bracketBasisN_Lof_Lof (n : ℕ) (u v w z : Fin (2 * n)) :
    iota0 n (Indexed.bracketBasisN n (Indexed.Lof u v) (Indexed.Lof w z))
      = iota0AB n (Indexed.Lof u v) * iota0AB n (Indexed.Lof w z)
        - iota0AB n (Indexed.Lof w z) * iota0AB n (Indexed.Lof u v) := by
  rw [Indexed.bracketBasisN_Lof_Lof]; exact iota0_bracketLLn n u v w z

theorem iota0_bracketBasisN_Lof_Fof (n : ℕ) (u v w : Fin (2 * n)) :
    iota0 n (Indexed.bracketBasisN n (Indexed.Lof u v) (Indexed.Fof w))
      = iota0AB n (Indexed.Lof u v) * iota0AB n (Indexed.Fof w)
        - iota0AB n (Indexed.Fof w) * iota0AB n (Indexed.Lof u v) := by
  rw [Indexed.bracketBasisN_Lof_Fof]; exact iota0_bracketLFn n u v w

theorem iota0_bracketBasisN_Fof_Lof (n : ℕ) (u v w : Fin (2 * n)) :
    iota0 n (Indexed.bracketBasisN n (Indexed.Fof w) (Indexed.Lof u v))
      = iota0AB n (Indexed.Fof w) * iota0AB n (Indexed.Lof u v)
        - iota0AB n (Indexed.Lof u v) * iota0AB n (Indexed.Fof w) := by
  rw [Indexed.bracketBasisN_Fof_Lof]; exact iota0_bracketFLn n u v w

theorem iota0_bracketBasisN_Fof_Fof (n : ℕ) (u v : Fin (2 * n)) :
    iota0 n (Indexed.bracketBasisN n (Indexed.Fof u) (Indexed.Fof v))
      = iota0AB n (Indexed.Fof u) * iota0AB n (Indexed.Fof v)
        + iota0AB n (Indexed.Fof v) * iota0AB n (Indexed.Fof u) := by
  rw [Indexed.bracketBasisN_Fof_Fof]; exact iota0_bracketFFn n u v

/-! ## General-element intertwining, via bilinear extension

**Statement-shape decision** (flagged explicitly, per the round's own instruction): because
`bracketBasisN`'s `FF` sector is an *anticommutator* while `LL`/`LF`/`FL` are *commutators*, there
is no single global expression "the commutator of `iota0 x`, `iota0 y`" equal to
`iota0 (bracketN x y)` for fully general (parity-mixed) `x y : IndexedMod n` -- forcing one would
misstate the FF contribution whenever `x` or `y` has any odd-basis component. The theorem below is
therefore stated as a `Finset.sum` of **basis-pair contributions** `iota0 (bracketBasisN n i j)`,
mirroring `bracketN`'s own defining double sum and `IndexedU.lean`'s `intertwiningU`'s proof shape
(`eR_decompose`/`Finset.sum_congr`, transposed here to `iota0`/`Indexed.eN` since `IndexedMod n`
has no ready-made `eN_decompose` -- `bracketN`'s own definition already *is* the double sum, so no
decomposition lemma is needed, only `iota0_sum`/`iota0_sum'` pushing `iota0` through it). Each
summand is fully characterized, case by case, by the four theorems just above
(`iota0_bracketLLn`/`iota0_bracketLFn`/`iota0_bracketFLn`/`iota0_bracketFFn`) -- a homogeneous `x`,
`y` pair picks out exactly one sector and the sum collapses to a single (anti)commutator. -/

theorem iota0_sum {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι)
    (f : ι → Indexed.IndexedMod n) :
    iota0 n (∑ i ∈ s, f i) = ∑ i ∈ s, iota0 n (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [iota0_zero]
  | @insert a s ha ih => rw [Finset.sum_insert ha, iota0_add, ih, Finset.sum_insert ha]

theorem iota0_sum' {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (c : ι → Indexed.Pn n)
    (f : ι → Indexed.IndexedMod n) :
    iota0 n (∑ i ∈ s, c i • f i)
      = ∑ i ∈ s, ((algebraMap (Indexed.Pn n) (RRing n) (c i)) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * iota0 n (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [iota0_zero]
  | @insert a s ha ih => rw [Finset.sum_insert ha, iota0_add, iota0_smul, ih, Finset.sum_insert ha]

/-- **The general-element intertwining theorem (stage 1's main deliverable).** -/
theorem iota0_bracketN (n : ℕ) (x y : Indexed.IndexedMod n) :
    iota0 n (Indexed.bracketN n x y)
      = ∑ i : Indexed.IndexedBasis n, ∑ j : Indexed.IndexedBasis n,
          ((algebraMap (Indexed.Pn n) (RRing n) (x i * y j)) ᵍ⊗ₜ[ℚ] (1 : A0 n))
            * iota0 n (Indexed.bracketBasisN n i j) := by
  unfold Indexed.bracketN
  rw [iota0_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [iota0_sum']

end Source
end InhomogeneousDeformations
