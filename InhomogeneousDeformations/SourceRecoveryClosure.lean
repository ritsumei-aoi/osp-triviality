import InhomogeneousDeformations.SourceRecoveryFinal

/-!
# I106 R6 — Z0, Z1, Z2, Z2b: `iota_beta` as a map, and the recovered bracket as a theorem

Ten-layer freeze (R5's nine layers plus its four `SourceRecovery*.lean` files) is read-only; this
file only reads their already-accepted definitions/theorems. Route proposed by Agent1c (see
`provided_inputs/I106_R6_PLAN.md`), independently sign-checked twice (Agent2c and an internal
second opinion, blind, zero discrepancy) before this file was written.
-/

namespace InhomogeneousDeformations
namespace Source

open scoped TensorProduct

/-! ## `kappaAB^2 = 0` -/

theorem kappaAB_sq_eq_zero (n : ℕ) : kappaAB n * kappaAB n = (0 : AB n) := by
  unfold kappaAB
  rw [GradedTensorProduct.tmul_zero_coe_mul_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
    (kappa n) (⟨1, SetLike.one_mem_graded (A0Grading n)⟩ : A0Grading n 0)
    (⟨kappa n, kappa_mem_RGradingQ_one n⟩ : RGradingQ n 1) (1 : A0 n)]
  rw [kappa_sq, mul_one, AB_zero_tmul]

/-! ## Some `RMod n` bookkeeping (trivial pointwise facts, not in the frozen layer) -/

theorem falsePart_sub {n : ℕ} (r1 r2 : Indexed.RMod n) :
    Indexed.falsePart (r1 - r2) = Indexed.falsePart r1 - Indexed.falsePart r2 := by
  funext i; simp [Indexed.falsePart]

theorem truePart_sub {n : ℕ} (r1 r2 : Indexed.RMod n) :
    Indexed.truePart (r1 - r2) = Indexed.truePart r1 - Indexed.truePart r2 := by
  funext i; simp [Indexed.truePart]

/-! ## Z0 — `liftBeta` (the `Pn n`-linear extension of `liftsFamilyBeta`) and `iotaBetaR` -/

noncomputable def liftBeta (n : ℕ) (z : Indexed.IndexedMod n) : AB n :=
  ∑ b : Indexed.IndexedBasis n,
    ((algebraMap (Indexed.Pn n) (RRing n) (z b)) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * liftsFamilyBeta n b

theorem liftBeta_add (n : ℕ) (x y : Indexed.IndexedMod n) :
    liftBeta n (x + y) = liftBeta n x + liftBeta n y := by
  unfold liftBeta
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro b _
  rw [Indexed.indexedMod_add_apply, map_add]
  show (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
      ((algebraMap (Indexed.Pn n) (RRing n) (x b) + algebraMap (Indexed.Pn n) (RRing n) (y b))
        ⊗ₜ[ℚ] (1 : A0 n))) * liftsFamilyBeta n b
    = (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
        ((algebraMap (Indexed.Pn n) (RRing n) (x b)) ⊗ₜ[ℚ] (1 : A0 n))) * liftsFamilyBeta n b
      + (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
          ((algebraMap (Indexed.Pn n) (RRing n) (y b)) ⊗ₜ[ℚ] (1 : A0 n))) * liftsFamilyBeta n b
  rw [TensorProduct.add_tmul, map_add, add_mul]

theorem liftBeta_smul (n : ℕ) (c : Indexed.Pn n) (x : Indexed.IndexedMod n) :
    liftBeta n (c • x) =
      ((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * liftBeta n x := by
  unfold liftBeta
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro b _
  rw [Indexed.indexedMod_smul_apply, map_mul, ← mul_assoc]
  congr 1
  exact (GradedTensorProduct.tmul_one_mul_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
    (algebraMap (Indexed.Pn n) (RRing n) c)
    (⟨algebraMap (Indexed.Pn n) (RRing n) (x b), algebraMap_mem_RGradingQ_zero n (x b)⟩ :
      RGradingQ n 0) (1 : A0 n)).symm

theorem liftBeta_zero (n : ℕ) : liftBeta n (0 : Indexed.IndexedMod n) = 0 := by
  unfold liftBeta
  simp only [Indexed.indexedMod_zero_apply, map_zero, AB_zero_tmul, zero_mul,
    Finset.sum_const_zero]

theorem liftBeta_eN (n : ℕ) (b : Indexed.IndexedBasis n) :
    liftBeta n (Indexed.eN b) = liftsFamilyBeta n b := by
  unfold liftBeta
  rw [Finset.sum_eq_single b]
  · unfold Indexed.eN
    rw [if_pos rfl, map_one]
    show (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) ((1 : RRing n) ⊗ₜ[ℚ] (1 : A0 n)))
        * liftsFamilyBeta n b = liftsFamilyBeta n b
    rw [← Algebra.TensorProduct.one_def, GradedTensorProduct.of_one, one_mul]
  · intro c _ hc
    unfold Indexed.eN
    rw [if_neg hc, map_zero, AB_zero_tmul, zero_mul]
  · intro h; exact absurd (Finset.mem_univ b) h

theorem liftBeta_neg (n : ℕ) (x : Indexed.IndexedMod n) : liftBeta n (-x) = -liftBeta n x := by
  have h : liftBeta n (-x) + liftBeta n x = 0 := by
    rw [← liftBeta_add, neg_add_cancel, liftBeta_zero]
  exact eq_neg_of_add_eq_zero_left h

theorem liftBeta_sum' {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (c : ι → Indexed.Pn n)
    (f : ι → Indexed.IndexedMod n) :
    liftBeta n (∑ i ∈ s, c i • f i)
      = ∑ i ∈ s, ((algebraMap (Indexed.Pn n) (RRing n) (c i)) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * liftBeta n (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [liftBeta_zero]
  | @insert a s ha ih => rw [Finset.sum_insert ha, liftBeta_add, liftBeta_smul, ih, Finset.sum_insert ha]

noncomputable def iotaBetaR (n : ℕ) (r : Indexed.RMod n) : AB n :=
  liftBeta n (Indexed.falsePart r) + kappaAB n * liftBeta n (Indexed.truePart r)

theorem iotaBetaR_add (n : ℕ) (r1 r2 : Indexed.RMod n) :
    iotaBetaR n (r1 + r2) = iotaBetaR n r1 + iotaBetaR n r2 := by
  unfold iotaBetaR
  rw [Indexed.falsePart_add, Indexed.truePart_add, liftBeta_add, liftBeta_add, mul_add]
  abel

theorem iotaBetaR_smul (n : ℕ) (c : Indexed.Pn n) (r : Indexed.RMod n) :
    iotaBetaR n (c • r) =
      ((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * iotaBetaR n r := by
  unfold iotaBetaR
  rw [Indexed.falsePart_smul, Indexed.truePart_smul, liftBeta_smul, liftBeta_smul, mul_add,
    algebraMap_tmul_mul_kappaAB_mul]

/-! ## Z1 — `iota0 = iotaBetaR ∘ UMapInv ∘ iotaR`, `UMapInv` appearing literally -/

/-- On a single basis vector, checked directly on both sum-type branches (the raw match of
`IndexedBasis n`, exactly as `Indexed.hBasis`/`liftsFamilyBeta`/`iota0Basis` are themselves
defined). Pure `iota0`/`liftBeta`/`hMap` content -- `UMapInv` enters afterward, once, at the
general-element theorem below, since `UMapInv_eq` needs no linearity to apply. -/
theorem iota0_eN_sub_eq (n : ℕ) (b : Indexed.IndexedBasis n) :
    iota0 n (Indexed.eN b) = liftBeta n (Indexed.eN b) - kappaAB n * liftBeta n (Indexed.hMap n (Indexed.eN b)) := by
  rw [Indexed.hMap_eN]
  rcases b with ⟨⟨u, v⟩, huv⟩ | u
  · show iota0 n (Indexed.eN (Sum.inl ⟨(u, v), huv⟩))
        = liftBeta n (Indexed.eN (Sum.inl ⟨(u, v), huv⟩))
          - kappaAB n * liftBeta n (Indexed.hBasis n (Sum.inl ⟨(u, v), huv⟩))
    have hh : Indexed.hBasis n (Sum.inl (⟨(u, v), huv⟩ : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}))
        = Indexed.betaN n u • Indexed.eN (Indexed.Fof v) + Indexed.betaN n v • Indexed.eN (Indexed.Fof u) := rfl
    rw [hh, liftBeta_add, liftBeta_smul, liftBeta_smul, liftBeta_eN, liftBeta_eN, liftBeta_eN]
    have hL : liftsFamilyBeta n (Sum.inl (⟨(u, v), huv⟩ : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}))
        = L0hatBeta n u v := rfl
    have hFv : liftsFamilyBeta n (Indexed.Fof v : Indexed.IndexedBasis n) = F0hatBeta n v := rfl
    have hFu : liftsFamilyBeta n (Indexed.Fof u : Indexed.IndexedBasis n) = F0hatBeta n u := rfl
    rw [hL, hFv, hFu, ← L0AB_eq_L0hatBeta_sub, iota0_eN]
    have hi0 : iota0AB n (Sum.inl (⟨(u, v), huv⟩ : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}))
        = L0AB n u v := rfl
    rw [hi0]
  · show iota0 n (Indexed.eN (Sum.inr u))
        = liftBeta n (Indexed.eN (Sum.inr u)) - kappaAB n * liftBeta n (Indexed.hBasis n (Sum.inr u))
    have hh : Indexed.hBasis n (Sum.inr u : Indexed.IndexedBasis n) = 0 := rfl
    rw [hh, liftBeta_zero, mul_zero, sub_zero, liftBeta_eN]
    have hF : liftsFamilyBeta n (Sum.inr u : Indexed.IndexedBasis n) = F0hatBeta n u := rfl
    rw [hF, iota0_eN]
    have hi0 : iota0AB n (Sum.inr u : Indexed.IndexedBasis n) = F0AB n u := rfl
    rw [hi0]
    exact (F0hatBeta_eq_F0AB n u).symm

/-- The basis-vector identity extended to all of `IndexedMod n` by `Pn n`-linearity, exactly the
`Finset.sum`-of-basis-terms pattern `iota0_bracketN`/`liftsBracket_general` already used. -/
theorem iota0_eq_liftBeta_sub (n : ℕ) (z : Indexed.IndexedMod n) :
    iota0 n z = liftBeta n z - kappaAB n * liftBeta n (Indexed.hMap n z) := by
  conv_lhs => rw [Indexed.expand_basisN z]
  conv_rhs => rw [Indexed.expand_basisN z]
  rw [iota0_sum', liftBeta_sum', Indexed.hMap_sum', liftBeta_sum', Finset.mul_sum,
    ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro b _
  rw [algebraMap_tmul_mul_kappaAB_mul, ← mul_sub]
  congr 1
  exact iota0_eN_sub_eq n b

/-- The closed form: `liftBeta` expressed purely via `iota0`, no self-reference -- proved by one
self-substitution of `iota0_eq_liftBeta_sub`, `kappaAB_sq_eq_zero` killing the doubly-`kappa`
remainder. Both independent sign-checks confirmed this step explicitly before implementation. -/
theorem liftBeta_eq_iota0_add (n : ℕ) (z : Indexed.IndexedMod n) :
    liftBeta n z = iota0 n z + kappaAB n * iota0 n (Indexed.hMap n z) := by
  have h1 := iota0_eq_liftBeta_sub n z
  have h2 := iota0_eq_liftBeta_sub n (Indexed.hMap n z)
  have e : liftBeta n z = iota0 n z + kappaAB n * liftBeta n (Indexed.hMap n z) := by
    rw [h1]; abel
  have e' : liftBeta n (Indexed.hMap n z)
      = iota0 n (Indexed.hMap n z)
        + kappaAB n * liftBeta n (Indexed.hMap n (Indexed.hMap n z)) := by
    rw [h2]; abel
  rw [e, e', mul_add, ← mul_assoc, kappaAB_sq_eq_zero, zero_mul, add_zero]

/-- **Z1, the round's point**: `UMapInv` appears literally in the statement. -/
theorem iota0_eq_iotaBetaR_UMapInv_iotaR (n : ℕ) (z : Indexed.IndexedMod n) :
    iota0 n z = iotaBetaR n (Indexed.UMapInv n (Indexed.iotaR z)) := by
  rw [iota0_eq_liftBeta_sub, Indexed.UMapInv_eq, Indexed.falsePart_iotaR]
  unfold iotaBetaR
  rw [falsePart_sub, truePart_sub, Indexed.falsePart_iotaR, Indexed.falsePart_kappaEmbed,
    Indexed.truePart_iotaR, Indexed.truePart_kappaEmbed, sub_zero, zero_sub, liftBeta_neg, mul_neg]
  abel

/-! ## Z2 — P4, the recovered bracket as a formal object -/

theorem prop_recovery (n : ℕ) (x y : Indexed.IndexedMod n) :
    ∑ i : Indexed.IndexedBasis n, ∑ j : Indexed.IndexedBasis n,
        ((algebraMap (Indexed.Pn n) (RRing n) (x i * y j)) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * liftsBracket n i j
      = iotaBetaR n (Indexed.iotaR (Indexed.bracketN n x y)
          + Indexed.kappaEmbed (Indexed.GammaBetaN n x y)) := by
  have hRHS : iotaBetaR n (Indexed.iotaR (Indexed.bracketN n x y)
        + Indexed.kappaEmbed (Indexed.GammaBetaN n x y))
      = liftBeta n (Indexed.bracketN n x y) + kappaAB n * liftBeta n (Indexed.GammaBetaN n x y) := by
    unfold iotaBetaR
    rw [Indexed.falsePart_add, Indexed.truePart_add, Indexed.falsePart_iotaR,
      Indexed.falsePart_kappaEmbed, Indexed.truePart_iotaR, Indexed.truePart_kappaEmbed, add_zero,
      zero_add]
  rw [hRHS]
  have h1 := iota0_eq_liftBeta_sub n (Indexed.bracketN n x y)
  have h2 := iota0_eq_liftBeta_sub n (Indexed.GammaBetaN n x y)
  have e1 : liftBeta n (Indexed.bracketN n x y)
      = iota0 n (Indexed.bracketN n x y)
        + kappaAB n * liftBeta n (Indexed.hMap n (Indexed.bracketN n x y)) := by
    rw [h1]; abel
  have e2 : kappaAB n * liftBeta n (Indexed.GammaBetaN n x y)
      = kappaAB n * iota0 n (Indexed.GammaBetaN n x y)
        + kappaAB n * (kappaAB n * liftBeta n (Indexed.hMap n (Indexed.GammaBetaN n x y))) := by
    rw [h2, mul_sub]; abel
  have e3 : kappaAB n * liftBeta n (Indexed.hMap n (Indexed.bracketN n x y))
      = kappaAB n * iota0 n (Indexed.hMap n (Indexed.bracketN n x y)) := by
    rw [liftBeta_eq_iota0_add, mul_add, ← mul_assoc, kappaAB_sq_eq_zero, zero_mul, add_zero]
  rw [e1, e2, e3, ← mul_assoc, kappaAB_sq_eq_zero, zero_mul, add_zero, liftsBracket_general]
  abel

/-! ## Z2b — tied to the frozen `bracketRBeta` -/

theorem prop_recovery_bracketRBeta (n : ℕ) (x y : Indexed.IndexedMod n) :
    iotaBetaR n (Indexed.iotaR (Indexed.bracketN n x y) + Indexed.kappaEmbed (Indexed.GammaBetaN n x y))
      = iotaBetaR n (Indexed.bracketRBeta n (Indexed.iotaR x) (Indexed.iotaR y)) := by
  rw [Indexed.bracketRBeta_iotaR_iotaR]

/-! ## Z3 — injectivity of `iotaBetaR`, uniqueness of the coefficient -/

/-- `{iota0Basis b}` is `ℚ`-linearly independent in `A_0 n` -- the un-corrected analogue of
`liftsFamilyBeta_linearIndependent`, but landing directly in `A_0 n` (no `R`-factor, no
`Corr`/`ABGradingR` machinery needed: `iota0Basis` has no deformation term). -/
theorem iota0Basis_linearIndependent (n : ℕ) :
    LinearIndependent ℚ (Source.iota0Basis n) := by
  rw [Fintype.linearIndependent_iff]
  intro g hg
  have hsplit0 := Fintype.sum_sum_type (fun y => g y • Source.iota0Basis n y)
  have hL0F0eq : (∑ p, g (Sum.inl p) • L0 n p.1.1 p.1.2) + (∑ u, g (Sum.inr u) • F0 n u) = 0 :=
    hsplit0.symm.trans hg
  have hLmem : (∑ p, g (Sum.inl p) • L0 n p.1.1 p.1.2) ∈ A0Grading n 0 :=
    Submodule.sum_mem _ (fun p _ => Submodule.smul_mem _ _ (L0_mem_A0Grading_zero n p.1.1 p.1.2))
  have hFmem : (∑ u, g (Sum.inr u) • F0 n u) ∈ A0Grading n 1 :=
    Submodule.sum_mem _ (fun u _ => Submodule.smul_mem _ _ (F0_mem_A0Grading_one n u))
  have hLeq0 : (∑ p, g (Sum.inl p) • L0 n p.1.1 p.1.2) = 0 :=
    (Submodule.disjoint_def.mp (A0Grading_disjoint_zero_one n)) _ hLmem
      (by rw [show (∑ p, g (Sum.inl p) • L0 n p.1.1 p.1.2) = -(∑ u, g (Sum.inr u) • F0 n u) from by
            rw [eq_neg_iff_add_eq_zero]; exact hL0F0eq]
          exact (Submodule.neg_mem_iff _).mpr hFmem)
  have hFeq0 : (∑ u, g (Sum.inr u) • F0 n u) = 0 := by
    rw [hLeq0, zero_add] at hL0F0eq; exact hL0F0eq
  intro x
  cases x with
  | inl p => exact (Fintype.linearIndependent_iff.mp (L0FamilyIndependent_proved n)) (fun p => g (Sum.inl p)) hLeq0 p
  | inr u => exact (Fintype.linearIndependent_iff.mp (F0_linearIndependent n)) (fun u => g (Sum.inr u)) hFeq0 u

/-- The `ℚ`-linear combination map for `iota0Basis`, as an explicit `LinearMap`, needed to invoke
`LinearMap.exists_leftInverse_of_injective` (a left inverse is the cleanest way to extract each
coordinate `g b` back out of a value `∑ g b • iota0Basis b`, without separately constructing a
full basis extension). -/
noncomputable def iota0BasisCombo (n : ℕ) : (Indexed.IndexedBasis n → ℚ) →ₗ[ℚ] A0 n where
  toFun g := ∑ b, g b • Source.iota0Basis n b
  map_add' g g' := by simp [add_smul, Finset.sum_add_distrib]
  map_smul' c g := by simp [smul_smul, Finset.smul_sum]

theorem iota0BasisCombo_injective (n : ℕ) : Function.Injective (iota0BasisCombo n) := by
  rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
  intro g hg
  funext b
  exact (Fintype.linearIndependent_iff.mp (iota0Basis_linearIndependent n)) g hg b

/-- The same local algebra map `SourceTensor.lean`'s `kappa_ne_zero` builds (via
`CliffordAlgebra.lift`), exposed here so it can be composed with `TrivSqZeroExt.fstHom`. -/
noncomputable def RRingToTriv (n : ℕ) :
    RRing n →ₐ[Indexed.Pn n] TrivSqZeroExt (Indexed.Pn n) (Indexed.Pn n) :=
  CliffordAlgebra.lift (Qzero n)
    ⟨TrivSqZeroExt.inrHom (Indexed.Pn n) (Indexed.Pn n), fun m => by
      rw [TrivSqZeroExt.inrHom, LinearMap.coe_mk, AddHom.coe_mk, TrivSqZeroExt.inr_mul_inr,
        show Qzero n m = 0 from rfl, map_zero]⟩

/-- A retraction `RRing n →ₗ[ℚ] Pn n` of `algebraMap`: `RRingToTriv` followed by projection to
the first (`Pn n`) coordinate. Since `RRingToTriv` is a `Pn n`-algebra map,
`RRingToTriv (algebraMap c) = c • RRingToTriv 1 = c • 1`, whose first coordinate is `c`. -/
noncomputable def RRingRetract (n : ℕ) : RRing n →ₗ[ℚ] Indexed.Pn n :=
  ((TrivSqZeroExt.fstHom (Indexed.Pn n) (Indexed.Pn n) (Indexed.Pn n)).toLinearMap.restrictScalars ℚ)
    ∘ₗ ((RRingToTriv n).toLinearMap.restrictScalars ℚ)

theorem RRingRetract_algebraMap (n : ℕ) (c : Indexed.Pn n) :
    RRingRetract n (algebraMap (Indexed.Pn n) (RRing n) c) = c := by
  have key : (TrivSqZeroExt.fstHom (Indexed.Pn n) (Indexed.Pn n) (Indexed.Pn n))
      ((RRingToTriv n) (algebraMap (Indexed.Pn n) (RRing n) c)) = c := by
    rw [Algebra.algebraMap_eq_smul_one, map_smul, map_one, map_smul, map_one, smul_eq_mul, mul_one]
  exact key

/-- For each `b0`, extracts `x b0` from `iota0 n x` -- built from a left inverse `ginv` of
`iota0BasisCombo` (peeling off the `A0 n` factor down to a single coordinate) composed with
`RRingRetract` (peeling the `R`-factor's `algebraMap` back to the original `Pn n`-scalar). -/
noncomputable def iota0Coord (n : ℕ) (ginv : A0 n →ₗ[ℚ] (Indexed.IndexedBasis n → ℚ))
    (b0 : Indexed.IndexedBasis n) : AB n →ₗ[ℚ] Indexed.Pn n :=
  (TensorProduct.rid ℚ (Indexed.Pn n)).toLinearMap ∘ₗ
    (TensorProduct.map (RRingRetract n) ((LinearMap.proj b0 : (Indexed.IndexedBasis n → ℚ) →ₗ[ℚ] ℚ) ∘ₗ ginv)) ∘ₗ
    (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)).symm.toLinearMap

theorem iota0Coord_apply (n : ℕ) (ginv : A0 n →ₗ[ℚ] (Indexed.IndexedBasis n → ℚ))
    (b0 : Indexed.IndexedBasis n) (c : Indexed.Pn n) (y : A0 n) :
    iota0Coord n ginv b0 ((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] y) = ginv y b0 • c := by
  unfold iota0Coord
  show (TensorProduct.rid ℚ (Indexed.Pn n))
      (TensorProduct.map (RRingRetract n) ((LinearMap.proj b0 : (Indexed.IndexedBasis n → ℚ) →ₗ[ℚ] ℚ) ∘ₗ ginv)
        ((GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)).symm
          (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
            ((algebraMap (Indexed.Pn n) (RRing n) c) ⊗ₜ[ℚ] y))))
      = ginv y b0 • c
  rw [LinearEquiv.symm_apply_apply, TensorProduct.map_tmul]
  show (TensorProduct.rid ℚ (Indexed.Pn n))
      ((RRingRetract n (algebraMap (Indexed.Pn n) (RRing n) c)) ⊗ₜ[ℚ]
        ((LinearMap.proj b0 : (Indexed.IndexedBasis n → ℚ) →ₗ[ℚ] ℚ) (ginv y)))
      = ginv y b0 • c
  rw [RRingRetract_algebraMap]
  show ginv y b0 • c = ginv y b0 • c
  rfl

theorem iota0_eq_zero_imp (n : ℕ) (z : Indexed.IndexedMod n) (hz : iota0 n z = 0) : z = 0 := by
  have hginv := LinearMap.exists_leftInverse_of_injective (iota0BasisCombo n)
    (LinearMap.ker_eq_bot.mpr (iota0BasisCombo_injective n))
  obtain ⟨ginv, hginv⟩ := hginv
  have hcoord : ∀ b : Indexed.IndexedBasis n,
      ginv (Source.iota0Basis n b) = (Pi.single b (1 : ℚ) : Indexed.IndexedBasis n → ℚ) := by
    intro b
    have hcombo : iota0BasisCombo n (Pi.single b (1 : ℚ)) = Source.iota0Basis n b := by
      unfold iota0BasisCombo
      show ∑ b' : Indexed.IndexedBasis n,
          (Pi.single b (1 : ℚ) : Indexed.IndexedBasis n → ℚ) b' • Source.iota0Basis n b'
          = Source.iota0Basis n b
      rw [Finset.sum_eq_single b]
      · simp
      · intro b' _ hb'
        simp [hb']
      · intro h; exact absurd (Finset.mem_univ b) h
    have := congrFun (congrArg DFunLike.coe hginv) (Pi.single b (1 : ℚ))
    simpa [hcombo] using this
  funext b0
  have hcompute : iota0Coord n ginv b0 (iota0 n z) = z b0 := by
    unfold iota0
    rw [map_sum]
    have hterm : ∀ b : Indexed.IndexedBasis n,
        iota0Coord n ginv b0
          ((algebraMap (Indexed.Pn n) (RRing n) (z b)) ᵍ⊗ₜ[ℚ] (Source.iota0Basis n b))
          = if b = b0 then z b else 0 := by
      intro b
      rw [iota0Coord_apply, hcoord b]
      by_cases h : b = b0
      · subst h; simp
      · simp [h]
    simp_rw [hterm]
    rw [Finset.sum_ite_eq' Finset.univ b0 z, if_pos (Finset.mem_univ b0)]
  rw [hz, map_zero] at hcompute
  simpa using hcompute.symm

theorem iota0_injective (n : ℕ) : Function.Injective (iota0 n) := by
  intro x y hxy
  have hxy0 : iota0 n (x - y) = 0 := by
    rw [sub_eq_add_neg, iota0_add, iota0_neg, hxy, add_neg_cancel]
  have := iota0_eq_zero_imp n (x - y) hxy0
  exact sub_eq_zero.mp this

/-! ## Z3b — `kappaAB n * iota0 n z = 0 → z = 0`

A retraction of `c ↦ kappa n * algebraMap c` (not of `algebraMap` alone, and not claiming
`kappa`-multiplication injective on anything wider -- `kappaMulR_not_injective` (K0c, frozen
`IndexedKappa.lean`) is left completely undisturbed): `RRingToTriv (kappa n) = TrivSqZeroExt.inr 1`
(exactly as `kappa_ne_zero`'s own proof establishes) and `RRingToTriv (algebraMap c) = (c, 0)`
(`TrivSqZeroExt.inl c`, the ring unit scaled by `c`), so `RRingToTriv (kappa n * algebraMap c) =
inr 1 * inl c`, whose `snd` component is exactly `c` (`TrivSqZeroExt.snd_mul`, using `fst (inr 1) =
0`, `snd (inr 1) = 1`, `fst (inl c) = c`, `snd (inl c) = 0`). -/

noncomputable def RRingRetractKappa (n : ℕ) : RRing n →ₗ[ℚ] Indexed.Pn n :=
  (TrivSqZeroExt.sndHom (R := Indexed.Pn n) (M := Indexed.Pn n)).restrictScalars ℚ
    ∘ₗ ((RRingToTriv n).toLinearMap.restrictScalars ℚ)

theorem RRingRetractKappa_kappa_mul_algebraMap (n : ℕ) (c : Indexed.Pn n) :
    RRingRetractKappa n (kappa n * algebraMap (Indexed.Pn n) (RRing n) c) = c := by
  have key : (TrivSqZeroExt.sndHom (R := Indexed.Pn n) (M := Indexed.Pn n))
      ((RRingToTriv n) (kappa n * algebraMap (Indexed.Pn n) (RRing n) c)) = c := by
    rw [map_mul]
    have hk : (RRingToTriv n) (kappa n) = TrivSqZeroExt.inr (1 : Indexed.Pn n) := by
      unfold RRingToTriv kappa
      rw [CliffordAlgebra.lift_ι_apply]
      rfl
    have hc : (RRingToTriv n) (algebraMap (Indexed.Pn n) (RRing n) c)
        = TrivSqZeroExt.inl c := by
      rw [Algebra.algebraMap_eq_smul_one, map_smul, map_one]
      show c • (1 : TrivSqZeroExt (Indexed.Pn n) (Indexed.Pn n)) = TrivSqZeroExt.inl c
      rw [show (1 : TrivSqZeroExt (Indexed.Pn n) (Indexed.Pn n)) = TrivSqZeroExt.inl 1 from rfl,
        ← TrivSqZeroExt.inl_smul, smul_eq_mul, mul_one]
    rw [hk, hc]
    show TrivSqZeroExt.sndHom (R := Indexed.Pn n) (M := Indexed.Pn n)
        (TrivSqZeroExt.inr (1 : Indexed.Pn n) * TrivSqZeroExt.inl c) = c
    show (TrivSqZeroExt.inr (1 : Indexed.Pn n) * TrivSqZeroExt.inl c).snd = c
    rw [TrivSqZeroExt.snd_mul, TrivSqZeroExt.fst_inr, TrivSqZeroExt.snd_inr,
      TrivSqZeroExt.fst_inl, TrivSqZeroExt.snd_inl]
    simp
  exact key

noncomputable def kappaCoord (n : ℕ) (ginv : A0 n →ₗ[ℚ] (Indexed.IndexedBasis n → ℚ))
    (b0 : Indexed.IndexedBasis n) : AB n →ₗ[ℚ] Indexed.Pn n :=
  (TensorProduct.rid ℚ (Indexed.Pn n)).toLinearMap ∘ₗ
    (TensorProduct.map (RRingRetractKappa n) ((LinearMap.proj b0 : (Indexed.IndexedBasis n → ℚ) →ₗ[ℚ] ℚ) ∘ₗ ginv)) ∘ₗ
    (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)).symm.toLinearMap

theorem kappaCoord_apply (n : ℕ) (ginv : A0 n →ₗ[ℚ] (Indexed.IndexedBasis n → ℚ))
    (b0 : Indexed.IndexedBasis n) (c : Indexed.Pn n) (y : A0 n) :
    kappaCoord n ginv b0 ((kappa n * algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] y)
      = ginv y b0 • c := by
  unfold kappaCoord
  show (TensorProduct.rid ℚ (Indexed.Pn n))
      (TensorProduct.map (RRingRetractKappa n) ((LinearMap.proj b0 : (Indexed.IndexedBasis n → ℚ) →ₗ[ℚ] ℚ) ∘ₗ ginv)
        ((GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)).symm
          (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
            ((kappa n * algebraMap (Indexed.Pn n) (RRing n) c) ⊗ₜ[ℚ] y))))
      = ginv y b0 • c
  rw [LinearEquiv.symm_apply_apply, TensorProduct.map_tmul]
  show (TensorProduct.rid ℚ (Indexed.Pn n))
      ((RRingRetractKappa n (kappa n * algebraMap (Indexed.Pn n) (RRing n) c)) ⊗ₜ[ℚ]
        ((LinearMap.proj b0 : (Indexed.IndexedBasis n → ℚ) →ₗ[ℚ] ℚ) (ginv y)))
      = ginv y b0 • c
  rw [RRingRetractKappa_kappa_mul_algebraMap]
  rfl

theorem kappaAB_iota0_eq_zero_imp (n : ℕ) (z : Indexed.IndexedMod n)
    (hz : kappaAB n * iota0 n z = 0) : z = 0 := by
  have hginv := LinearMap.exists_leftInverse_of_injective (iota0BasisCombo n)
    (LinearMap.ker_eq_bot.mpr (iota0BasisCombo_injective n))
  obtain ⟨ginv, hginv⟩ := hginv
  have hcoord : ∀ b : Indexed.IndexedBasis n,
      ginv (Source.iota0Basis n b) = (Pi.single b (1 : ℚ) : Indexed.IndexedBasis n → ℚ) := by
    intro b
    have hcombo : iota0BasisCombo n (Pi.single b (1 : ℚ)) = Source.iota0Basis n b := by
      unfold iota0BasisCombo
      show ∑ b' : Indexed.IndexedBasis n,
          (Pi.single b (1 : ℚ) : Indexed.IndexedBasis n → ℚ) b' • Source.iota0Basis n b'
          = Source.iota0Basis n b
      rw [Finset.sum_eq_single b]
      · simp
      · intro b' _ hb'
        simp [hb']
      · intro h; exact absurd (Finset.mem_univ b) h
    have := congrFun (congrArg DFunLike.coe hginv) (Pi.single b (1 : ℚ))
    simpa [hcombo] using this
  funext b0
  have hcompute : kappaCoord n ginv b0 (kappaAB n * iota0 n z) = z b0 := by
    have hexpand : kappaAB n * iota0 n z
        = ∑ b : Indexed.IndexedBasis n,
            (kappa n * algebraMap (Indexed.Pn n) (RRing n) (z b)) ᵍ⊗ₜ[ℚ] (Source.iota0Basis n b) := by
      unfold iota0
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl (fun b _ => kappaAB_mul_algebraMap_tmul n (z b) (Source.iota0Basis n b))
    rw [hexpand, map_sum]
    have hterm : ∀ b : Indexed.IndexedBasis n,
        kappaCoord n ginv b0
          ((kappa n * algebraMap (Indexed.Pn n) (RRing n) (z b)) ᵍ⊗ₜ[ℚ] (Source.iota0Basis n b))
          = if b = b0 then z b else 0 := by
      intro b
      rw [kappaCoord_apply, hcoord b]
      by_cases h : b = b0
      · subst h; simp
      · simp [h]
    simp_rw [hterm]
    rw [Finset.sum_ite_eq' Finset.univ b0 z, if_pos (Finset.mem_univ b0)]
  rw [hz, map_zero] at hcompute
  simpa using hcompute.symm

/-! ## Z3c — `iotaBetaR` injective, uniqueness of the coefficient -/

theorem iota0_mem_ABGradingR_zero (n : ℕ) (z : Indexed.IndexedMod n) :
    iota0 n z ∈ ABGradingR n 0 := by
  unfold iota0
  exact Submodule.sum_mem _ (fun b _ =>
    ABGradingR_mem_of_tmul n 0
      (⟨algebraMap (Indexed.Pn n) (RRing n) (z b), algebraMap_mem_RGradingQ_zero n (z b)⟩ :
        RGradingQ n 0)
      (Source.iota0Basis n b))

theorem kappa_mul_algebraMap_mem_RGradingQ_one (n : ℕ) (c : Indexed.Pn n) :
    kappa n * algebraMap (Indexed.Pn n) (RRing n) c ∈ RGradingQ n 1 := by
  have hmem := SetLike.mul_mem_graded (kappa_mem_RGradingQ_one n) (algebraMap_mem_RGradingQ_zero n c)
  simpa using hmem

set_option maxHeartbeats 1000000 in
theorem kappaAB_mul_iota0_mem_ABGradingR_one (n : ℕ) (z : Indexed.IndexedMod n) :
    kappaAB n * iota0 n z ∈ ABGradingR n 1 := by
  have hexpand : kappaAB n * iota0 n z
      = ∑ b : Indexed.IndexedBasis n,
          (kappa n * algebraMap (Indexed.Pn n) (RRing n) (z b)) ᵍ⊗ₜ[ℚ] (Source.iota0Basis n b) := by
    unfold iota0
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun b _ => kappaAB_mul_algebraMap_tmul n (z b) (Source.iota0Basis n b))
  rw [hexpand]
  exact Submodule.sum_mem _ (fun b _ =>
    ABGradingR_mem_of_tmul n 1
      (⟨kappa n * algebraMap (Indexed.Pn n) (RRing n) (z b),
        kappa_mul_algebraMap_mem_RGradingQ_one n (z b)⟩ : RGradingQ n 1)
      (Source.iota0Basis n b))

theorem iotaBetaR_zero (n : ℕ) : iotaBetaR n (0 : Indexed.RMod n) = 0 := by
  unfold iotaBetaR
  have hf : Indexed.falsePart (0 : Indexed.RMod n) = 0 := by
    funext i; simp [Indexed.falsePart]
  have ht : Indexed.truePart (0 : Indexed.RMod n) = 0 := by
    funext i; simp [Indexed.truePart]
  rw [hf, ht, liftBeta_zero, mul_zero, add_zero]

theorem iotaBetaR_neg (n : ℕ) (r : Indexed.RMod n) : iotaBetaR n (-r) = -iotaBetaR n r := by
  have h : iotaBetaR n (-r) + iotaBetaR n r = 0 := by
    rw [← iotaBetaR_add, neg_add_cancel, iotaBetaR_zero]
  exact eq_neg_of_add_eq_zero_left h

/-- **The key decomposition**: `iotaBetaR n r` splits cleanly into an `ABGradingR n 0` part
(`iota0 n (falsePart r)`) and an `ABGradingR n 1` part (`kappaAB n * iota0 n (...)`), obtained by
applying `liftBeta_eq_iota0_add` to both `falsePart r` and `truePart r` and using
`kappaAB_sq_eq_zero` to kill the resulting double-`kappa` cross term. -/
theorem iotaBetaR_eq_iota0_add (n : ℕ) (r : Indexed.RMod n) :
    iotaBetaR n r = iota0 n (Indexed.falsePart r)
      + kappaAB n * iota0 n (Indexed.hMap n (Indexed.falsePart r) + Indexed.truePart r) := by
  unfold iotaBetaR
  have e1 := liftBeta_eq_iota0_add n (Indexed.falsePart r)
  have e2 := liftBeta_eq_iota0_add n (Indexed.truePart r)
  rw [e1, e2, mul_add, ← mul_assoc, kappaAB_sq_eq_zero, zero_mul, add_zero, iota0_add, mul_add]
  abel

/-- **Z3c**: `iotaBetaR` is injective. -/
theorem iotaBetaR_injective (n : ℕ) : Function.Injective (iotaBetaR n) := by
  intro r1 r2 hr
  rw [← sub_eq_zero]
  have h : iotaBetaR n (r1 - r2) = 0 := by
    rw [sub_eq_add_neg, iotaBetaR_add, iotaBetaR_neg, hr, add_neg_cancel]
  have hfalse0 : Indexed.falsePart (r1 - r2) = 0 := by
    have hmem0 : iota0 n (Indexed.falsePart (r1 - r2)) ∈ ABGradingR n 0 :=
      iota0_mem_ABGradingR_zero n _
    have hmem1 : iota0 n (Indexed.falsePart (r1 - r2)) ∈ ABGradingR n 1 := by
      have heq := iotaBetaR_eq_iota0_add n (r1 - r2)
      rw [h] at heq
      have hzero : iota0 n (Indexed.falsePart (r1 - r2))
          = - (kappaAB n * iota0 n (Indexed.hMap n (Indexed.falsePart (r1 - r2))
              + Indexed.truePart (r1 - r2))) := by
        rw [eq_neg_iff_add_eq_zero]; exact heq.symm
      rw [hzero]
      exact (Submodule.neg_mem_iff _).mpr (kappaAB_mul_iota0_mem_ABGradingR_one n _)
    have hcombo0 : iota0 n (Indexed.falsePart (r1 - r2)) = 0 :=
      (Submodule.disjoint_def.mp (ABGradingR_disjoint_zero_one n)) _ hmem0 hmem1
    exact iota0_eq_zero_imp n _ hcombo0
  have htrue0 : Indexed.truePart (r1 - r2) = 0 := by
    have heq := iotaBetaR_eq_iota0_add n (r1 - r2)
    rw [h, hfalse0, iota0_zero, zero_add, Indexed.hMap_zero, zero_add] at heq
    exact kappaAB_iota0_eq_zero_imp n _ heq.symm
  rw [Indexed.decompose (r1 - r2), hfalse0, htrue0, Indexed.iotaR_zero, Indexed.kappaEmbed_zero,
    add_zero]

/-- **Uniqueness of the coefficient** (the manuscript's own "`g_R=g_P⊕\kappa g_P`, so the
coefficient is unique", lines 270-272), via `iotaBetaR_injective` and the `iotaR`/`kappaEmbed`
splitting: two candidate `IndexedMod n`-valued coefficients giving the same `iotaBetaR`-value on
`iotaR (bracketN x y) + kappaEmbed w` must coincide. -/
theorem coefficient_unique (n : ℕ) (x y w1 w2 : Indexed.IndexedMod n)
    (h : iotaBetaR n (Indexed.iotaR (Indexed.bracketN n x y) + Indexed.kappaEmbed w1)
        = iotaBetaR n (Indexed.iotaR (Indexed.bracketN n x y) + Indexed.kappaEmbed w2)) :
    w1 = w2 := by
  have hr := iotaBetaR_injective n h
  have h1 := congrArg Indexed.truePart hr
  rwa [Indexed.truePart_add, Indexed.truePart_add, Indexed.truePart_iotaR, zero_add, zero_add,
    Indexed.truePart_kappaEmbed, Indexed.truePart_kappaEmbed] at h1

end Source
end InhomogeneousDeformations
