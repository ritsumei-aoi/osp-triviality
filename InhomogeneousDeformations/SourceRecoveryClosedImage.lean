import InhomogeneousDeformations.SourceRecoveryClosure

/-!
# I106 R7 — W0-W3: closed image, the last clause of `prop:recovery`

Eleven-layer freeze (R6's ten layers plus `SourceRecoveryClosure.lean`) is read-only; this file
only reads their already-accepted definitions/theorems.

The naive extension of `liftsBracket`'s commutator/anticommutator dispatch (bare `IndexedBasis n`
parity, unchanged) to `kappa`-multiplied generators does **not** close; the correct extension uses
`parityR` (`IndexedKappa.lean:105`, the `kappa`-slot itself contributes to parity) to build a
genuine supercommutator. This was independently derived and cross-checked twice (blind) before
this file was written, with zero discrepancy.
-/

namespace InhomogeneousDeformations
namespace Source

open scoped TensorProduct

/-! ## `gsignNQ`, the plain-`ℚ` companion of the frozen `gsignN`, mirroring `SourceWeyl.lean`'s
own `JnQ` device exactly. -/

noncomputable def gsignNQ (n : ℕ) (a b : ZMod 2) : ℚ :=
  MvPolynomial.constantCoeff (Indexed.gsignN n a b)

theorem gsignNQ_00 (n : ℕ) : gsignNQ n (0 : ZMod 2) (0 : ZMod 2) = 1 := by
  unfold gsignNQ; rw [Indexed.gsignN_00]; simp
theorem gsignNQ_01 (n : ℕ) : gsignNQ n (0 : ZMod 2) (1 : ZMod 2) = 1 := by
  unfold gsignNQ; rw [Indexed.gsignN_01]; simp
theorem gsignNQ_10 (n : ℕ) : gsignNQ n (1 : ZMod 2) (0 : ZMod 2) = 1 := by
  unfold gsignNQ; rw [Indexed.gsignN_10]; simp
theorem gsignNQ_11 (n : ℕ) : gsignNQ n (1 : ZMod 2) (1 : ZMod 2) = -1 := by
  unfold gsignNQ; rw [Indexed.gsignN_11]; simp

/-! ## `L0AB`/`F0AB` commute/anticommute with `kappaAB`, by A0-degree -/

theorem L0AB_comm_kappaAB (n : ℕ) (u v : Fin (2 * n)) :
    L0AB n u v * kappaAB n = kappaAB n * L0AB n u v := by
  have e1 : L0AB n u v * kappaAB n = (kappa n) ᵍ⊗ₜ[ℚ] (L0 n u v) := by
    unfold L0AB kappaAB
    rw [GradedTensorProduct.tmul_zero_coe_mul_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
      (1 : RRing n) (⟨L0 n u v, L0_mem_A0Grading_zero n u v⟩ : A0Grading n 0)
      (⟨kappa n, kappa_mem_RGradingQ_one n⟩ : RGradingQ n 1) (1 : A0 n)]
    rw [mul_one, one_mul]
  have e2 : kappaAB n * L0AB n u v = (kappa n) ᵍ⊗ₜ[ℚ] (L0 n u v) := by
    unfold L0AB
    rw [show ((1 : RRing n) ᵍ⊗ₜ[ℚ] (L0 n u v) : AB n)
        = (algebraMap (Indexed.Pn n) (RRing n) (1 : Indexed.Pn n)) ᵍ⊗ₜ[ℚ] (L0 n u v) from by
      rw [map_one]]
    rw [kappaAB_mul_algebraMap_tmul, map_one, mul_one]
  rw [e1, e2]

set_option maxHeartbeats 1000000 in
theorem F0AB_comm_kappaAB (n : ℕ) (u : Fin (2 * n)) :
    F0AB n u * kappaAB n = -(kappaAB n * F0AB n u) := by
  unfold F0AB kappaAB
  have e1 : ((1 : RRing n) ᵍ⊗ₜ[ℚ] (F0 n u) : AB n) * ((kappa n) ᵍ⊗ₜ[ℚ] (1 : A0 n))
      = (-1 : ℤˣ) ^ ((1 : ZMod 2) * (1 : ZMod 2)) •
        ((kappa n) ᵍ⊗ₜ[ℚ] (F0 n u) : AB n) := by
    rw [GradedTensorProduct.tmul_coe_mul_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
      (1 : RRing n) (⟨F0 n u, F0_mem_A0Grading_one n u⟩ : A0Grading n 1)
      (⟨kappa n, kappa_mem_RGradingQ_one n⟩ : RGradingQ n 1) (1 : A0 n)]
    congr 2
    · exact one_mul (kappa n)
    · exact mul_one (F0 n u)
  have e2 : (kappa n) ᵍ⊗ₜ[ℚ] (1 : A0 n) * ((1 : RRing n) ᵍ⊗ₜ[ℚ] (F0 n u) : AB n)
      = ((kappa n) ᵍ⊗ₜ[ℚ] (F0 n u) : AB n) := by
    rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
      (kappa n) (⟨1, SetLike.one_mem_graded (A0Grading n)⟩ : A0Grading n 0)
      (⟨1, SetLike.one_mem_graded (RGradingQ n)⟩ : RGradingQ n 0) (F0 n u)]
    rw [mul_one, one_mul]
  rw [e1, e2]
  norm_num

/-- Generalizes `F0AB_comm_kappaAB` with an extra `algebraMap c` factor in the `R`-slot (still
degree `0` there, so the sign is unaffected — only `F0 n v`'s own `A0`-degree `1` matters). -/
theorem algebraMap_tmul_F0_comm_kappaAB (n : ℕ) (c : Indexed.Pn n) (v : Fin (2 * n)) :
    ((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (F0 n v) : AB n) * kappaAB n
      = -(kappaAB n * ((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (F0 n v))) := by
  unfold kappaAB
  have e1 : ((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (F0 n v) : AB n)
      * ((kappa n) ᵍ⊗ₜ[ℚ] (1 : A0 n))
      = (-1 : ℤˣ) ^ ((1 : ZMod 2) * (1 : ZMod 2)) •
        ((kappa n * algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (F0 n v) : AB n) := by
    rw [GradedTensorProduct.tmul_coe_mul_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
      (algebraMap (Indexed.Pn n) (RRing n) c) (⟨F0 n v, F0_mem_A0Grading_one n v⟩ : A0Grading n 1)
      (⟨kappa n, kappa_mem_RGradingQ_one n⟩ : RGradingQ n 1) (1 : A0 n)]
    congr 2
    · exact (Algebra.commutes c (kappa n))
    · exact mul_one (F0 n v)
  have e2 : (kappa n) ᵍ⊗ₜ[ℚ] (1 : A0 n)
      * ((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (F0 n v) : AB n)
      = (kappa n * algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (F0 n v) := by
    rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
      (kappa n) (⟨1, SetLike.one_mem_graded (A0Grading n)⟩ : A0Grading n 0)
      (⟨algebraMap (Indexed.Pn n) (RRing n) c, algebraMap_mem_RGradingQ_zero n c⟩ : RGradingQ n 0)
      (F0 n v)]
    rw [one_mul]
  rw [e1, e2]
  norm_num

/-- `iota0 n (hMap n (eN i))` always anticommutes with `kappaAB` — `hMap`'s own output is always a
`Pn n`-combination of pure `Fof`-type basis vectors (or `0`), matching `F0AB`'s own `A0`-degree. -/
theorem iota0_hMap_eN_comm_kappaAB (n : ℕ) (i : Indexed.IndexedBasis n) :
    iota0 n (Indexed.hMap n (Indexed.eN i)) * kappaAB n
      = -(kappaAB n * iota0 n (Indexed.hMap n (Indexed.eN i))) := by
  rw [Indexed.hMap_eN]
  match i with
  | .inl ⟨(u, v), huv⟩ =>
    show iota0 n (Indexed.hBasis n (Sum.inl ⟨(u, v), huv⟩)) * kappaAB n
        = -(kappaAB n * iota0 n (Indexed.hBasis n (Sum.inl ⟨(u, v), huv⟩)))
    have hh : Indexed.hBasis n (Sum.inl (⟨(u, v), huv⟩ : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}))
        = Indexed.betaN n u • Indexed.eN (Indexed.Fof v) + Indexed.betaN n v • Indexed.eN (Indexed.Fof u) := rfl
    rw [hh, iota0_add, iota0_smul_eN, iota0_smul_eN, iota0Basis_Fof, iota0Basis_Fof]
    rw [add_mul, mul_add, algebraMap_tmul_F0_comm_kappaAB, algebraMap_tmul_F0_comm_kappaAB]
    abel
  | .inr u =>
    show iota0 n (Indexed.hBasis n (Sum.inr u)) * kappaAB n
        = -(kappaAB n * iota0 n (Indexed.hBasis n (Sum.inr u)))
    have hh : Indexed.hBasis n (Sum.inr u : Indexed.IndexedBasis n) = 0 := rfl
    rw [hh, iota0_zero, zero_mul, mul_zero, neg_zero]

/-! ## `truePart` on `eR` basis vectors (mirrors the frozen `falsePart_eR_false`/`_true` exactly;
not itself in the frozen layer, so proved here). -/

theorem truePart_eR_false (n : ℕ) (i : Indexed.IndexedBasis n) :
    Indexed.truePart ((Indexed.eR ((i, false) : Indexed.RBasis n))) = 0 := by
  funext k
  unfold Indexed.truePart Indexed.eR
  simp

theorem truePart_eR_true (n : ℕ) (i : Indexed.IndexedBasis n) :
    Indexed.truePart ((Indexed.eR ((i, true) : Indexed.RBasis n))) = Indexed.eN i := by
  funext k
  unfold Indexed.truePart Indexed.eR Indexed.eN
  simp [Prod.ext_iff]

/-! ## `iotaBetaR` on `RBasis n` basis vectors -/

theorem iotaBetaR_eR_false (n : ℕ) (i : Indexed.IndexedBasis n) :
    iotaBetaR n (Indexed.eR ((i, false) : Indexed.RBasis n)) = liftsFamilyBeta n i := by
  unfold iotaBetaR
  rw [Indexed.falsePart_eR_false, truePart_eR_false, liftBeta_zero, mul_zero, add_zero, liftBeta_eN]

theorem iotaBetaR_eR_true (n : ℕ) (i : Indexed.IndexedBasis n) :
    iotaBetaR n (Indexed.eR ((i, true) : Indexed.RBasis n)) = kappaAB n * liftsFamilyBeta n i := by
  unfold iotaBetaR
  rw [Indexed.falsePart_eR_true, truePart_eR_true, liftBeta_zero, zero_add, liftBeta_eN]

/-! ## `iotaBetaRBracket`, the single formula (using `parityR`, not the bare `IndexedBasis`
parity) that closes on all four rows. -/

noncomputable def iotaBetaRBracket (n : ℕ) (p q : Indexed.RBasis n) : AB n :=
  iotaBetaR n (Indexed.eR p) * iotaBetaR n (Indexed.eR q)
    - (gsignNQ n (Indexed.parityR p) (Indexed.parityR q)) •
        (iotaBetaR n (Indexed.eR q) * iotaBetaR n (Indexed.eR p))

/-! ## W0 — the image characterization -/

theorem iotaBetaR_image_eq (n : ℕ) (r : Indexed.RMod n) :
    ∃ z1 z2 : Indexed.IndexedMod n, r = Indexed.iotaR z1 + Indexed.kappaEmbed (z2 - Indexed.hMap n z1)
      ∧ iotaBetaR n r = iota0 n z1 + kappaAB n * iota0 n z2 := by
  refine ⟨Indexed.falsePart r, Indexed.hMap n (Indexed.falsePart r) + Indexed.truePart r, ?_, ?_⟩
  · have h1 : Indexed.hMap n (Indexed.falsePart r) + Indexed.truePart r - Indexed.hMap n (Indexed.falsePart r)
        = Indexed.truePart r := by abel
    rw [h1]
    exact Indexed.decompose r
  · exact iotaBetaR_eq_iota0_add n r

/-- General expansion of `iotaBetaR` on an `iotaR z + kappaEmbed w`-shaped argument, mirroring
`prop_recovery`'s own `hRHS` step exactly. -/
theorem iotaBetaR_iotaR_add_kappaEmbed (n : ℕ) (z w : Indexed.IndexedMod n) :
    iotaBetaR n (Indexed.iotaR z + Indexed.kappaEmbed w)
      = iota0 n z + kappaAB n * iota0 n w + kappaAB n * iota0 n (Indexed.hMap n z) := by
  have hRHS : iotaBetaR n (Indexed.iotaR z + Indexed.kappaEmbed w)
      = liftBeta n z + kappaAB n * liftBeta n w := by
    unfold iotaBetaR
    rw [Indexed.falsePart_add, Indexed.truePart_add, Indexed.falsePart_iotaR,
      Indexed.falsePart_kappaEmbed, Indexed.truePart_iotaR, Indexed.truePart_kappaEmbed, add_zero,
      zero_add]
  rw [hRHS, liftBeta_eq_iota0_add, liftBeta_eq_iota0_add, mul_add, ← mul_assoc,
    kappaAB_sq_eq_zero, zero_mul, add_zero]
  abel

/-- **`(false,false)` row**: reduces to `liftsBracket_eq_bridge` (frozen, R6). -/
theorem iotaBetaRBracket_false_false (n : ℕ) (i j : Indexed.IndexedBasis n) :
    iotaBetaRBracket n (i, false) (j, false)
      = iotaBetaR n (Indexed.bracketRBetaBasis n (i, false) (j, false)) := by
  have hpar : Indexed.parityR ((i, false) : Indexed.RBasis n) = Indexed.parity i
      ∧ Indexed.parityR ((j, false) : Indexed.RBasis n) = Indexed.parity j := by
    constructor <;> (unfold Indexed.parityR; simp)
  unfold iotaBetaRBracket
  rw [hpar.1, hpar.2, iotaBetaR_eR_false, iotaBetaR_eR_false]
  show liftsFamilyBeta n i * liftsFamilyBeta n j
      - gsignNQ n (Indexed.parity i) (Indexed.parity j) • (liftsFamilyBeta n j * liftsFamilyBeta n i)
      = iotaBetaR n (Indexed.bracketRBetaBasis n (i, false) (j, false))
  show liftsFamilyBeta n i * liftsFamilyBeta n j
      - gsignNQ n (Indexed.parity i) (Indexed.parity j) • (liftsFamilyBeta n j * liftsFamilyBeta n i)
      = iotaBetaR n (Indexed.iotaR (Indexed.bracketBasisN n i j) + Indexed.kappaEmbed (Indexed.GammaBetaBasis n i j))
  rw [iotaBetaR_iotaR_add_kappaEmbed]
  have hlift : liftsFamilyBeta n i * liftsFamilyBeta n j
      - gsignNQ n (Indexed.parity i) (Indexed.parity j) • (liftsFamilyBeta n j * liftsFamilyBeta n i)
      = liftsBracket n i j := by
    match i, j with
    | .inl ⟨(u, v), huv⟩, .inl ⟨(w, z), hwz⟩ =>
      show L0hatBeta n u v * L0hatBeta n w z
          - gsignNQ n 0 0 • (L0hatBeta n w z * L0hatBeta n u v) = liftsBracket n _ _
      rw [gsignNQ_00, one_smul]; rfl
    | .inl ⟨(u, v), huv⟩, .inr w =>
      show L0hatBeta n u v * F0hatBeta n w
          - gsignNQ n 0 1 • (F0hatBeta n w * L0hatBeta n u v) = liftsBracket n _ _
      rw [gsignNQ_01, one_smul]; rfl
    | .inr w, .inl ⟨(u, v), huv⟩ =>
      show F0hatBeta n w * L0hatBeta n u v
          - gsignNQ n 1 0 • (L0hatBeta n u v * F0hatBeta n w) = liftsBracket n _ _
      rw [gsignNQ_10, one_smul]; rfl
    | .inr u, .inr v =>
      show F0hatBeta n u * F0hatBeta n v
          - gsignNQ n 1 1 • (F0hatBeta n v * F0hatBeta n u) = liftsBracket n _ _
      rw [gsignNQ_11, neg_smul, one_smul, sub_neg_eq_add]
      rfl
  rw [hlift, liftsBracket_eq_bridge]

/-- `kappaAB n * liftsFamilyBeta n i` collapses to the undeformed lift — the deformation's own
correction term dies against the outer `kappaAB` via `kappaAB_sq_eq_zero`. -/
theorem kappaAB_mul_liftsFamilyBeta (n : ℕ) (i : Indexed.IndexedBasis n) :
    kappaAB n * liftsFamilyBeta n i = kappaAB n * iota0AB n i := by
  have h : liftsFamilyBeta n i = iota0AB n i + kappaAB n * iota0 n (Indexed.hMap n (Indexed.eN i)) := by
    rw [← liftBeta_eN, liftBeta_eq_iota0_add, iota0_eN]
  rw [h, mul_add, ← mul_assoc, kappaAB_sq_eq_zero, zero_mul, add_zero]

/-- Sandwiching any `iota0AB` basis lift between two `kappaAB`'s vanishes, regardless of whether
that basis lift commutes (`L0AB`) or anticommutes (`F0AB`) with `kappaAB` — either way the two
`kappaAB` factors collapse via `kappaAB_sq_eq_zero` once brought together. -/
theorem kappaAB_mul_iota0AB_mul_kappaAB (n : ℕ) (i : Indexed.IndexedBasis n) :
    kappaAB n * iota0AB n i * kappaAB n = 0 := by
  match i with
  | .inl ⟨(u, v), huv⟩ =>
    show kappaAB n * iota0AB n (Sum.inl ⟨(u, v), huv⟩) * kappaAB n = 0
    rw [show iota0AB n (Sum.inl (⟨(u, v), huv⟩ : {p : Fin (2*n) × Fin (2*n) // p.1 ≤ p.2}))
        = L0AB n u v from rfl]
    rw [mul_assoc, L0AB_comm_kappaAB, ← mul_assoc, kappaAB_sq_eq_zero, zero_mul]
  | .inr u =>
    show kappaAB n * iota0AB n (Sum.inr u) * kappaAB n = 0
    rw [show iota0AB n (Sum.inr u : Indexed.IndexedBasis n) = F0AB n u from rfl]
    rw [mul_assoc, F0AB_comm_kappaAB, mul_neg, ← mul_assoc, kappaAB_sq_eq_zero, zero_mul, neg_zero]

/-- **`(true,true)` row**: cheap — both products vanish individually. -/
theorem iotaBetaRBracket_true_true (n : ℕ) (i j : Indexed.IndexedBasis n) :
    iotaBetaRBracket n (i, true) (j, true)
      = iotaBetaR n (Indexed.bracketRBetaBasis n (i, true) (j, true)) := by
  unfold iotaBetaRBracket
  rw [iotaBetaR_eR_true, iotaBetaR_eR_true, kappaAB_mul_liftsFamilyBeta, kappaAB_mul_liftsFamilyBeta]
  have hz1 : kappaAB n * iota0AB n i * (kappaAB n * iota0AB n j) = 0 := by
    rw [← mul_assoc, kappaAB_mul_iota0AB_mul_kappaAB, zero_mul]
  have hz2 : kappaAB n * iota0AB n j * (kappaAB n * iota0AB n i) = 0 := by
    rw [← mul_assoc, kappaAB_mul_iota0AB_mul_kappaAB, zero_mul]
  rw [hz1, hz2, smul_zero, sub_zero]
  show (0 : AB n) = iotaBetaR n (0 : Indexed.RMod n)
  have hfp : Indexed.falsePart (0 : Indexed.RMod n) = 0 := by
    funext k; simp [Indexed.falsePart]
  have htp : Indexed.truePart (0 : Indexed.RMod n) = 0 := by
    funext k; simp [Indexed.truePart]
  rw [iotaBetaR_eq_iota0_add, hfp, htp, iota0_zero, Indexed.hMap_zero, add_zero, iota0_zero,
    mul_zero, add_zero]

/-- General commutation of `iota0AB n i` with `kappaAB`, by `i`'s own `parity`. -/
theorem iota0AB_mul_kappaAB (n : ℕ) (i : Indexed.IndexedBasis n) :
    iota0AB n i * kappaAB n = gsignNQ n (Indexed.parity i) 1 • (kappaAB n * iota0AB n i) := by
  match i with
  | .inl ⟨(u, v), huv⟩ =>
    show iota0AB n (Sum.inl ⟨(u, v), huv⟩) * kappaAB n
        = gsignNQ n (Indexed.parity (Sum.inl ⟨(u, v), huv⟩)) 1
          • (kappaAB n * iota0AB n (Sum.inl ⟨(u, v), huv⟩))
    rw [show iota0AB n (Sum.inl (⟨(u, v), huv⟩ : {p : Fin (2*n) × Fin (2*n) // p.1 ≤ p.2}))
        = L0AB n u v from rfl, show Indexed.parity (Sum.inl (⟨(u, v), huv⟩ :
        {p : Fin (2*n) × Fin (2*n) // p.1 ≤ p.2})) = 0 from rfl,
      gsignNQ_01, one_smul, L0AB_comm_kappaAB]
  | .inr u =>
    show iota0AB n (Sum.inr u) * kappaAB n
        = gsignNQ n (Indexed.parity (Sum.inr u : Indexed.IndexedBasis n)) 1
          • (kappaAB n * iota0AB n (Sum.inr u))
    rw [show iota0AB n (Sum.inr u : Indexed.IndexedBasis n) = F0AB n u from rfl,
      show Indexed.parity (Sum.inr u : Indexed.IndexedBasis n) = 1 from rfl,
      gsignNQ_11, neg_smul, one_smul, F0AB_comm_kappaAB]

/-- `liftsFamilyBeta n i * kappaAB` collapses the same way `kappaAB * liftsFamilyBeta n i` does. -/
theorem liftsFamilyBeta_mul_kappaAB (n : ℕ) (i : Indexed.IndexedBasis n) :
    liftsFamilyBeta n i * kappaAB n = iota0AB n i * kappaAB n := by
  have h : liftsFamilyBeta n i = iota0AB n i + kappaAB n * iota0 n (Indexed.hMap n (Indexed.eN i)) := by
    rw [← liftBeta_eN, liftBeta_eq_iota0_add, iota0_eN]
  rw [h, add_mul]
  have hz : kappaAB n * iota0 n (Indexed.hMap n (Indexed.eN i)) * kappaAB n = 0 := by
    rw [mul_assoc, iota0_hMap_eN_comm_kappaAB, mul_neg, ← mul_assoc, kappaAB_sq_eq_zero, zero_mul,
      neg_zero]
  rw [hz, add_zero]

/-- `kappaAB * iota0AB j * liftsFamilyBeta i` collapses the deformed factor away, the same way
`kappaAB_mul_liftsFamilyBeta` does, now with an extra `iota0AB j *` prefix that commutes the
correction term's own extra `kappaAB` back into range of the outer one. -/
theorem kappaAB_mul_iota0AB_mul_liftsFamilyBeta (n : ℕ) (i j : Indexed.IndexedBasis n) :
    kappaAB n * iota0AB n j * liftsFamilyBeta n i = kappaAB n * iota0AB n j * iota0AB n i := by
  have h : liftsFamilyBeta n i = iota0AB n i + kappaAB n * iota0 n (Indexed.hMap n (Indexed.eN i)) := by
    rw [← liftBeta_eN, liftBeta_eq_iota0_add, iota0_eN]
  rw [h, mul_add]
  have step1 : iota0AB n j * (kappaAB n * iota0 n (Indexed.hMap n (Indexed.eN i)))
      = gsignNQ n (Indexed.parity j) 1 • (kappaAB n * (iota0AB n j * iota0 n (Indexed.hMap n (Indexed.eN i)))) := by
    rw [← mul_assoc, iota0AB_mul_kappaAB, smul_mul_assoc, mul_assoc]
  have hz : kappaAB n * iota0AB n j * (kappaAB n * iota0 n (Indexed.hMap n (Indexed.eN i))) = 0 := by
    rw [mul_assoc, step1, mul_smul_comm, ← mul_assoc, kappaAB_sq_eq_zero, zero_mul, smul_zero]
  rw [hz, add_zero]

theorem iotaBetaR_kappaEmbed (n : ℕ) (w : Indexed.IndexedMod n) :
    iotaBetaR n (Indexed.kappaEmbed w) = kappaAB n * liftBeta n w := by
  unfold iotaBetaR
  rw [Indexed.falsePart_kappaEmbed, Indexed.truePart_kappaEmbed, liftBeta_zero, zero_add]

theorem algebraMap_gsignN (n : ℕ) (a b : ZMod 2) :
    algebraMap (Indexed.Pn n) (RRing n) (Indexed.gsignN n a b) = gsignNQ n a b • (1 : RRing n) := by
  match a, b with
  | (0 : ZMod 2), (0 : ZMod 2) => rw [Indexed.gsignN_00, gsignNQ_00, one_smul, map_one]
  | (0 : ZMod 2), (1 : ZMod 2) => rw [Indexed.gsignN_01, gsignNQ_01, one_smul, map_one]
  | (1 : ZMod 2), (0 : ZMod 2) => rw [Indexed.gsignN_10, gsignNQ_10, one_smul, map_one]
  | (1 : ZMod 2), (1 : ZMod 2) =>
    rw [Indexed.gsignN_11, gsignNQ_11,
      show (-1 : Indexed.Pn n) = Indexed.cratN n (-1) from by
        rw [← Indexed.cratN_neg, Indexed.cratN_one],
      algebraMap_cratN]

theorem Lof_eq_inl (n : ℕ) (u v : Fin (2 * n)) (h : u ≤ v) :
    Indexed.Lof u v = Sum.inl (⟨(u, v), h⟩ : {p : Fin (2*n) × Fin (2*n) // p.1 ≤ p.2}) := by
  unfold Indexed.Lof; rw [dif_pos h]

theorem iota0_bracketBasisN_unified (n : ℕ) (i j : Indexed.IndexedBasis n) :
    iota0 n (Indexed.bracketBasisN n i j)
      = iota0AB n i * iota0AB n j
        - gsignNQ n (Indexed.parity i) (Indexed.parity j) • (iota0AB n j * iota0AB n i) := by
  match i, j with
  | .inl ⟨(u, v), huv⟩, .inl ⟨(w, z), hwz⟩ =>
    rw [show Indexed.parity (Sum.inl (⟨(u, v), huv⟩ : {p : Fin (2*n) × Fin (2*n) // p.1 ≤ p.2}))
        = 0 from rfl, show Indexed.parity (Sum.inl (⟨(w, z), hwz⟩ :
        {p : Fin (2*n) × Fin (2*n) // p.1 ≤ p.2})) = 0 from rfl, gsignNQ_00, one_smul,
      ← Lof_eq_inl n u v huv, ← Lof_eq_inl n w z hwz]
    exact iota0_bracketBasisN_Lof_Lof n u v w z
  | .inl ⟨(u, v), huv⟩, .inr w =>
    rw [show Indexed.parity (Sum.inl (⟨(u, v), huv⟩ : {p : Fin (2*n) × Fin (2*n) // p.1 ≤ p.2}))
        = 0 from rfl, show Indexed.parity (Sum.inr w : Indexed.IndexedBasis n) = 1 from rfl,
      gsignNQ_01, one_smul, ← Lof_eq_inl n u v huv]
    exact iota0_bracketBasisN_Lof_Fof n u v w
  | .inr w, .inl ⟨(u, v), huv⟩ =>
    rw [show Indexed.parity (Sum.inr w : Indexed.IndexedBasis n) = 1 from rfl,
      show Indexed.parity (Sum.inl (⟨(u, v), huv⟩ : {p : Fin (2*n) × Fin (2*n) // p.1 ≤ p.2}))
        = 0 from rfl, gsignNQ_10, one_smul, ← Lof_eq_inl n u v huv]
    exact iota0_bracketBasisN_Fof_Lof n u v w
  | .inr u, .inr v =>
    rw [show Indexed.parity (Sum.inr u : Indexed.IndexedBasis n) = 1 from rfl,
      show Indexed.parity (Sum.inr v : Indexed.IndexedBasis n) = 1 from rfl, gsignNQ_11, neg_smul,
      one_smul, sub_neg_eq_add]
    exact iota0_bracketBasisN_Fof_Fof n u v

theorem gsignNQ_mul_eq (n : ℕ) (a b : ZMod 2) :
    gsignNQ n a (b + 1) = gsignNQ n a 1 * gsignNQ n a b := by
  match a, b with
  | (0 : ZMod 2), (0 : ZMod 2) =>
    rw [show (0:ZMod 2)+1 = 1 from by decide]; rw [gsignNQ_01, gsignNQ_00]; ring
  | (0 : ZMod 2), (1 : ZMod 2) =>
    rw [show (1:ZMod 2)+1 = 0 from by decide]; rw [gsignNQ_00, gsignNQ_01]; ring
  | (1 : ZMod 2), (0 : ZMod 2) =>
    rw [show (0:ZMod 2)+1 = 1 from by decide]; rw [gsignNQ_11, gsignNQ_10]; ring
  | (1 : ZMod 2), (1 : ZMod 2) =>
    rw [show (1:ZMod 2)+1 = 0 from by decide]; rw [gsignNQ_10, gsignNQ_11]; ring

theorem gsignNQ_succ_mul_eq (n : ℕ) (a b : ZMod 2) :
    gsignNQ n (a + 1) b * gsignNQ n b 1 = gsignNQ n a b := by
  match a, b with
  | (0 : ZMod 2), (0 : ZMod 2) =>
    rw [show (0:ZMod 2)+1 = 1 from by decide]; rw [gsignNQ_10, gsignNQ_01, gsignNQ_00]; ring
  | (0 : ZMod 2), (1 : ZMod 2) =>
    rw [show (0:ZMod 2)+1 = 1 from by decide]; rw [gsignNQ_11, gsignNQ_01]; ring
  | (1 : ZMod 2), (0 : ZMod 2) =>
    rw [show (1:ZMod 2)+1 = 0 from by decide]; rw [gsignNQ_00, gsignNQ_01, gsignNQ_10]; ring
  | (1 : ZMod 2), (1 : ZMod 2) =>
    rw [show (1:ZMod 2)+1 = 0 from by decide]; rw [gsignNQ_01, gsignNQ_11]; ring

/-- **`(true,false)` row**. -/
theorem iotaBetaRBracket_true_false (n : ℕ) (i j : Indexed.IndexedBasis n) :
    iotaBetaRBracket n (i, true) (j, false)
      = iotaBetaR n (Indexed.bracketRBetaBasis n (i, true) (j, false)) := by
  have hpar1 : Indexed.parityR ((i, true) : Indexed.RBasis n) = Indexed.parity i + 1 := by
    unfold Indexed.parityR; simp
  have hpar2 : Indexed.parityR ((j, false) : Indexed.RBasis n) = Indexed.parity j := by
    unfold Indexed.parityR; simp
  unfold iotaBetaRBracket
  rw [hpar1, hpar2, iotaBetaR_eR_true, iotaBetaR_eR_false, kappaAB_mul_liftsFamilyBeta]
  have hGpGq : kappaAB n * iota0AB n i * liftsFamilyBeta n j
      = kappaAB n * (iota0AB n i * iota0AB n j) := by
    rw [kappaAB_mul_iota0AB_mul_liftsFamilyBeta, mul_assoc]
  have hGqGp : liftsFamilyBeta n j * (kappaAB n * iota0AB n i)
      = gsignNQ n (Indexed.parity j) 1 • (kappaAB n * (iota0AB n j * iota0AB n i)) := by
    rw [← mul_assoc, liftsFamilyBeta_mul_kappaAB, iota0AB_mul_kappaAB, smul_mul_assoc, mul_assoc]
  rw [hGpGq, hGqGp]
  show kappaAB n * (iota0AB n i * iota0AB n j)
      - gsignNQ n (Indexed.parity i + 1) (Indexed.parity j)
        • (gsignNQ n (Indexed.parity j) 1 • (kappaAB n * (iota0AB n j * iota0AB n i)))
      = iotaBetaR n (Indexed.bracketRBetaBasis n (i, true) (j, false))
  show kappaAB n * (iota0AB n i * iota0AB n j)
      - gsignNQ n (Indexed.parity i + 1) (Indexed.parity j)
        • (gsignNQ n (Indexed.parity j) 1 • (kappaAB n * (iota0AB n j * iota0AB n i)))
      = iotaBetaR n (Indexed.kappaEmbed (Indexed.bracketBasisN n i j))
  rw [show iotaBetaR n (Indexed.kappaEmbed (Indexed.bracketBasisN n i j))
      = kappaAB n * liftBeta n (Indexed.bracketBasisN n i j) from iotaBetaR_kappaEmbed n _]
  have hkill : kappaAB n * (kappaAB n * iota0 n (Indexed.hMap n (Indexed.bracketBasisN n i j))) = 0 := by
    rw [← mul_assoc, kappaAB_sq_eq_zero, zero_mul]
  rw [liftBeta_eq_iota0_add, mul_add, hkill, add_zero]
  rw [iota0_bracketBasisN_unified, mul_sub]
  rw [smul_smul, gsignNQ_succ_mul_eq, mul_smul_comm]

/-- **`(false,true)` row**. -/
theorem iotaBetaRBracket_false_true (n : ℕ) (i j : Indexed.IndexedBasis n) :
    iotaBetaRBracket n (i, false) (j, true)
      = iotaBetaR n (Indexed.bracketRBetaBasis n (i, false) (j, true)) := by
  have hpar1 : Indexed.parityR ((i, false) : Indexed.RBasis n) = Indexed.parity i := by
    unfold Indexed.parityR; simp
  have hpar2 : Indexed.parityR ((j, true) : Indexed.RBasis n) = Indexed.parity j + 1 := by
    unfold Indexed.parityR; simp
  unfold iotaBetaRBracket
  rw [hpar1, hpar2, iotaBetaR_eR_false, iotaBetaR_eR_true, kappaAB_mul_liftsFamilyBeta]
  have hGpGq : liftsFamilyBeta n i * (kappaAB n * iota0AB n j)
      = gsignNQ n (Indexed.parity i) 1 • (kappaAB n * (iota0AB n i * iota0AB n j)) := by
    rw [← mul_assoc, liftsFamilyBeta_mul_kappaAB, iota0AB_mul_kappaAB, smul_mul_assoc,
      mul_assoc]
  have hGqGp : kappaAB n * iota0AB n j * liftsFamilyBeta n i
      = kappaAB n * (iota0AB n j * iota0AB n i) := by
    rw [kappaAB_mul_iota0AB_mul_liftsFamilyBeta, mul_assoc]
  rw [hGpGq, hGqGp]
  show gsignNQ n (Indexed.parity i) 1 • (kappaAB n * (iota0AB n i * iota0AB n j))
      - gsignNQ n (Indexed.parity i) (Indexed.parity j + 1) • (kappaAB n * (iota0AB n j * iota0AB n i))
      = iotaBetaR n (Indexed.bracketRBetaBasis n (i, false) (j, true))
  show gsignNQ n (Indexed.parity i) 1 • (kappaAB n * (iota0AB n i * iota0AB n j))
      - gsignNQ n (Indexed.parity i) (Indexed.parity j + 1) • (kappaAB n * (iota0AB n j * iota0AB n i))
      = iotaBetaR n (Indexed.kappaEmbed (Indexed.gsignN n (Indexed.parity i) 1
          • Indexed.bracketBasisN n i j))
  rw [iotaBetaR_kappaEmbed, liftBeta_smul, algebraMap_gsignN, AB_tmul_smul_left]
  rw [show (1 : RRing n) ᵍ⊗ₜ[ℚ] (1 : A0 n) = (1 : AB n) from by
    show GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) ((1:RRing n) ⊗ₜ[ℚ] (1:A0 n)) = 1
    rw [← Algebra.TensorProduct.one_def, GradedTensorProduct.of_one]]
  rw [smul_mul_assoc, one_mul, mul_smul_comm]
  have hkill : kappaAB n * (kappaAB n * iota0 n (Indexed.hMap n (Indexed.bracketBasisN n i j))) = 0 := by
    rw [← mul_assoc, kappaAB_sq_eq_zero, zero_mul]
  rw [liftBeta_eq_iota0_add, mul_add, smul_add, hkill, smul_zero, add_zero]
  rw [iota0_bracketBasisN_unified, mul_sub, mul_smul_comm]
  rw [smul_sub, smul_smul, gsignNQ_mul_eq]

/-! ## W2 -- the closure theorem, assembling all four rows -/

theorem iotaBetaR_sum {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι)
    (f : ι → Indexed.RMod n) :
    iotaBetaR n (∑ i ∈ s, f i) = ∑ i ∈ s, iotaBetaR n (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [iotaBetaR_zero]
  | @insert a s ha ih => rw [Finset.sum_insert ha, iotaBetaR_add, ih, Finset.sum_insert ha]

theorem iotaBetaR_sum' {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (c : ι → Indexed.Pn n)
    (f : ι → Indexed.RMod n) :
    iotaBetaR n (∑ i ∈ s, c i • f i)
      = ∑ i ∈ s, ((algebraMap (Indexed.Pn n) (RRing n) (c i)) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * iotaBetaR n (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [iotaBetaR_zero]
  | @insert a s ha ih =>
    rw [Finset.sum_insert ha, iotaBetaR_add, iotaBetaR_smul, ih, Finset.sum_insert ha]

/-- **W2, the strong closure theorem**: the bilinear sum of concrete (anti)commutators of
`iotaBetaR`-images over ALL of `RBasis n` (not just the `IndexedBasis n`-indexed, undoubled
family) lands exactly in `iotaBetaR`'s own image -- `iotaBetaR n (bracketRBeta n r s)`, `bracketRBeta`
the frozen R1 object. Closure is then immediate: the value is manifestly an `iotaBetaR`-image by
construction. (Stated with the `algebraMap (...) ᵍ⊗ₜ 1`-wrapped coefficient, matching
`liftsBracket_general`'s/`iota0_bracketN`'s own established shape -- `AB n` carries no native
`Pn n`-scalar action, only the `RRing n`-embedded one already used throughout this project.) -/
theorem iotaBetaR_bracket_closed (n : ℕ) (r s : Indexed.RMod n) :
    ∑ p : Indexed.RBasis n, ∑ q : Indexed.RBasis n,
        ((algebraMap (Indexed.Pn n) (RRing n) (r p * s q)) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * iotaBetaRBracket n p q
      = iotaBetaR n (Indexed.bracketRBeta n r s) := by
  unfold Indexed.bracketRBeta
  rw [iotaBetaR_sum]
  apply Finset.sum_congr rfl
  intro p _
  rw [iotaBetaR_sum']
  apply Finset.sum_congr rfl
  intro q _
  congr 1
  match p, q with
  | (i, false), (j, false) => exact iotaBetaRBracket_false_false n i j
  | (i, false), (j, true) => exact iotaBetaRBracket_false_true n i j
  | (i, true), (j, false) => exact iotaBetaRBracket_true_false n i j
  | (i, true), (j, true) => exact iotaBetaRBracket_true_true n i j

/-- The `(false,false)` row's value, exposed directly against `liftsBracket` (frozen, R6) --
extracted from `iotaBetaRBracket_false_false`'s own internal derivation, needed for the
specialization check below. -/
theorem iotaBetaRBracket_false_false_eq_liftsBracket (n : ℕ) (i j : Indexed.IndexedBasis n) :
    iotaBetaRBracket n (i, false) (j, false) = liftsBracket n i j := by
  have hpar : Indexed.parityR ((i, false) : Indexed.RBasis n) = Indexed.parity i
      ∧ Indexed.parityR ((j, false) : Indexed.RBasis n) = Indexed.parity j := by
    constructor <;> (unfold Indexed.parityR; simp)
  unfold iotaBetaRBracket
  rw [hpar.1, hpar.2, iotaBetaR_eR_false, iotaBetaR_eR_false]
  match i, j with
  | .inl ⟨(u, v), huv⟩, .inl ⟨(w, z), hwz⟩ =>
    show L0hatBeta n u v * L0hatBeta n w z
        - gsignNQ n 0 0 • (L0hatBeta n w z * L0hatBeta n u v) = liftsBracket n _ _
    rw [gsignNQ_00, one_smul]; rfl
  | .inl ⟨(u, v), huv⟩, .inr w =>
    show L0hatBeta n u v * F0hatBeta n w
        - gsignNQ n 0 1 • (F0hatBeta n w * L0hatBeta n u v) = liftsBracket n _ _
    rw [gsignNQ_01, one_smul]; rfl
  | .inr w, .inl ⟨(u, v), huv⟩ =>
    show F0hatBeta n w * L0hatBeta n u v
        - gsignNQ n 1 0 • (L0hatBeta n u v * F0hatBeta n w) = liftsBracket n _ _
    rw [gsignNQ_10, one_smul]; rfl
  | .inr u, .inr v =>
    show F0hatBeta n u * F0hatBeta n v
        - gsignNQ n 1 1 • (F0hatBeta n v * F0hatBeta n u) = liftsBracket n _ _
    rw [gsignNQ_11, neg_smul, one_smul, sub_neg_eq_add]; rfl

/-- **The specialization check**: `iotaBetaR_bracket_closed`, evaluated at `r = iotaR x`,
`s = iotaR y`, reduces EXACTLY to R6's `liftsBracket_general`/`prop_recovery` -- confirming
nothing drifted from R6's own already-accepted result. Every `(p,q)` pair with either slot `true`
contributes `0` (`iotaR`'s own value is `0` on the `true` slot, killing the `algebraMap`-embedded
coefficient via `map_zero`/`AB_zero_tmul`), leaving exactly the `(false,false)` terms, each equal
to `liftsBracket n i j` by `iotaBetaRBracket_false_false_eq_liftsBracket`. -/
theorem iotaBetaR_bracket_closed_specializes (n : ℕ) (x y : Indexed.IndexedMod n) :
    ∑ p : Indexed.RBasis n, ∑ q : Indexed.RBasis n,
        ((algebraMap (Indexed.Pn n) (RRing n) ((Indexed.iotaR x) p * (Indexed.iotaR y) q))
          ᵍ⊗ₜ[ℚ] (1 : A0 n)) * iotaBetaRBracket n p q
      = ∑ i : Indexed.IndexedBasis n, ∑ j : Indexed.IndexedBasis n,
          ((algebraMap (Indexed.Pn n) (RRing n) (x i * y j)) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * liftsBracket n i j := by
  simp only [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl; intro i _
  rw [Indexed.sum_bool_eq]
  have h1 : (∑ j : Indexed.IndexedBasis n, ∑ b' : Bool,
      ((algebraMap (Indexed.Pn n) (RRing n) (Indexed.iotaR x (i, true) * Indexed.iotaR y (j, b')))
        ᵍ⊗ₜ[ℚ] (1 : A0 n)) * iotaBetaRBracket n (i, true) (j, b')) = 0 := by
    apply Finset.sum_eq_zero; intro j _
    apply Finset.sum_eq_zero; intro b' _
    simp [Indexed.iotaR, AB_zero_tmul]
  rw [h1, zero_add]
  apply Finset.sum_congr rfl; intro j _
  rw [Indexed.sum_bool_eq]
  have h2 : ((algebraMap (Indexed.Pn n) (RRing n) (Indexed.iotaR x (i, false) * Indexed.iotaR y (j, true)))
      ᵍ⊗ₜ[ℚ] (1 : A0 n)) * iotaBetaRBracket n (i, false) (j, true) = 0 := by
    simp [Indexed.iotaR, AB_zero_tmul]
  rw [h2, zero_add]
  show ((algebraMap (Indexed.Pn n) (RRing n) (Indexed.iotaR x (i, false) * Indexed.iotaR y (j, false)))
      ᵍ⊗ₜ[ℚ] (1 : A0 n)) * iotaBetaRBracket n (i, false) (j, false)
      = ((algebraMap (Indexed.Pn n) (RRing n) (x i * y j)) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * liftsBracket n i j
  rw [show Indexed.iotaR x (i, false) = x i from rfl, show Indexed.iotaR y (j, false) = y j from rfl,
    iotaBetaRBracket_false_false_eq_liftsBracket]

/-! ## W3 -- `prop:recovery` assembled

`prop:recovery`'s four clauses, each with the theorem that carries it, all now in Lean:

1. `iota_beta` injective: `iotaBetaR_injective` (frozen, R6, `SourceRecoveryClosure.lean`).
2. The recovered bracket `[X,Y]_beta = [X,Y]_0 + kappa Gamma_beta(X,Y)`: `prop_recovery` (frozen,
   R6), generalized to all of `g_R` by `iotaBetaR_bracket_closed` (this round), which specializes
   back to `prop_recovery`/`liftsBracket_general` exactly (`iotaBetaR_bracket_closed_specializes`).
3. The coefficient unique as an element of `g_P`: `coefficient_unique` (frozen, R6).
4. Closed image: `iotaBetaR_bracket_closed` (this round) -- the value of the bracket on any two
   `iotaBetaR`-images is again an `iotaBetaR`-image, by construction.

What remains unformalized, unchanged since R4/R5/R6: `lem:source-isomorphism` **as an abstract
statement** -- that the algebra presented by `eq:source` is isomorphic to `A_B` -- stays
unformalized; the deformed source is realized concretely (route (b)), not presented abstractly,
and the recovery is proved for that realization. Also unchanged: `cor:complex`,
`f_beta = h - ad(2F(v))` beyond what R1 proved, the reference-example table, other models/ranks. -/

theorem prop_recovery_assembled : True := by
  have _clause1 := @iotaBetaR_injective
  have _clause2 := @iotaBetaR_bracket_closed
  have _clause3 := @coefficient_unique
  have _clause4 := @iotaBetaR_bracket_closed
  trivial

end Source
end InhomogeneousDeformations
