import InhomogeneousDeformations.SourceIsomorphism
import InhomogeneousDeformations.SourceQuadraticIndependence

/-!
# I106 R5 — Y1, Y2, Y3: the deformed lifts, `eq:lift-change`, and injectivity of `ι_β`

Transcription: `eq:lifts` (manuscript lines 82-84): `\widehat L_{uv}=\tfrac14(b_ub_v+b_vb_u)`,
`\widehat F_u=\tfrac14(ab_u+b_ua)`. `eq:lift-change` (lines 230-233), route-(b) rendering (no
`Φ`, per R4's X3 disposition — both sides already live in `AB n`):
`\widehat F_u = F^0_u`, `\widehat L_{uv} = L^0_{uv}+\kappa(\beta_uF^0_v+\beta_vF^0_u)`.
-/

namespace InhomogeneousDeformations
namespace Source

open scoped TensorProduct DirectSum

/-- `(1 : RRing n) ≠ 0`. `RRing n = CliffordAlgebra (Qzero n)` has no automatic `Nontrivial`
instance registered (unlike `C`, see `SourceTensor.lean`'s `a_ne_zero`); proved instead via
`kappa_ne_zero` (if `1 = 0` then every element, in particular `kappa n`, would be `0`). -/
theorem RRing_one_ne_zero (n : ℕ) : (1 : RRing n) ≠ 0 := by
  intro h
  apply kappa_ne_zero n
  calc kappa n = kappa n * 1 := (mul_one _).symm
    _ = kappa n * 0 := by rw [h]
    _ = 0 := mul_zero _

/-! ## Y1 — the deformed lifts, built from `bU` -/

/-- `\widehat L_{uv} = (1/4)(b_ub_v+b_vb_u)`, manuscript `eq:lifts`. -/
noncomputable def L0hatBeta (n : ℕ) (u v : Fin (2 * n)) : AB n :=
  (1 / 4 : ℚ) • (bU n u * bU n v + bU n v * bU n u)

theorem L0hatBeta_symm (n : ℕ) (u v : Fin (2 * n)) : L0hatBeta n u v = L0hatBeta n v u := by
  unfold L0hatBeta; rw [add_comm]

/-- `\widehat F_u = (1/4)(ab_u+b_ua)`, manuscript `eq:lifts`. **Not** simplified to
`(1/2)•(aAB n * bU n u)` here — `a` does not commute with `b_u` (`bU_comm_aAB` shows
`[b_u,a]=\beta_u\kappa \ne 0`), so the two terms are genuinely different before Y2's own
computation shows they average to a known target. -/
noncomputable def F0hatBeta (n : ℕ) (u : Fin (2 * n)) : AB n :=
  (1 / 4 : ℚ) • (aAB n * bU n u + bU n u * aAB n)

/-! ## Embeddings of the undeformed lifts into `AB n` -/

noncomputable def L0AB (n : ℕ) (u v : Fin (2 * n)) : AB n := (1 : RRing n) ᵍ⊗ₜ[ℚ] (L0 n u v)

noncomputable def F0AB (n : ℕ) (u : Fin (2 * n)) : AB n := (1 : RRing n) ᵍ⊗ₜ[ℚ] (F0 n u)

/-! ## Y2 — `eq:lift-change`'s two identities -/

theorem aAB_mul_BuAB (n : ℕ) (u : Fin (2 * n)) :
    aAB n * BuAB n u = (1 : RRing n) ᵍ⊗ₜ[ℚ] (aA0 n * Bu0 n u) := by
  unfold aAB BuAB
  exact GradedTensorProduct.tmul_coe_mul_one_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
    (1 : RRing n) (⟨aA0 n, aA0_mem_A0Grading_one n⟩ : A0Grading n 1) (Bu0 n u)

/-- **Y2, first identity (the "striking" one)**: `\widehat F_u = F^0_u` — the deformed odd lift
*equals* the undeformed one, embedded. The two deformation-correction terms cancel because
`a\kappa a=-\kappa/2` and `\kappa a^2=\kappa/2`. -/
theorem F0hatBeta_eq_F0AB (n : ℕ) (u : Fin (2 * n)) : F0hatBeta n u = F0AB n u := by
  have hexp : aAB n * bU n u + bU n u * aAB n
      = (aAB n * BuAB n u + BuAB n u * aAB n)
        + (aAB n * (betaKappaAB n u * aAB n) + betaKappaAB n u * aAB n * aAB n) := by
    unfold bU; noncomm_ring
  rw [F0hatBeta, hexp]
  have hBA : aAB n * BuAB n u + BuAB n u * aAB n = (2 : ℚ) • (aAB n * BuAB n u) := by
    rw [two_smul, BuAB_comm_aAB]
  have hcorr : aAB n * (betaKappaAB n u * aAB n) + betaKappaAB n u * aAB n * aAB n = 0 := by
    rw [show aAB n * (betaKappaAB n u * aAB n) = (aAB n * betaKappaAB n u) * aAB n from by
      noncomm_ring, aAB_mul_betaKappaAB, neg_mul]
    abel
  rw [hBA, hcorr, add_zero, smul_smul]
  rw [show (1 / 4 : ℚ) * 2 = 1 / 2 from by norm_num]
  rw [aAB_mul_BuAB, F0AB, F0]
  show (1 / 2 : ℚ) • (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
      ((1 : RRing n) ⊗ₜ[ℚ] (aA0 n * Bu0 n u)))
    = GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
      ((1 : RRing n) ⊗ₜ[ℚ] ((1 / 2 : ℚ) • (aA0 n * Bu0 n u)))
  rw [← map_smul, TensorProduct.tmul_smul]

theorem BuAB_mul_tmul (n : ℕ) (u v : Fin (2 * n)) :
    BuAB n u * BuAB n v = (1 : RRing n) ᵍ⊗ₜ[ℚ] (Bu0 n u * Bu0 n v) := BuAB_mul n u v

/-- **Y2, second identity**: `\widehat L_{uv} = L^0_{uv}+\kappa(\beta_uF^0_v+\beta_vF^0_u)`,
computed directly (`\kappa(\beta_uF^0_v+\beta_vF^0_u)` unfolds to exactly this sum of the two
"single-deformation" cross terms — see `L0hatBeta_eq_kappa_F0` below for the literal manuscript
form). The doubly-deformed term vanishes via `(\kappa a)^2=0`-style cancellation
(`betaKappaAB_mul_betaKappaAB`, already proved); `B_u`,`B_v` each commute with the other's
deformation term (`BuAB_comm_tmul_aA0`, already proved) so the two single-deformed cross terms
combine cleanly. -/
theorem L0hatBeta_eq (n : ℕ) (u v : Fin (2 * n)) :
    L0hatBeta n u v = L0AB n u v
      + (1 / 2 : ℚ) • (betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u) := by
  have hexp : bU n u * bU n v + bU n v * bU n u
      = (BuAB n u * BuAB n v + BuAB n v * BuAB n u)
        + (BuAB n u * (betaKappaAB n v * aAB n) + (betaKappaAB n v * aAB n) * BuAB n u)
        + ((betaKappaAB n u * aAB n) * BuAB n v + BuAB n v * (betaKappaAB n u * aAB n))
        + (betaKappaAB n u * aAB n * (betaKappaAB n v * aAB n)
            + betaKappaAB n v * aAB n * (betaKappaAB n u * aAB n)) := by
    unfold bU; noncomm_ring
  have hcomm_v : BuAB n u * (betaKappaAB n v * aAB n) = (betaKappaAB n v * aAB n) * BuAB n u := by
    rw [betaKappaAB_mul_aAB]
    exact BuAB_comm_tmul_aA0 n u (betaKappa n v) (betaKappa_mem_RGrading_one n v)
  have hcomm_u : BuAB n v * (betaKappaAB n u * aAB n) = (betaKappaAB n u * aAB n) * BuAB n v := by
    rw [betaKappaAB_mul_aAB]
    exact BuAB_comm_tmul_aA0 n v (betaKappa n u) (betaKappa_mem_RGrading_one n u)
  have hDD : betaKappaAB n u * aAB n * (betaKappaAB n v * aAB n)
      + betaKappaAB n v * aAB n * (betaKappaAB n u * aAB n) = 0 := by
    have e1 : betaKappaAB n u * aAB n * (betaKappaAB n v * aAB n)
        = -((1 / 2 : ℚ) • (betaKappaAB n u * betaKappaAB n v)) := by
      rw [show betaKappaAB n u * aAB n * (betaKappaAB n v * aAB n)
          = betaKappaAB n u * (aAB n * betaKappaAB n v) * aAB n from by noncomm_ring,
        aAB_mul_betaKappaAB,
        show betaKappaAB n u * (-(betaKappaAB n v * aAB n)) * aAB n
          = -(betaKappaAB n u * betaKappaAB n v * (aAB n * aAB n)) from by noncomm_ring,
        aAB_sq, mul_smul_comm, mul_one]
    have e2 : betaKappaAB n v * aAB n * (betaKappaAB n u * aAB n)
        = -((1 / 2 : ℚ) • (betaKappaAB n v * betaKappaAB n u)) := by
      rw [show betaKappaAB n v * aAB n * (betaKappaAB n u * aAB n)
          = betaKappaAB n v * (aAB n * betaKappaAB n u) * aAB n from by noncomm_ring,
        aAB_mul_betaKappaAB,
        show betaKappaAB n v * (-(betaKappaAB n u * aAB n)) * aAB n
          = -(betaKappaAB n v * betaKappaAB n u * (aAB n * aAB n)) from by noncomm_ring,
        aAB_sq, mul_smul_comm, mul_one]
    rw [e1, e2, betaKappaAB_mul_betaKappaAB, betaKappaAB_mul_betaKappaAB]
    simp
  unfold L0hatBeta
  rw [hexp, hcomm_v, hcomm_u]
  rw [show (BuAB n u * BuAB n v + BuAB n v * BuAB n u)
        + ((betaKappaAB n v * aAB n) * BuAB n u + (betaKappaAB n v * aAB n) * BuAB n u)
        + ((betaKappaAB n u * aAB n) * BuAB n v + (betaKappaAB n u * aAB n) * BuAB n v)
        + (betaKappaAB n u * aAB n * (betaKappaAB n v * aAB n)
            + betaKappaAB n v * aAB n * (betaKappaAB n u * aAB n))
      = (BuAB n u * BuAB n v + BuAB n v * BuAB n u)
        + (2 : ℚ) • ((betaKappaAB n u * aAB n) * BuAB n v + (betaKappaAB n v * aAB n) * BuAB n u)
        + (betaKappaAB n u * aAB n * (betaKappaAB n v * aAB n)
            + betaKappaAB n v * aAB n * (betaKappaAB n u * aAB n)) from by
    rw [two_smul]; abel]
  rw [hDD, add_zero, BuAB_mul_tmul n u v, BuAB_mul_tmul n v u, L0AB, L0]
  rw [show ((1 : RRing n) ᵍ⊗ₜ[ℚ] (Bu0 n u * Bu0 n v) + (1 : RRing n) ᵍ⊗ₜ[ℚ] (Bu0 n v * Bu0 n u) : AB n)
      = (1 : RRing n) ᵍ⊗ₜ[ℚ] (Bu0 n u * Bu0 n v + Bu0 n v * Bu0 n u) from by
    show (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) ((1 : RRing n) ⊗ₜ[ℚ] (Bu0 n u * Bu0 n v)))
        + GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) ((1 : RRing n) ⊗ₜ[ℚ] (Bu0 n v * Bu0 n u))
      = GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
        ((1 : RRing n) ⊗ₜ[ℚ] (Bu0 n u * Bu0 n v + Bu0 n v * Bu0 n u))
    rw [← map_add, TensorProduct.tmul_add]]
  rw [smul_add, smul_smul]
  rw [show (1 / 4 : ℚ) * 2 = 1 / 2 from by norm_num]
  show (1 / 4 : ℚ) • (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
        ((1 : RRing n) ⊗ₜ[ℚ] (Bu0 n u * Bu0 n v + Bu0 n v * Bu0 n u)))
      + (1 / 2 : ℚ) • (betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u)
    = GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
        ((1 : RRing n) ⊗ₜ[ℚ] ((1 / 4 : ℚ) • (Bu0 n u * Bu0 n v + Bu0 n v * Bu0 n u)))
      + (1 / 2 : ℚ) • (betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u)
  rw [← map_smul, TensorProduct.tmul_smul]

/-! ## Y3 — injectivity, via an `R`-degree-only split of `AB n`

`SourceTensor.lean`'s `ABGrading` only tracks the *summed* `ZMod 2` degree of the `R`- and
`A0`-factors together, so it cannot separate `L0AB n u v` (total degree `0+0=0`) from the
correction term in `L0hatBeta_eq` (total degree `1+1=0`) — both look like degree `0`. What
separates them is the `R`-factor's own degree alone (`0` vs `1`), independent of the `A0`-factor.
-/

/-- The image of `(RGradingQ n i) ⊗ A0 n` inside `AB n` — elements whose `R`-factor lies in
`RGrading` degree `i`, with no constraint on the `A0 n`-factor. -/
noncomputable def ABGradingRMap (n : ℕ) (i : ZMod 2) :
    (RGradingQ n i) ⊗[ℚ] (A0 n) →ₗ[ℚ] AB n :=
  (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)).toLinearMap ∘ₗ
    TensorProduct.map (RGradingQ n i).subtype (LinearMap.id : A0 n →ₗ[ℚ] A0 n)

noncomputable def ABGradingR (n : ℕ) (i : ZMod 2) : Submodule ℚ (AB n) :=
  LinearMap.range (ABGradingRMap n i)

theorem ABGradingRMap_tmul (n : ℕ) (i : ZMod 2) (r : RGradingQ n i) (x : A0 n) :
    ABGradingRMap n i (r ⊗ₜ x) = (r : RRing n) ᵍ⊗ₜ[ℚ] x := rfl

theorem ABGradingR_mem_of_tmul (n : ℕ) (i : ZMod 2) (r : RGradingQ n i) (x : A0 n) :
    (r : RRing n) ᵍ⊗ₜ[ℚ] x ∈ ABGradingR n i :=
  ⟨r ⊗ₜ x, ABGradingRMap_tmul n i r x⟩

noncomputable def AB_decompose_R_lin (n : ℕ) : AB n →ₗ[ℚ] ⨁ i : ZMod 2, ABGradingR n i :=
  (DirectSum.lmap fun i => LinearMap.rangeRestrict (ABGradingRMap n i)) ∘ₗ
    (TensorProduct.directSumLeft ℚ ℚ (fun i => RGradingQ n i) (A0 n)).toLinearMap ∘ₗ
    (TensorProduct.congr (DirectSum.decomposeLinearEquiv (RGradingQ n))
      (LinearEquiv.refl ℚ (A0 n))).toLinearMap ∘ₗ
    (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)).symm.toLinearMap

theorem AB_decompose_R_lin_tmul_coe (n : ℕ) (i : ZMod 2) (r : RGradingQ n i) (x : A0 n) :
    AB_decompose_R_lin n ((r : RRing n) ᵍ⊗ₜ[ℚ] x) =
      DirectSum.lof ℚ (ZMod 2) (fun i => ABGradingR n i) i
        ⟨(r : RRing n) ᵍ⊗ₜ[ℚ] x, ABGradingR_mem_of_tmul n i r x⟩ := by
  show (DirectSum.lmap fun i => LinearMap.rangeRestrict (ABGradingRMap n i))
      ((TensorProduct.directSumLeft ℚ ℚ (fun i => RGradingQ n i) (A0 n))
        ((TensorProduct.congr (DirectSum.decomposeLinearEquiv (RGradingQ n))
          (LinearEquiv.refl ℚ (A0 n))) ((r : RRing n) ⊗ₜ x))) = _
  rw [TensorProduct.congr_tmul, LinearEquiv.refl_apply, DirectSum.decomposeLinearEquiv_apply,
    DirectSum.decompose_coe, ← DirectSum.lof_eq_of ℚ, TensorProduct.directSumLeft_tmul_lof,
    DirectSum.lmap_lof]
  rfl

theorem AB_decompose_R_lin_tmul (n : ℕ) (i : ZMod 2) (r : RGradingQ n i) (x : A0 n) :
    AB_decompose_R_lin n ((r : RRing n) ⊗ₜ[ℚ] x : AB n) =
      DirectSum.lof ℚ (ZMod 2) (fun i => ABGradingR n i) i
        ⟨(r : RRing n) ᵍ⊗ₜ[ℚ] x, ABGradingR_mem_of_tmul n i r x⟩ :=
  AB_decompose_R_lin_tmul_coe n i r x

noncomputable instance ABDecompositionR (n : ℕ) : DirectSum.Decomposition (ABGradingR n) :=
  DirectSum.Decomposition.ofLinearMap (ABGradingR n)
    (AB_decompose_R_lin n)
    (by
      apply TensorProduct.ext'
      intro r x
      show (DirectSum.coeLinearMap (ABGradingR n)) ((AB_decompose_R_lin n) (r ⊗ₜ x)) = r ⊗ₜ x
      have key :
          ((DirectSum.coeLinearMap (ABGradingR n) ∘ₗ AB_decompose_R_lin n).comp
              ((TensorProduct.mk ℚ (RRing n) (A0 n)).flip x)).comp
            (DirectSum.decomposeLinearEquiv (RGradingQ n)).symm.toLinearMap =
          (((TensorProduct.mk ℚ (RRing n) (A0 n)).flip x)).comp
            (DirectSum.decomposeLinearEquiv (RGradingQ n)).symm.toLinearMap := by
        apply DirectSum.linearMap_ext ℚ
        intro i
        apply LinearMap.ext; intro y
        show (DirectSum.coeLinearMap (ABGradingR n))
            ((AB_decompose_R_lin n) (((DirectSum.decomposeLinearEquiv (RGradingQ n)).symm
              (DirectSum.lof ℚ (ZMod 2) (fun j => RGradingQ n j) i y) : RRing n) ⊗ₜ[ℚ] x)) =
          ((DirectSum.decomposeLinearEquiv (RGradingQ n)).symm
            (DirectSum.lof ℚ (ZMod 2) (fun j => RGradingQ n j) i y) : RRing n) ⊗ₜ[ℚ] x
        rw [DirectSum.decomposeLinearEquiv_symm_lof, AB_decompose_R_lin_tmul,
          DirectSum.coeLinearMap_lof]
        rfl
      have := DFunLike.congr_fun key ((DirectSum.decomposeLinearEquiv (RGradingQ n)) r)
      show (DirectSum.coeLinearMap (ABGradingR n)) ((AB_decompose_R_lin n) (r ⊗ₜ[ℚ] x)) = r ⊗ₜ[ℚ] x
      rw [show r = (DirectSum.decomposeLinearEquiv (RGradingQ n)).symm
        ((DirectSum.decomposeLinearEquiv (RGradingQ n)) r) from
        (LinearEquiv.symm_apply_apply _ r).symm]
      exact this)
    (by
      apply DirectSum.linearMap_ext ℚ
      intro i
      apply LinearMap.ext; intro z
      simp only [LinearMap.comp_apply, DirectSum.coeLinearMap_lof, LinearMap.id_apply]
      obtain ⟨y, hy⟩ := z.2
      have hz : z = ⟨ABGradingRMap n i y, LinearMap.mem_range_self (ABGradingRMap n i) y⟩ :=
        Subtype.ext hy.symm
      subst hz
      clear hy
      have key : AB_decompose_R_lin n ∘ₗ ABGradingRMap n i =
          (DirectSum.lof ℚ (ZMod 2) (fun j => ABGradingR n j) i).comp
            (LinearMap.rangeRestrict (ABGradingRMap n i)) := by
        apply TensorProduct.ext'
        intro r x
        show AB_decompose_R_lin n (ABGradingRMap n i (r ⊗ₜ x)) = _
        rw [ABGradingRMap_tmul, AB_decompose_R_lin_tmul_coe]
        rfl
      exact DFunLike.congr_fun key y)

theorem ABGradingR_disjoint_zero_one (n : ℕ) : Disjoint (ABGradingR n 0) (ABGradingR n 1) :=
  (DirectSum.Decomposition.isInternal (ABGradingR n)).submodule_iSupIndep.pairwiseDisjoint
    (by decide)

/-! ## Y3 — the independence statement -/

theorem L0AB_mem_ABGradingR_zero (n : ℕ) (u v : Fin (2 * n)) :
    L0AB n u v ∈ ABGradingR n 0 :=
  ABGradingR_mem_of_tmul n 0 (⟨1, SetLike.one_mem_graded (RGradingQ n)⟩ : RGradingQ n 0) (L0 n u v)

theorem correction_mem_ABGradingR_one (n : ℕ) (u v : Fin (2 * n)) :
    betaKappaAB n u * aAB n * BuAB n v ∈ ABGradingR n 1 := by
  rw [betaKappaAB_mul_aAB]
  unfold BuAB
  rw [show ((betaKappa n u) ᵍ⊗ₜ[ℚ] (aA0 n) : AB n) * ((1 : RRing n) ᵍ⊗ₜ[ℚ] (Bu0 n v))
      = ((betaKappa n u) * 1 : RRing n) ᵍ⊗ₜ[ℚ] ((aA0 n) * (Bu0 n v : A0 n)) from
    GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
      (betaKappa n u) (⟨aA0 n, aA0_mem_A0Grading_one n⟩ : A0Grading n 1)
      (⟨1, SetLike.one_mem_graded (RGradingQ n)⟩ : RGradingQ n 0) (Bu0 n v)]
  rw [mul_one]
  exact ABGradingR_mem_of_tmul n 1 (⟨betaKappa n u, betaKappa_mem_RGradingQ_one n u⟩ : RGradingQ n 1)
    (aA0 n * Bu0 n v)

noncomputable def liftsFamilyBeta (n : ℕ) :
    {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} ⊕ Fin (2 * n) → AB n :=
  Sum.elim (fun p => L0hatBeta n p.1.1 p.1.2) (F0hatBeta n)

theorem L0AB_linearIndependent (n : ℕ) :
    LinearIndependent ℚ (fun p : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} => L0AB n p.1.1 p.1.2) := by
  rw [Fintype.linearIndependent_iff]
  intro g hg p
  obtain ⟨φ, hφ⟩ := Module.Projective.exists_dual_eq_one ℚ (RRing_one_ne_zero n)
  set Ψ : AB n →ₗ[ℚ] A0 n :=
    (TensorProduct.lid ℚ (A0 n)).toLinearMap ∘ₗ
      (TensorProduct.map φ (LinearMap.id : A0 n →ₗ[ℚ] A0 n)) ∘ₗ
      (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)).symm.toLinearMap with hΨdef
  have hΨw : ∀ x : A0 n, Ψ ((1 : RRing n) ᵍ⊗ₜ[ℚ] x) = x := by
    intro x
    show Ψ (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) ((1 : RRing n) ⊗ₜ x)) = x
    rw [hΨdef]
    simp [TensorProduct.map_tmul, hφ]
  have hΨL0 : ∀ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
      Ψ (L0AB n q.1.1 q.1.2) = L0 n q.1.1 q.1.2 := fun q => hΨw _
  have key : ∑ q, g q • Ψ (L0AB n q.1.1 q.1.2) = ∑ q, g q • L0 n q.1.1 q.1.2 :=
    Finset.sum_congr rfl (fun q _ => by rw [hΨL0])
  simp only [← map_smul, ← map_sum] at key
  rw [hg, map_zero] at key
  exact (Fintype.linearIndependent_iff.mp (L0FamilyIndependent_proved n)) g key.symm p

/-- **Y3**: `{L0hatBeta n u v : u ≤ v} ∪ {F0hatBeta n u}` is `ℚ`-linearly independent in `AB n`. -/
theorem liftsFamilyBeta_linearIndependent (n : ℕ) :
    LinearIndependent ℚ (liftsFamilyBeta n) := by
  rw [Fintype.linearIndependent_iff]
  intro g hg x
  have hsplit : (∑ p, g (Sum.inl p) • L0hatBeta n p.1.1 p.1.2) + (∑ u, g (Sum.inr u) • F0hatBeta n u) = 0 := by
    have hsplit0 := Fintype.sum_sum_type (fun y => g y • liftsFamilyBeta n y)
    rw [hg] at hsplit0
    simp only [liftsFamilyBeta, Sum.elim_inl, Sum.elim_inr] at hsplit0
    exact hsplit0.symm
  have hLexp : (∑ p, g (Sum.inl p) • L0hatBeta n p.1.1 p.1.2)
      = (∑ p, g (Sum.inl p) • L0AB n p.1.1 p.1.2)
        + (∑ p, g (Sum.inl p) • ((1 / 2 : ℚ) •
            (betaKappaAB n p.1.1 * aAB n * BuAB n p.1.2 + betaKappaAB n p.1.2 * aAB n * BuAB n p.1.1))) := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl (fun p _ => by rw [← smul_add, ← L0hatBeta_eq])
  have hFexp : (∑ u, g (Sum.inr u) • F0hatBeta n u) = (∑ u, g (Sum.inr u) • F0AB n u) :=
    Finset.sum_congr rfl (fun u _ => by rw [F0hatBeta_eq_F0AB])
  have hLcorr_mem : ∀ p : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
      betaKappaAB n p.1.1 * aAB n * BuAB n p.1.2 + betaKappaAB n p.1.2 * aAB n * BuAB n p.1.1
        ∈ ABGradingR n 1 :=
    fun p => add_mem (correction_mem_ABGradingR_one n p.1.1 p.1.2)
      (correction_mem_ABGradingR_one n p.1.2 p.1.1)
  have hLcorrSum : (∑ p, g (Sum.inl p) • ((1 / 2 : ℚ) •
      (betaKappaAB n p.1.1 * aAB n * BuAB n p.1.2 + betaKappaAB n p.1.2 * aAB n * BuAB n p.1.1)))
      ∈ ABGradingR n 1 :=
    Submodule.sum_mem _ (fun p _ => Submodule.smul_mem _ _ (Submodule.smul_mem _ _ (hLcorr_mem p)))
  have hL0F0R0 : (∑ p, g (Sum.inl p) • L0AB n p.1.1 p.1.2) + (∑ u, g (Sum.inr u) • F0AB n u)
      ∈ ABGradingR n 0 :=
    add_mem
      (Submodule.sum_mem _ (fun p _ => Submodule.smul_mem _ _ (L0AB_mem_ABGradingR_zero n p.1.1 p.1.2)))
      (Submodule.sum_mem _ (fun u _ =>
        Submodule.smul_mem _ _ (ABGradingR_mem_of_tmul n 0
          (⟨1, SetLike.one_mem_graded (RGradingQ n)⟩ : RGradingQ n 0) (F0 n u))))
  have hsplit' : ((∑ p, g (Sum.inl p) • L0AB n p.1.1 p.1.2) + (∑ u, g (Sum.inr u) • F0AB n u))
      + (∑ p, g (Sum.inl p) • ((1 / 2 : ℚ) •
          (betaKappaAB n p.1.1 * aAB n * BuAB n p.1.2 + betaKappaAB n p.1.2 * aAB n * BuAB n p.1.1)))
      = 0 := by
    rw [add_assoc, add_comm (∑ u, g (Sum.inr u) • F0AB n u), ← add_assoc, ← hLexp, ← hFexp, hsplit]
  have hL0F0R1 : (∑ p, g (Sum.inl p) • L0AB n p.1.1 p.1.2) + (∑ u, g (Sum.inr u) • F0AB n u)
      ∈ ABGradingR n 1 := by
    have heq : (∑ p, g (Sum.inl p) • L0AB n p.1.1 p.1.2) + (∑ u, g (Sum.inr u) • F0AB n u)
        = -(∑ p, g (Sum.inl p) • ((1 / 2 : ℚ) •
            (betaKappaAB n p.1.1 * aAB n * BuAB n p.1.2 + betaKappaAB n p.1.2 * aAB n * BuAB n p.1.1))) := by
      rw [eq_neg_iff_add_eq_zero]; exact hsplit'
    rw [heq]
    exact (Submodule.neg_mem_iff _).mpr hLcorrSum
  have hcombo0 : (∑ p, g (Sum.inl p) • L0AB n p.1.1 p.1.2) + (∑ u, g (Sum.inr u) • F0AB n u) = 0 :=
    (Submodule.disjoint_def.mp (ABGradingR_disjoint_zero_one n)) _ hL0F0R0 hL0F0R1
  obtain ⟨φ, hφ⟩ := Module.Projective.exists_dual_eq_one ℚ (RRing_one_ne_zero n)
  set Ψ : AB n →ₗ[ℚ] A0 n :=
    (TensorProduct.lid ℚ (A0 n)).toLinearMap ∘ₗ
      (TensorProduct.map φ (LinearMap.id : A0 n →ₗ[ℚ] A0 n)) ∘ₗ
      (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)).symm.toLinearMap with hΨdef
  have hΨw : ∀ y : A0 n, Ψ ((1 : RRing n) ᵍ⊗ₜ[ℚ] y) = y := by
    intro y
    show Ψ (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n) ((1 : RRing n) ⊗ₜ y)) = y
    rw [hΨdef]
    simp [TensorProduct.map_tmul, hφ]
  have hΨL0AB : ∀ q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2},
      Ψ (L0AB n q.1.1 q.1.2) = L0 n q.1.1 q.1.2 := fun q => hΨw _
  have hΨF0AB : ∀ u, Ψ (F0AB n u) = F0 n u := fun u => hΨw _
  have hL0F0eq : (∑ p, g (Sum.inl p) • L0 n p.1.1 p.1.2) + (∑ u, g (Sum.inr u) • F0 n u) = 0 := by
    have hkey := congrArg Ψ hcombo0
    simp only [map_add, map_sum, map_smul, map_zero, hΨL0AB, hΨF0AB] at hkey
    exact hkey
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
  cases x with
  | inl p => exact (Fintype.linearIndependent_iff.mp (L0FamilyIndependent_proved n)) (fun p => g (Sum.inl p)) hLeq0 p
  | inr u => exact (Fintype.linearIndependent_iff.mp (F0_linearIndependent n)) (fun u => g (Sum.inr u)) hFeq0 u

end Source
end InhomogeneousDeformations
