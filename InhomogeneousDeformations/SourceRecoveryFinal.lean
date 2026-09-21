import InhomogeneousDeformations.SourceRecoveryBridge
import InhomogeneousDeformations.IndexedU

/-!
# I106 R5 — Y4 Stage 2 and Y5: the recovery theorem and closed image

Continues Stage 1 (`SourceRecoveryBridge.lean`). `IndexedU.lean` (frozen, R1) supplies `hMap`,
`UMap`, `UMapInv`, `bracketR`, and the already-proved abstract gate theorem
`intertwiningU_coordinate_corollary`. This file works out the *concrete* analogue.

## Why the target is not simply `iota0 (bracketN X Y) + kappaAB * iota0 (GammaBetaN X Y)`

Unwinding `intertwiningU_coordinate_corollary` at the level of `RMod n`: writing
`Z := bracketR n (UMap n (iotaR X)) (UMap n (iotaR Y))`, one has `UMapInv n Z = Z - kappaEmbed
(hMap n (falsePart Z))` (`UMapInv_eq`), and `falsePart Z = bracketN n X Y` (the only summand of
`Z`'s defining expansion landing in the `false` slot is `bracketR n (iotaR X) (iotaR Y) = iotaR
(bracketN n X Y)`; the two cross terms and the double-`kappa` term all land in the `true` slot or
vanish). So the *raw* concrete commutator of deformed lifts -- which realizes `Z` itself, before
`UMapInv`'s correction -- equals
`iota0 (bracketN X Y) + kappaAB * iota0 (GammaBetaN X Y) + kappaAB * iota0 (hMap n (bracketN X Y))`,
**not** just the first two terms. The extra `hMap`-of-the-output term is genuine, already-accepted
structure (the very map used to build `UMap`/`UMapInv` in `IndexedU.lean`), not a new invention.
It vanishes on the `LF`/`FL` sectors (`hMap` kills `F`-basis vectors, `hBasis_Fof`) and exactly
*cancels* `GammaBetaBasis`'s own `FF`-sector value (checked below), matching the independently
verified fact that `F0hatBeta = F0AB` has *zero* deformation correction at the lift level
(`F0hatBeta_eq_F0AB`) even though `Gamma_beta(F_u,F_v) \ne 0`. On `LL` it does not vanish; the
theorem below states it explicitly rather than forcing a false "zero correction" reading of
`Gamma_beta(L_uv,L_wz) = 0`.
-/

namespace InhomogeneousDeformations
namespace Source

open scoped TensorProduct

/-! ## Generic combinators for `AB n`, ported from `SourceRecoveryBase.lean`'s `Module.End`-level
versions (`comm_mul_left`/`comm_mul_right`/`symm_comm_symm`) -- purely ring-theoretic, so the
same proofs go through verbatim for the associative ring `AB n`. -/

theorem comm_mul_right_AB (n : ℕ) (x y z : AB n) :
    x*(y*z) - (y*z)*x = (x*y-y*x)*z + y*(x*z-z*x) := by
  rw [sub_mul, mul_sub, ← mul_assoc, ← mul_assoc, ← mul_assoc]
  abel

theorem comm_mul_left_AB (n : ℕ) (x y z : AB n) :
    (x*y)*z - z*(x*y) = x*(y*z-z*y) + (x*z-z*x)*y := by
  rw [mul_sub, sub_mul, mul_assoc, mul_assoc, mul_assoc]
  abel

theorem symm_comm_symm_AB (n : ℕ) (a b c d : AB n) (p q r s : ℚ)
    (hac : a*c - c*a = p • (1 : AB n))
    (had : a*d - d*a = q • (1 : AB n))
    (hbc : b*c - c*b = r • (1 : AB n))
    (hbd : b*d - d*b = s • (1 : AB n)) :
    (a*b+b*a)*(c*d+d*c) - (c*d+d*c)*(a*b+b*a)
      = (2:ℚ) • (r • (a*d+d*a) + s • (a*c+c*a) + p • (b*d+d*b) + q • (b*c+c*b)) := by
  have hab_cd : a*b*(c*d) - (c*d)*(a*b) = a*(b*(c*d)-(c*d)*b) + (a*(c*d)-(c*d)*a)*b :=
    comm_mul_left_AB n a b (c*d)
  have hbcd : b*(c*d) - (c*d)*b = (b*c-c*b)*d + c*(b*d-d*b) := comm_mul_right_AB n b c d
  have hacd : a*(c*d) - (c*d)*a = (a*c-c*a)*d + c*(a*d-d*a) := comm_mul_right_AB n a c d
  have hba_cd : b*a*(c*d) - (c*d)*(b*a) = b*(a*(c*d)-(c*d)*a) + (b*(c*d)-(c*d)*b)*a :=
    comm_mul_left_AB n b a (c*d)
  have hab_dc : a*b*(d*c) - (d*c)*(a*b) = a*(b*(d*c)-(d*c)*b) + (a*(d*c)-(d*c)*a)*b :=
    comm_mul_left_AB n a b (d*c)
  have hbdc : b*(d*c) - (d*c)*b = (b*d-d*b)*c + d*(b*c-c*b) := comm_mul_right_AB n b d c
  have hadc : a*(d*c) - (d*c)*a = (a*d-d*a)*c + d*(a*c-c*a) := comm_mul_right_AB n a d c
  have hba_dc : b*a*(d*c) - (d*c)*(b*a) = b*(a*(d*c)-(d*c)*a) + (b*(d*c)-(d*c)*b)*a :=
    comm_mul_left_AB n b a (d*c)
  have key : (a*b+b*a)*(c*d+d*c) - (c*d+d*c)*(a*b+b*a)
      = (a*b*(c*d) - (c*d)*(a*b)) + (a*b*(d*c) - (d*c)*(a*b))
        + (b*a*(c*d) - (c*d)*(b*a)) + (b*a*(d*c) - (d*c)*(b*a)) := by noncomm_ring
  rw [key, hab_cd, hab_dc, hba_cd, hba_dc, hacd, hbcd, hadc, hbdc, hac, had, hbc, hbd]
  simp only [mul_add, add_mul, smul_mul_assoc, mul_smul_comm, one_mul, mul_one]
  module

/-! ## `L0hatBeta`'s own bracket relation (the `LL`, "own-algebra" identity)

`bU` satisfies exactly the same commutator-with-`J` relation as `BuAB` (`bU_comm_bU`), and
`L0hatBeta` has exactly `L0hat`'s shape with `bU` in place of `B` -- so `symm_comm_symm_AB`
applies directly, with **no explicit correction term**: this is the fact that the deformed
generators obey their own copy of the same algebra. -/
theorem L0hatBeta_comm (n : ℕ) (u v w z : Fin (2 * n)) :
    L0hatBeta n u v * L0hatBeta n w z - L0hatBeta n w z * L0hatBeta n u v
      = (1/2 : ℚ) • (JnQ n v w • L0hatBeta n u z + JnQ n u w • L0hatBeta n v z
          + JnQ n v z • L0hatBeta n u w + JnQ n u z • L0hatBeta n v w) := by
  unfold L0hatBeta
  rw [show (1/4:ℚ) • (bU n u * bU n v + bU n v * bU n u) * ((1/4:ℚ) • (bU n w * bU n z + bU n z * bU n w))
        - (1/4:ℚ) • (bU n w * bU n z + bU n z * bU n w) * ((1/4:ℚ) • (bU n u * bU n v + bU n v * bU n u))
      = (1/16 : ℚ) • ((bU n u * bU n v + bU n v * bU n u) * (bU n w * bU n z + bU n z * bU n w)
          - (bU n w * bU n z + bU n z * bU n w) * (bU n u * bU n v + bU n v * bU n u)) from by
    rw [smul_mul_smul_comm, smul_mul_smul_comm]; module]
  rw [symm_comm_symm_AB n (bU n u) (bU n v) (bU n w) (bU n z)
    (JnQ n u w) (JnQ n u z) (JnQ n v w) (JnQ n v z)
    (bU_comm_bU n u w) (bU_comm_bU n u z) (bU_comm_bU n v w) (bU_comm_bU n v z)]
  module

/-! ## `F0AB` as `(1/2) aAB BuAB`, and the `LF`/`FL`/`FF` sectors -/

theorem F0AB_eq_aAB_mul_BuAB (n : ℕ) (u : Fin (2 * n)) :
    F0AB n u = (1/2 : ℚ) • (aAB n * BuAB n u) := by
  unfold F0AB Source.F0
  rw [AB_tmul_smul, aAB_mul_BuAB]

/-! ## Connecting `iota0AB n (Lof/Fof _)` to `L0AB`/`F0AB` -/

theorem iota0AB_Fof_eq_F0AB (n : ℕ) (u : Fin (2 * n)) :
    iota0AB n (Indexed.Fof u) = F0AB n u := by
  unfold iota0AB F0AB
  rw [iota0Basis_Fof]

theorem iota0AB_Lof_eq_L0AB (n : ℕ) (u v : Fin (2 * n)) :
    iota0AB n (Indexed.Lof u v) = L0AB n u v := by
  unfold iota0AB L0AB
  rw [iota0Basis_Lof]

/-! ## FF sector -- transported, zero net correction (confirms the header comment's cancellation
claim by direct computation, not just via the abstract `hMap`/`GammaBetaBasis` bookkeeping). -/

theorem F0hatBeta_bracket_FF (n : ℕ) (u v : Fin (2 * n)) :
    F0hatBeta n u * F0hatBeta n v + F0hatBeta n v * F0hatBeta n u
      = iota0 n (Indexed.bracketBasisN n (Indexed.Fof u) (Indexed.Fof v)) := by
  rw [F0hatBeta_eq_F0AB, F0hatBeta_eq_F0AB, iota0_bracketBasisN_Fof_Fof,
    iota0AB_Fof_eq_F0AB, iota0AB_Fof_eq_F0AB]

/-! ## LF/FL sector -- transported through `iota0`/`kappaAB`

The correction term `Corr(u,v) := (1/2)•(betaKappaAB u*aAB*BuAB v + betaKappaAB v*aAB*BuAB u)`
(the deformation piece of `L0hatBeta_eq`) commutes against `F0AB w = (1/2)•(aAB*BuAB w)` up to a
`betaKappaAB`-weighted *anticommutator* of `BuAB`'s, which collapses to `L0AB` via
`Bu0_mul_L0hat_shape`; the result matches `kappaAB n * iota0 n (gammaLFn n u v w)` exactly, term
for term, once `iota0` is pushed through `gammaLFn`'s own `cratN (1/2) • (betaN _ • eN (Lof _ _) +
betaN _ • eN (Lof _ _))` shape and the two `algebraMap`-degree-`0`/`kappa` tensor factors are
recombined via `GradedTensorProduct.tmul_coe_mul_zero_coe_tmul`. -/

theorem BuAB_sandwich_aAB (n : ℕ) (y : Fin (2 * n)) :
    aAB n * BuAB n y * aAB n = (1 / 2 : ℚ) • BuAB n y := by
  rw [mul_assoc, BuAB_comm_aAB, ← mul_assoc, aAB_sq, smul_mul_assoc, one_mul]

theorem BuAB_mul_BuAB_add_swap (n : ℕ) (y w : Fin (2 * n)) :
    BuAB n y * BuAB n w + BuAB n w * BuAB n y = (4 : ℚ) • L0AB n y w := by
  unfold L0AB
  rw [BuAB_mul, BuAB_mul, ← AB_tmul_add, Bu0_mul_L0hat_shape, AB_tmul_smul]

theorem D_mul_aBuAB (n : ℕ) (x y w : Fin (2 * n)) :
    (betaKappaAB n x * aAB n * BuAB n y) * (aAB n * BuAB n w)
      = (1 / 2 : ℚ) • (betaKappaAB n x * BuAB n y * BuAB n w) := by
  rw [show betaKappaAB n x * aAB n * BuAB n y * (aAB n * BuAB n w)
      = betaKappaAB n x * (aAB n * BuAB n y * aAB n) * BuAB n w from by noncomm_ring,
    BuAB_sandwich_aAB, mul_smul_comm, smul_mul_assoc]

theorem aBuAB_mul_D (n : ℕ) (x y w : Fin (2 * n)) :
    (aAB n * BuAB n w) * (betaKappaAB n x * aAB n * BuAB n y)
      = -((1 / 2 : ℚ) • (betaKappaAB n x * BuAB n w * BuAB n y)) := by
  rw [show aAB n * BuAB n w * (betaKappaAB n x * aAB n * BuAB n y)
      = aAB n * (BuAB n w * (betaKappaAB n x * aAB n)) * BuAB n y from by noncomm_ring,
    BuAB_comm_betaKappaAB_mul_aAB,
    show aAB n * (betaKappaAB n x * aAB n * BuAB n w) * BuAB n y
      = (aAB n * betaKappaAB n x) * aAB n * BuAB n w * BuAB n y from by noncomm_ring,
    aAB_mul_betaKappaAB,
    show (-(betaKappaAB n x * aAB n)) * aAB n * BuAB n w * BuAB n y
      = -(betaKappaAB n x * (aAB n * aAB n) * BuAB n w * BuAB n y) from by noncomm_ring,
    aAB_sq]
  rw [show betaKappaAB n x * ((1 / 2 : ℚ) • (1 : AB n)) * BuAB n w * BuAB n y
      = (1 / 2 : ℚ) • (betaKappaAB n x * BuAB n w * BuAB n y) from by
    rw [mul_smul_comm, mul_one, smul_mul_assoc, smul_mul_assoc]]

theorem LF_correction (n : ℕ) (u v w : Fin (2 * n)) :
    ((1 / 2 : ℚ) • (betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u))
        * F0AB n w
      - F0AB n w
        * ((1 / 2 : ℚ) • (betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u))
      = (1 / 2 : ℚ) • (betaKappaAB n u * L0AB n v w + betaKappaAB n v * L0AB n u w) := by
  have e1 : betaKappaAB n u * BuAB n v * BuAB n w + betaKappaAB n u * BuAB n w * BuAB n v
      = betaKappaAB n u * ((4 : ℚ) • L0AB n v w) := by
    rw [mul_assoc, mul_assoc, ← mul_add, BuAB_mul_BuAB_add_swap]
  have e2 : betaKappaAB n v * BuAB n u * BuAB n w + betaKappaAB n v * BuAB n w * BuAB n u
      = betaKappaAB n v * ((4 : ℚ) • L0AB n u w) := by
    rw [mul_assoc, mul_assoc, ← mul_add, BuAB_mul_BuAB_add_swap]
  rw [F0AB_eq_aAB_mul_BuAB]
  rw [show ((1 / 2 : ℚ) • (betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u))
        * ((1 / 2 : ℚ) • (aAB n * BuAB n w))
      - ((1 / 2 : ℚ) • (aAB n * BuAB n w))
        * ((1 / 2 : ℚ) • (betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u))
      = (1 / 4 : ℚ) • ((betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u)
            * (aAB n * BuAB n w)
          - (aAB n * BuAB n w)
            * (betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u)) from by
    rw [smul_mul_smul_comm, smul_mul_smul_comm]; module]
  rw [add_mul, mul_add, D_mul_aBuAB, D_mul_aBuAB, aBuAB_mul_D, aBuAB_mul_D]
  rw [show (1 / 2 : ℚ) • (betaKappaAB n u * BuAB n v * BuAB n w)
        + (1 / 2 : ℚ) • (betaKappaAB n v * BuAB n u * BuAB n w)
        - (-((1 / 2 : ℚ) • (betaKappaAB n u * BuAB n w * BuAB n v))
            + -((1 / 2 : ℚ) • (betaKappaAB n v * BuAB n w * BuAB n u)))
      = (1 / 2 : ℚ) • (betaKappaAB n u * BuAB n v * BuAB n w + betaKappaAB n u * BuAB n w * BuAB n v)
        + (1 / 2 : ℚ) • (betaKappaAB n v * BuAB n u * BuAB n w + betaKappaAB n v * BuAB n w * BuAB n u)
      from by module]
  rw [e1, e2]
  rw [show (1 / 4 : ℚ) • ((1 / 2 : ℚ) • (betaKappaAB n u * ((4 : ℚ) • L0AB n v w))
        + (1 / 2 : ℚ) • (betaKappaAB n v * ((4 : ℚ) • L0AB n u w)))
      = (1 / 2 : ℚ) • (betaKappaAB n u * L0AB n v w + betaKappaAB n v * L0AB n u w) from by
    rw [mul_smul_comm, mul_smul_comm]; module]

theorem kappaAB_mul_algebraMap_tmul (n : ℕ) (c : Indexed.Pn n) (a0 : A0 n) :
    kappaAB n * ((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] a0)
      = (kappa n * algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] a0 := by
  unfold kappaAB
  rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
    (kappa n) (⟨1, SetLike.one_mem_graded (A0Grading n)⟩ : A0Grading n 0)
    (⟨algebraMap (Indexed.Pn n) (RRing n) c, algebraMap_mem_RGradingQ_zero n c⟩ : RGradingQ n 0) a0]
  simp

theorem kappaAB_mul_algebraMap_tmul_L0 (n : ℕ) (x y z : Fin (2 * n)) :
    kappaAB n * ((algebraMap (Indexed.Pn n) (RRing n) (Indexed.betaN n x)) ᵍ⊗ₜ[ℚ] (L0 n y z))
      = betaKappaAB n x * L0AB n y z := by
  rw [kappaAB_mul_algebraMap_tmul, ← Algebra.commutes]
  unfold betaKappaAB betaKappa L0AB
  rw [GradedTensorProduct.tmul_coe_mul_one_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
    (algebraMap (Indexed.Pn n) (RRing n) (Indexed.betaN n x) * kappa n)
    (⟨1, SetLike.one_mem_graded (A0Grading n)⟩ : A0Grading n 0) (L0 n y z)]
  simp

theorem iota0_gammaLFn (n : ℕ) (u v w : Fin (2 * n)) :
    kappaAB n * iota0 n (Indexed.gammaLFn n u v w)
      = (1 / 2 : ℚ) • (betaKappaAB n u * L0AB n v w + betaKappaAB n v * L0AB n u w) := by
  unfold Indexed.gammaLFn
  rw [iota0_smul, iota0_add, iota0_smul_eN, iota0_smul_eN, iota0Basis_Lof, iota0Basis_Lof]
  rw [show ((algebraMap (Indexed.Pn n) (RRing n) (Indexed.cratN n (1 / 2))) ᵍ⊗ₜ[ℚ] (1 : A0 n) : AB n)
      = (1 / 2 : ℚ) • (1 : AB n) from by
    rw [algebraMap_cratN, AB_tmul_smul_left]
    show (1 / 2 : ℚ) • (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
        ((1 : RRing n) ⊗ₜ[ℚ] (1 : A0 n)) : AB n) = (1 / 2 : ℚ) • (1 : AB n)
    rw [← Algebra.TensorProduct.one_def, GradedTensorProduct.of_one]]
  rw [smul_mul_assoc, one_mul, mul_smul_comm, mul_add, kappaAB_mul_algebraMap_tmul_L0,
    kappaAB_mul_algebraMap_tmul_L0]

theorem L0hatBeta_bracket_LF (n : ℕ) (u v w : Fin (2 * n)) :
    L0hatBeta n u v * F0hatBeta n w - F0hatBeta n w * L0hatBeta n u v
      = iota0 n (Indexed.bracketBasisN n (Indexed.Lof u v) (Indexed.Fof w))
        + kappaAB n * iota0 n (Indexed.gammaLFn n u v w) := by
  rw [iota0_gammaLFn, iota0_bracketBasisN_Lof_Fof, iota0AB_Lof_eq_L0AB, iota0AB_Fof_eq_F0AB,
    L0hatBeta_eq, F0hatBeta_eq_F0AB]
  rw [show (L0AB n u v
        + (1 / 2 : ℚ) • (betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u))
      * F0AB n w
      - F0AB n w
        * (L0AB n u v
          + (1 / 2 : ℚ) • (betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u))
      = (L0AB n u v * F0AB n w - F0AB n w * L0AB n u v)
        + (((1 / 2 : ℚ) • (betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u))
            * F0AB n w
          - F0AB n w
            * ((1 / 2 : ℚ) •
                (betaKappaAB n u * aAB n * BuAB n v + betaKappaAB n v * aAB n * BuAB n u)))
      from by noncomm_ring]
  rw [LF_correction]

/-- **FL case**, by negation of `L0hatBeta_bracket_LF` (not a fresh computation). -/
theorem F0hatBeta_bracket_FL (n : ℕ) (u v w : Fin (2 * n)) :
    F0hatBeta n w * L0hatBeta n u v - L0hatBeta n u v * F0hatBeta n w
      = iota0 n (Indexed.bracketBasisN n (Indexed.Fof w) (Indexed.Lof u v))
        + kappaAB n * iota0 n (Indexed.GammaBetaBasis n (Indexed.Fof w) (Indexed.Lof u v)) := by
  have hFL : Indexed.bracketBasisN n (Indexed.Fof w) (Indexed.Lof u v)
      = -(Indexed.bracketBasisN n (Indexed.Lof u v) (Indexed.Fof w)) := by
    rw [Indexed.bracketBasisN_Fof_Lof, Indexed.bracketBasisN_Lof_Fof]
  have hGamma : Indexed.GammaBetaBasis n (Indexed.Fof w) (Indexed.Lof u v)
      = -(Indexed.gammaLFn n u v w) := Indexed.GammaBetaBasis_Fof_Lof n w u v
  rw [hFL, hGamma, iota0_neg, iota0_neg, mul_neg, ← neg_add, ← L0hatBeta_bracket_LF]
  abel

/-! ## The abstract `hMap`-of-output correction, all four sectors

Pure `Pn n` combinatorics (`Indexed`/`IndexedU` layer only, no `GradedTensorProduct`), computing
`hMap n (bracketBasisN n X Y)` on each of the four basis sectors and comparing it against
`GammaBetaBasis n X Y`. This is the abstract half of the correction described in the header
comment; the transported `LL` sector (fully general `hMap`-of-output term through `iota0`) is
completed below, after these are proved. -/

end Source

namespace Indexed

theorem hMap_bracketFFn (n : ℕ) (u v : Fin (2 * n)) :
    hMap n (bracketFFn n u v) = -(GammaBetaBasis n (Fof u) (Fof v)) := by
  unfold bracketFFn GammaBetaBasis gammaFFn
  rw [hMap_smul, hMap_Lof]
  show cratN n (1/2) • (betaN n u • eN (Fof v) + betaN n v • eN (Fof u))
      = -(-(cratN n (1 / 2) • (betaN n u • eN (Fof v) + betaN n v • eN (Fof u))))
  rw [neg_neg]

theorem hMap_bracketLFn (n : ℕ) (u v w : Fin (2 * n)) :
    hMap n (bracketLFn n u v w) = 0 := by
  unfold bracketLFn
  rw [hMap_add, hMap_smul, hMap_smul, hMap_Fof, hMap_Fof, smul_zero, smul_zero, add_zero]

theorem hMap_bracketLLn (n : ℕ) (u v w z : Fin (2 * n)) :
    hMap n (bracketLLn n u v w z)
      = (cratN n (1/2) * Jn n v w) • (betaN n u • eN (Fof z) + betaN n z • eN (Fof u))
        + (cratN n (1/2) * Jn n u w) • (betaN n v • eN (Fof z) + betaN n z • eN (Fof v))
        + (cratN n (1/2) * Jn n v z) • (betaN n u • eN (Fof w) + betaN n w • eN (Fof u))
        + (cratN n (1/2) * Jn n u z) • (betaN n v • eN (Fof w) + betaN n w • eN (Fof v)) := by
  unfold bracketLLn
  rw [hMap_add, hMap_add, hMap_add, hMap_smul, hMap_smul, hMap_smul, hMap_smul,
    hMap_Lof, hMap_Lof, hMap_Lof, hMap_Lof]

end Indexed

namespace Source

open scoped TensorProduct

/-! ## LL sector -- transported through `iota0`/`kappaAB`

Route: substitute `L0hatBeta_eq` into `L0hatBeta_comm`'s RHS (a plain *sum* of shifted
`L0hatBeta`s, not a fresh commutator) rather than expanding `[L0hatBeta uv, L0hatBeta wz]`
directly -- this avoids ever computing a `Corr`-vs-`Corr` commutator. Each `Corr` term simplifies
first via `betaKappaAB_mul_aAB_mul_BuAB` (`betaKappaAB x*aAB*BuAB y = 2•(betaKappaAB x*F0AB y)`,
from `F0AB_eq_aAB_mul_BuAB` alone) to `Corr(x,y) = betaKappaAB x*F0AB y + betaKappaAB y*F0AB x`,
which then transports through `iota0`/`kappaAB` by the same `algebraMap`/`GradedTensorProduct`
mechanics as the `LF` sector's `iota0_gammaLFn`, applied to `hMap_bracketLLn`'s own four terms
(each already shaped `(cratN(1/2)*Jn _ _)•(betaN _•eN(Fof _)+betaN _•eN(Fof _))`, matching
`gammaLFn`'s shape exactly, term for term). -/

theorem betaKappaAB_mul_aAB_mul_BuAB (n : ℕ) (x y : Fin (2 * n)) :
    betaKappaAB n x * aAB n * BuAB n y = (2 : ℚ) • (betaKappaAB n x * F0AB n y) := by
  rw [F0AB_eq_aAB_mul_BuAB, mul_smul_comm, smul_smul, mul_assoc]
  norm_num

theorem Corr_eq (n : ℕ) (x y : Fin (2 * n)) :
    (1 / 2 : ℚ) • (betaKappaAB n x * aAB n * BuAB n y + betaKappaAB n y * aAB n * BuAB n x)
      = betaKappaAB n x * F0AB n y + betaKappaAB n y * F0AB n x := by
  rw [betaKappaAB_mul_aAB_mul_BuAB, betaKappaAB_mul_aAB_mul_BuAB, smul_add, smul_smul, smul_smul]
  norm_num

theorem kappaAB_mul_algebraMap_tmul_F0 (n : ℕ) (x y : Fin (2 * n)) :
    kappaAB n * ((algebraMap (Indexed.Pn n) (RRing n) (Indexed.betaN n x)) ᵍ⊗ₜ[ℚ] (F0 n y))
      = betaKappaAB n x * F0AB n y := by
  rw [kappaAB_mul_algebraMap_tmul, ← Algebra.commutes]
  unfold betaKappaAB betaKappa F0AB
  rw [GradedTensorProduct.tmul_coe_mul_one_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
    (algebraMap (Indexed.Pn n) (RRing n) (Indexed.betaN n x) * kappa n)
    (⟨1, SetLike.one_mem_graded (A0Grading n)⟩ : A0Grading n 0) (F0 n y)]
  simp

theorem iota0_hMap_term (n : ℕ) (a b c d : Fin (2 * n)) :
    kappaAB n * iota0 n ((Indexed.cratN n (1 / 2) * Indexed.Jn n a b) •
        (Indexed.betaN n c • Indexed.eN (Indexed.Fof d)
          + Indexed.betaN n d • Indexed.eN (Indexed.Fof c)))
      = ((1 / 2 : ℚ) * JnQ n a b) • (betaKappaAB n c * F0AB n d + betaKappaAB n d * F0AB n c) := by
  rw [iota0_smul, iota0_add, iota0_smul_eN, iota0_smul_eN, iota0Basis_Fof, iota0Basis_Fof]
  rw [show ((algebraMap (Indexed.Pn n) (RRing n) (Indexed.cratN n (1 / 2) * Indexed.Jn n a b))
        ᵍ⊗ₜ[ℚ] (1 : A0 n) : AB n)
      = ((1 / 2 : ℚ) * JnQ n a b) • (1 : AB n) from by
    rw [algebraMap_cratN_half_mul_Jn, AB_tmul_smul_left]
    show ((1 / 2 : ℚ) * JnQ n a b) • (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)
        ((1 : RRing n) ⊗ₜ[ℚ] (1 : A0 n)) : AB n) = ((1 / 2 : ℚ) * JnQ n a b) • (1 : AB n)
    rw [← Algebra.TensorProduct.one_def, GradedTensorProduct.of_one]]
  rw [smul_mul_assoc, one_mul, mul_smul_comm, mul_add, kappaAB_mul_algebraMap_tmul_F0,
    kappaAB_mul_algebraMap_tmul_F0]

theorem iota0_hMap_bracketLLn (n : ℕ) (u v w z : Fin (2 * n)) :
    kappaAB n * iota0 n (Indexed.hMap n (Indexed.bracketLLn n u v w z))
      = (1 / 2 : ℚ) • (JnQ n v w • (betaKappaAB n u * F0AB n z + betaKappaAB n z * F0AB n u)
          + JnQ n u w • (betaKappaAB n v * F0AB n z + betaKappaAB n z * F0AB n v)
          + JnQ n v z • (betaKappaAB n u * F0AB n w + betaKappaAB n w * F0AB n u)
          + JnQ n u z • (betaKappaAB n v * F0AB n w + betaKappaAB n w * F0AB n v)) := by
  rw [Indexed.hMap_bracketLLn, iota0_add, iota0_add, iota0_add, mul_add, mul_add, mul_add,
    iota0_hMap_term, iota0_hMap_term, iota0_hMap_term, iota0_hMap_term]
  module

/-- **LL case, transported (Y4 Stage 2's fourth and final basis-pair sector).** -/
theorem L0hatBeta_bracket_LL (n : ℕ) (u v w z : Fin (2 * n)) :
    L0hatBeta n u v * L0hatBeta n w z - L0hatBeta n w z * L0hatBeta n u v
      = iota0 n (Indexed.bracketBasisN n (Indexed.Lof u v) (Indexed.Lof w z))
        + kappaAB n * iota0 n (Indexed.hMap n (Indexed.bracketLLn n u v w z)) := by
  rw [L0hatBeta_comm, iota0_hMap_bracketLLn, iota0_bracketBasisN_Lof_Lof, iota0AB_comm_LL,
    iota0AB_Lof_eq_L0AB, iota0AB_Lof_eq_L0AB, iota0AB_Lof_eq_L0AB, iota0AB_Lof_eq_L0AB,
    L0hatBeta_eq, L0hatBeta_eq, L0hatBeta_eq, L0hatBeta_eq, Corr_eq, Corr_eq, Corr_eq, Corr_eq]
  module

/-! ## The general-element recovery theorem

Unified per-basis-pair bridge: on every basis pair `i j : IndexedBasis n`, the concrete
bracket-or-anticommutator of `liftsFamilyBeta` values (`liftsBracket`, defined by the *same*
four-way match as `Indexed.bracketBasisN`/`Indexed.GammaBetaBasis` themselves) equals `iota0
(bracketBasisN i j) + kappaAB * iota0 (GammaBetaBasis i j) + kappaAB * iota0 (hMap (bracketBasisN i
j))` -- i.e. the four already-proved sector theorems (`L0hatBeta_bracket_LL/_LF`,
`F0hatBeta_bracket_FL/_FF`), each restated so that `GammaBetaBasis` and `hMap ∘ bracketBasisN` both
appear (folding the FF/LF/FL sectors' already-zero `hMap` or `GammaBetaBasis` contribution back in
explicitly, verified case by case below). This is what makes the general-element assembly a single
`Finset.sum` linearity argument, exactly mirroring `iota0_bracketN`'s own proof shape. -/

/-- The concrete counterpart of `bracketBasisN`/`GammaBetaBasis`'s own four-way match: a commutator
on every sector except `Fof,Fof` (anticommutator there), applied to `liftsFamilyBeta`. -/
noncomputable def liftsBracket (n : ℕ) : Indexed.IndexedBasis n → Indexed.IndexedBasis n → AB n
  | .inl ⟨(u, v), _⟩, .inl ⟨(w, z), _⟩ =>
      L0hatBeta n u v * L0hatBeta n w z - L0hatBeta n w z * L0hatBeta n u v
  | .inl ⟨(u, v), _⟩, .inr w => L0hatBeta n u v * F0hatBeta n w - F0hatBeta n w * L0hatBeta n u v
  | .inr w, .inl ⟨(u, v), _⟩ => F0hatBeta n w * L0hatBeta n u v - L0hatBeta n u v * F0hatBeta n w
  | .inr u, .inr v => F0hatBeta n u * F0hatBeta n v + F0hatBeta n v * F0hatBeta n u

theorem liftsBracket_eq_bridge (n : ℕ) (i j : Indexed.IndexedBasis n) :
    liftsBracket n i j
      = iota0 n (Indexed.bracketBasisN n i j) + kappaAB n * iota0 n (Indexed.GammaBetaBasis n i j)
        + kappaAB n * iota0 n (Indexed.hMap n (Indexed.bracketBasisN n i j)) := by
  match i, j with
  | .inl ⟨(u, v), huv⟩, .inl ⟨(w, z), hwz⟩ =>
    have h := L0hatBeta_bracket_LL n u v w z
    rw [Indexed.bracketBasisN_Lof_Lof] at h
    show L0hatBeta n u v * L0hatBeta n w z - L0hatBeta n w z * L0hatBeta n u v
        = iota0 n (Indexed.bracketLLn n u v w z) + kappaAB n * iota0 n (0 : Indexed.IndexedMod n)
          + kappaAB n * iota0 n (Indexed.hMap n (Indexed.bracketLLn n u v w z))
    rw [iota0_zero, mul_zero, add_zero]
    exact h
  | .inl ⟨(u, v), huv⟩, .inr w =>
    have h := L0hatBeta_bracket_LF n u v w
    rw [Indexed.bracketBasisN_Lof_Fof] at h
    show L0hatBeta n u v * F0hatBeta n w - F0hatBeta n w * L0hatBeta n u v
        = iota0 n (Indexed.bracketLFn n u v w) + kappaAB n * iota0 n (Indexed.gammaLFn n u v w)
          + kappaAB n * iota0 n (Indexed.hMap n (Indexed.bracketLFn n u v w))
    rw [Indexed.hMap_bracketLFn, iota0_zero, mul_zero, add_zero]
    exact h
  | .inr w, .inl ⟨(u, v), huv⟩ =>
    have h := F0hatBeta_bracket_FL n u v w
    rw [Indexed.bracketBasisN_Fof_Lof, Indexed.GammaBetaBasis_Fof_Lof] at h
    show F0hatBeta n w * L0hatBeta n u v - L0hatBeta n u v * F0hatBeta n w
        = iota0 n (-Indexed.bracketLFn n u v w)
          + kappaAB n * iota0 n (-Indexed.gammaLFn n u v w)
          + kappaAB n * iota0 n (Indexed.hMap n (-Indexed.bracketLFn n u v w))
    rw [Indexed.hMap_neg, Indexed.hMap_bracketLFn, neg_zero, iota0_zero, mul_zero, add_zero]
    exact h
  | .inr u, .inr v =>
    have h := F0hatBeta_bracket_FF n u v
    show F0hatBeta n u * F0hatBeta n v + F0hatBeta n v * F0hatBeta n u
        = iota0 n (Indexed.bracketFFn n u v)
          + kappaAB n * iota0 n (Indexed.GammaBetaBasis n (Indexed.Fof u) (Indexed.Fof v))
          + kappaAB n * iota0 n (Indexed.hMap n (Indexed.bracketFFn n u v))
    rw [Indexed.hMap_bracketFFn, iota0_neg, mul_neg, h, Indexed.bracketBasisN_Fof_Fof]
    abel

end Source

namespace Indexed

/-- `hMap` distributes over a finite sum, exactly as `iota0_sum` does for `iota0`
(`SourceRecoveryBridge.lean`). -/
theorem hMap_sum {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (f : ι → IndexedMod n) :
    hMap n (∑ i ∈ s, f i) = ∑ i ∈ s, hMap n (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [hMap_zero]
  | @insert a s ha ih => rw [Finset.sum_insert ha, hMap_add, ih, Finset.sum_insert ha]

/-- `hMap` distributes over a finite `Pn n`-scaled sum, exactly as `iota0_sum'` does for `iota0`. -/
theorem hMap_sum' {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (c : ι → Pn n)
    (f : ι → IndexedMod n) :
    hMap n (∑ i ∈ s, c i • f i) = ∑ i ∈ s, c i • hMap n (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [hMap_zero]
  | @insert a s ha ih => rw [Finset.sum_insert ha, hMap_add, hMap_smul, ih, Finset.sum_insert ha]

end Indexed

namespace Source
open scoped TensorProduct

/-- `iota0` distributes over the `hMap ∘ bracketN` double sum, exactly as `iota0_sum`/`iota0_sum'`
do for `bracketN` itself in `iota0_bracketN` (Stage 1). -/
theorem hMap_bracketN_eq (n : ℕ) (x y : Indexed.IndexedMod n) :
    Indexed.hMap n (Indexed.bracketN n x y)
      = ∑ i : Indexed.IndexedBasis n, ∑ j : Indexed.IndexedBasis n,
          (x i * y j) • Indexed.hMap n (Indexed.bracketBasisN n i j) := by
  unfold Indexed.bracketN
  rw [Indexed.hMap_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [Indexed.hMap_sum']

theorem GammaBetaN_eq_sum (n : ℕ) (x y : Indexed.IndexedMod n) :
    Indexed.GammaBetaN n x y
      = ∑ i : Indexed.IndexedBasis n, ∑ j : Indexed.IndexedBasis n,
          (x i * y j) • Indexed.GammaBetaBasis n i j := rfl

theorem algebraMap_tmul_comm_kappaAB (n : ℕ) (c : Indexed.Pn n) :
    kappaAB n * ((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (1 : A0 n) : AB n)
      = ((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (1 : A0 n) : AB n) * kappaAB n := by
  rw [kappaAB_mul_algebraMap_tmul, ← Algebra.commutes]
  unfold kappaAB
  rw [GradedTensorProduct.tmul_one_mul_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
    (algebraMap (Indexed.Pn n) (RRing n) c)
    (⟨kappa n, kappa_mem_RGrading_one n⟩ : RGradingQ n 1) (1 : A0 n)]

theorem algebraMap_tmul_mul_kappaAB_mul (n : ℕ) (c : Indexed.Pn n) (z : AB n) :
    kappaAB n * (((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (1 : A0 n) : AB n) * z)
      = ((algebraMap (Indexed.Pn n) (RRing n) c) ᵍ⊗ₜ[ℚ] (1 : A0 n) : AB n) * (kappaAB n * z) := by
  rw [← mul_assoc, algebraMap_tmul_comm_kappaAB, mul_assoc]

/-- Generic form of `iota0_bracketN`'s own proof shape, for any bilinearly-extended `F`. -/
theorem iota0_bilinearExtend (n : ℕ) (F : Indexed.IndexedBasis n → Indexed.IndexedBasis n → Indexed.IndexedMod n)
    (x y : Indexed.IndexedMod n) :
    iota0 n (∑ i : Indexed.IndexedBasis n, ∑ j : Indexed.IndexedBasis n, (x i * y j) • F i j)
      = ∑ i : Indexed.IndexedBasis n, ∑ j : Indexed.IndexedBasis n,
          ((algebraMap (Indexed.Pn n) (RRing n) (x i * y j)) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * iota0 n (F i j) := by
  rw [iota0_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [iota0_sum']

/-- **The general-element recovery theorem (R5's own deliverable, P4)**: the concrete
bracket-or-anticommutator of the deformed lifts, summed over every basis pair weighted by `x`,
`y`'s own coordinates, equals the `iota0`-image of the *undeformed* bracket `Indexed.bracketN`,
plus a `kappaAB`-linear correction built from the frozen, accepted `Indexed.GammaBetaN` -- appearing
literally, per the round's anti-circularity obligation -- plus the already-identified extra
`hMap`-of-the-output term (real structure, not an error; see this file's header comment). -/
theorem liftsBracket_general (n : ℕ) (x y : Indexed.IndexedMod n) :
    ∑ i : Indexed.IndexedBasis n, ∑ j : Indexed.IndexedBasis n,
        ((algebraMap (Indexed.Pn n) (RRing n) (x i * y j)) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * liftsBracket n i j
      = iota0 n (Indexed.bracketN n x y)
        + kappaAB n * iota0 n (Indexed.GammaBetaN n x y)
        + kappaAB n * iota0 n (Indexed.hMap n (Indexed.bracketN n x y)) := by
  rw [iota0_bracketN, GammaBetaN_eq_sum, hMap_bracketN_eq, iota0_bilinearExtend,
    iota0_bilinearExtend]
  simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  rw [algebraMap_tmul_mul_kappaAB_mul, algebraMap_tmul_mul_kappaAB_mul, ← mul_add, ← mul_add,
    ← liftsBracket_eq_bridge]

/-- **Y4, in exactly the shape S requires** -- mirroring the manuscript's own `eq:recover-by-u`,
`[X,Y]_beta = U^{-1}[UX,UY]_0`: the manuscript's *recovered* bracket is the raw commutator with a
`U^{-1}`-style correction already applied, not the raw commutator itself. Concretely, `U^{-1}`'s own
definition (`UMapInv_eq`, `IndexedU.lean`) subtracts exactly `kappaEmbed (hMap n (falsePart Z))`
from the raw bracket `Z`; `falsePart Z = bracketN n x y` (proved in this file's header derivation),
so the concrete analogue of the *recovered* bracket is `liftsBracket_general`'s raw sum with
`kappaAB n * iota0 n (hMap n (bracketN n x y))` subtracted -- exactly the term
`liftsBracket_general` already isolates. Moving it across the equality (pure ring arithmetic on an
already-proved identity, not a new definition or a fresh proof obligation) leaves **exactly** the
two-term shape S demands, `GammaBetaN` appearing literally and nothing else: -/
theorem liftsBracket_recovery (n : ℕ) (x y : Indexed.IndexedMod n) :
    (∑ i : Indexed.IndexedBasis n, ∑ j : Indexed.IndexedBasis n,
        ((algebraMap (Indexed.Pn n) (RRing n) (x i * y j)) ᵍ⊗ₜ[ℚ] (1 : A0 n)) * liftsBracket n i j)
      - kappaAB n * iota0 n (Indexed.hMap n (Indexed.bracketN n x y))
      = iota0 n (Indexed.bracketN n x y) + kappaAB n * iota0 n (Indexed.GammaBetaN n x y) := by
  rw [liftsBracket_general]
  abel

/-! ## Y5 -- closed image: a precise obstruction, not a closure theorem

**Finding (partial, honestly not closed as stated): the bare `Pn n`-span of `liftsFamilyBeta`
alone is *not* closed under the bracket; closure needs the family extended by `kappaAB` multiples,
mirroring the abstract `RMod n`'s own `IndexedMod n`-doubling (`iotaR`/`kappaEmbed`,
`IndexedKappa.lean`) at the concrete level.**

The obstruction is visible already at Y2: `betaKappaAB` (the deformation piece of `L0hatBeta_eq`)
is *itself* `kappaAB` times an `algebraMap`-embedded scalar (`betaKappaAB_eq_kappaAB_mul`
below), so `L0hatBeta_eq` rearranges to express the *undeformed* `L0AB` as `L0hatBeta` minus a
`kappaAB`-multiple of `F0hatBeta`-combinations (`L0AB_eq_L0hatBeta_sub`). Since
`liftsBracket_general`'s correction terms are built from `iota0 (bracketN x y)` -- a combination of
*undeformed* `L0`/`F0`, via `L0AB`, not `L0hatBeta` -- the `L0AB`-parts of that correction do **not**
lie in the bare span of `{L0hatBeta, F0hatBeta}` alone; `L0AB_eq_L0hatBeta_sub` shows exactly what is
missing: a `kappaAB`-multiple of the family. `F0hatBeta_eq_F0AB` (Y2) already shows the parallel `F0`
part poses no such obstruction (it is already exactly `F0hatBeta`, no correction). This is a typed
account of the gap, not a decision that closure is impossible: extending the spanning family to
`liftsFamilyBeta n b` together with `kappaAB n * liftsFamilyBeta n b` (for every basis vector `b`) is
the natural fix, mirroring `RMod n = IndexedMod n × Bool`'s own shape, but re-running Y3's
linear-independence argument and a full closure theorem for that doubled family is not attempted in
this pass. -/

theorem betaKappaAB_eq_kappaAB_mul (n : ℕ) (u : Fin (2 * n)) :
    betaKappaAB n u = kappaAB n * ((algebraMap (Indexed.Pn n) (RRing n) (Indexed.betaN n u))
      ᵍ⊗ₜ[ℚ] (1 : A0 n)) := by
  unfold betaKappaAB Source.betaKappa
  rw [Algebra.commutes, ← kappaAB_mul_algebraMap_tmul]

/-- **The Y5 obstruction, stated precisely**: the undeformed `L0AB n u v` equals `L0hatBeta n u v`
minus a `kappaAB`-multiple of an `F0hatBeta`-combination -- a term outside the bare (undoubled)
span of `liftsFamilyBeta`. -/
theorem L0AB_eq_L0hatBeta_sub (n : ℕ) (u v : Fin (2 * n)) :
    L0AB n u v = L0hatBeta n u v
      - kappaAB n * (((algebraMap (Indexed.Pn n) (RRing n) (Indexed.betaN n u)) ᵍ⊗ₜ[ℚ] (1 : A0 n))
          * F0hatBeta n v
        + ((algebraMap (Indexed.Pn n) (RRing n) (Indexed.betaN n v)) ᵍ⊗ₜ[ℚ] (1 : A0 n))
          * F0hatBeta n u) := by
  rw [L0hatBeta_eq, Corr_eq, F0hatBeta_eq_F0AB, F0hatBeta_eq_F0AB, betaKappaAB_eq_kappaAB_mul,
    betaKappaAB_eq_kappaAB_mul, mul_add, mul_assoc, mul_assoc]
  abel

end Source
end InhomogeneousDeformations
