import InhomogeneousDeformations.SourceC
import InhomogeneousDeformations.SourceWeyl
import Mathlib.LinearAlgebra.TensorProduct.Graded.Internal
import Mathlib.Algebra.Algebra.RestrictScalars
import Mathlib.Algebra.TrivSqZeroExt.Basic
import Mathlib.Algebra.CharP.Invertible
import Mathlib.LinearAlgebra.CliffordAlgebra.Contraction
import Mathlib.LinearAlgebra.Dual.Lemmas
import Mathlib.Tactic.NoncommRing

/-!
# I106 R2, W2 — `A_0 = W_n ⊗ C` and `A_B = R ⊗ A_0`, via `GradedTensorProduct`

`R = P ⊕ κP` (manuscript line 55-56) is built here as `CliffordAlgebra (0 : QuadraticForm
(Pn n) (Pn n))` — the same recipe `SourceC.lean`'s `C` uses (a Clifford algebra realizing a
"square-zero extension by one odd generator"), now with the **zero** quadratic form (giving
`κ² = 0` instead of `a² = 1/2`) and over the real coefficient ring `Pn n` instead of `ℚ`. This
is deliberately **not** `TrivSqZeroExt`: an earlier attempt built `R`'s grading by hand for
`TrivSqZeroExt (Pn n) (Pn n)`, mirroring an abandoned draft from R0's F1 investigation, and it
was fragile (several broken proof steps, unresolved goals) for no offsetting benefit —
`CliffordAlgebra` gives the `ZMod 2` grading **for free** via the library's own
`CliffordAlgebra.gradedAlgebra`, exactly as it already does for `C`. Reusing one recipe for
both odd generators is also simpler to audit than hand-building a second one.

**A3**: `κ` odd, `κ² = 0` (both supplied by the library, as for `a`), and `κ`, `a` independent
in `A_B` (the two-sided sign is genuinely needed, not decorative) — tested the same way R0's
F1 experiment did, now on the real objects instead of a toy.
**A4**: `A_0`'s and `A_B`'s gradings restrict correctly to their tensor factors — by
construction, since both are built as `GradedTensorProduct`s of already-graded algebras.
-/

namespace InhomogeneousDeformations
namespace Source

open scoped TensorProduct DirectSum

/-! ## `R = P ⊕ κP`, via `CliffordAlgebra` with the zero quadratic form -/

noncomputable def Qzero (n : ℕ) : QuadraticForm (Indexed.Pn n) (Indexed.Pn n) := 0

/-- `R`, built as `CliffordAlgebra (Qzero n)`. -/
noncomputable abbrev RRing (n : ℕ) : Type := CliffordAlgebra (Qzero n)

/-- The odd generator `κ`. -/
noncomputable def kappa (n : ℕ) : RRing n := CliffordAlgebra.ι (Qzero n) 1

/-- **A3, first part**: `κ² = 0` — supplied directly by the library
(`CliffordAlgebra.ι_sq_scalar`), exactly as `a² = 1/2` was for `C`. -/
theorem kappa_sq (n : ℕ) : kappa n * kappa n = 0 := by
  unfold kappa
  rw [CliffordAlgebra.ι_sq_scalar]
  rw [show Qzero n 1 = 0 from rfl]
  exact map_zero _

/-- `R`'s `ZMod 2` grading is `CliffordAlgebra.evenOdd (Qzero n)`, from the library. -/
noncomputable abbrev RGrading (n : ℕ) : ZMod 2 → Submodule (Indexed.Pn n) (RRing n) :=
  CliffordAlgebra.evenOdd (Qzero n)

/-- **A3, second part**: `κ` is odd. -/
theorem kappa_mem_RGrading_one (n : ℕ) : kappa n ∈ RGrading n 1 :=
  CliffordAlgebra.ι_mem_evenOdd_one (Qzero n) 1

/-- `κ ≠ 0` — needed so A3's independence identity is a genuine, non-vacuous statement. Proved
by lifting `ι (Qzero n)` into `TrivSqZeroExt (Pn n) (Pn n)` via `CliffordAlgebra.lift` (using
`TrivSqZeroExt.inr`'s own square-zero property, `inr_mul_inr`, to match the `Qzero n = 0`
relation exactly) and observing the image of `κ` is `inr 1 ≠ 0`. -/
theorem kappa_ne_zero (n : ℕ) : kappa n ≠ 0 := by
  set F : RRing n →ₐ[Indexed.Pn n] TrivSqZeroExt (Indexed.Pn n) (Indexed.Pn n) :=
    CliffordAlgebra.lift (Qzero n)
      ⟨TrivSqZeroExt.inrHom (Indexed.Pn n) (Indexed.Pn n), fun m => by
        rw [TrivSqZeroExt.inrHom, LinearMap.coe_mk, AddHom.coe_mk, TrivSqZeroExt.inr_mul_inr,
          show Qzero n m = 0 from rfl, map_zero]⟩ with hF
  have hFkappa : F (kappa n) = TrivSqZeroExt.inr 1 := by
    rw [kappa, hF, CliffordAlgebra.lift_ι_apply, TrivSqZeroExt.inrHom, LinearMap.coe_mk,
      AddHom.coe_mk]
  intro h
  rw [h, map_zero] at hFkappa
  exact one_ne_zero (by
    have := congrArg TrivSqZeroExt.snd hFkappa.symm
    rwa [TrivSqZeroExt.snd_inr, TrivSqZeroExt.snd_zero] at this)

noncomputable instance RGradedAlgebra (n : ℕ) : GradedAlgebra (RGrading n) :=
  CliffordAlgebra.gradedAlgebra (Qzero n)

/-- `Source.a ≠ 0` — needed so A3's independence identity is non-vacuous. `C` is nontrivial
(`CliffordAlgebra`'s own `instNontrivial`, needing only `Invertible (2 : ℚ)`, automatic since
`ℚ` is a field of characteristic zero), so `1 ≠ 0` in `C`; combined with `Source.a_sq` (`a*a =
1/2 • 1`) this forces `a ≠ 0`, since `a = 0` would force `1/2 • (1:C) = 0`, i.e. `1 = 0`. -/
theorem a_ne_zero : Source.a ≠ 0 := by
  have : Nontrivial C := inferInstance
  intro h
  apply (one_ne_zero (α := C))
  have h2 := Source.a_sq
  rw [h, mul_zero] at h2
  have h3 := congrArg (fun x => (2 : ℚ) • x) h2
  simp only [smul_smul, smul_zero] at h3
  norm_num at h3

/-- `R` is a `ℚ`-algebra (not just a `Pn n`-algebra), via `Pn n` itself being one — needed to
tensor `R` with the `ℚ`-algebra `A_0` in W2. Not automatic (`Algebra.restrictScalars` is
deliberately not an instance, to avoid diamonds), so declared once here. -/
noncomputable instance RRing_algebra_rat (n : ℕ) : Algebra ℚ (RRing n) :=
  Algebra.restrictScalars ℚ (Indexed.Pn n) (RRing n)

noncomputable instance RRing_isScalarTower (n : ℕ) :
    IsScalarTower ℚ (Indexed.Pn n) (RRing n) :=
  IsScalarTower.of_algebraMap_eq fun _ => rfl

/-- `R`'s grading, viewed over `ℚ` instead of `Pn n` — needed for `GradedTensorProduct ℚ`.
Restricting scalars on an already-graded algebra is graded "for free"
(`RingTheory.GradedAlgebra.Basic`'s own `restrictScalars` instance). -/
noncomputable abbrev RGradingQ (n : ℕ) : ZMod 2 → Submodule ℚ (RRing n) :=
  fun i => (RGrading n i).restrictScalars ℚ

noncomputable instance RGradedAlgebraQ (n : ℕ) : GradedAlgebra (RGradingQ n) :=
  inferInstance

/-! ## `W_n`, graded trivially -- "since `W_n` is purely even" (manuscript line 104) -/

/-- `W_n`'s grading: everything in degree `0`. -/
noncomputable def WGrading (n : ℕ) : ZMod 2 → Submodule ℚ (Module.End ℚ (WPoly n))
  | 0 => ⊤
  | 1 => ⊥

theorem ZMod2_eq_zero_or_one (i : ZMod 2) : i = 0 ∨ i = 1 := by revert i; decide

theorem ZMod2_add_cases (i j : ZMod 2) :
    (i = 0 ∧ j = 0 ∧ i + j = 0) ∨ (i = 0 ∧ j = 1 ∧ i + j = 1) ∨
    (i = 1 ∧ j = 0 ∧ i + j = 1) ∨ (i = 1 ∧ j = 1 ∧ i + j = 0) := by
  rcases ZMod2_eq_zero_or_one i with hi | hi <;> rcases ZMod2_eq_zero_or_one j with hj | hj <;>
    subst hi <;> subst hj <;> simp_all <;> decide

/-- The four `(i', j')` pairs with `i' + j' = k` for a total degree `k = i + j` split as
`i = i' + i''`, `j = j' + j''` across two factors: standalone (no other hypotheses in context)
so `revert i j; decide` only reverts `i, j` themselves. -/
theorem ZMod2_total_degree_cases (i j : ZMod 2) :
    (i + j = (0 : ZMod 2) + 0 + (i + j)) ∧ (i + j = (0 : ZMod 2) + 1 + (i + (j + 1))) ∧
    (i + j = (1 : ZMod 2) + 0 + (i + 1 + j)) ∧ (i + j = (1 : ZMod 2) + 1 + (i + 1 + (j + 1))) := by
  revert i j; decide

/-- Standalone (no other hypotheses in context) so `revert k; decide` only reverts `k` itself. -/
theorem ZMod2_one_add_succ (k : ZMod 2) : (1 : ZMod 2) + (k + 1) = k := by revert k; decide

instance WGrading_setLike (n : ℕ) : SetLike.GradedMonoid (WGrading n) where
  one_mem := by simp [WGrading]
  mul_mem := by
    intro i j gi gj hgi hgj
    rcases ZMod2_add_cases i j with ⟨hi, hj, hij⟩ | ⟨hi, hj, hij⟩ | ⟨hi, hj, hij⟩ | ⟨hi, hj, hij⟩ <;>
      subst hi <;> subst hj <;> rw [hij] <;> simp_all [WGrading]

/-- The decomposition map, as a plain `LinearMap` — `DirectSum.Decomposition` (unlike
`GradedAlgebra.ofAlgHom`) needs no ring-multiplicativity proof at all, only additivity and
`ℚ`-linearity, which sidesteps the several fragile steps a full `AlgHom` construction needed
here for no benefit (`SetLike.GradedMonoid`, proved separately above, already carries the
multiplicative content `GradedRing` needs). -/
noncomputable def WGrading_decompose_lin (n : ℕ) :
    Module.End ℚ (WPoly n) →ₗ[ℚ] (⨁ i : ZMod 2, WGrading n i) :=
  (DirectSum.lof ℚ (ZMod 2) (fun i => WGrading n i) 0).comp
    (LinearMap.codRestrict (WGrading n 0) LinearMap.id (fun f => by simp [WGrading]))

noncomputable instance WGradedAlgebra (n : ℕ) : GradedAlgebra (WGrading n) :=
  { WGrading_setLike n with
    toDecomposition := DirectSum.Decomposition.ofLinearMap (WGrading n)
      (WGrading_decompose_lin n)
      (by
        ext f
        simp [WGrading_decompose_lin])
      (by
        apply DirectSum.linearMap_ext ℚ
        intro i
        rcases ZMod2_eq_zero_or_one i with hi | hi <;> subst hi
        · apply LinearMap.ext; intro x
          obtain ⟨x, hx⟩ := x
          simp only [LinearMap.comp_apply, DirectSum.coeLinearMap_lof, WGrading_decompose_lin,
            LinearMap.id_apply]
          rfl
        · apply LinearMap.ext; intro x
          obtain ⟨x, hx⟩ := x
          simp only [WGrading, Submodule.mem_bot] at hx
          subst hx
          simp only [LinearMap.comp_apply, DirectSum.coeLinearMap_lof, WGrading_decompose_lin,
            LinearMap.id_apply, map_zero]
          rw [show (⟨0, hx⟩ : (WGrading n 1 : Submodule ℚ (Module.End ℚ (WPoly n)))) = 0 from rfl,
            map_zero]) }

/-! ## `A_0 = W_n ⊗ C` and `A_B = R ⊗ A_0` -/

noncomputable abbrev A0 (n : ℕ) : Type :=
  GradedTensorProduct ℚ (WGrading n) (CGrading)

/-! ### `A_0`'s own `ZMod 2` grading

Mathlib does not supply "the tensor product of two graded algebras is itself graded" (see the
module docstring TODO in `Mathlib.LinearAlgebra.TensorProduct.Graded.Internal`), so this is built
by hand here. It is much easier than the general case because `W_n` is *purely even*
(`WGrading n 0 = ⊤`): `GradedTensorProduct.tmul_coe_mul_zero_coe_tmul` shows the graded
multiplication picks up **no sign** whenever the second tensor factor's `𝒜`-degree is `0`, and
every element of `Module.End ℚ (WPoly n)` qualifies. So `A_0`'s grading is exactly `C`'s own
grading, pulled back along the tensor factor: `A_0`'s degree-`k` piece is
`(Module.End ℚ (WPoly n)) ⊗[ℚ] (CGrading k)`, viewed inside `A0 n`. -/

/-- The inclusion of `(Module.End ℚ (WPoly n)) ⊗[ℚ] (CGrading k)` into `A0 n`. -/
noncomputable def A0GradingMap (n : ℕ) (k : ZMod 2) :
    (Module.End ℚ (WPoly n)) ⊗[ℚ] (CGrading k) →ₗ[ℚ] A0 n :=
  (GradedTensorProduct.of ℚ (WGrading n) CGrading).toLinearMap ∘ₗ
    TensorProduct.map LinearMap.id (CGrading k).subtype

theorem A0GradingMap_tmul (n : ℕ) (k : ZMod 2) (w : Module.End ℚ (WPoly n)) (c : CGrading k) :
    A0GradingMap n k (w ⊗ₜ c) = w ᵍ⊗ₜ[ℚ] (c : C) := rfl

/-- `A_0`'s `ZMod 2` grading: degree `k` is the image of `(Module.End ℚ (WPoly n)) ⊗[ℚ]
(CGrading k)`, i.e. exactly `C`'s own grading pulled back along the tensor factor. -/
noncomputable def A0Grading (n : ℕ) (k : ZMod 2) : Submodule ℚ (A0 n) :=
  LinearMap.range (A0GradingMap n k)

theorem A0Grading_mem_of_tmul (n : ℕ) (k : ZMod 2) (w : Module.End ℚ (WPoly n)) (c : CGrading k) :
    w ᵍ⊗ₜ[ℚ] (c : C) ∈ A0Grading n k :=
  ⟨w ⊗ₜ c, A0GradingMap_tmul n k w c⟩

instance A0Grading_setLike (n : ℕ) : SetLike.GradedMonoid (A0Grading n) where
  one_mem := by
    have h := A0Grading_mem_of_tmul n 0 1 (⟨1, SetLike.one_mem_graded CGrading⟩ : CGrading 0)
    simpa [GradedTensorProduct.tmul, ← GradedTensorProduct.of_one,
      Algebra.TensorProduct.one_def] using h
  mul_mem := by
    intro i j gi gj hgi hgj
    obtain ⟨xi, hxi⟩ := hgi
    obtain ⟨xj, hxj⟩ := hgj
    subst hxi
    subst hxj
    induction xi using TensorProduct.induction_on with
    | zero => simp
    | add x y hx hy => simp only [map_add, add_mul]; exact add_mem hx hy
    | tmul w c =>
      induction xj using TensorProduct.induction_on with
      | zero => simp
      | add x y hx hy => simp only [map_add, mul_add]; exact add_mem hx hy
      | tmul w' c' =>
        rw [A0GradingMap_tmul, A0GradingMap_tmul,
          GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := WGrading n) (ℬ := CGrading)
            w c (⟨w', trivial⟩ : WGrading n 0) c']
        exact A0Grading_mem_of_tmul n (i + j) (w * w')
          (⟨(c : C) * (c' : C), SetLike.mul_mem_graded c.2 c'.2⟩ : CGrading (i + j))

/-- The decomposition map for `A0Grading`, built from `C`'s own decomposition: `A0 n = (Module.End
ℚ (WPoly n)) ⊗[ℚ] C ≃ (Module.End ℚ (WPoly n)) ⊗[ℚ] (⨁ k, CGrading k) ≃ ⨁ k, (Module.End ℚ
(WPoly n)) ⊗[ℚ] (CGrading k)`, corestricted degreewise into `A0Grading n k`. -/
noncomputable def A0_decompose_lin (n : ℕ) :
    A0 n →ₗ[ℚ] ⨁ k : ZMod 2, A0Grading n k :=
  (DirectSum.lmap fun k => LinearMap.rangeRestrict (A0GradingMap n k)) ∘ₗ
    (TensorProduct.directSumRight ℚ ℚ (Module.End ℚ (WPoly n)) (fun k => CGrading k)).toLinearMap ∘ₗ
    (TensorProduct.congr (LinearEquiv.refl ℚ (Module.End ℚ (WPoly n)))
      (DirectSum.decomposeLinearEquiv CGrading)).toLinearMap ∘ₗ
    (GradedTensorProduct.of ℚ (WGrading n) CGrading).symm.toLinearMap

/-- `A0_decompose_lin` on a pure tensor `w ᵍ⊗ₜ (c : C)` with `c` already homogeneous of degree
`k`: it lands exactly on the `k`-th summand, tagged by the obvious membership witness. This is
the computational core both `left_inv` and `right_inv` reduce to. -/
theorem A0_decompose_lin_tmul_coe (n : ℕ) (k : ZMod 2) (w : Module.End ℚ (WPoly n))
    (c : CGrading k) :
    A0_decompose_lin n (w ᵍ⊗ₜ[ℚ] (c : C)) =
      DirectSum.lof ℚ (ZMod 2) (fun k => A0Grading n k) k
        ⟨w ᵍ⊗ₜ[ℚ] (c : C), A0Grading_mem_of_tmul n k w c⟩ := by
  show (DirectSum.lmap fun k => LinearMap.rangeRestrict (A0GradingMap n k))
      ((TensorProduct.directSumRight ℚ ℚ (Module.End ℚ (WPoly n)) (fun k => CGrading k))
        ((TensorProduct.congr (LinearEquiv.refl ℚ (Module.End ℚ (WPoly n)))
          (DirectSum.decomposeLinearEquiv CGrading)) (w ⊗ₜ (c : C)))) = _
  rw [TensorProduct.congr_tmul, LinearEquiv.refl_apply, DirectSum.decomposeLinearEquiv_apply,
    DirectSum.decompose_coe, ← DirectSum.lof_eq_of ℚ, TensorProduct.directSumRight_tmul_lof,
    DirectSum.lmap_lof]
  rfl

/-- `A0_decompose_lin_tmul_coe`, restated with the tensor factor written as the *bare*
`⊗ₜ` (rather than `ᵍ⊗ₜ`): defeq to the original (`GradedTensorProduct.tmul`'s `of` is not
reducible, so `simp`/`rw` cannot bridge the two notations on their own — this restatement is
needed wherever a term arrives already in bare-tensor form, e.g. from `TensorProduct.tmul_sum`). -/
theorem A0_decompose_lin_tmul (n : ℕ) (k : ZMod 2) (w : Module.End ℚ (WPoly n))
    (c : CGrading k) :
    A0_decompose_lin n (w ⊗ₜ[ℚ] (c : C) : A0 n) =
      DirectSum.lof ℚ (ZMod 2) (fun k => A0Grading n k) k
        ⟨(w ⊗ₜ[ℚ] (c : C) : A0 n), A0Grading_mem_of_tmul n k w c⟩ :=
  A0_decompose_lin_tmul_coe n k w c

noncomputable instance A0GradedAlgebra (n : ℕ) : GradedAlgebra (A0Grading n) :=
  { A0Grading_setLike n with
    toDecomposition := DirectSum.Decomposition.ofLinearMap (A0Grading n)
      (A0_decompose_lin n)
      (by
        apply TensorProduct.ext'
        intro w c
        show (DirectSum.coeLinearMap (A0Grading n)) ((A0_decompose_lin n) (w ⊗ₜ c)) = w ⊗ₜ c
        have key :
            ((DirectSum.coeLinearMap (A0Grading n) ∘ₗ A0_decompose_lin n).comp
                (TensorProduct.mk ℚ (Module.End ℚ (WPoly n)) C w)).comp
              (DirectSum.decomposeLinearEquiv CGrading).symm.toLinearMap =
            ((TensorProduct.mk ℚ (Module.End ℚ (WPoly n)) C w)).comp
              (DirectSum.decomposeLinearEquiv CGrading).symm.toLinearMap := by
          apply DirectSum.linearMap_ext ℚ
          intro k
          apply LinearMap.ext; intro x
          show (DirectSum.coeLinearMap (A0Grading n))
              ((A0_decompose_lin n) (w ⊗ₜ[ℚ] ((DirectSum.decomposeLinearEquiv CGrading).symm
                (DirectSum.lof ℚ (ZMod 2) (fun i => CGrading i) k x)))) =
            w ⊗ₜ[ℚ] ((DirectSum.decomposeLinearEquiv CGrading).symm
              (DirectSum.lof ℚ (ZMod 2) (fun i => CGrading i) k x))
          rw [DirectSum.decomposeLinearEquiv_symm_lof, A0_decompose_lin_tmul,
            DirectSum.coeLinearMap_lof]
        have := DFunLike.congr_fun key ((DirectSum.decomposeLinearEquiv CGrading) c)
        show (DirectSum.coeLinearMap (A0Grading n)) ((A0_decompose_lin n) (w ⊗ₜ[ℚ] c)) = w ⊗ₜ[ℚ] c
        rw [show c = (DirectSum.decomposeLinearEquiv CGrading).symm
          ((DirectSum.decomposeLinearEquiv CGrading) c) from
          (LinearEquiv.symm_apply_apply _ c).symm]
        exact this)
      (by
        apply DirectSum.linearMap_ext ℚ
        intro k
        apply LinearMap.ext; intro z
        simp only [LinearMap.comp_apply, DirectSum.coeLinearMap_lof, LinearMap.id_apply]
        obtain ⟨y, hy⟩ := z.2
        have hz : z = ⟨A0GradingMap n k y, LinearMap.mem_range_self (A0GradingMap n k) y⟩ :=
          Subtype.ext hy.symm
        subst hz
        clear hy
        have key : A0_decompose_lin n ∘ₗ A0GradingMap n k =
            (DirectSum.lof ℚ (ZMod 2) (fun i => A0Grading n i) k).comp
              (LinearMap.rangeRestrict (A0GradingMap n k)) := by
          apply TensorProduct.ext'
          intro w c
          show A0_decompose_lin n (A0GradingMap n k (w ⊗ₜ c)) = _
          rw [A0GradingMap_tmul, A0_decompose_lin_tmul_coe]
          rfl
        exact DFunLike.congr_fun key y) }

/-! ## `A_B = R ⊗ A_0`, and **A3**'s independence witness -/

/-- `A_B = R ⊗ A_0`, over `ℚ` (via `RGradingQ`, `R`'s grading restricted to `ℚ`-scalars). -/
noncomputable abbrev AB (n : ℕ) : Type :=
  GradedTensorProduct ℚ (RGradingQ n) (A0Grading n)

/-! ### `A_B`'s own `ZMod 2` grading

Unlike `A_0` (where `W_n`'s triviality let the grading come entirely from `C`), *neither* factor
of `A_B = R ⊗ A_0` is trivial here, so the grading is a genuine total-degree sum:
`A_B`'s degree-`k` piece is `⋃_{i+j=k} (RGradingQ n i) ⊗ (A0Grading n j)`. Since `i` ranges over
`ZMod 2 = {0, 1}`, `j = k + i` is determined by `i`, so this is a join of exactly two ranges. -/

/-- The inclusion of `(RGradingQ n i) ⊗[ℚ] (A0Grading n j)` into `AB n`. -/
noncomputable def ABGradingMap (n : ℕ) (i j : ZMod 2) :
    (RGradingQ n i) ⊗[ℚ] (A0Grading n j) →ₗ[ℚ] AB n :=
  (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)).toLinearMap ∘ₗ
    TensorProduct.map (RGradingQ n i).subtype (A0Grading n j).subtype

theorem ABGradingMap_tmul (n : ℕ) (i j : ZMod 2) (r : RGradingQ n i) (x : A0Grading n j) :
    ABGradingMap n i j (r ⊗ₜ x) = (r : RRing n) ᵍ⊗ₜ[ℚ] (x : A0 n) := rfl

/-- `A_B`'s `ZMod 2` grading: degree `k` is generated by pure tensors `r ᵍ⊗ₜ x` with
`deg_R(r) + deg_{A0}(x) = k`, i.e. the join of the `(0, k)` and `(1, k+1)` pieces (these are the
only two pairs summing to `k` in `ZMod 2`). -/
noncomputable def ABGrading (n : ℕ) (k : ZMod 2) : Submodule ℚ (AB n) :=
  LinearMap.range (ABGradingMap n 0 k) ⊔ LinearMap.range (ABGradingMap n 1 (k + 1))

theorem ABGradingMap_range_le (n : ℕ) (i j : ZMod 2) :
    LinearMap.range (ABGradingMap n i j) ≤ ABGrading n (i + j) := by
  rcases ZMod2_eq_zero_or_one i with hi | hi <;> subst hi
  · rw [zero_add]; exact le_sup_left
  · show LinearMap.range (ABGradingMap n 1 j) ≤
      LinearMap.range (ABGradingMap n 0 (1 + j)) ⊔ LinearMap.range (ABGradingMap n 1 ((1 + j) + 1))
    rw [show (1 + j : ZMod 2) + 1 = j from by revert j; decide]
    exact le_sup_right

theorem ABGrading_mem_of_tmul (n : ℕ) (i j : ZMod 2) (r : RGradingQ n i) (x : A0Grading n j) :
    (r : RRing n) ᵍ⊗ₜ[ℚ] (x : A0 n) ∈ ABGrading n (i + j) :=
  ABGradingMap_range_le n i j ⟨r ⊗ₜ x, rfl⟩

instance ABGrading_setLike (n : ℕ) : SetLike.GradedMonoid (ABGrading n) where
  one_mem := by
    have h : (1 : RRing n) ᵍ⊗ₜ[ℚ] (1 : A0 n) ∈ ABGrading n (0 + 0) :=
      ABGrading_mem_of_tmul n 0 0
        (⟨1, SetLike.one_mem_graded (RGradingQ n)⟩ : RGradingQ n 0)
        (⟨1, SetLike.one_mem_graded (A0Grading n)⟩ : A0Grading n 0)
    have h1 : (1 : AB n) = (1 : RRing n) ᵍ⊗ₜ[ℚ] (1 : A0 n) := by
      rw [GradedTensorProduct.tmul, ← GradedTensorProduct.of_one, Algebra.TensorProduct.one_def]
    rw [h1]
    simpa using h
  mul_mem := by
    intro i j gi gj hgi hgj
    have mul_mem_aux : ∀ {i' j' i'' j'' : ZMod 2} (x y : AB n),
        x ∈ LinearMap.range (ABGradingMap n i' j') →
        y ∈ LinearMap.range (ABGradingMap n i'' j'') →
        x * y ∈ ABGrading n ((i' + i'') + (j' + j'')) := by
      intro i' j' i'' j'' x y hx hy
      obtain ⟨x, rfl⟩ := hx
      obtain ⟨y, rfl⟩ := hy
      induction x using TensorProduct.induction_on with
      | zero => simp
      | add x1 x2 hx1 hx2 => simp only [map_add, add_mul]; exact add_mem hx1 hx2
      | tmul r1 x1 =>
        induction y using TensorProduct.induction_on with
        | zero => simp
        | add y1 y2 hy1 hy2 => simp only [map_add, mul_add]; exact add_mem hy1 hy2
        | tmul r2 x2 =>
          rw [ABGradingMap_tmul, ABGradingMap_tmul,
            GradedTensorProduct.tmul_coe_mul_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
              (r1 : RRing n) x1 r2 (x2 : A0 n)]
          exact Submodule.smul_mem _ _
            (ABGrading_mem_of_tmul n (i' + i'') (j' + j'')
              (⟨(r1 : RRing n) * (r2 : RRing n), SetLike.mul_mem_graded r1.2 r2.2⟩ :
                RGradingQ n (i' + i''))
              (⟨(x1 : A0 n) * (x2 : A0 n), SetLike.mul_mem_graded x1.2 x2.2⟩ :
                A0Grading n (j' + j'')))
    obtain ⟨deg00, deg01, deg10, deg11⟩ := ZMod2_total_degree_cases i j
    obtain ⟨gi0, hgi0, gi1, hgi1, rfl⟩ := Submodule.mem_sup.mp hgi
    obtain ⟨gj0, hgj0, gj1, hgj1, rfl⟩ := Submodule.mem_sup.mp hgj
    have expand : (gi0 + gi1) * (gj0 + gj1) = gi0 * gj0 + gi0 * gj1 + gi1 * gj0 + gi1 * gj1 := by
      noncomm_ring
    rw [expand]
    refine add_mem (add_mem (add_mem ?_ ?_) ?_) ?_
    · rw [deg00]; exact mul_mem_aux gi0 gj0 hgi0 hgj0
    · rw [deg01]; exact mul_mem_aux gi0 gj1 hgi0 hgj1
    · rw [deg10]; exact mul_mem_aux gi1 gj0 hgi1 hgj0
    · rw [deg11]; exact mul_mem_aux gi1 gj1 hgi1 hgj1

/-- The decomposition map for `ABGrading`, built from `R`'s and `A_0`'s own decompositions:
`AB n = RRing n ⊗[ℚ] A0 n ≃ (⨁ i, RGradingQ n i) ⊗[ℚ] (⨁ j, A0Grading n j) ≃ ⨁ p : ZMod 2 × ZMod
2, RGradingQ n p.1 ⊗[ℚ] A0Grading n p.2`, folded down to `⨁ k, ABGrading n k` by summing each
pair's two indices via `DirectSum.toModule`. -/
noncomputable def AB_decompose_lin (n : ℕ) : AB n →ₗ[ℚ] ⨁ k : ZMod 2, ABGrading n k :=
  (DirectSum.toModule ℚ (ZMod 2 × ZMod 2) (⨁ k : ZMod 2, ABGrading n k)
    (fun p => (DirectSum.lof ℚ (ZMod 2) (fun k => ABGrading n k) (p.1 + p.2)).comp
      ((Submodule.inclusion (ABGradingMap_range_le n p.1 p.2)).comp
        (LinearMap.rangeRestrict (ABGradingMap n p.1 p.2))))) ∘ₗ
    (TensorProduct.directSum ℚ ℚ (fun i => RGradingQ n i) (fun j => A0Grading n j)).toLinearMap ∘ₗ
    (TensorProduct.congr (DirectSum.decomposeLinearEquiv (RGradingQ n))
      (DirectSum.decomposeLinearEquiv (A0Grading n))).toLinearMap ∘ₗ
    (GradedTensorProduct.of ℚ (RGradingQ n) (A0Grading n)).symm.toLinearMap

theorem AB_decompose_lin_tmul (n : ℕ) (i j : ZMod 2) (r : RGradingQ n i) (x : A0Grading n j) :
    AB_decompose_lin n ((r : RRing n) ᵍ⊗ₜ[ℚ] (x : A0 n)) =
      DirectSum.lof ℚ (ZMod 2) (fun k => ABGrading n k) (i + j)
        ⟨(r : RRing n) ᵍ⊗ₜ[ℚ] (x : A0 n), ABGrading_mem_of_tmul n i j r x⟩ := by
  show (DirectSum.toModule ℚ (ZMod 2 × ZMod 2) (⨁ k : ZMod 2, ABGrading n k)
      (fun p => (DirectSum.lof ℚ (ZMod 2) (fun k => ABGrading n k) (p.1 + p.2)).comp
        ((Submodule.inclusion (ABGradingMap_range_le n p.1 p.2)).comp
          (LinearMap.rangeRestrict (ABGradingMap n p.1 p.2)))))
      ((TensorProduct.directSum ℚ ℚ (fun i => RGradingQ n i) (fun j => A0Grading n j))
        ((TensorProduct.congr (DirectSum.decomposeLinearEquiv (RGradingQ n))
          (DirectSum.decomposeLinearEquiv (A0Grading n))) (r ⊗ₜ x))) = _
  rw [TensorProduct.congr_tmul, DirectSum.decomposeLinearEquiv_apply,
    DirectSum.decomposeLinearEquiv_apply, DirectSum.decompose_coe, DirectSum.decompose_coe,
    ← DirectSum.lof_eq_of ℚ, ← DirectSum.lof_eq_of ℚ, TensorProduct.directSum_lof_tmul_lof,
    DirectSum.toModule_lof]
  rfl

/-- `lof` at two propositionally-equal indices, tagging the same ambient value, agree — used to
bridge `lof (0+k)` (what `AB_decompose_lin_tmul` naturally produces) against `lof k` (what
`ABGrading n k`'s own name expects), without rewriting inside the dependent membership proof
(which trips Lean's "motive is not type correct" check). -/
theorem DirectSum_lof_ABGrading_congr (n : ℕ) {i i' : ZMod 2} (h : i = i') (v : AB n)
    (hi : v ∈ ABGrading n i) (hi' : v ∈ ABGrading n i') :
    DirectSum.lof ℚ (ZMod 2) (fun k => ABGrading n k) i ⟨v, hi⟩ =
      DirectSum.lof ℚ (ZMod 2) (fun k => ABGrading n k) i' ⟨v, hi'⟩ := by
  subst h; rfl

theorem ABGradingMap0_range_le (n : ℕ) (k : ZMod 2) :
    LinearMap.range (ABGradingMap n 0 k) ≤ ABGrading n k := by
  have h := ABGradingMap_range_le n 0 k; rwa [zero_add] at h

theorem ABGradingMap1_range_le (n : ℕ) (k : ZMod 2) :
    LinearMap.range (ABGradingMap n 1 (k + 1)) ≤ ABGrading n k := by
  have h := ABGradingMap_range_le n 1 (k + 1); rwa [ZMod2_one_add_succ] at h

/-- The `(i, j) = (0, k)` summand of `ABGrading n k`, corestricted from `ABGradingMap n 0 k`'s
range — a genuine `LinearMap`, so `map_zero`/`map_add` apply to it directly with no ad-hoc
membership-tag bookkeeping. -/
noncomputable def ABGradingMap0Restrict (n : ℕ) (k : ZMod 2) :
    RGradingQ n 0 ⊗[ℚ] A0Grading n k →ₗ[ℚ] ABGrading n k :=
  (Submodule.inclusion (ABGradingMap0_range_le n k)).comp
    (LinearMap.rangeRestrict (ABGradingMap n 0 k))

/-- The `(i, j) = (1, k+1)` summand of `ABGrading n k`. -/
noncomputable def ABGradingMap1Restrict (n : ℕ) (k : ZMod 2) :
    RGradingQ n 1 ⊗[ℚ] A0Grading n (k + 1) →ₗ[ℚ] ABGrading n k :=
  (Submodule.inclusion (ABGradingMap1_range_le n k)).comp
    (LinearMap.rangeRestrict (ABGradingMap n 1 (k + 1)))

noncomputable instance ABGradedAlgebra (n : ℕ) : GradedAlgebra (ABGrading n) :=
  { ABGrading_setLike n with
    toDecomposition := DirectSum.Decomposition.ofLinearMap (ABGrading n)
      (AB_decompose_lin n)
      (by
        apply TensorProduct.ext'
        intro r x
        have keyR : ∀ (i : ZMod 2) (r' : RGradingQ n i),
            (DirectSum.coeLinearMap (ABGrading n))
                ((AB_decompose_lin n) ((r' : RRing n) ⊗ₜ[ℚ] x)) =
              (r' : RRing n) ⊗ₜ[ℚ] x := by
          intro i r'
          have hx : x = (DirectSum.decomposeLinearEquiv (A0Grading n)).symm
              (DirectSum.decomposeLinearEquiv (A0Grading n) x) :=
            (LinearEquiv.symm_apply_apply _ x).symm
          rw [hx]
          generalize (DirectSum.decomposeLinearEquiv (A0Grading n) x) = dx
          clear hx x
          induction dx using DirectSum.induction_on with
          | zero =>
            rw [LinearEquiv.map_zero, TensorProduct.tmul_zero]
            exact (congrArg (DirectSum.coeLinearMap (ABGrading n))
              (map_zero (AB_decompose_lin n))).trans (map_zero _)
          | of j y =>
            rw [← DirectSum.lof_eq_of ℚ, DirectSum.decomposeLinearEquiv_symm_lof]
            show (DirectSum.coeLinearMap (ABGrading n))
                ((AB_decompose_lin n) ((r' : RRing n) ᵍ⊗ₜ[ℚ] (y : A0 n))) =
              (r' : RRing n) ᵍ⊗ₜ[ℚ] (y : A0 n)
            rw [AB_decompose_lin_tmul, DirectSum.coeLinearMap_lof]
          | add d1 d2 hd1 hd2 =>
            rw [LinearEquiv.map_add, TensorProduct.tmul_add]
            refine (congrArg (DirectSum.coeLinearMap (ABGrading n))
              (map_add (AB_decompose_lin n) _ _)).trans ?_
            rw [LinearMap.map_add, hd1, hd2]
            rfl
        have key2 : ∀ (i : ZMod 2) (r' : RGradingQ n i),
            (DirectSum.coeLinearMap (ABGrading n)) ((AB_decompose_lin n) (r' ⊗ₜ[ℚ] x)) =
              (r' : RRing n) ⊗ₜ[ℚ] x := keyR
        have final : ∀ r'' : RRing n,
            (DirectSum.coeLinearMap (ABGrading n)) ((AB_decompose_lin n) (r'' ⊗ₜ[ℚ] x)) =
              r'' ⊗ₜ[ℚ] x := by
          intro r''
          have hr : r'' = (DirectSum.decomposeLinearEquiv (RGradingQ n)).symm
              (DirectSum.decomposeLinearEquiv (RGradingQ n) r'') :=
            (LinearEquiv.symm_apply_apply _ r'').symm
          rw [hr]
          generalize (DirectSum.decomposeLinearEquiv (RGradingQ n) r'') = dr
          clear hr r''
          induction dr using DirectSum.induction_on with
          | zero =>
            rw [LinearEquiv.map_zero, TensorProduct.zero_tmul]
            exact (congrArg (DirectSum.coeLinearMap (ABGrading n))
              (map_zero (AB_decompose_lin n))).trans (map_zero _)
          | of i r' =>
            rw [← DirectSum.lof_eq_of ℚ, DirectSum.decomposeLinearEquiv_symm_lof]
            exact key2 i r'
          | add d1 d2 hd1 hd2 =>
            rw [LinearEquiv.map_add, TensorProduct.add_tmul]
            refine (congrArg (DirectSum.coeLinearMap (ABGrading n))
              (map_add (AB_decompose_lin n) _ _)).trans ?_
            rw [LinearMap.map_add, hd1, hd2]
            rfl
        exact final r)
      (by
        apply DirectSum.linearMap_ext ℚ
        intro k
        have key0 : (AB_decompose_lin n).comp (ABGradingMap n 0 k) =
            (DirectSum.lof ℚ (ZMod 2) (fun i => ABGrading n i) k).comp
              (ABGradingMap0Restrict n k) := by
          apply TensorProduct.ext'
          intro r0 x0
          show AB_decompose_lin n (ABGradingMap n 0 k (r0 ⊗ₜ x0)) =
            DirectSum.lof ℚ (ZMod 2) (fun i => ABGrading n i) k
              (ABGradingMap0Restrict n k (r0 ⊗ₜ x0))
          rw [ABGradingMap_tmul, AB_decompose_lin_tmul]
          apply DirectSum_lof_ABGrading_congr n (zero_add k)
        have key1 : (AB_decompose_lin n).comp (ABGradingMap n 1 (k + 1)) =
            (DirectSum.lof ℚ (ZMod 2) (fun i => ABGrading n i) k).comp
              (ABGradingMap1Restrict n k) := by
          apply TensorProduct.ext'
          intro r1 x1
          show AB_decompose_lin n (ABGradingMap n 1 (k + 1) (r1 ⊗ₜ x1)) =
            DirectSum.lof ℚ (ZMod 2) (fun i => ABGrading n i) k
              (ABGradingMap1Restrict n k (r1 ⊗ₜ x1))
          rw [ABGradingMap_tmul, AB_decompose_lin_tmul]
          apply DirectSum_lof_ABGrading_congr n (ZMod2_one_add_succ k)
        apply LinearMap.ext; intro z
        simp only [LinearMap.comp_apply, DirectSum.coeLinearMap_lof, LinearMap.id_apply]
        obtain ⟨z0, hz0, z1, hz1, hz⟩ := Submodule.mem_sup.mp z.2
        obtain ⟨y0, hy0⟩ := hz0
        obtain ⟨y1, hy1⟩ := hz1
        have e0 : AB_decompose_lin n z0 =
            DirectSum.lof ℚ (ZMod 2) (fun i => ABGrading n i) k (ABGradingMap0Restrict n k y0) :=
          hy0 ▸ DFunLike.congr_fun key0 y0
        have e1 : AB_decompose_lin n z1 =
            DirectSum.lof ℚ (ZMod 2) (fun i => ABGrading n i) k (ABGradingMap1Restrict n k y1) :=
          hy1 ▸ DFunLike.congr_fun key1 y1
        rw [← hz]
        calc AB_decompose_lin n (z0 + z1)
            = AB_decompose_lin n z0 + AB_decompose_lin n z1 := map_add _ _ _
          _ = DirectSum.lof ℚ (ZMod 2) (fun i => ABGrading n i) k (ABGradingMap0Restrict n k y0) +
                DirectSum.lof ℚ (ZMod 2) (fun i => ABGrading n i) k
                  (ABGradingMap1Restrict n k y1) := by rw [e0, e1]
          _ = DirectSum.lof ℚ (ZMod 2) (fun i => ABGrading n i) k
                (ABGradingMap0Restrict n k y0 + ABGradingMap1Restrict n k y1) :=
                (map_add (DirectSum.lof ℚ (ZMod 2) (fun i => ABGrading n i) k) _ _).symm
          _ = DirectSum.lof ℚ (ZMod 2) (fun i => ABGrading n i) k z := by
                congr 1
                apply Subtype.ext
                show (ABGradingMap0Restrict n k y0 : AB n) + (ABGradingMap1Restrict n k y1 : AB n)
                  = z
                rw [show (ABGradingMap0Restrict n k y0 : AB n) = ABGradingMap n 0 k y0 from rfl,
                  show (ABGradingMap1Restrict n k y1 : AB n) = ABGradingMap n 1 (k + 1) y1 from
                    rfl, hy0, hy1, hz]) }

noncomputable def aA0 (n : ℕ) : A0 n := (1 : Module.End ℚ (WPoly n)) ᵍ⊗ₜ[ℚ] (Source.a : C)

theorem aA0_mem_A0Grading_one (n : ℕ) : aA0 n ∈ A0Grading n 1 :=
  A0Grading_mem_of_tmul n 1 1 (⟨Source.a, Source.a_odd⟩ : CGrading 1)

theorem aA0_ne_zero (n : ℕ) : aA0 n ≠ 0 := by
  intro h
  obtain ⟨f, hf⟩ := Module.Projective.exists_dual_eq_one ℚ
    (one_ne_zero (α := Module.End ℚ (WPoly n)))
  obtain ⟨g, hg⟩ := Module.Projective.exists_dual_eq_one ℚ Source.a_ne_zero
  have hΦ : ((TensorProduct.lid ℚ ℚ).toLinearMap ∘ₗ TensorProduct.map f g)
      ((1 : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (Source.a : C)) = 1 := by
    simp [TensorProduct.map_tmul, hf, hg]
  rw [show ((1 : Module.End ℚ (WPoly n)) ⊗ₜ[ℚ] (Source.a : C) : A0 n) = aA0 n from rfl, h] at hΦ
  simp only [LinearMap.comp_apply, map_zero] at hΦ
  exact one_ne_zero hΦ.symm

/-- `κ`, embedded into `A_B` as `κ ᵍ⊗ₜ 1`. -/
noncomputable def kappaAB (n : ℕ) : AB n := (kappa n) ᵍ⊗ₜ[ℚ] (1 : A0 n)

/-- `a`, embedded into `A_B` as `1 ᵍ⊗ₜ (a embedded into A_0)`. -/
noncomputable def aAB (n : ℕ) : AB n := (1 : RRing n) ᵍ⊗ₜ[ℚ] (aA0 n)

theorem kappa_mem_RGradingQ_one (n : ℕ) : kappa n ∈ RGradingQ n 1 :=
  kappa_mem_RGrading_one n

/-- **A3**: `κ`'s and `a`'s embeddings anti-commute in `A_B`, genuinely (via the Koszul sign,
since both are odd — `(-1)^(1*1) = -1`), and their product is nonzero, so the sign is not
decorative: it is the only thing distinguishing `κ_AB * a_AB` from `a_AB * κ_AB`. -/
theorem A3_independence (n : ℕ) :
    kappaAB n * aAB n = -(aAB n * kappaAB n) ∧ kappaAB n * aAB n ≠ 0 := by
  have e1 : kappaAB n * aAB n = (kappa n) ᵍ⊗ₜ[ℚ] (aA0 n) := by
    unfold kappaAB aAB
    rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
      (kappa n) (⟨1, SetLike.one_mem_graded (A0Grading n)⟩ : A0Grading n 0)
      (⟨1, SetLike.one_mem_graded (RGradingQ n)⟩ : RGradingQ n 0) (aA0 n)]
    rw [mul_one, one_mul]
  have e2 : aAB n * kappaAB n =
      (-1 : ℤˣ) ^ ((1 : ZMod 2) * (1 : ZMod 2)) •
        ((kappa n) ᵍ⊗ₜ[ℚ] (aA0 n) : AB n) := by
    unfold kappaAB aAB
    rw [GradedTensorProduct.tmul_coe_mul_coe_tmul (𝒜 := RGradingQ n) (ℬ := A0Grading n)
      (1 : RRing n) (⟨aA0 n, aA0_mem_A0Grading_one n⟩ : A0Grading n 1)
      (⟨kappa n, kappa_mem_RGradingQ_one n⟩ : RGradingQ n 1) (1 : A0 n)]
    congr 2
    · exact one_mul (kappa n)
    · exact mul_one (aA0 n)
  refine ⟨?_, ?_⟩
  · rw [e1, e2]
    norm_num
  · rw [e1]
    intro h
    obtain ⟨f, hf⟩ := Module.Projective.exists_dual_eq_one ℚ (kappa_ne_zero n)
    obtain ⟨g, hg⟩ := Module.Projective.exists_dual_eq_one ℚ (aA0_ne_zero n)
    have hΦ : ((TensorProduct.lid ℚ ℚ).toLinearMap ∘ₗ TensorProduct.map f g)
        ((kappa n) ⊗ₜ[ℚ] (aA0 n)) = 1 := by
      simp [TensorProduct.map_tmul, hf, hg]
    rw [show ((kappa n) ⊗ₜ[ℚ] (aA0 n) : AB n) =
      ((kappa n) ᵍ⊗ₜ[ℚ] (aA0 n) : AB n) from rfl, h] at hΦ
    simp only [LinearMap.comp_apply, map_zero] at hΦ
    exact one_ne_zero hΦ.symm

/-! ## **A4**: `A_0`'s and `A_B`'s gradings restrict correctly to their tensor factors -/

/-- `A4`, for `A_0`: the natural inclusions of `W_n` and `C` into `A_0` respect the gradings —
`w ↦ w ᵍ⊗ₜ 1` sends everything to degree `0` (`W_n` is purely even), and `c ↦ 1 ᵍ⊗ₜ c` sends
`CGrading k` into `A0Grading k` exactly. -/
theorem A4_W_into_A0 (n : ℕ) (w : Module.End ℚ (WPoly n)) :
    (w ᵍ⊗ₜ[ℚ] (1 : C) : A0 n) ∈ A0Grading n 0 :=
  A0Grading_mem_of_tmul n 0 w (⟨1, SetLike.one_mem_graded CGrading⟩ : CGrading 0)

theorem A4_C_into_A0 (n : ℕ) (k : ZMod 2) (c : CGrading k) :
    ((1 : Module.End ℚ (WPoly n)) ᵍ⊗ₜ[ℚ] (c : C) : A0 n) ∈ A0Grading n k :=
  A0Grading_mem_of_tmul n k 1 c

/-- `A4`, for `A_B`: the natural inclusions of `R` and `A_0` into `A_B` respect `A_B`'s *own*
`ZMod 2` grading (`ABGrading`, a genuine total-degree grading built from `RGradingQ` and
`A0Grading` together — see `ABGradedAlgebra` — not merely "some ad-hoc range containing the
image"). `r ↦ r ᵍ⊗ₜ 1` sends `RGradingQ n k` into `ABGrading n k` (the `A_0`-side contributes
degree `0`), and `x ↦ 1 ᵍ⊗ₜ x` sends `A0Grading n k` into `ABGrading n k` (the `R`-side
contributes degree `0`); both are the `k = k + 0` and `k = 0 + k` special cases of
`ABGrading_mem_of_tmul`. -/
theorem A4_R_into_AB (n : ℕ) (k : ZMod 2) (r : RGradingQ n k) :
    ((r : RRing n) ᵍ⊗ₜ[ℚ] (1 : A0 n) : AB n) ∈ ABGrading n k := by
  have h := ABGrading_mem_of_tmul n k 0 r (⟨1, SetLike.one_mem_graded (A0Grading n)⟩ : A0Grading n 0)
  rwa [add_zero] at h

theorem A4_A0_into_AB (n : ℕ) (k : ZMod 2) (x : A0Grading n k) :
    ((1 : RRing n) ᵍ⊗ₜ[ℚ] (x : A0 n) : AB n) ∈ ABGrading n k := by
  have h := ABGrading_mem_of_tmul n 0 k (⟨1, SetLike.one_mem_graded (RGradingQ n)⟩ : RGradingQ n 0) x
  rwa [zero_add] at h

end Source
end InhomogeneousDeformations
