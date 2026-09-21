import InhomogeneousDeformations.SourceTensor

/-!
# I106 R2, W3 — the undeformed lifts `L^0_{uv}` and `F^0_u`, in `A_0`

Transcription (manuscript lines 128-131):
```
L^0_{uv} = (1/4)(B_u B_v + B_v B_u),   F^0_u = (1/2) a B_u
```
in `A_0`. **The undeformed ones only** — `\widehat L_{uv}`, `\widehat F_u` live in `A_\beta`
(the *deformed* source) and are out of scope for this round (W2's own scope note: "`A_β` is
out of scope — it is the deformed source, related to `A_B` through `Φ`, which is R3. Do not
define it.").

`B_u` is embedded into `A_0` as `B_u ᵍ⊗ₜ 1` (`Bu0`); `a` is already embedded as `aA0` in
`SourceTensor.lean`. `L^0_{uv}` lives entirely in `A0Grading n 0` (both factors are
`B`-products, degree `0` in `C`); `F^0_u` lives entirely in `A0Grading n 1` (the single factor
`a`, degree `1`, contributes the whole `C`-degree) — recorded below since it is the exact
"different `A_0`-degree" fact `W4`'s independence argument turns on.

Manuscript line 105-106 ("Since `W_n` is purely even, `aB_u = B_u a` in `A_0` and `A_B`") is
transcribed literally as `a_comm_Bu0` below — not needed to *define* `F^0_u`, but recorded
because it is the manuscript's own stated reason the order in `F^0_u = (1/2) a B_u` does not
matter, and it is a one-line consequence of the same `tmul_coe_mul_zero_coe_tmul` fact `A_0`'s
whole construction rests on.
-/

namespace InhomogeneousDeformations
namespace Source

/-- `B_u`, embedded into `A_0` as `B_u ᵍ⊗ₜ 1`. -/
noncomputable def Bu0 (n : ℕ) (u : Fin (2 * n)) : A0 n :=
  (B n u) ᵍ⊗ₜ[ℚ] (1 : C)

theorem Bu0_mem_A0Grading_zero (n : ℕ) (u : Fin (2 * n)) : Bu0 n u ∈ A0Grading n 0 :=
  A0Grading_mem_of_tmul n 0 (B n u) (⟨1, SetLike.one_mem_graded CGrading⟩ : CGrading 0)

/-- Manuscript line 105-106, transcribed literally: "Since `W_n` is purely even, `aB_u = B_u a`
in `A_0`" — both sides equal `(B n u) ᵍ⊗ₜ a` via `tmul_coe_mul_zero_coe_tmul` in either order,
since the `W_n`-side factor of either `1` or `B n u` is always (trivially) of degree `0`. -/
theorem a_comm_Bu0 (n : ℕ) (u : Fin (2 * n)) : aA0 n * Bu0 n u = Bu0 n u * aA0 n := by
  unfold aA0 Bu0
  rw [GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := WGrading n) (ℬ := CGrading)
      (1 : Module.End ℚ (WPoly n)) (⟨Source.a, Source.a_odd⟩ : CGrading 1)
      (⟨B n u, trivial⟩ : WGrading n 0) (1 : C),
    GradedTensorProduct.tmul_coe_mul_zero_coe_tmul (𝒜 := WGrading n) (ℬ := CGrading)
      (B n u) (⟨1, SetLike.one_mem_graded CGrading⟩ : CGrading 0)
      (⟨1, trivial⟩ : WGrading n 0) (Source.a : C)]
  simp

/-- `L^0_{uv} = (1/4)(B_uB_v+B_vB_u)`, in `A_0` (manuscript line 129-130). Manifestly symmetric
in `u`, `v` by construction, matching `\mathfrak g`'s own convention `L_{vu}=L_{uv}`
(manuscript line 133-134). -/
noncomputable def L0 (n : ℕ) (u v : Fin (2 * n)) : A0 n :=
  (1 / 4 : ℚ) • (Bu0 n u * Bu0 n v + Bu0 n v * Bu0 n u)

theorem L0_symm (n : ℕ) (u v : Fin (2 * n)) : L0 n u v = L0 n v u := by
  unfold L0; rw [add_comm]

theorem L0_mem_A0Grading_zero (n : ℕ) (u v : Fin (2 * n)) : L0 n u v ∈ A0Grading n 0 := by
  unfold L0
  have huv : Bu0 n u * Bu0 n v ∈ A0Grading n (0 + 0) :=
    SetLike.mul_mem_graded (Bu0_mem_A0Grading_zero n u) (Bu0_mem_A0Grading_zero n v)
  have hvu : Bu0 n v * Bu0 n u ∈ A0Grading n (0 + 0) :=
    SetLike.mul_mem_graded (Bu0_mem_A0Grading_zero n v) (Bu0_mem_A0Grading_zero n u)
  rw [add_zero] at huv hvu
  exact Submodule.smul_mem _ _ (add_mem huv hvu)

/-- `F^0_u = (1/2) a B_u`, in `A_0` (manuscript line 130-131). -/
noncomputable def F0 (n : ℕ) (u : Fin (2 * n)) : A0 n :=
  (1 / 2 : ℚ) • (aA0 n * Bu0 n u)

theorem F0_mem_A0Grading_one (n : ℕ) (u : Fin (2 * n)) : F0 n u ∈ A0Grading n 1 := by
  unfold F0
  refine Submodule.smul_mem _ _ ?_
  have h1 : aA0 n ∈ A0Grading n 1 := aA0_mem_A0Grading_one n
  have h0 : Bu0 n u ∈ A0Grading n 0 := Bu0_mem_A0Grading_zero n u
  have := SetLike.mul_mem_graded h1 h0
  rwa [add_zero] at this

end Source
end InhomogeneousDeformations
