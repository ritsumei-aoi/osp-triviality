import InhomogeneousDeformations.N1Proofs

/-!
# C2.1 — general homogeneous degree preservation on the retained n=1 model

`C2.1`: for arbitrary `x y : Mod` (not merely basis vectors or single-support
elements) and `dx dy : ZMod 2`, `IsHomog x dx → IsHomog y dy →
IsHomog (bracket x y) (dx + dy)`. Uses only the already-audited basis-level
`bracket_degree0` and the finite-sum bilinear expansion `bracket_bilinear_expand`;
no new axiom, no coefficient-domain assumption, no source injectivity.
-/

namespace InhomogeneousDeformations

/-- `C2.1`: the native bracket preserves homogeneous degree on arbitrary
(not merely basis-supported) module elements. -/
theorem bracket_isHomog (x y : Mod) (dx dy : ZMod 2)
    (hx : IsHomog x dx) (hy : IsHomog y dy) :
    IsHomog (bracket x y) (dx + dy) := by
  intro k hk
  by_contra hne
  apply hk
  rw [bracket_bilinear_expand]
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  apply Finset.sum_eq_zero; intro i _
  apply Finset.sum_eq_zero; intro j _
  by_cases hxi : x i = 0
  · simp [hxi]
  · by_cases hyj : y j = 0
    · simp [hyj]
    · have hpi := hx i hxi
      have hpj := hy j hyj
      have hz : bracket (e i) (e j) k = 0 := by
        by_contra hc
        exact hne (by rw [bracket_degree0 i j k hc, hpi, hpj])
      simp [hz]

end InhomogeneousDeformations
