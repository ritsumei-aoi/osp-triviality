import InhomogeneousDeformations.IndexedU
import Mathlib.LinearAlgebra.CliffordAlgebra.Basic
import Mathlib.LinearAlgebra.CliffordAlgebra.Grading

/-!
# I106 R2, W0 — `C = Q[a]/(a^2 - 1/2)`, `a` odd

Transcription: `C = \mathbb Q[a]/(a^2-\tfrac12)` (manuscript lines 98-101). Built as
`CliffordAlgebra Qhalf` with `Qhalf : QuadraticForm ℚ ℚ`, `Qhalf x = x^2/2` — the same
construction R0's F1 experiment used (there with the zero form) and Agent1c recompiled,
now with the actual quadratic form the manuscript needs. `CliffordAlgebra.ι_sq_scalar`
gives `a^2 = 1/2` **by the library**, not by a fresh axiom or computation of our own.

**A2**: `a^2 = 1/2` and `a` odd, both proved below.
-/

namespace InhomogeneousDeformations
namespace Source

/-- `Q(x) = x^2/2`, the quadratic form whose Clifford algebra is `C`. -/
noncomputable def Qhalf : QuadraticForm ℚ ℚ := (1 / 2 : ℚ) • QuadraticMap.sq

/-- `C = Q[a]/(a^2 - 1/2)`, built as `CliffordAlgebra Qhalf`. -/
noncomputable abbrev C : Type := CliffordAlgebra Qhalf

/-- The odd generator `a`. -/
noncomputable def a : C := CliffordAlgebra.ι Qhalf 1

theorem Qhalf_apply (x : ℚ) : Qhalf x = x ^ 2 / 2 := by
  simp [Qhalf, QuadraticMap.sq_apply]
  ring

/-- **A2, first half**: `a^2 = 1/2 * 1` — supplied directly by the library
(`CliffordAlgebra.ι_sq_scalar`), not a fresh computation. -/
theorem a_sq : a * a = (1 / 2 : ℚ) • (1 : C) := by
  have h := CliffordAlgebra.ι_sq_scalar Qhalf (1 : ℚ)
  rw [Qhalf_apply] at h
  rw [show ((1 : ℚ) ^ 2 / 2) = (1 / 2 : ℚ) from by ring] at h
  rw [Algebra.algebraMap_eq_smul_one] at h
  exact h

/-- **A2, second half**: `a` is odd, i.e. lies in `CliffordAlgebra.evenOdd Qhalf 1` — via the
library's own `ι_mem_evenOdd_one`. -/
theorem a_odd : a ∈ CliffordAlgebra.evenOdd Qhalf 1 :=
  CliffordAlgebra.ι_mem_evenOdd_one Qhalf 1

/-- `C`'s `ZMod 2` grading is `CliffordAlgebra.evenOdd Qhalf`, an instance the library already
supplies (`CliffordAlgebra.gradedAlgebra`) — noted here under a local name for W2 to cite. -/
noncomputable abbrev CGrading : ZMod 2 → Submodule ℚ C := CliffordAlgebra.evenOdd Qhalf

noncomputable instance CGrading_gradedAlgebra : GradedAlgebra CGrading :=
  CliffordAlgebra.gradedAlgebra Qhalf

end Source
end InhomogeneousDeformations
