import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Fintype.Basic

/-!
# n=1 basis

Five-element basis for the n=1 pilot: `L11, L12, L22` (even, degree 0) and
`F1, F2` (odd, degree 1). This directly names the module basis fixed by
PILOT_CONTRACT.md §1/§2 and I105_R1A_N1_ORACLE_AND_TESTS.md §1.
-/

namespace InhomogeneousDeformations

inductive Basis5 : Type
  | L11 : Basis5
  | L12 : Basis5
  | L22 : Basis5
  | F1  : Basis5
  | F2  : Basis5
  deriving DecidableEq, Repr

namespace Basis5

instance : Fintype Basis5 where
  elems := {L11, L12, L22, F1, F2}
  complete := by intro x; cases x <;> decide

/-- Z/2 parity: `0` for `L11,L12,L22`, `1` for `F1,F2`, matching the
stated degrees `0,0,0,1,1`. -/
def parity : Basis5 → ZMod 2
  | L11 => 0
  | L12 => 0
  | L22 => 0
  | F1 => 1
  | F2 => 1

end Basis5

end InhomogeneousDeformations
