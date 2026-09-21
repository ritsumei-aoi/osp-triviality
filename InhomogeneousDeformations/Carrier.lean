import Mathlib.Algebra.MvPolynomial.Basic
import Mathlib.Algebra.MvPolynomial.CommRing

/-!
# Coefficient ring P = Q[beta1, beta2]

`Coeff` is the polynomial ring over the two even parameters, used only by
the structured-decoder layer (T5-T7). The n=1 native bracket structure
constants (T0-T4) are themselves parameter-free rationals (see Native.lean),
matching I105_R1A_N1_ORACLE_AND_TESTS.md's remark that "the table has no
parameter-dependent coefficients."
-/

namespace InhomogeneousDeformations

/-- `P = Q[beta1, beta2]`, variables indexed by `Fin 2` (`0 = beta1`, `1 = beta2`). -/
abbrev Coeff : Type := MvPolynomial (Fin 2) ℚ

noncomputable def beta1 : Coeff := MvPolynomial.X 0
noncomputable def beta2 : Coeff := MvPolynomial.X 1

/-- Rational-constant embedding into `Coeff`, used uniformly as the `Coeff`
scalar for structure constants (avoids mixing `ℚ`-smul with `Coeff`-smul). -/
noncomputable def crat (q : ℚ) : Coeff := MvPolynomial.C q

end InhomogeneousDeformations
