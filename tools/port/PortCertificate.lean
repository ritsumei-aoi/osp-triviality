import InhomogeneousDeformations

/-!
# Port certificate (Lean v4.34.1)

mathlib `631b214f7c` (#41427) deleted `MvPolynomial.coeff m p`; the new form is `p.coeff m`
(`AddMonoidAlgebra.coeff`). This file re-defines the old name in terms of the new one and
re-states the seven statements of `SourceQuadraticIndependence.lean` at `de410b6` **verbatim**,
each closed by the ported theorem. It compiling shows that each ported statement is the old
statement, reading the old `coeff` as the new projection. Not part of the package.
-/

/-- The removed API, re-defined through the new one (arguments in the old order). -/
@[reducible] def MvPolynomial.coeff {σ : Type*} {R : Type*} [CommSemiring R]
    (m : σ →₀ ℕ) (p : MvPolynomial σ R) : R :=
  -- written with the full name: inside this definition `p.coeff` would resolve to the
  -- definition itself
  AddMonoidAlgebra.coeff p m

namespace InhomogeneousDeformations
namespace Source
namespace PortCertificate

open scoped TensorProduct

/-- `coeff_smul_one_eq_zero`, the statement of `de410b6` verbatim. -/
theorem cert_coeff_smul_one_eq_zero (n : ℕ) (c : ℚ) {m : Fin n →₀ ℕ} (hm : m ≠ 0) :
    MvPolynomial.coeff m (c • (1 : WPoly n)) = 0 :=
  coeff_smul_one_eq_zero n c hm

/-- `coeff_single_add_single_X_mul_X`, the statement of `de410b6` verbatim. -/
theorem cert_coeff_single_add_single_X_mul_X (n : ℕ) (k l i j : Fin n) (hij : i ≤ j) (hkl : k ≤ l)
    (hne : ¬ (i = k ∧ j = l)) :
    MvPolynomial.coeff (Finsupp.single k 1 + Finsupp.single l 1)
      (MvPolynomial.X i * MvPolynomial.X j : WPoly n) = 0 :=
  coeff_single_add_single_X_mul_X n k l i j hij hkl hne

/-- `hval1`, the statement of `de410b6` verbatim. -/
theorem cert_hval1 (n : ℕ) (k l : Fin n) (hkl : k ≤ l)
    (q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}) :
    MvPolynomial.coeff (Finsupp.single k 1 + Finsupp.single l 1)
      ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (1 : WPoly n))
      = if q.1.1 = Indexed.oddIdx n k ∧ q.1.2 = Indexed.oddIdx n l then (2 : ℚ) else 0 :=
  hval1 n k l hkl q

/-- `hval2`, the statement of `de410b6` verbatim. -/
theorem cert_hval2 (n : ℕ) (m p : Fin n) (hmp : m ≠ p)
    (q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}) :
    MvPolynomial.coeff (Finsupp.single p 1)
      ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (MvPolynomial.X m : WPoly n))
      = (if q.1.1 = Indexed.evenIdx n m ∧ q.1.2 = Indexed.oddIdx n p then (2 : ℚ) else 0)
        + (if q.1.1 = Indexed.oddIdx n p ∧ q.1.2 = Indexed.evenIdx n m then 2 else 0) :=
  hval2 n m p hmp q

/-- `hval0`, the statement of `de410b6` verbatim. -/
theorem cert_hval0 (n : ℕ) (q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}) :
    MvPolynomial.coeff 0 ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (1 : WPoly n))
      = if ∃ i : Fin n, q.1.1 = Indexed.evenIdx n i ∧ q.1.2 = Indexed.oddIdx n i then (1 : ℚ)
        else 0 :=
  hval0 n q

/-- `hval3`, the statement of `de410b6` verbatim. -/
theorem cert_hval3 (n : ℕ) (k : Fin n) (q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}) :
    MvPolynomial.coeff (Finsupp.single k 1)
      ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1) (MvPolynomial.X k : WPoly n))
      = if q.1.1 = Indexed.evenIdx n k ∧ q.1.2 = Indexed.oddIdx n k then (3 : ℚ)
        else if ∃ i : Fin n, q.1.1 = Indexed.evenIdx n i ∧ q.1.2 = Indexed.oddIdx n i then 1
        else 0 :=
  hval3 n k q

/-- `hval4`, the statement of `de410b6` verbatim. -/
theorem cert_hval4 (n : ℕ) (k l : Fin n) (hkl : k ≤ l)
    (q : {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2}) :
    MvPolynomial.coeff 0
      ((B n q.1.1 * B n q.1.2 + B n q.1.2 * B n q.1.1)
        (MvPolynomial.X k * MvPolynomial.X l : WPoly n))
      = if q.1.1 = Indexed.evenIdx n k ∧ q.1.2 = Indexed.evenIdx n l then
          (if k = l then (4 : ℚ) else 2) else 0 :=
  hval4 n k l hkl q

end PortCertificate
end Source
end InhomogeneousDeformations

#print axioms InhomogeneousDeformations.Source.PortCertificate.cert_coeff_smul_one_eq_zero
#print axioms InhomogeneousDeformations.Source.PortCertificate.cert_coeff_single_add_single_X_mul_X
#print axioms InhomogeneousDeformations.Source.PortCertificate.cert_hval1
#print axioms InhomogeneousDeformations.Source.PortCertificate.cert_hval2
#print axioms InhomogeneousDeformations.Source.PortCertificate.cert_hval0
#print axioms InhomogeneousDeformations.Source.PortCertificate.cert_hval3
#print axioms InhomogeneousDeformations.Source.PortCertificate.cert_hval4
