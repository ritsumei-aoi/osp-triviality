import InhomogeneousDeformations.IndexedU
import Mathlib.Algebra.MvPolynomial.PDeriv
import Mathlib.Algebra.Module.LinearMap.Defs

/-!
# I106 R2, W1 — `W_n` as operators on `MvPolynomial (Fin n) ℚ`

**The open choice, resolved**: `W_n` is realized as `Module.End ℚ (WPoly n)`, the even
generators `B_u` as multiplication and differentiation operators, `Module.End`'s own
composition standing for the algebra product. Rejected: a free-algebra-modulo-relations
model (`FreeAlgebra`/`RingQuot`) — mathlib has no Weyl algebra to quotient toward, and the
same `RingQuot`-normalization stall F2 (R0) already named for the source algebra applies
here too, with no offsetting benefit; the operator model instead turns the one relation
that matters (`eq:source`'s `[b_u,b_v]=J_{uv}1`) into a *computation*
(`MvPolynomial.pderiv_mul`/`pderiv_X`), not a postulate.

**A1**: `[B_u, B_v] = Jn n u v • 1`, `Jn` **imported unchanged from the frozen
`Indexed.lean`**, proved below by direct computation, case-split on the even/odd shape of
`u` and `v` — never a fresh matrix.
-/

namespace InhomogeneousDeformations
namespace Source

/-- The `n` polynomial variables `W_n` acts on — **not** `Indexed.lean`'s `Pn n` (which has
`2n` variables `beta_1,...,beta_{2n}` and a different role: the coordinate ring of the target
side, not the operator domain here). -/
noncomputable abbrev WPoly (n : ℕ) : Type := MvPolynomial (Fin n) ℚ

/-- The index `j : Fin n` a source index `u : Fin (2n)` corresponds to: `u = 2j` or `u = 2j+1`.
Matches `Indexed.lean`'s own `evenIdx`/`oddIdx` inverse (`u.val / 2`), reused rather than
redefined. -/
noncomputable def wIndex (n : ℕ) (u : Fin (2 * n)) : Fin n :=
  ⟨(u : ℕ) / 2, by have := u.isLt; omega⟩

theorem wIndex_evenIdx (n : ℕ) (j : Fin n) : wIndex n (Indexed.evenIdx n j) = j := by
  apply Fin.ext
  unfold wIndex Indexed.evenIdx
  simp

theorem wIndex_oddIdx (n : ℕ) (j : Fin n) : wIndex n (Indexed.oddIdx n j) = j := by
  apply Fin.ext
  unfold wIndex Indexed.oddIdx
  simp
  omega

/-- `B_u`: differentiation (`pderiv`) at the even indices, multiplication by the
corresponding variable at the odd indices — the assignment `B_{2j} = \partial_j`,
`B_{2j+1} = X_j` fixed by matching `Jn n (evenIdx n j) (oddIdx n j) = 1` (checked below,
`Jn_B_even_odd`) against `[\partial_j, X_j\cdot] = 1` (a computation, `pderiv_X_self`). -/
noncomputable def B (n : ℕ) (u : Fin (2 * n)) : Module.End ℚ (WPoly n) :=
  if (u : ℕ) % 2 = 0 then (MvPolynomial.pderiv (wIndex n u)).toLinearMap
  else LinearMap.mulLeft ℚ (MvPolynomial.X (wIndex n u))

theorem B_evenIdx (n : ℕ) (j : Fin n) :
    B n (Indexed.evenIdx n j) = (MvPolynomial.pderiv j).toLinearMap := by
  unfold B
  rw [if_pos (by unfold Indexed.evenIdx; simp), wIndex_evenIdx]

theorem B_oddIdx (n : ℕ) (j : Fin n) :
    B n (Indexed.oddIdx n j) = LinearMap.mulLeft ℚ (MvPolynomial.X j) := by
  unfold B
  rw [if_neg (by unfold Indexed.oddIdx; simp), wIndex_oddIdx]

/-- The commutator of two operators on `WPoly n`, as an element of `Module.End ℚ (WPoly n)`
(a `Ring`, composition as multiplication). -/
noncomputable def opComm (f g : Module.End ℚ (WPoly n)) : Module.End ℚ (WPoly n) := f * g - g * f

theorem pderiv_mulLeft_comm (n : ℕ) (i j : Fin n) :
    opComm (MvPolynomial.pderiv i).toLinearMap (LinearMap.mulLeft ℚ (MvPolynomial.X j))
      = (if i = j then (1 : ℚ) else 0) • (1 : Module.End ℚ (WPoly n)) := by
  unfold opComm
  apply LinearMap.ext; intro f
  simp only [LinearMap.sub_apply, Module.End.mul_apply, LinearMap.mulLeft_apply,
    Derivation.coeFn_coe]
  rw [MvPolynomial.pderiv_mul]
  by_cases hij : i = j
  · subst hij
    rw [MvPolynomial.pderiv_X_self]
    simp
  · rw [MvPolynomial.pderiv_X_of_ne (Ne.symm hij), if_neg hij]
    simp

/-- Mixed partial derivatives commute, for every polynomial (not just monomials) — proved by
`MvPolynomial.induction_on` (the `C`/`add`/`mul_X` cases), since mathlib does not supply this
directly. -/
theorem pderiv_pderiv_X_zero (n : ℕ) (i j k : Fin n) :
    MvPolynomial.pderiv i (MvPolynomial.pderiv j (MvPolynomial.X k) : WPoly n) = 0 := by
  by_cases hjk : j = k
  · subst hjk; simp
  · rw [MvPolynomial.pderiv_X_of_ne (Ne.symm hjk)]; simp

theorem pderiv_pderiv_comm (n : ℕ) (i j : Fin n) (p : WPoly n) :
    MvPolynomial.pderiv i (MvPolynomial.pderiv j p)
      = MvPolynomial.pderiv j (MvPolynomial.pderiv i p) := by
  induction p using MvPolynomial.induction_on with
  | C a => simp
  | add p q hp hq => simp [hp, hq]
  | mul_X p k hp =>
    simp only [MvPolynomial.pderiv_mul, map_add, pderiv_pderiv_X_zero]
    rw [hp]
    ring

theorem opComm_pderiv_pderiv (n : ℕ) (i j : Fin n) :
    opComm ((MvPolynomial.pderiv i).toLinearMap : Module.End ℚ (WPoly n))
        (MvPolynomial.pderiv j).toLinearMap = 0 := by
  unfold opComm
  apply LinearMap.ext; intro f
  simp only [LinearMap.sub_apply, Module.End.mul_apply, Derivation.coeFn_coe,
    LinearMap.zero_apply]
  rw [pderiv_pderiv_comm, sub_self]

theorem opComm_mulLeft_mulLeft (n : ℕ) (i j : Fin n) :
    opComm (LinearMap.mulLeft ℚ (MvPolynomial.X i) : Module.End ℚ (WPoly n))
        (LinearMap.mulLeft ℚ (MvPolynomial.X j)) = 0 := by
  unfold opComm
  apply LinearMap.ext; intro f
  simp only [LinearMap.sub_apply, Module.End.mul_apply, LinearMap.mulLeft_apply,
    LinearMap.zero_apply]
  rw [show MvPolynomial.X i * (MvPolynomial.X j * f) = MvPolynomial.X j * (MvPolynomial.X i * f)
      from by ring, sub_self]

theorem opComm_swap (f g : Module.End ℚ (WPoly n)) : opComm f g = -opComm g f := by
  unfold opComm; abel

theorem mulLeft_pderiv_comm (n : ℕ) (i j : Fin n) :
    opComm (LinearMap.mulLeft ℚ (MvPolynomial.X j) : Module.End ℚ (WPoly n))
        (MvPolynomial.pderiv i).toLinearMap
      = -((if i = j then (1 : ℚ) else 0) • (1 : Module.End ℚ (WPoly n))) := by
  rw [opComm_swap, pderiv_mulLeft_comm]

theorem u_eq_evenIdx_of_even (n : ℕ) (u : Fin (2 * n)) (h : (u : ℕ) % 2 = 0) :
    u = Indexed.evenIdx n (wIndex n u) := by
  apply Fin.ext; unfold Indexed.evenIdx wIndex; dsimp only; omega

theorem u_eq_oddIdx_of_odd (n : ℕ) (u : Fin (2 * n)) (h : ¬ (u : ℕ) % 2 = 0) :
    u = Indexed.oddIdx n (wIndex n u) := by
  apply Fin.ext; unfold Indexed.oddIdx wIndex; dsimp only; omega

/-- `Jn`'s value as a plain `ℚ` scalar — `Jn` only ever takes the constant values `1`, `-1` or
`0` (`cratN n c = MvPolynomial.C c`), so `constantCoeff` recovers exactly that value; `Jn`
itself appears literally as the argument, not redefined. -/
noncomputable def JnQ (n : ℕ) (u v : Fin (2 * n)) : ℚ :=
  MvPolynomial.constantCoeff (Indexed.Jn n u v)

theorem JnQ_eq_zero_of_same_parity (n : ℕ) (u v : Fin (2 * n))
    (h : (u : ℕ) % 2 = (v : ℕ) % 2) : JnQ n u v = 0 := by
  unfold JnQ Indexed.Jn
  rw [if_neg (by omega), if_neg (by omega)]
  simp

theorem JnQ_evenIdx_oddIdx (n : ℕ) (i j : Fin n) :
    JnQ n (Indexed.evenIdx n i) (Indexed.oddIdx n j) = if i = j then 1 else 0 := by
  unfold JnQ Indexed.Jn Indexed.evenIdx Indexed.oddIdx
  dsimp only
  by_cases hij : i = j
  · subst hij; rw [if_pos (by omega)]; simp
  · rw [if_neg (by omega), if_neg (by omega)]
    simp [hij]

theorem JnQ_oddIdx_evenIdx (n : ℕ) (i j : Fin n) :
    JnQ n (Indexed.oddIdx n i) (Indexed.evenIdx n j) = if i = j then (-1 : ℚ) else 0 := by
  unfold JnQ Indexed.Jn Indexed.evenIdx Indexed.oddIdx
  dsimp only
  by_cases hij : i = j
  · subst hij; rw [if_neg (by omega), if_pos (by omega)]
    unfold Indexed.cratN; simp
  · rw [if_neg (by omega), if_neg (by omega)]
    simp [hij]

/-- **A1**: `[B_u, B_v] = Jn n u v • 1`, `Jn` imported unchanged from the frozen
`Indexed.lean` and appearing literally in the statement (via `JnQ`, its constant-coefficient
value). Proved by a four-way case split on the even/odd shape of `u` and `v` (never a fresh
matrix), citing the four commutator facts above and matching each against `Jn`'s own case
split. -/
theorem B_comm (n : ℕ) (u v : Fin (2 * n)) :
    opComm (B n u) (B n v) = JnQ n u v • (1 : Module.End ℚ (WPoly n)) := by
  by_cases hu : (u : ℕ) % 2 = 0 <;> by_cases hv : (v : ℕ) % 2 = 0
  · -- both even: [B_u,B_v] = [pderiv,pderiv] = 0 = JnQ (same parity)
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
    rw [B_evenIdx, B_evenIdx, opComm_pderiv_pderiv,
      JnQ_eq_zero_of_same_parity n _ _ (by omega), zero_smul]
  · -- u even, v odd: the substantive sector
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.evenIdx n i := ⟨wIndex n u, u_eq_evenIdx_of_even n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
    rw [B_evenIdx, B_oddIdx, pderiv_mulLeft_comm, JnQ_evenIdx_oddIdx]
  · -- u odd, v even
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.evenIdx n j := ⟨wIndex n v, u_eq_evenIdx_of_even n v hv⟩
    rw [B_oddIdx, B_evenIdx, mulLeft_pderiv_comm, JnQ_oddIdx_evenIdx]
    by_cases hij : i = j
    · subst hij
      rw [if_pos rfl, if_pos rfl]
      apply LinearMap.ext; intro x
      simp
    · rw [if_neg hij, if_neg (Ne.symm hij)]; simp
  · -- both odd: [B_u,B_v] = [mulLeft,mulLeft] = 0 = JnQ (same parity)
    obtain ⟨i, rfl⟩ : ∃ i, u = Indexed.oddIdx n i := ⟨wIndex n u, u_eq_oddIdx_of_odd n u hu⟩
    obtain ⟨j, rfl⟩ : ∃ j, v = Indexed.oddIdx n j := ⟨wIndex n v, u_eq_oddIdx_of_odd n v hv⟩
    rw [B_oddIdx, B_oddIdx, opComm_mulLeft_mulLeft,
      JnQ_eq_zero_of_same_parity n _ _ (by omega), zero_smul]

end Source
end InhomogeneousDeformations
