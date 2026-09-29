import Mathlib.Algebra.MvPolynomial.Basic
import Mathlib.Algebra.MvPolynomial.CommRing
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Sum
import Mathlib.Data.Prod.Lex
import Mathlib.Order.Basic

/-!
# C1.1 — genuinely rank-indexed native family definitions

For each `n : ℕ`, `I_n = Fin (2*n)` (mathematical index `u ∈ {1,...,2n}` maps
to zero-based `u-1`), `P_n = Q[beta_0,...,beta_(2n-1)]`. `IndexedBasis n` is
the disjoint union of even sorted pairs `(u,v)` with `u ≤ v` and odd indices
`u`; parity `0` for even pairs, `1` for odd indices. `J_n` uses natural
(non-modular) representative indices. The bracket is defined by the
manuscript's rank-indexed formulas directly (not via the old table/oracle),
extended `P_n`-bilinearly by finite coefficient sums over the whole module.
No proof at general rank is attempted here (`n=1` alone is proved, in
`N1Specialization.lean`); `n=0` is an admissible empty implementation case
with no rank-zero scientific claim.
-/

namespace InhomogeneousDeformations
namespace Indexed

/-- `P_n = Q[beta_0,...,beta_(2n-1)]`. -/
abbrev Pn (n : ℕ) : Type := MvPolynomial (Fin (2 * n)) ℚ

noncomputable def cratN (n : ℕ) (q : ℚ) : Pn n := MvPolynomial.C q

@[simp] lemma cratN_zero (n : ℕ) : cratN n (0 : ℚ) = 0 := by simp [cratN]
@[simp] lemma cratN_one (n : ℕ) : cratN n (1 : ℚ) = 1 := by simp [cratN]
@[simp] lemma cratN_add (n : ℕ) (a b : ℚ) : cratN n a + cratN n b = cratN n (a + b) := by
  simp [cratN]
@[simp] lemma cratN_mul (n : ℕ) (a b : ℚ) : cratN n a * cratN n b = cratN n (a * b) := by
  simp [cratN]
@[simp] lemma cratN_neg (n : ℕ) (a : ℚ) : -cratN n a = cratN n (-a) := by simp [cratN]

/-- The genuinely rank-indexed basis: the disjoint union of even sorted pairs
`(u,v)`, `u ≤ v`, and odd indices `u`, both ranging over `I_n = Fin (2n)`. -/
def IndexedBasis (n : ℕ) : Type :=
  {p : Fin (2 * n) × Fin (2 * n) // p.1 ≤ p.2} ⊕ Fin (2 * n)

instance (n : ℕ) : DecidableEq (IndexedBasis n) := by
  unfold IndexedBasis; infer_instance

instance (n : ℕ) : Fintype (IndexedBasis n) := by
  unfold IndexedBasis; infer_instance

/-- Parity: `0` on even pairs, `1` on odd indices. -/
def parity {n : ℕ} : IndexedBasis n → ZMod 2
  | .inl _ => 0
  | .inr _ => 1

/-- `L(u,v)`: sorts the two indices, no sign. -/
def Lof {n : ℕ} (u v : Fin (2 * n)) : IndexedBasis n :=
  if h : u ≤ v then Sum.inl ⟨(u, v), h⟩
  else Sum.inl ⟨(v, u), (not_le.mp h).le⟩

def Fof {n : ℕ} (u : Fin (2 * n)) : IndexedBasis n := Sum.inr u

/-- `J_n`, natural (non-modular) representative indices:
`+1` when `u` is even and `v = u+1`; `-1` when `v` is even and `u = v+1`;
`0` otherwise. -/
noncomputable def Jn (n : ℕ) (u v : Fin (2 * n)) : Pn n :=
  if (u : ℕ) % 2 = 0 ∧ (v : ℕ) = (u : ℕ) + 1 then cratN n 1
  else if (v : ℕ) % 2 = 0 ∧ (u : ℕ) = (v : ℕ) + 1 then cratN n (-1)
  else cratN n 0

abbrev IndexedMod (n : ℕ) : Type := IndexedBasis n → Pn n

@[simp] lemma indexedMod_zero_apply {n : ℕ} (k : IndexedBasis n) :
    (0 : IndexedMod n) k = 0 := rfl
@[simp] lemma indexedMod_add_apply {n : ℕ} (x y : IndexedMod n) (k : IndexedBasis n) :
    (x + y) k = x k + y k := rfl
@[simp] lemma indexedMod_smul_apply {n : ℕ} (c : Pn n) (x : IndexedMod n) (k : IndexedBasis n) :
    (c • x) k = c * x k := rfl
@[simp] lemma indexedMod_neg_apply {n : ℕ} (x : IndexedMod n) (k : IndexedBasis n) :
    (-x) k = -x k := rfl

noncomputable def eN {n : ℕ} (b : IndexedBasis n) : IndexedMod n :=
  fun k => if k = b then 1 else 0

noncomputable def bracketFFn (n : ℕ) (u v : Fin (2 * n)) : IndexedMod n :=
  (cratN n (1 / 2)) • eN (Lof u v)

noncomputable def bracketLFn (n : ℕ) (u v w : Fin (2 * n)) : IndexedMod n :=
  (cratN n (1 / 2) * Jn n v w) • eN (Fof u) + (cratN n (1 / 2) * Jn n u w) • eN (Fof v)

noncomputable def bracketLLn (n : ℕ) (u v w z : Fin (2 * n)) : IndexedMod n :=
  (cratN n (1 / 2) * Jn n v w) • eN (Lof u z) + (cratN n (1 / 2) * Jn n u w) • eN (Lof v z)
  + (cratN n (1 / 2) * Jn n v z) • eN (Lof u w) + (cratN n (1 / 2) * Jn n u z) • eN (Lof v w)

/-- Structure constants on all ordered basis pairs, from the rank-indexed
formulas: `[F_u,F_v]_n=(1/2)L_uv`, `[L_uv,F_w]_n=(1/2)(J_vw F_u+J_uw F_v)`,
`[L_uv,L_wz]_n=(1/2)(J_vw L_uz+J_uw L_vz+J_vz L_uw+J_uz L_vw)`,
`[F_w,L_uv]_n=-[L_uv,F_w]_n`. -/
noncomputable def bracketBasisN (n : ℕ) : IndexedBasis n → IndexedBasis n → IndexedMod n
  | .inl ⟨(u, v), _⟩, .inl ⟨(w, z), _⟩ => bracketLLn n u v w z
  | .inl ⟨(u, v), _⟩, .inr w => bracketLFn n u v w
  | .inr w, .inl ⟨(u, v), _⟩ => -bracketLFn n u v w
  | .inr u, .inr v => bracketFFn n u v

/-- `C1.1`: the total `P_n`-bilinear bracket, by finite coefficient sums over
the whole indexed module, for every rank `n`. -/
noncomputable def bracketN (n : ℕ) (x y : IndexedMod n) : IndexedMod n :=
  ∑ i : IndexedBasis n, ∑ j : IndexedBasis n, (x i * y j) • bracketBasisN n i j

/-! ## C1 (correction002): explicit canonical indexed basis order

An injective key into the lexicographic product `ℕ ×ₗ ℕ ×ₗ ℕ`, pulled back
to a genuine `LinearOrder (IndexedBasis n)` via `LinearOrder.lift'`. The key
places every even pair (tag `0`) before every odd index (tag `1`), orders
even pairs lexicographically by `(u.val, v.val)`, and orders odd indices
increasingly by `u.val`. The key and the order it induces are defined
purely in terms of `IndexedBasis n`'s own `Fin`/`Subtype`/`Sum` structure --
neither mentions `Pn n`, `bracketN`, or any coefficient -- so independence
from coefficients and bracket computation holds definitionally, by the type
signature alone, not as a separate theorem to prove. -/

/-- The canonical order key: `0` tag + `(u.val, v.val)` for even pairs (so
these precede everything with tag `1`, and are lexicographically ordered
among themselves), `1` tag + `(u.val, 0)` for odd indices (so these are
ordered increasingly by `u.val`). -/
def orderKey {n : ℕ} (p : IndexedBasis n) : ℕ ×ₗ ℕ ×ₗ ℕ :=
  match p with
  | .inl ⟨(u, v), _⟩ => toLex (0, toLex (u.val, v.val))
  | .inr u => toLex (1, toLex (u.val, 0))

theorem orderKey_injective {n : ℕ} : Function.Injective (orderKey (n := n)) := by
  intro p q hpq
  match p, q with
  | .inl ⟨(u1, v1), h1⟩, .inl ⟨(u2, v2), h2⟩ =>
    simp only [orderKey, toLex_inj, Prod.mk.injEq] at hpq
    have hu' : u1 = u2 := Fin.ext (by omega)
    have hv' : v1 = v2 := Fin.ext (by omega)
    subst hu'; subst hv'; rfl
  | .inl ⟨(u1, v1), h1⟩, .inr u2 =>
    exfalso
    simp only [orderKey, toLex_inj, Prod.mk.injEq] at hpq
    omega
  | .inr u1, .inl ⟨(u2, v2), h2⟩ =>
    exfalso
    simp only [orderKey, toLex_inj, Prod.mk.injEq] at hpq
    omega
  | .inr u1, .inr u2 =>
    simp only [orderKey, toLex_inj, Prod.mk.injEq] at hpq
    have hu' : u1 = u2 := Fin.ext (by omega)
    subst hu'; rfl

/-- `C1`: the canonical order on `IndexedBasis n`, pulled back from the key
above. -/
noncomputable instance instLinearOrderIndexedBasis {n : ℕ} : LinearOrder (IndexedBasis n) :=
  LinearOrder.lift' orderKey orderKey_injective

/-- `C1`: every even pair precedes every odd index. -/
theorem orderKey_even_lt_odd {n : ℕ} (p : {pr : Fin (2 * n) × Fin (2 * n) // pr.1 ≤ pr.2})
    (w : Fin (2 * n)) : orderKey (n := n) (Sum.inl p) < orderKey (n := n) (Sum.inr w) := by
  obtain ⟨⟨u, v⟩, h⟩ := p
  simp only [orderKey, Prod.Lex.toLex_lt_toLex]
  omega

/-- `C1`: even pairs are ordered lexicographically by `(u.val, v.val)`
(disjunctive form: first by `u.val`, then, when equal, by `v.val`). -/
theorem orderKey_even_lex {n : ℕ} (u1 v1 u2 v2 : Fin (2 * n)) (h1 : u1 ≤ v1) (h2 : u2 ≤ v2) :
    orderKey (n := n) (Sum.inl ⟨(u1, v1), h1⟩) < orderKey (n := n) (Sum.inl ⟨(u2, v2), h2⟩)
      ↔ u1.val < u2.val ∨ (u1.val = u2.val ∧ v1.val < v2.val) := by
  simp [orderKey, Prod.Lex.toLex_lt_toLex]

/-- `C1`: odd indices are ordered increasingly by `u.val`. -/
theorem orderKey_odd_lt {n : ℕ} (u1 u2 : Fin (2 * n)) :
    orderKey (n := n) (Sum.inr u1) < orderKey (n := n) (Sum.inr u2) ↔ u1.val < u2.val := by
  simp [orderKey, Prod.Lex.toLex_lt_toLex]

end Indexed
end InhomogeneousDeformations
