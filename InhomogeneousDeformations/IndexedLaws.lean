import InhomogeneousDeformations.Indexed

/-!
# R2-C-1 (U0-U3) — general-rank interface, degree law and super-skew

`H1.2`: every declaration here is stated for general `n : ℕ`, with indices
in `Fin (2 * n)`; no `n = 1` specialization, `Fin 2` instance, or literal
five-element basis appears anywhere. `Indexed.lean` (frozen) is never
reopened; every law here is a genuinely new statement/proof about its
already-accepted definitions.

`H1.3` (verbatim, do not upgrade): a successful R2-C-1 establishes, for
every `n ≥ 1`, that the frozen indexed bracket satisfies the degree law
and super-skew symmetry -- half of P1. It does NOT establish super-Jacobi
(U4/U5, R2-C-2) or the oscillator realization correspondence (P1's own
declared non-goal).
-/

namespace InhomogeneousDeformations
namespace Indexed

/-! ## U0 — `Jn`/`Lof` interface at general `n`

Transposed from the `n=1` route Agent1 specified: antisymmetry follows by
case analysis on the two mutually-exclusive decidable branch conditions,
introducing no term containing `2 * n` as a factor (the omega-nonlinearity
trap from correction002). -/

/-- `Jn`'s vanishing diagonal: `J_{uu} = 0`. Both of `Jn`'s branch
conditions reduce, at `v = u`, to the impossible `u = u + 1`. -/
theorem Jn_diag (n : ℕ) (u : Fin (2 * n)) : Jn n u u = 0 := by
  have h : (u : ℕ) ≠ (u : ℕ) + 1 := by omega
  simp [Jn, h]

/-- `Jn` antisymmetry: `J_{vu} = -J_{uv}`. Decided purely by which of the
two mutually-exclusive branch conditions (`u` even & `v = u+1`, or `v` even
& `u = v+1`) holds -- never by arithmetic in `n`. -/
theorem Jn_antisymm (n : ℕ) (u v : Fin (2 * n)) : Jn n v u = -(Jn n u v) := by
  unfold Jn
  split_ifs <;>
    first
    | rfl
    | (exfalso; omega)
    | (rw [cratN_neg] <;> norm_num)

/-- `Lof` commutativity: `L_{uv} = L_{vu}`. A case split on `u`'s and `v`'s
relative order, closed by antisymmetry of `≤` on `Fin`, `Subtype`/`Prod`
proof-irrelevance closing the matching-order cases directly. -/
theorem Lof_comm {n : ℕ} (u v : Fin (2 * n)) : Lof u v = Lof v u := by
  rcases eq_or_ne u v with heq | hne
  · subst heq; rfl
  · rcases lt_or_gt_of_ne hne with hlt | hgt
    · unfold Lof; rw [dif_pos hlt.le, dif_neg (not_le.mpr hlt)]
    · unfold Lof; rw [dif_pos hgt.le, dif_neg (not_le.mpr hgt)]

/-- Coordinate lemma: `Lof`'s value determines, and is determined by, the
unordered pair of its two arguments -- lets later proofs compare
`eN (Lof _ _)` terms by comparing components directly, without
re-unfolding `Lof`'s `dite` each time. -/
theorem Lof_eq_iff {n : ℕ} {u v w z : Fin (2 * n)} :
    Lof u v = Lof w z ↔ (u = w ∧ v = z) ∨ (u = z ∧ v = w) := by
  constructor
  · intro h
    unfold Lof at h
    by_cases huv : u ≤ v
    · by_cases hwz : w ≤ z
      · rw [dif_pos huv, dif_pos hwz] at h
        injection h with h2
        injection congrArg Subtype.val h2 with e1 e2
        exact Or.inl ⟨e1, e2⟩
      · rw [dif_pos huv, dif_neg hwz] at h
        injection h with h2
        injection congrArg Subtype.val h2 with e1 e2
        exact Or.inr ⟨e1, e2⟩
    · by_cases hwz : w ≤ z
      · rw [dif_neg huv, dif_pos hwz] at h
        injection h with h2
        injection congrArg Subtype.val h2 with e1 e2
        exact Or.inr ⟨e2, e1⟩
      · rw [dif_neg huv, dif_neg hwz] at h
        injection h with h2
        injection congrArg Subtype.val h2 with e1 e2
        exact Or.inl ⟨e2, e1⟩
  · rintro (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
    · rfl
    · exact Lof_comm u v

/-- `Lof` always lands in the even (`.inl`) sector; needed so that `eN`
of an `Lof` value is supported nowhere in the odd (`.inr`) sector. -/
theorem Lof_ne_inr {n : ℕ} (u v w : Fin (2 * n)) : Lof u v ≠ Sum.inr w := by
  unfold Lof; split_ifs <;> simp

@[simp] theorem eN_Lof_apply_inr {n : ℕ} (a b w : Fin (2 * n)) :
    eN (Lof a b : IndexedBasis n) (Sum.inr w) = 0 := by
  unfold eN
  rw [if_neg (fun h => Lof_ne_inr a b w h.symm)]

/-! ## U1 — degree law at general `n` -/

/-- Degree law at basis level: FF yields an `L` (`1+1=0`), LF/FL yield an
`F` (`0+1=1`/`1+0=1`), LL yields an `L` (`0+0=0`) -- direct from the four
shapes `bracketBasisN` matches on. -/
theorem bracketBasisN_degree0 {n : ℕ} (i j k : IndexedBasis n)
    (h : bracketBasisN n i j k ≠ 0) : parity k = parity i + parity j := by
  match i, j with
  | .inl ⟨(u, v), _⟩, .inl ⟨(w, z), _⟩ =>
    match k with
    | .inl _ => rfl
    | .inr w' => exfalso; apply h; simp [bracketBasisN, bracketLLn]
  | .inl ⟨(u, v), _⟩, .inr w =>
    match k with
    | .inr _ => rfl
    | .inl p => exfalso; apply h; simp [bracketBasisN, bracketLFn, eN, Fof]
  | .inr w, .inl ⟨(u, v), _⟩ =>
    match k with
    | .inr _ => rfl
    | .inl p => exfalso; apply h; simp [bracketBasisN, bracketLFn, eN, Fof]
  | .inr u, .inr v =>
    match k with
    | .inl _ => rfl
    | .inr w' => exfalso; apply h; simp [bracketBasisN, bracketFFn]

/-- General-rank homogeneity predicate, transposed from `Native.lean`'s
`n=1`-only `IsHomog` (which is stated over the fixed `Mod = Basis5 → Coeff`
and cannot itself be reused here). -/
def IsHomogN {n : ℕ} (x : IndexedMod n) (d : ZMod 2) : Prop :=
  ∀ i : IndexedBasis n, x i ≠ 0 → parity i = d

/-- Degree law extended to arbitrary homogeneous elements of `IndexedMod n`.
Transposed from `HomogeneousDegree.lean`'s `bracket_isHomog`: that argument
is rank-independent (it only uses the basis-level degree law and the
double-sum shape `bracketN` itself already has), so it transposes with no
structural change beyond the renamed general-rank objects. -/
theorem bracketN_isHomogN {n : ℕ} (x y : IndexedMod n) (dx dy : ZMod 2)
    (hx : IsHomogN x dx) (hy : IsHomogN y dy) :
    IsHomogN (bracketN n x y) (dx + dy) := by
  intro k hk
  by_contra hne
  apply hk
  unfold bracketN
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  apply Finset.sum_eq_zero; intro i _
  apply Finset.sum_eq_zero; intro j _
  by_cases hxi : x i = 0
  · simp [hxi]
  · by_cases hyj : y j = 0
    · simp [hyj]
    · have hpi := hx i hxi
      have hpj := hy j hyj
      have hz : bracketBasisN n i j k = 0 := by
        by_contra hc
        exact hne (by rw [bracketBasisN_degree0 i j k hc, hpi, hpj])
      simp [hz]

/-! ## U2 — super-skew at basis level, general `n`

`gsignN` is the rank-indexed analogue of `N1Proofs.lean`'s `Coeff`-valued
`gsign`, valued in `Pn n` here since general-rank structure constants live
there, not in the `n=1`-only `Coeff`. -/

noncomputable def gsignN (n : ℕ) (p q : ZMod 2) : Pn n :=
  if p = 1 ∧ q = 1 then cratN n (-1) else cratN n 1

theorem gsignN_00 (n : ℕ) : gsignN n (0 : ZMod 2) (0 : ZMod 2) = 1 := by
  unfold gsignN; rw [if_neg (by decide), cratN_one]
theorem gsignN_01 (n : ℕ) : gsignN n (0 : ZMod 2) (1 : ZMod 2) = 1 := by
  unfold gsignN; rw [if_neg (by decide), cratN_one]
theorem gsignN_10 (n : ℕ) : gsignN n (1 : ZMod 2) (0 : ZMod 2) = 1 := by
  unfold gsignN; rw [if_neg (by decide), cratN_one]
theorem gsignN_11 (n : ℕ) : gsignN n (1 : ZMod 2) (1 : ZMod 2) = -1 := by
  unfold gsignN
  rw [if_pos (by decide)]
  rw [show (-1 : ℚ) = -(1 : ℚ) from rfl, ← cratN_neg, cratN_one]

/-- **FF sector**: both parities odd, so the requirement `[F_u,F_v] =
+[F_v,F_u]` is exactly `Lof` commutativity. -/
theorem superskew_FF {n : ℕ} (u v : Fin (2 * n)) : bracketFFn n u v = bracketFFn n v u := by
  unfold bracketFFn; rw [Lof_comm u v]

/-- **LF sector**: `p(L)p(F) = 0`, so the requirement is `[L,F] = -[F,L]`;
since `bracketBasisN`'s own FL branch is defined as `-`(the LF branch),
this direction is immediate. -/
theorem superskew_LF {n : ℕ} (u v w : Fin (2 * n)) :
    bracketLFn n u v w = -(gsignN n (0 : ZMod 2) (1 : ZMod 2)) • (-bracketLFn n u v w) := by
  rw [gsignN_01, neg_smul, one_smul, neg_neg]

/-- **FL sector**: the other direction of the same definitional fact. -/
theorem superskew_FL {n : ℕ} (u v w : Fin (2 * n)) :
    (-bracketLFn n u v w) = -(gsignN n (1 : ZMod 2) (0 : ZMod 2)) • bracketLFn n u v w := by
  rw [gsignN_10, neg_smul, one_smul]

/-- **LL sector**, the one with content: a four-term matching using only
`Jn_antisymm` and `Lof_comm` (§3.1's two facts), exactly as designed --
after rewriting, both sides consist of literally the same four terms in a
different order, closed by `abel` on the additive group `IndexedMod n`. -/
theorem superskew_LL {n : ℕ} (u v w z : Fin (2 * n)) :
    bracketLLn n u v w z = -(gsignN n (0 : ZMod 2) (0 : ZMod 2)) • bracketLLn n w z u v := by
  rw [gsignN_00, neg_smul, one_smul]
  unfold bracketLLn
  rw [Jn_antisymm n u z, Jn_antisymm n u w, Jn_antisymm n v z, Jn_antisymm n v w,
      Lof_comm w v, Lof_comm z v, Lof_comm w u, Lof_comm z u]
  simp only [mul_neg, neg_smul]
  abel

/-- `U2`: super-skew symmetry on all ordered basis pairs, general `n`,
assembled from the four sectors above. -/
theorem bracketBasisN_super_skew {n : ℕ} (i j : IndexedBasis n) :
    bracketBasisN n i j = -(gsignN n (parity i) (parity j)) • bracketBasisN n j i := by
  match i, j with
  | .inl ⟨(u, v), _⟩, .inl ⟨(w, z), _⟩ => exact superskew_LL u v w z
  | .inl ⟨(u, v), _⟩, .inr w => exact superskew_LF u v w
  | .inr w, .inl ⟨(u, v), _⟩ => exact superskew_FL u v w
  | .inr u, .inr v =>
    show bracketFFn n u v = -(gsignN n (1 : ZMod 2) (1 : ZMod 2)) • bracketFFn n v u
    rw [gsignN_11, neg_neg, one_smul]
    exact superskew_FF u v

/-! ## U3 — super-skew for arbitrary module elements, general `n` -/

/-- `U3`: super-skew symmetry extended to arbitrary homogeneous elements
of `IndexedMod n`, by the finite-sum bilinear expansion `bracketN` itself
already uses -- transposed from `N1Proofs.lean`'s `bracket_super_skew_homog`
(`T4`), the same shape, with `U2`'s `bracketBasisN_super_skew` in place of
`bracket_super_skew`. No graded sign beyond `U2`'s own is introduced by the
bilinear extension, since every coefficient of `Pn n` is even. -/
theorem bracketN_super_skew_homog {n : ℕ} (x y : IndexedMod n) (dx dy : ZMod 2)
    (hx : IsHomogN x dx) (hy : IsHomogN y dy) :
    bracketN n x y = -(gsignN n dx dy) • bracketN n y x := by
  have hswap : bracketN n y x
      = ∑ i : IndexedBasis n, ∑ j : IndexedBasis n, (x i * y j) • bracketBasisN n j i := by
    unfold bracketN
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl; intro j _
    apply Finset.sum_congr rfl; intro i _
    congr 1; ring
  rw [hswap]
  unfold bracketN
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro i _
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro j _
  by_cases hxy : x i * y j = 0
  · simp [hxy]
  · have hxi : x i ≠ 0 := fun h0 => hxy (by rw [h0]; ring)
    have hyj : y j ≠ 0 := fun h0 => hxy (by rw [h0]; ring)
    rw [bracketBasisN_super_skew i j, hx i hxi, hy j hyj, smul_smul, smul_smul, neg_mul]
    congr 1
    ring

end Indexed
end InhomogeneousDeformations
