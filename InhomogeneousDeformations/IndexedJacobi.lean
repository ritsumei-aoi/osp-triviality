import InhomogeneousDeformations.IndexedLaws

/-!
# R2-C-2 (U4-U5) — general-rank super-Jacobi

`H1.2`: every declaration here is stated for general `n : ℕ`, with indices
in `Fin (2 * n)`; no `n = 1` specialization, `Fin 2` instance, or literal
five-element basis appears anywhere. `Indexed.lean` and `IndexedLaws.lean`
(both frozen) are never reopened; every law here is a new statement/proof
about their already-accepted definitions, using only `Jn_antisymm` and
`Lof_comm` (`IndexedLaws.lean`'s U0) as the reusable facts about `Jn`/`Lof`.

`H1.3` (verbatim, do not upgrade): a successful R2-C-2 establishes
super-Jacobi for the frozen indexed bracket for every `n ≥ 1`, which
together with the accepted R2-C-1 completes P1: the specified coordinate
structure is a Lie superalgebra of the stated shape at every rank. It does
NOT establish the oscillator realization correspondence, which is P1's own
declared non-goal and remains so even once P1 is complete.

Separately: the manuscript obtains super-Jacobi for free from the
associative realization (`lem:base`'s proof, P4, deferred). This module
proves Jacobi directly from the coordinate formulas; the proof below does
not mirror the paper's argument.

## Sector table (U4)

All four rotation-class representatives close on `IndexedLaws.lean`'s
`Jn_antisymm`/`Lof_comm` alone, exactly as Agent1's hand analysis predicted:

* `jacobiN_FFF` -- reduces to `Jn_antisymm` alone.
* `jacobiN_LFF` -- reduces to `Lof_comm` alone.
* `jacobiN_LLF` -- reduces to `Jn_antisymm` plus commutativity of `Pn n`.
* `jacobiN_LLL` -- the stage's real work (48-term expansion); closes by the
  same two facts plus `ring_nf`'s ring-level normalization of the `Pn n`
  coefficients once every `Lof`/`Jn` occurrence is brought to one canonical
  argument order.

`jacobiN_basis` assembles all eight ordered basis-triple shapes from these
four representatives via `jacobiSum_rot` (the cyclic symmetry of the
Jacobi sum itself, immediate from reassociating its own three summands).
`jacobiN_homog` (U5) extends this, unconditionally, to arbitrary
homogeneous elements of `IndexedMod n`, transposed from `N1Proofs.lean`'s
`jacobi_homog` (the `n=1` shape of the same extension).
-/

namespace InhomogeneousDeformations
namespace Indexed

/-! ## Bilinearity toolkit for `bracketN`, general `n`

None of these exist in the frozen modules: `IndexedLaws.lean` (U0-U3) never
needed the full bilinearity of `bracketN` in both arguments, only enough
structure for the degree law and super-skew. U4 needs it to distribute the
bracket over the sums the basis-level structure constants produce. -/

theorem bracketN_add_left {n : ℕ} (x1 x2 y : IndexedMod n) :
    bracketN n (x1 + x2) y = bracketN n x1 y + bracketN n x2 y := by
  funext k
  simp only [bracketN, Finset.sum_apply, Pi.add_apply]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro i _
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro j _
  simp only [Pi.smul_apply, smul_eq_mul]; ring

theorem bracketN_add_right {n : ℕ} (x y1 y2 : IndexedMod n) :
    bracketN n x (y1 + y2) = bracketN n x y1 + bracketN n x y2 := by
  funext k
  simp only [bracketN, Finset.sum_apply, Pi.add_apply]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro i _
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro j _
  simp only [Pi.smul_apply, smul_eq_mul]; ring

theorem bracketN_smul_left {n : ℕ} (c : Pn n) (x y : IndexedMod n) :
    bracketN n (c • x) y = c • bracketN n x y := by
  funext k
  simp only [bracketN, Finset.sum_apply, Pi.smul_apply]
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro i _
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro j _
  simp only [smul_eq_mul]; ring

theorem bracketN_smul_right {n : ℕ} (c : Pn n) (x y : IndexedMod n) :
    bracketN n x (c • y) = c • bracketN n x y := by
  funext k
  simp only [bracketN, Finset.sum_apply, Pi.smul_apply]
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro i _
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro j _
  simp only [smul_eq_mul]; ring

theorem bracketN_neg_right {n : ℕ} (x y : IndexedMod n) :
    bracketN n x (-y) = -bracketN n x y := by
  funext k
  simp only [bracketN, Finset.sum_apply, Pi.neg_apply, Pi.smul_apply, smul_eq_mul]
  rw [← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl; intro i _
  rw [← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl; intro j _
  ring

theorem bracketN_eN_eN {n : ℕ} (X Y : IndexedBasis n) :
    bracketN n (eN X) (eN Y) = bracketBasisN n X Y := by
  unfold bracketN
  rw [Finset.sum_eq_single X]
  · rw [Finset.sum_eq_single Y]
    · simp [eN]
    · intro b _ hb
      have : eN Y b = 0 := by unfold eN; rw [if_neg hb]
      simp [this]
    · intro h; exact absurd (Finset.mem_univ Y) h
  · intro b _ hb
    have : eN X b = 0 := by unfold eN; rw [if_neg hb]
    simp [this]
  · intro h; exact absurd (Finset.mem_univ X) h

/-! ## Basis-shape reduction toolkit, general `n`

`bracketBasisN` is defined by a raw pattern match on the underlying `Sum`
type; these lemmas restate its four branches in terms of `Lof`/`Fof`
directly, resolving `Lof`'s internal `dite` explicitly via `dif_pos`/
`dif_neg` (never via a bare `show`, which gets stuck at `whnf` trying to
decide an abstract `u ≤ v`). U4's proofs rewrite through these rather than
unfolding `bracketBasisN` blindly. -/

theorem parity_Lof {n : ℕ} (u v : Fin (2 * n)) : parity (Lof u v : IndexedBasis n) = 0 := by
  rcases eq_or_ne u v with heq | hne
  · subst heq; rw [Lof, dif_pos (le_refl u)]; rfl
  · rcases lt_or_gt_of_ne hne with hlt | hgt
    · rw [Lof, dif_pos hlt.le]; rfl
    · rw [Lof, dif_neg (not_le.mpr hgt)]; rfl

theorem parity_Fof {n : ℕ} (u : Fin (2 * n)) : parity (Fof u : IndexedBasis n) = 1 := rfl

theorem bracketBasisN_Fof_Fof {n : ℕ} (u v : Fin (2 * n)) :
    bracketBasisN n (Fof u) (Fof v) = bracketFFn n u v := rfl

theorem bracketLFn_comm {n : ℕ} (u v w : Fin (2 * n)) :
    bracketLFn n u v w = bracketLFn n v u w := by unfold bracketLFn; abel

theorem bracketBasisN_Lof_Fof {n : ℕ} (a b x : Fin (2 * n)) :
    bracketBasisN n (Lof a b) (Fof x) = bracketLFn n a b x := by
  rcases eq_or_ne a b with heq | hne
  · subst heq; rw [Lof, dif_pos (le_refl a)]; rfl
  · rcases lt_or_gt_of_ne hne with hlt | hgt
    · rw [Lof, dif_pos hlt.le]; rfl
    · rw [Lof, dif_neg (not_le.mpr hgt)]; exact bracketLFn_comm b a x

theorem bracketBasisN_Fof_Lof {n : ℕ} (x a b : Fin (2 * n)) :
    bracketBasisN n (Fof x) (Lof a b) = -bracketLFn n a b x := by
  rcases eq_or_ne a b with heq | hne
  · subst heq; rw [Lof, dif_pos (le_refl a)]; rfl
  · rcases lt_or_gt_of_ne hne with hlt | hgt
    · rw [Lof, dif_pos hlt.le]; rfl
    · rw [Lof, dif_neg (not_le.mpr hgt)]
      show -bracketLFn n b a x = -bracketLFn n a b x
      rw [bracketLFn_comm]

theorem bracketLLn_comm12 {n : ℕ} (u v w z : Fin (2 * n)) :
    bracketLLn n v u w z = bracketLLn n u v w z := by unfold bracketLLn; abel

theorem bracketLLn_comm34 {n : ℕ} (u v w z : Fin (2 * n)) :
    bracketLLn n u v z w = bracketLLn n u v w z := by unfold bracketLLn; abel

theorem bracketBasisN_Lof_Lof {n : ℕ} (a b c d : Fin (2 * n)) :
    bracketBasisN n (Lof a b) (Lof c d) = bracketLLn n a b c d := by
  rcases eq_or_ne a b with hab | hab
  · subst hab
    rcases eq_or_ne c d with hcd | hcd
    · subst hcd; rw [Lof, dif_pos (le_refl a), Lof, dif_pos (le_refl c)]; rfl
    · rcases lt_or_gt_of_ne hcd with hlt | hgt
      · rw [Lof, dif_pos (le_refl a), Lof, dif_pos hlt.le]; rfl
      · rw [Lof, dif_pos (le_refl a), Lof, dif_neg (not_le.mpr hgt)]
        show bracketLLn n a a d c = bracketLLn n a a c d
        rw [bracketLLn_comm34]
  · rcases lt_or_gt_of_ne hab with hablt | habgt
    · rw [Lof, dif_pos hablt.le]
      rcases eq_or_ne c d with hcd | hcd
      · subst hcd; rw [Lof, dif_pos (le_refl c)]; rfl
      · rcases lt_or_gt_of_ne hcd with hlt | hgt
        · rw [Lof, dif_pos hlt.le]; rfl
        · rw [Lof, dif_neg (not_le.mpr hgt)]
          show bracketLLn n a b d c = bracketLLn n a b c d
          rw [bracketLLn_comm34]
    · rw [Lof, dif_neg (not_le.mpr habgt)]
      rcases eq_or_ne c d with hcd | hcd
      · subst hcd; rw [Lof, dif_pos (le_refl c)]
        show bracketLLn n b a c c = bracketLLn n a b c c
        rw [bracketLLn_comm12]
      · rcases lt_or_gt_of_ne hcd with hlt | hgt
        · rw [Lof, dif_pos hlt.le]
          show bracketLLn n b a c d = bracketLLn n a b c d
          rw [bracketLLn_comm12]
        · rw [Lof, dif_neg (not_le.mpr hgt)]
          show bracketLLn n b a d c = bracketLLn n a b c d
          rw [bracketLLn_comm34, bracketLLn_comm12]

/-! ## U4 -- super-Jacobi at basis level, general `n` -/

/-- The cyclic Jacobi sum on ordered basis triples, `Pn n`-valued, exactly
the left side of U4's identity. -/
noncomputable def jacobiSum {n : ℕ} (X Y Z : IndexedBasis n) : IndexedMod n :=
  gsignN n (parity X) (parity Z) • bracketN n (eN X) (bracketN n (eN Y) (eN Z))
    + gsignN n (parity Y) (parity X) • bracketN n (eN Y) (bracketN n (eN Z) (eN X))
    + gsignN n (parity Z) (parity Y) • bracketN n (eN Z) (bracketN n (eN X) (eN Y))

/-- **FFF sector**: all three signs are `-1`, so the identity is the cyclic
sum of `[F_u,[F_v,F_w]]`. Reduces to `Jn_antisymm` alone, as Agent1's hand
analysis predicted. -/
theorem jacobiN_FFF {n : ℕ} (u v w : Fin (2 * n)) :
    jacobiSum (Fof u : IndexedBasis n) (Fof v) (Fof w) = 0 := by
  unfold jacobiSum
  rw [bracketN_eN_eN (Fof v : IndexedBasis n) (Fof w),
      bracketN_eN_eN (Fof w : IndexedBasis n) (Fof u),
      bracketN_eN_eN (Fof u : IndexedBasis n) (Fof v),
      bracketBasisN_Fof_Fof, bracketBasisN_Fof_Fof, bracketBasisN_Fof_Fof]
  unfold bracketFFn
  rw [bracketN_smul_right, bracketN_smul_right, bracketN_smul_right,
      bracketN_eN_eN (Fof u : IndexedBasis n) (Lof v w),
      bracketN_eN_eN (Fof v : IndexedBasis n) (Lof w u),
      bracketN_eN_eN (Fof w : IndexedBasis n) (Lof u v),
      bracketBasisN_Fof_Lof u v w, bracketBasisN_Fof_Lof v w u, bracketBasisN_Fof_Lof w u v]
  simp only [Fof, parity, gsignN]
  unfold bracketLFn
  simp only [smul_add, smul_smul, smul_neg]
  rw [Jn_antisymm n w v, Jn_antisymm n u w, Jn_antisymm n v u]
  simp only [mul_neg, neg_smul]
  abel

/-- **LFF sector**, representative `X = L_uv, Y = F_w, Z = F_z`. Reduces to
`Lof_comm` alone, as Agent1's hand analysis predicted. -/
theorem jacobiN_LFF {n : ℕ} (u v w z : Fin (2 * n)) :
    jacobiSum (Lof u v : IndexedBasis n) (Fof w) (Fof z) = 0 := by
  unfold jacobiSum
  rw [bracketN_eN_eN (Fof w : IndexedBasis n) (Fof z),
      bracketN_eN_eN (Fof z : IndexedBasis n) (Lof u v),
      bracketN_eN_eN (Lof u v : IndexedBasis n) (Fof w),
      bracketBasisN_Fof_Fof, bracketBasisN_Fof_Lof z u v, bracketBasisN_Lof_Fof u v w,
      parity_Lof, parity_Fof w, parity_Fof z]
  simp (config := { decide := true }) only [gsignN, if_true, if_false]
  rw [bracketN_neg_right]
  unfold bracketFFn bracketLFn
  rw [bracketN_smul_right,
      show bracketN n (eN (Lof u v : IndexedBasis n)) (eN (Lof w z : IndexedBasis n))
        = bracketLLn n u v w z from (bracketN_eN_eN (Lof u v) (Lof w z)).trans (bracketBasisN_Lof_Lof u v w z)]
  rw [bracketN_add_right, bracketN_add_right, bracketN_smul_right, bracketN_smul_right,
      bracketN_smul_right, bracketN_smul_right]
  rw [bracketN_eN_eN (Fof w : IndexedBasis n) (Fof u), bracketN_eN_eN (Fof w : IndexedBasis n) (Fof v),
      bracketN_eN_eN (Fof z : IndexedBasis n) (Fof u), bracketN_eN_eN (Fof z : IndexedBasis n) (Fof v),
      bracketBasisN_Fof_Fof, bracketBasisN_Fof_Fof, bracketBasisN_Fof_Fof, bracketBasisN_Fof_Fof]
  unfold bracketFFn
  simp only [smul_add, smul_smul, smul_neg]
  rw [show Lof w u = Lof u w from Lof_comm w u, show Lof w v = Lof v w from Lof_comm w v,
      show Lof z u = Lof u z from Lof_comm z u, show Lof z v = Lof v z from Lof_comm z v]
  unfold bracketLLn
  simp only [cratN_one, ← cratN_neg, neg_mul, one_mul, neg_add, neg_smul, smul_add, smul_smul]
  ring_nf

/-- **LLF sector**, representative `X = L_uv, Y = L_wz, Z = F_t`. Reduces
to `Jn_antisymm` plus commutativity of multiplication in `Pn n`, as
Agent1's hand analysis predicted. -/
theorem jacobiN_LLF {n : ℕ} (u v w z t : Fin (2 * n)) :
    jacobiSum (Lof u v : IndexedBasis n) (Lof w z) (Fof t) = 0 := by
  unfold jacobiSum
  rw [bracketN_eN_eN (Lof w z : IndexedBasis n) (Fof t),
      bracketN_eN_eN (Fof t : IndexedBasis n) (Lof u v),
      bracketN_eN_eN (Lof u v : IndexedBasis n) (Lof w z),
      bracketBasisN_Lof_Fof w z t, bracketBasisN_Fof_Lof t u v, bracketBasisN_Lof_Lof u v w z,
      parity_Lof u v, parity_Lof w z, parity_Fof t]
  simp (config := { decide := true }) only [gsignN, if_false]
  simp only [cratN_one, one_smul]
  rw [bracketN_neg_right]
  unfold bracketLFn bracketLLn
  rw [bracketN_add_right, bracketN_add_right, bracketN_add_right, bracketN_add_right,
      bracketN_add_right,
      bracketN_smul_right, bracketN_smul_right, bracketN_smul_right, bracketN_smul_right,
      bracketN_smul_right, bracketN_smul_right, bracketN_smul_right, bracketN_smul_right]
  rw [bracketN_eN_eN (Lof u v : IndexedBasis n) (Fof w), bracketN_eN_eN (Lof u v : IndexedBasis n) (Fof z),
      bracketN_eN_eN (Lof w z : IndexedBasis n) (Fof u), bracketN_eN_eN (Lof w z : IndexedBasis n) (Fof v),
      bracketN_eN_eN (Fof t : IndexedBasis n) (Lof u z), bracketN_eN_eN (Fof t : IndexedBasis n) (Lof v z),
      bracketN_eN_eN (Fof t : IndexedBasis n) (Lof u w), bracketN_eN_eN (Fof t : IndexedBasis n) (Lof v w),
      bracketBasisN_Lof_Fof u v w, bracketBasisN_Lof_Fof u v z,
      bracketBasisN_Lof_Fof w z u, bracketBasisN_Lof_Fof w z v,
      bracketBasisN_Fof_Lof t u z, bracketBasisN_Fof_Lof t v z,
      bracketBasisN_Fof_Lof t u w, bracketBasisN_Fof_Lof t v w]
  unfold bracketLFn
  simp only [smul_neg, smul_add, smul_smul]
  rw [Jn_antisymm n u z, Jn_antisymm n u w, Jn_antisymm n v z, Jn_antisymm n v w]
  simp only [mul_neg, neg_neg, neg_add, neg_smul]
  ring_nf

/-- **LLL sector**, representative `X = L_uv, Y = L_wz, Z = L_st`, all
signs `+1`. The stage's real work: each of the three cyclic terms expands
to sixteen `L`-basis terms (forty-eight total), organized here by full
mechanical expansion followed by normalizing every `Lof` occurrence to one
canonical argument order (via `Lof_comm`) and every `Jn` occurrence to one
canonical order (via `Jn_antisymm`), after which `ring_nf` closes the
identity by recognizing the like `Pn n`-coefficient terms cancel. -/
theorem jacobiN_LLL {n : ℕ} (u v w z s t : Fin (2 * n)) :
    jacobiSum (Lof u v : IndexedBasis n) (Lof w z) (Lof s t) = 0 := by
  unfold jacobiSum
  rw [bracketN_eN_eN (Lof w z : IndexedBasis n) (Lof s t),
      bracketN_eN_eN (Lof s t : IndexedBasis n) (Lof u v),
      bracketN_eN_eN (Lof u v : IndexedBasis n) (Lof w z),
      bracketBasisN_Lof_Lof w z s t, bracketBasisN_Lof_Lof s t u v, bracketBasisN_Lof_Lof u v w z,
      parity_Lof u v, parity_Lof w z, parity_Lof s t]
  simp (config := { decide := true }) only [gsignN, if_false]
  simp only [cratN_one, one_smul]
  unfold bracketLLn
  rw [bracketN_add_right, bracketN_add_right, bracketN_add_right,
      bracketN_add_right, bracketN_add_right, bracketN_add_right,
      bracketN_add_right, bracketN_add_right, bracketN_add_right,
      bracketN_smul_right, bracketN_smul_right, bracketN_smul_right, bracketN_smul_right,
      bracketN_smul_right, bracketN_smul_right, bracketN_smul_right, bracketN_smul_right,
      bracketN_smul_right, bracketN_smul_right, bracketN_smul_right, bracketN_smul_right]
  rw [bracketN_eN_eN (Lof u v : IndexedBasis n) (Lof w t), bracketN_eN_eN (Lof u v : IndexedBasis n) (Lof z t),
      bracketN_eN_eN (Lof u v : IndexedBasis n) (Lof w s), bracketN_eN_eN (Lof u v : IndexedBasis n) (Lof z s),
      bracketN_eN_eN (Lof w z : IndexedBasis n) (Lof s v), bracketN_eN_eN (Lof w z : IndexedBasis n) (Lof t v),
      bracketN_eN_eN (Lof w z : IndexedBasis n) (Lof s u), bracketN_eN_eN (Lof w z : IndexedBasis n) (Lof t u),
      bracketN_eN_eN (Lof s t : IndexedBasis n) (Lof u z), bracketN_eN_eN (Lof s t : IndexedBasis n) (Lof v z),
      bracketN_eN_eN (Lof s t : IndexedBasis n) (Lof u w), bracketN_eN_eN (Lof s t : IndexedBasis n) (Lof v w),
      bracketBasisN_Lof_Lof u v w t, bracketBasisN_Lof_Lof u v z t,
      bracketBasisN_Lof_Lof u v w s, bracketBasisN_Lof_Lof u v z s,
      bracketBasisN_Lof_Lof w z s v, bracketBasisN_Lof_Lof w z t v,
      bracketBasisN_Lof_Lof w z s u, bracketBasisN_Lof_Lof w z t u,
      bracketBasisN_Lof_Lof s t u z, bracketBasisN_Lof_Lof s t v z,
      bracketBasisN_Lof_Lof s t u w, bracketBasisN_Lof_Lof s t v w]
  unfold bracketLLn
  simp only [smul_add, smul_smul]
  rw [show Lof w v = Lof v w from Lof_comm w v, show Lof z v = Lof v z from Lof_comm z v,
      show Lof w u = Lof u w from Lof_comm w u, show Lof z u = Lof u z from Lof_comm z u,
      show Lof s z = Lof z s from Lof_comm s z, show Lof t z = Lof z t from Lof_comm t z,
      show Lof s u = Lof u s from Lof_comm s u, show Lof t u = Lof u t from Lof_comm t u,
      show Lof s v = Lof v s from Lof_comm s v, show Lof t v = Lof v t from Lof_comm t v,
      show Lof s w = Lof w s from Lof_comm s w, show Lof t w = Lof w t from Lof_comm t w]
  rw [Jn_antisymm n u t, Jn_antisymm n v z, Jn_antisymm n v w, Jn_antisymm n u s,
      Jn_antisymm n v t, Jn_antisymm n u z, Jn_antisymm n u w, Jn_antisymm n v s,
      Jn_antisymm n z t, Jn_antisymm n z s, Jn_antisymm n w t, Jn_antisymm n w s]
  simp only [mul_neg, neg_mul, neg_neg, neg_smul]
  ring_nf

/-- The Jacobi sum is invariant under cyclic rotation of its three
arguments -- immediate, since rotating `X,Y,Z` to `Y,Z,X` merely reorders
`jacobiSum`'s own three summands. This is what makes "four sectors up to
rotation" exact: every one of the eight ordered basis-triple shapes is
either one of the four representatives above or a rotation of one. -/
theorem jacobiSum_rot {n : ℕ} (X Y Z : IndexedBasis n) :
    jacobiSum X Y Z = jacobiSum Y Z X := by
  unfold jacobiSum; abel

/-- `U4`: super-Jacobi on all eight ordered basis-triple shapes, general
`n`, assembled from the four representatives above via `jacobiSum_rot`.
The three non-representative "2L1F" shapes (`LFL`, `FLL`) and "1L2F"
shapes (`FFL`, `FLF`) are rotations of `jacobiN_LLF`/`jacobiN_LFF`
respectively; `LLL` and `FFF` are already rotation-invariant. -/
theorem jacobiN_basis {n : ℕ} (i j k : IndexedBasis n) : jacobiSum i j k = 0 := by
  match i, j, k with
  | .inl ⟨(u, v), h1⟩, .inl ⟨(w, z), h2⟩, .inl ⟨(s, t), h3⟩ =>
      have e1 : (Sum.inl ⟨(u, v), h1⟩ : IndexedBasis n) = Lof u v := by rw [Lof, dif_pos h1]
      have e2 : (Sum.inl ⟨(w, z), h2⟩ : IndexedBasis n) = Lof w z := by rw [Lof, dif_pos h2]
      have e3 : (Sum.inl ⟨(s, t), h3⟩ : IndexedBasis n) = Lof s t := by rw [Lof, dif_pos h3]
      rw [e1, e2, e3]; exact jacobiN_LLL u v w z s t
  | .inl ⟨(u, v), h1⟩, .inl ⟨(w, z), h2⟩, .inr t =>
      have e1 : (Sum.inl ⟨(u, v), h1⟩ : IndexedBasis n) = Lof u v := by rw [Lof, dif_pos h1]
      have e2 : (Sum.inl ⟨(w, z), h2⟩ : IndexedBasis n) = Lof w z := by rw [Lof, dif_pos h2]
      rw [e1, e2]; exact jacobiN_LLF u v w z t
  | .inl ⟨(u, v), h1⟩, .inr w, .inl ⟨(s, t), h3⟩ =>
      have e1 : (Sum.inl ⟨(u, v), h1⟩ : IndexedBasis n) = Lof u v := by rw [Lof, dif_pos h1]
      have e3 : (Sum.inl ⟨(s, t), h3⟩ : IndexedBasis n) = Lof s t := by rw [Lof, dif_pos h3]
      rw [e1, e3]
      have h := jacobiN_LLF s t u v w
      rwa [jacobiSum_rot (Lof s t : IndexedBasis n) (Lof u v) (Fof w)] at h
  | .inr w, .inl ⟨(u, v), h1⟩, .inl ⟨(s, t), h3⟩ =>
      have e1 : (Sum.inl ⟨(u, v), h1⟩ : IndexedBasis n) = Lof u v := by rw [Lof, dif_pos h1]
      have e3 : (Sum.inl ⟨(s, t), h3⟩ : IndexedBasis n) = Lof s t := by rw [Lof, dif_pos h3]
      rw [e1, e3]
      have h := jacobiN_LLF u v s t w
      rw [jacobiSum_rot (Lof u v : IndexedBasis n) (Lof s t) (Fof w)] at h
      rwa [jacobiSum_rot (Lof s t : IndexedBasis n) (Fof w) (Lof u v)] at h
  | .inl ⟨(u, v), h1⟩, .inr w, .inr z =>
      have e1 : (Sum.inl ⟨(u, v), h1⟩ : IndexedBasis n) = Lof u v := by rw [Lof, dif_pos h1]
      rw [e1]; exact jacobiN_LFF u v w z
  | .inr w, .inr z, .inl ⟨(u, v), h1⟩ =>
      have e1 : (Sum.inl ⟨(u, v), h1⟩ : IndexedBasis n) = Lof u v := by rw [Lof, dif_pos h1]
      rw [e1]
      have h := jacobiN_LFF u v w z
      rwa [jacobiSum_rot (Lof u v : IndexedBasis n) (Fof w) (Fof z)] at h
  | .inr w, .inl ⟨(u, v), h1⟩, .inr z =>
      have e1 : (Sum.inl ⟨(u, v), h1⟩ : IndexedBasis n) = Lof u v := by rw [Lof, dif_pos h1]
      rw [e1]
      have h := jacobiN_LFF u v z w
      rw [jacobiSum_rot (Lof u v : IndexedBasis n) (Fof z) (Fof w)] at h
      rwa [jacobiSum_rot (Fof z : IndexedBasis n) (Fof w) (Lof u v)] at h
  | .inr u, .inr v, .inr w => exact jacobiN_FFF u v w

/-! ## Finite-sum expansion toolkit for `bracketN`, general `n`

Transposed from `N1Proofs.lean`'s `bracket_sum_left`/`bracket_sum_right`/
`expand_basis`/`bracket_bilinear_expand`/`bracket_expand_left`: the same
shapes, over `IndexedBasis n`/`IndexedMod n`/`eN`/`bracketN` in place of
`Basis5`/`Mod`/`e`/`bracket`. -/

theorem bracketN_sum_left {n : ℕ} (s : Finset (IndexedBasis n)) (f : IndexedBasis n → IndexedMod n)
    (y : IndexedMod n) :
    bracketN n (∑ i ∈ s, f i) y = ∑ i ∈ s, bracketN n (f i) y := by
  induction s using Finset.induction with
  | empty => simp [bracketN]
  | @insert a s ha ih => rw [Finset.sum_insert ha, bracketN_add_left, ih, Finset.sum_insert ha]

theorem bracketN_sum_right {n : ℕ} (s : Finset (IndexedBasis n)) (f : IndexedBasis n → IndexedMod n)
    (x : IndexedMod n) :
    bracketN n x (∑ j ∈ s, f j) = ∑ j ∈ s, bracketN n x (f j) := by
  induction s using Finset.induction with
  | empty => simp [bracketN]
  | @insert a s ha ih => rw [Finset.sum_insert ha, bracketN_add_right, ih, Finset.sum_insert ha]

theorem expand_basisN {n : ℕ} (x : IndexedMod n) : x = ∑ i : IndexedBasis n, x i • eN i := by
  funext k
  rw [Finset.sum_apply]
  rw [Finset.sum_eq_single k]
  · simp [eN]
  · intro b _ hb; simp [eN, Ne.symm hb]
  · intro h; exact absurd (Finset.mem_univ k) h

theorem bracketN_bilinear_expand {n : ℕ} (x y : IndexedMod n) :
    bracketN n x y = ∑ i : IndexedBasis n, ∑ j : IndexedBasis n, (x i * y j) • bracketN n (eN i) (eN j) := by
  simp_rw [bracketN_eN_eN]
  rfl

theorem bracketN_expand_left {n : ℕ} (v w : IndexedMod n) :
    bracketN n v w = ∑ i : IndexedBasis n, (v i) • bracketN n (eN i) w := by
  conv_lhs => rw [expand_basisN v]
  rw [bracketN_sum_left]
  apply Finset.sum_congr rfl; intro i _
  rw [bracketN_smul_left]

/-! ## U5 -- super-Jacobi for arbitrary homogeneous elements, general `n` -/

/-- `U5`: the graded Jacobi identity for arbitrary homogeneous elements of
`IndexedMod n`, general `n`, **unconditional** since `U4`'s `jacobiN_basis`
covers all eight basis-triple shapes. Transposed from `N1Proofs.lean`'s
`jacobi_homog`: the same triple-sum expansion and reduction to the
basis-level identity, with `jacobiN_basis` in place of `jacobi125`. On
`Pn n` every coefficient is even, so this bilinear/trilinear extension
introduces no graded sign of its own beyond U4's own `gsignN` factors. -/
theorem jacobiN_homog {n : ℕ} (x y z : IndexedMod n) (dx dy dz : ZMod 2)
    (hx : IsHomogN x dx) (hy : IsHomogN y dy) (hz : IsHomogN z dz) :
    gsignN n dx dz • bracketN n x (bracketN n y z) + gsignN n dy dx • bracketN n y (bracketN n z x)
      + gsignN n dz dy • bracketN n z (bracketN n x y) = 0 := by
  have key : ∀ i j k : IndexedBasis n, x i ≠ 0 → y j ≠ 0 → z k ≠ 0 →
      gsignN n dx dz • bracketN n (eN i) (bracketN n (eN j) (eN k))
        + gsignN n dy dx • bracketN n (eN j) (bracketN n (eN k) (eN i))
        + gsignN n dz dy • bracketN n (eN k) (bracketN n (eN i) (eN j)) = 0 := by
    intro i j k hi hj hk
    rw [← hx i hi, ← hy j hj, ← hz k hk]
    exact jacobiN_basis i j k
  have expand1 : bracketN n x (bracketN n y z)
      = ∑ i, ∑ j, ∑ k, (x i * y j * z k) • bracketN n (eN i) (bracketN n (eN j) (eN k)) := by
    rw [bracketN_expand_left x (bracketN n y z)]
    apply Finset.sum_congr rfl; intro i _
    rw [bracketN_bilinear_expand y z, bracketN_sum_right, Finset.smul_sum]
    apply Finset.sum_congr rfl; intro j _
    rw [bracketN_sum_right, Finset.smul_sum]
    apply Finset.sum_congr rfl; intro k _
    rw [bracketN_smul_right, smul_smul]
    congr 1; ring
  have expand2 : bracketN n y (bracketN n z x)
      = ∑ i, ∑ j, ∑ k, (x i * y j * z k) • bracketN n (eN j) (bracketN n (eN k) (eN i)) := by
    have natural : bracketN n y (bracketN n z x)
        = ∑ j, ∑ k, ∑ i, (x i * y j * z k) • bracketN n (eN j) (bracketN n (eN k) (eN i)) := by
      rw [bracketN_expand_left y (bracketN n z x)]
      apply Finset.sum_congr rfl; intro j _
      rw [bracketN_bilinear_expand z x, bracketN_sum_right, Finset.smul_sum]
      apply Finset.sum_congr rfl; intro k _
      rw [bracketN_sum_right, Finset.smul_sum]
      apply Finset.sum_congr rfl; intro i _
      rw [bracketN_smul_right, smul_smul]
      congr 1; ring
    rw [natural]
    rw [show (∑ j, ∑ k, ∑ i, (x i * y j * z k) • bracketN n (eN j) (bracketN n (eN k) (eN i)))
          = ∑ j, ∑ i, ∑ k, (x i * y j * z k) • bracketN n (eN j) (bracketN n (eN k) (eN i)) from by
        apply Finset.sum_congr rfl; intro j _; rw [Finset.sum_comm]]
    rw [Finset.sum_comm]
  have expand3 : bracketN n z (bracketN n x y)
      = ∑ i, ∑ j, ∑ k, (x i * y j * z k) • bracketN n (eN k) (bracketN n (eN i) (eN j)) := by
    have natural : bracketN n z (bracketN n x y)
        = ∑ k, ∑ i, ∑ j, (x i * y j * z k) • bracketN n (eN k) (bracketN n (eN i) (eN j)) := by
      rw [bracketN_expand_left z (bracketN n x y)]
      apply Finset.sum_congr rfl; intro k _
      rw [bracketN_bilinear_expand x y, bracketN_sum_right, Finset.smul_sum]
      apply Finset.sum_congr rfl; intro i _
      rw [bracketN_sum_right, Finset.smul_sum]
      apply Finset.sum_congr rfl; intro j _
      rw [bracketN_smul_right, smul_smul]
      congr 1; ring
    rw [natural, Finset.sum_comm]
    apply Finset.sum_congr rfl; intro i _
    rw [Finset.sum_comm]
  rw [expand1, expand2, expand3]
  simp_rw [Finset.smul_sum]
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_eq_zero; intro i _
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_eq_zero; intro j _
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_eq_zero; intro k _
  by_cases h : x i * y j * z k = 0
  · simp [h]
  · have hxi : x i ≠ 0 := fun h0 => h (by rw [h0]; ring)
    have hyj : y j ≠ 0 := fun h0 => h (by rw [h0]; ring)
    have hzk : z k ≠ 0 := fun h0 => h (by rw [h0]; ring)
    have hjac := key i j k hxi hyj hzk
    have hcomb : gsignN n dx dz • (x i * y j * z k) • bracketN n (eN i) (bracketN n (eN j) (eN k))
        + gsignN n dy dx • (x i * y j * z k) • bracketN n (eN j) (bracketN n (eN k) (eN i))
        + gsignN n dz dy • (x i * y j * z k) • bracketN n (eN k) (bracketN n (eN i) (eN j))
        = (x i * y j * z k) • (gsignN n dx dz • bracketN n (eN i) (bracketN n (eN j) (eN k))
            + gsignN n dy dx • bracketN n (eN j) (bracketN n (eN k) (eN i))
            + gsignN n dz dy • bracketN n (eN k) (bracketN n (eN i) (eN j))) := by
      simp_rw [smul_add, smul_smul, mul_comm (x i * y j * z k)]
    rw [hcomb, hjac, smul_zero]

end Indexed
end InhomogeneousDeformations
