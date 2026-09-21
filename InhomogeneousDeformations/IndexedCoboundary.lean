import InhomogeneousDeformations.IndexedJacobi
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Algebra.BigOperators.Fin

/-!
# R2-D (G0-G5) — the coefficient/coboundary identity `Gamma_beta = delta f_beta`

`H1.2`: every declaration here is stated for general `n : ℕ`, with indices
in `Fin (2 * n)`; no `n = 1` specialization, `Fin 2` instance, or literal
five-element basis appears anywhere. `Indexed.lean`, `IndexedLaws.lean` and
`IndexedJacobi.lean` (all three frozen) are never reopened; the freeze is
three layers deep and deliberate -- G4 rests on U0's `Jn_antisymm` and
`Lof_comm` and on P1's bracket formulas, and none of them are restated to
ease this proof. Any new lemma about `Jn`/`Lof` needed here lives in this
module, never retroactively in an accepted one.

`H1.3` (verbatim, do not upgrade): a successful R2-D establishes **P2**:
for every `n ≥ 1`, the coordinate-specified `Gamma_beta` equals the
coboundary `delta f_beta` of the coordinate-specified primitive, on the
free `P`-module. It does **not** establish that this coordinate
`Gamma_beta` is the coefficient actually recovered from the oscillator
source (that identification is **P4**, the declared gap of this stage);
the `T_beta` intertwining, and therefore not that the deformation is
*trivialized* (P2 says the coefficient is a coboundary; the step from
"coboundary" to "trivialized by an explicit even change of generators" is
**P3**); nor the manuscript's `thm:main`, nor a formalization of the
manuscript's own proof, which *derives* `Gamma_beta` from the source
rather than defining it.

## Transcription (Section T)

`Gamma_beta`'s basis values are transcribed from `eq:gamma-ll`,
`eq:gamma-lf`, `eq:gamma-ff`; `f_beta`'s values from `eq:primitive`; all
from `aoi2026_triviality_osp1_2n_revised.tex`. Neither is derived,
adjusted or "simplified" from the other. `eq:primitive` is 1-based
(`f_beta(F_u) = sum_{j=1}^n (beta_{2j-1} L_{2j,u} - beta_{2j} L_{2j-1,u})`);
its 0-based form, used throughout below, is
`sum_{j=0}^{n-1} (beta_{2j} L_{2j+1,u} - beta_{2j+1} L_{2j,u})`
(`fBetaBasis`'s `.inr` case, via `evenIdx`/`oddIdx`). `Gamma_beta`'s FL
sector has no manuscript formula: it is fixed by the declared convention
`Gamma_beta(Y,X) = -(-1)^{p(X)p(Y)} Gamma_beta(X,Y)`
(`GammaBetaBasis`'s `.inr, .inl` case, literally `-gammaLFn`), and G4's FL
proof is reported as corroboration of that convention, not as an
independent check.

## Sector table (G4)

All three substantive sectors close on `Jn_antisymm`, `Lof_comm` and G0's
contraction identity alone, exactly as Agent1's hand analysis predicted:

* `G4_LL` -- `f_beta` vanishes on the even part, so all three terms of
  `delta f_beta` vanish, matching `eq:gamma-ll`.
* `G4_LF` -- expanding `f_beta(F_w)` via G0's contraction identity and the
  accepted `LL` bracket formula reduces the `[L_{uv}, f_beta(F_w)]_0`
  term to `eq:gamma-lf` plus exactly `f_beta([L_{uv},F_w]_0)`, which then
  cancels against `delta`'s own third term.
* `G4_FF` -- the two `[f_beta(F_u),F_v]_0`/`[f_beta(F_v),F_u]_0` terms'
  `J`-parts cancel by `Jn_antisymm`, leaving `eq:gamma-ff`; the third term
  vanishes since `bracketFFn` lands purely in the (`f_beta`-killed) even
  part.
* `G4_FL` -- corroboration of the declared convention: computed the same
  way as `G4_LF`/`G4_LL`'s methods combined, and found to match.

`G4` assembles all four sectors into one statement on all ordered basis
pairs; `G5` extends it, unconditionally, to arbitrary elements of
`IndexedMod n` via the same `bilinearExtend` combinator both `GammaBetaN`
and `deltaFN` are built from -- since both sides are literally the
bilinear extension of basis functions that agree pointwise (G4), equality
on all of `IndexedMod n` follows with no separate homogeneous-expansion
argument, and in particular subsumes the homogeneous-hypothesis form the
design/packet describes.
-/

namespace InhomogeneousDeformations
namespace Indexed

/-! ## G0 -- `beta`, the vector `v`, and the contraction identity -/

/-- `beta_u`, the polynomial generator of `Pn n` at index `u`. -/
noncomputable def betaN (n : ℕ) (u : Fin (2 * n)) : Pn n := MvPolynomial.X u

/-- The index-`t` partner under `Jn`: `t+1` if `t` is even, `t-1` if odd.
Bounds proved by `omega` from `t.isLt` alone -- linear in `n`, never
multiplying two unknowns (the rank-nonlinearity trap named in every
earlier round). -/
noncomputable def partnerN (n : ℕ) (t : Fin (2 * n)) : Fin (2 * n) :=
  if h : (t : ℕ) % 2 = 0 then ⟨(t : ℕ) + 1, by have := t.isLt; omega⟩
  else ⟨(t : ℕ) - 1, by have := t.isLt; omega⟩

/-- The vector `v = sum_{j=0}^{n-1} (beta_{2j+1} e_{2j} - beta_{2j} e_{2j+1})`
(0-based `eq:v`), given componentwise via the partner index: `v_t =
beta_{t+1}` for `t` even, `-beta_{t-1}` for `t` odd -- the unique closed
form matching that sum termwise. -/
noncomputable def vN (n : ℕ) (t : Fin (2 * n)) : Pn n :=
  if (t : ℕ) % 2 = 0 then betaN n (partnerN n t) else -betaN n (partnerN n t)

/-- The 0-based even/odd index pair for loop variable `j : Fin n`:
`evenIdx n j = 2*j`, `oddIdx n j = 2*j+1`. -/
noncomputable def evenIdx (n : ℕ) (j : Fin n) : Fin (2 * n) := ⟨2 * (j : ℕ), by have := j.isLt; omega⟩
noncomputable def oddIdx (n : ℕ) (j : Fin n) : Fin (2 * n) := ⟨2 * (j : ℕ) + 1, by have := j.isLt; omega⟩

theorem Jn_ne_zero_iff_partner (n : ℕ) (t u : Fin (2 * n)) (h : Jn n t u ≠ 0) :
    t = partnerN n u := by
  unfold Jn at h
  split_ifs at h with h1 h2
  · obtain ⟨he, hs⟩ := h1
    apply Fin.ext
    unfold partnerN
    rw [dif_neg (show ¬ (u : ℕ) % 2 = 0 by omega)]
    dsimp only
    omega
  · obtain ⟨he, hs⟩ := h2
    apply Fin.ext
    unfold partnerN
    rw [dif_pos he]
    dsimp only
    omega
  · exact absurd (cratN_zero n) h

/-- `G0`'s boxed identity: `sum_t v_t J_{tu} = beta_u`. The sum collapses
to the single term `t = partnerN n u` via `Finset.sum_eq_single`, computed
by cases on `u`'s parity. -/
theorem contraction_identity (n : ℕ) (u : Fin (2 * n)) :
    ∑ t : Fin (2 * n), vN n t * Jn n t u = betaN n u := by
  rw [Finset.sum_eq_single (partnerN n u)]
  · rcases eq_or_ne ((u : ℕ) % 2) 0 with hu | hu
    · have e1 : partnerN n u = ⟨(u : ℕ) + 1, by have := u.isLt; omega⟩ := by
        unfold partnerN; rw [dif_pos hu]
      rw [e1]
      have hv : vN n (⟨(u : ℕ) + 1, by have := u.isLt; omega⟩ : Fin (2 * n)) = -betaN n u := by
        have hp : partnerN n (⟨(u : ℕ) + 1, by have := u.isLt; omega⟩ : Fin (2 * n)) = u := by
          apply Fin.ext
          unfold partnerN
          rw [dif_neg (show ¬ ((⟨(u : ℕ) + 1, by have := u.isLt; omega⟩ : Fin (2 * n)) : ℕ) % 2 = 0 by
            dsimp only; omega)]
          dsimp only; omega
        unfold vN
        rw [if_neg (show ¬ ((⟨(u : ℕ) + 1, by have := u.isLt; omega⟩ : Fin (2 * n)) : ℕ) % 2 = 0 by
          dsimp only; omega), hp]
      have hj : Jn n (⟨(u : ℕ) + 1, by have := u.isLt; omega⟩ : Fin (2 * n)) u = -1 := by
        unfold Jn
        rw [if_neg (by dsimp only; omega),
            if_pos (show (u : ℕ) % 2 = 0 ∧ ((⟨(u : ℕ) + 1, by have := u.isLt; omega⟩ : Fin (2 * n)) : ℕ)
                = (u : ℕ) + 1 from ⟨hu, by dsimp only⟩)]
        rw [show (-1 : ℚ) = -(1 : ℚ) from rfl, ← cratN_neg, cratN_one]
      rw [hv, hj]
      ring
    · have e1 : partnerN n u = ⟨(u : ℕ) - 1, by have := u.isLt; omega⟩ := by
        unfold partnerN; rw [dif_neg hu]
      rw [e1]
      have hv : vN n (⟨(u : ℕ) - 1, by have := u.isLt; omega⟩ : Fin (2 * n)) = betaN n u := by
        have hp : partnerN n (⟨(u : ℕ) - 1, by have := u.isLt; omega⟩ : Fin (2 * n)) = u := by
          apply Fin.ext
          unfold partnerN
          rw [dif_pos (show ((⟨(u : ℕ) - 1, by have := u.isLt; omega⟩ : Fin (2 * n)) : ℕ) % 2 = 0 by
            dsimp only; omega)]
          dsimp only; omega
        unfold vN
        rw [if_pos (show ((⟨(u : ℕ) - 1, by have := u.isLt; omega⟩ : Fin (2 * n)) : ℕ) % 2 = 0 by
          dsimp only; omega), hp]
      have hj : Jn n (⟨(u : ℕ) - 1, by have := u.isLt; omega⟩ : Fin (2 * n)) u = 1 := by
        unfold Jn
        rw [if_pos (show ((⟨(u : ℕ) - 1, by have := u.isLt; omega⟩ : Fin (2 * n)) : ℕ) % 2 = 0
              ∧ (u : ℕ) = ((⟨(u : ℕ) - 1, by have := u.isLt; omega⟩ : Fin (2 * n)) : ℕ) + 1 from
              ⟨by dsimp only; omega, by dsimp only; omega⟩)]
        exact cratN_one n
      rw [hv, hj]
      ring
  · intro t _ ht
    have : Jn n t u = 0 := by
      by_contra hne
      exact ht (Jn_ne_zero_iff_partner n t u hne)
    rw [this, mul_zero]
  · intro h; exact absurd (Finset.mem_univ _) h

/-- The antisymmetric form `sum_t v_t J_{ut} = -beta_u`, via
`contraction_identity` and `Jn_antisymm`. -/
theorem contraction_identity_neg (n : ℕ) (u : Fin (2 * n)) :
    ∑ t : Fin (2 * n), vN n t * Jn n u t = -betaN n u := by
  have heq : ∀ t, vN n t * Jn n u t = -(vN n t * Jn n t u) := by
    intro t
    rw [Jn_antisymm n t u]
    ring
  simp_rw [heq]
  rw [Finset.sum_neg_distrib, contraction_identity]

/-- The interleaving equivalence `Fin n × Fin 2 ≃ Fin (2*n)`, `(j,p) ↦
2*j+p`, used to split a `Fin (2*n)` sum into its even and odd parts. -/
noncomputable def pairEquivN (n : ℕ) : Fin n × Fin 2 ≃ Fin (2 * n) :=
  finProdFinEquiv.trans (finCongr (Nat.mul_comm n 2))

theorem sum_split_even_odd {M : Type*} [AddCommMonoid M] (n : ℕ) (g : Fin (2 * n) → M) :
    ∑ t : Fin (2 * n), g t
      = ∑ j : Fin n, (g (evenIdx n j) + g (oddIdx n j)) := by
  rw [← Equiv.sum_comp (pairEquivN n) g]
  rw [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro j _
  rw [Fin.sum_univ_two]
  congr 2
  · apply Fin.ext; simp [pairEquivN, finProdFinEquiv, finCongr, evenIdx]
  · apply Fin.ext; simp [pairEquivN, finProdFinEquiv, finCongr, oddIdx]; omega

/-! ## G1 -- the primitive `f_beta` -/

/-- `f_beta` on the basis: `0` on `L`'s, and `eq:primitive`'s 0-based form
on `F_u` (`H1` in `T`'s two hazards, converted from the manuscript's
1-based `sum_{j=1}^n`). -/
noncomputable def fBetaBasis (n : ℕ) : IndexedBasis n → IndexedMod n
  | .inl _ => 0
  | .inr u => ∑ j : Fin n,
      (betaN n (evenIdx n j) • eN (Lof (oddIdx n j) u) - betaN n (oddIdx n j) • eN (Lof (evenIdx n j) u))

/-- `G1`'s required transcription check: `eq:primitive`'s `sum_j` form
agrees with the `thm:main` proof's `f_beta(F_u) = -sum_t v_t L_{tu}`
form. Proved via `sum_split_even_odd` plus the `partnerN` computation
(never by deriving one form from the other). -/
theorem fBetaBasis_alt_form (n : ℕ) (u : Fin (2 * n)) :
    fBetaBasis n (Fof u) = -∑ t : Fin (2 * n), vN n t • eN (Lof t u) := by
  show (∑ j : Fin n,
      (betaN n (evenIdx n j) • eN (Lof (oddIdx n j) u) - betaN n (oddIdx n j) • eN (Lof (evenIdx n j) u)))
    = -∑ t : Fin (2 * n), vN n t • eN (Lof t u)
  rw [sum_split_even_odd n (fun t => vN n t • eN (Lof t u))]
  rw [← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl
  intro j _
  have hve : vN n (evenIdx n j) = betaN n (oddIdx n j) := by
    have he : (evenIdx n j : ℕ) % 2 = 0 := by unfold evenIdx; dsimp only; omega
    have hp : partnerN n (evenIdx n j) = oddIdx n j := by
      apply Fin.ext; unfold partnerN; rw [dif_pos he]; unfold evenIdx oddIdx; dsimp only
    unfold vN
    rw [if_pos he, hp]
  have hvo : vN n (oddIdx n j) = -betaN n (evenIdx n j) := by
    have ho : ¬ (oddIdx n j : ℕ) % 2 = 0 := by unfold oddIdx; dsimp only; omega
    have hp : partnerN n (oddIdx n j) = evenIdx n j := by
      apply Fin.ext; unfold partnerN; rw [dif_neg ho]; unfold evenIdx oddIdx; dsimp only; omega
    unfold vN
    rw [if_neg ho, hp]
  rw [hve, hvo]
  simp only [neg_smul]
  abel

/-- `G1`'s oddness law at basis level: `f_beta` is degree-raising by
exactly `1`. -/
theorem fBetaBasis_degree (n : ℕ) (i k : IndexedBasis n) (h : fBetaBasis n i k ≠ 0) :
    parity k = parity i + 1 := by
  match i with
  | .inl _ => simp [fBetaBasis] at h
  | .inr u =>
    match k with
    | .inl _ => simp [parity]; decide
    | .inr m => exfalso; apply h; simp [fBetaBasis]

/-- `f_beta` extended `P`-linearly to all of `IndexedMod n`. -/
noncomputable def fBetaN (n : ℕ) (x : IndexedMod n) : IndexedMod n :=
  ∑ i : IndexedBasis n, x i • fBetaBasis n i

theorem fBetaN_eN (n : ℕ) (i : IndexedBasis n) : fBetaN n (eN i) = fBetaBasis n i := by
  unfold fBetaN
  rw [Finset.sum_eq_single i]
  · simp [eN]
  · intro b _ hb
    have : eN i b = 0 := by unfold eN; rw [if_neg hb]
    simp [this]
  · intro h; exact absurd (Finset.mem_univ i) h

theorem fBetaN_add (n : ℕ) (x1 x2 : IndexedMod n) :
    fBetaN n (x1 + x2) = fBetaN n x1 + fBetaN n x2 := by
  unfold fBetaN
  funext k
  simp only [Finset.sum_apply, Pi.add_apply]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro i _
  simp only [Pi.smul_apply, smul_eq_mul]; ring

theorem fBetaN_smul (n : ℕ) (c : Pn n) (x : IndexedMod n) :
    fBetaN n (c • x) = c • fBetaN n x := by
  unfold fBetaN
  funext k
  simp only [Finset.sum_apply, Pi.smul_apply]
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro i _
  simp only [smul_eq_mul]; ring

theorem fBetaN_zero (n : ℕ) : fBetaN n (0 : IndexedMod n) = 0 := by
  unfold fBetaN; simp

theorem fBetaN_neg (n : ℕ) (x : IndexedMod n) : fBetaN n (-x) = -fBetaN n x := by
  have h1 : fBetaN n (-x) + fBetaN n x = 0 := by
    rw [← fBetaN_add, neg_add_cancel, fBetaN_zero]
  exact eq_neg_of_add_eq_zero_left h1

theorem fBetaBasis_Lof (n : ℕ) (a b : Fin (2 * n)) : fBetaBasis n (Lof a b) = 0 := by
  rcases eq_or_ne a b with hab | hab
  · subst hab; rw [Lof, dif_pos (le_refl a)]; rfl
  · rcases lt_or_gt_of_ne hab with hlt | hgt
    · rw [Lof, dif_pos hlt.le]; rfl
    · rw [Lof, dif_neg (not_le.mpr hgt)]; rfl

theorem fBetaN_Lof (n : ℕ) (a b : Fin (2 * n)) : fBetaN n (eN (Lof a b)) = 0 := by
  rw [fBetaN_eN, fBetaBasis_Lof]

/-- `G1`'s module-level oddness, transposed in shape from
`IndexedLaws.lean`'s `bracketN_isHomogN`. -/
theorem fBetaN_isHomogN (n : ℕ) (x : IndexedMod n) (dx : ZMod 2) (hx : IsHomogN x dx) :
    IsHomogN (fBetaN n x) (dx + 1) := by
  intro k hk
  by_contra hne
  apply hk
  unfold fBetaN
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  apply Finset.sum_eq_zero; intro i _
  by_cases hxi : x i = 0
  · simp [hxi]
  · have hpi := hx i hxi
    have hz : fBetaBasis n i k = 0 := by
      by_contra hc
      exact hne (by rw [fBetaBasis_degree n i k hc, hpi])
    simp [hz]

/-! ## A generic `P`-bilinear extension from a basis pairing

Shared by `GammaBetaN` (G2) and `deltaFN` (G3): both are `∑∑ (x i * y j) •
F i j` for their own basis function `F`, so this one combinator and its
bilinearity toolkit serve both, instead of duplicating `bracketN`'s own
toolkit (frozen, `Indexed.lean`) for each new pairing. -/

noncomputable def bilinearExtend {n : ℕ} (F : IndexedBasis n → IndexedBasis n → IndexedMod n)
    (x y : IndexedMod n) : IndexedMod n :=
  ∑ i : IndexedBasis n, ∑ j : IndexedBasis n, (x i * y j) • F i j

theorem bilinearExtend_eN_eN {n : ℕ} (F : IndexedBasis n → IndexedBasis n → IndexedMod n)
    (X Y : IndexedBasis n) : bilinearExtend F (eN X) (eN Y) = F X Y := by
  unfold bilinearExtend
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

/-! ## Finite-sum bilinearity toolkit for `bracketN`, general `n`

`IndexedJacobi.lean`'s own toolkit is specific to `IndexedBasis n`-indexed
sums (U5's own need); G4 needs the analogue for `Fin (2*n)`-indexed sums
(over `v`'s components), plus `bracketN_neg_left` (only the right-argument
form was needed before), plus the trivial `bracketN_zero_left/right`. -/

theorem bracketN_zero_left (n : ℕ) (y : IndexedMod n) : bracketN n 0 y = 0 := by
  unfold bracketN; simp

theorem bracketN_zero_right (n : ℕ) (x : IndexedMod n) : bracketN n x 0 = 0 := by
  unfold bracketN; simp

theorem bracketN_neg_left {n : ℕ} (x y : IndexedMod n) : bracketN n (-x) y = -bracketN n x y := by
  have h1 : bracketN n (-x) y + bracketN n x y = 0 := by
    rw [← bracketN_add_left, neg_add_cancel, bracketN_zero_left]
  exact eq_neg_of_add_eq_zero_left h1

theorem bracketN_sum_right' {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (f : ι → IndexedMod n)
    (x : IndexedMod n) :
    bracketN n x (∑ j ∈ s, f j) = ∑ j ∈ s, bracketN n x (f j) := by
  induction s using Finset.induction with
  | empty => simp [bracketN_zero_right]
  | @insert a s ha ih => rw [Finset.sum_insert ha, bracketN_add_right, ih, Finset.sum_insert ha]

theorem bracketN_sum_left' {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (f : ι → IndexedMod n)
    (y : IndexedMod n) :
    bracketN n (∑ i ∈ s, f i) y = ∑ i ∈ s, bracketN n (f i) y := by
  induction s using Finset.induction with
  | empty => simp [bracketN_zero_left]
  | @insert a s ha ih => rw [Finset.sum_insert ha, bracketN_add_left, ih, Finset.sum_insert ha]

/-! ## G2 -- the coefficient `Gamma_beta`, all four sectors -/

/-- `eq:gamma-lf`: `Gamma_beta(L_uv,F_w) = (1/2)(beta_u L_vw + beta_v L_uw)`. -/
noncomputable def gammaLFn (n : ℕ) (u v w : Fin (2 * n)) : IndexedMod n :=
  cratN n (1 / 2) • (betaN n u • eN (Lof v w) + betaN n v • eN (Lof u w))

/-- `eq:gamma-ff`: `Gamma_beta(F_u,F_v) = -(1/2)(beta_u F_v + beta_v F_u)`. -/
noncomputable def gammaFFn (n : ℕ) (u v : Fin (2 * n)) : IndexedMod n :=
  -(cratN n (1 / 2) • (betaN n u • eN (Fof v) + betaN n v • eN (Fof u)))

/-- `Gamma_beta` on all four basis sectors: `eq:gamma-ll` (`0`),
`gammaLFn` (LF, transcribed), `-gammaLFn` (FL, the declared convention of
G2 -- `Gamma_beta(Y,X) = -(-1)^{p(X)p(Y)} Gamma_beta(X,Y)` specialized to
`X=L_uv,Y=F_w`, giving `-(1/2)(beta_uL_vw+beta_vL_uw)` exactly), and
`gammaFFn` (FF, transcribed). -/
noncomputable def GammaBetaBasis (n : ℕ) : IndexedBasis n → IndexedBasis n → IndexedMod n
  | .inl ⟨(_, _), _⟩, .inl ⟨(_, _), _⟩ => 0
  | .inl ⟨(u, v), _⟩, .inr w => gammaLFn n u v w
  | .inr w, .inl ⟨(u, v), _⟩ => -gammaLFn n u v w
  | .inr u, .inr v => gammaFFn n u v

noncomputable def GammaBetaN (n : ℕ) (x y : IndexedMod n) : IndexedMod n :=
  bilinearExtend (GammaBetaBasis n) x y

/-- `G2`'s map degree: `Gamma_beta(g_{P,i},g_{P,j}) ⊆ g_{P,i+j+1}`. -/
theorem GammaBetaBasis_degree0 {n : ℕ} (i j k : IndexedBasis n)
    (h : GammaBetaBasis n i j k ≠ 0) : parity k = parity i + parity j + 1 := by
  match i, j with
  | .inl ⟨(_, _), _⟩, .inl ⟨(_, _), _⟩ => simp [GammaBetaBasis] at h
  | .inl ⟨(_, _), _⟩, .inr _ =>
    match k with
    | .inr _ => exfalso; apply h; simp [GammaBetaBasis, gammaLFn, eN_Lof_apply_inr]
    | .inl _ => simp [parity]; decide
  | .inr _, .inl ⟨(_, _), _⟩ =>
    match k with
    | .inr _ => exfalso; apply h; simp [GammaBetaBasis, gammaLFn, eN_Lof_apply_inr]
    | .inl _ => simp [parity]; decide
  | .inr _, .inr _ =>
    match k with
    | .inl _ => exfalso; apply h; simp [GammaBetaBasis, gammaFFn, Fof, eN]
    | .inr _ => simp [parity]; decide

/-! ## G3 -- the coboundary operator, for a general odd `P`-linear map `f` -/

/-- `(delta f)(X,Y) = [f(X),Y]_0 + (-1)^{p(X)}[X,f(Y)]_0 - f([X,Y]_0)` on
basis pairs, using the accepted `bracketN` and `gsignN n (parity X) 1 =
(-1)^{p(X)}` (`gsignN`'s own `q=1` slice). -/
noncomputable def deltaFBasis (n : ℕ) (f : IndexedMod n → IndexedMod n) (X Y : IndexedBasis n) :
    IndexedMod n :=
  bracketN n (f (eN X)) (eN Y) + gsignN n (parity X) 1 • bracketN n (eN X) (f (eN Y))
    - f (bracketN n (eN X) (eN Y))

noncomputable def deltaFN (n : ℕ) (f : IndexedMod n → IndexedMod n) (x y : IndexedMod n) :
    IndexedMod n :=
  bilinearExtend (deltaFBasis n f) x y

/-! ## G4, LL sector -/

theorem GammaBetaBasis_Lof_Lof (n : ℕ) (a b c d : Fin (2 * n)) :
    GammaBetaBasis n (Lof a b) (Lof c d) = 0 := by
  rcases eq_or_ne a b with hab | hab
  · subst hab
    rcases eq_or_ne c d with hcd | hcd
    · subst hcd; rw [Lof, dif_pos (le_refl a), Lof, dif_pos (le_refl c)]; rfl
    · rcases lt_or_gt_of_ne hcd with hlt | hgt
      · rw [Lof, dif_pos (le_refl a), Lof, dif_pos hlt.le]; rfl
      · rw [Lof, dif_pos (le_refl a), Lof, dif_neg (not_le.mpr hgt)]; rfl
  · rcases lt_or_gt_of_ne hab with hablt | habgt
    · rw [Lof, dif_pos hablt.le]
      rcases eq_or_ne c d with hcd | hcd
      · subst hcd; rw [Lof, dif_pos (le_refl c)]; rfl
      · rcases lt_or_gt_of_ne hcd with hlt | hgt
        · rw [Lof, dif_pos hlt.le]; rfl
        · rw [Lof, dif_neg (not_le.mpr hgt)]; rfl
    · rw [Lof, dif_neg (not_le.mpr habgt)]
      rcases eq_or_ne c d with hcd | hcd
      · subst hcd; rw [Lof, dif_pos (le_refl c)]; rfl
      · rcases lt_or_gt_of_ne hcd with hlt | hgt
        · rw [Lof, dif_pos hlt.le]; rfl
        · rw [Lof, dif_neg (not_le.mpr hgt)]; rfl

/-- **LL sector**: reduces to "`f_beta` vanishes on the even part" plus
`P`-linearity, as Agent1's hand analysis predicted. -/
theorem G4_LL (n : ℕ) (a b c d : Fin (2 * n)) :
    GammaBetaBasis n (Lof a b) (Lof c d) = deltaFBasis n (fBetaN n) (Lof a b) (Lof c d) := by
  rw [GammaBetaBasis_Lof_Lof]
  unfold deltaFBasis
  rw [fBetaN_Lof, bracketN_zero_left, fBetaN_Lof, bracketN_zero_right, smul_zero]
  rw [bracketN_eN_eN, bracketBasisN_Lof_Lof]
  unfold bracketLLn
  rw [fBetaN_add, fBetaN_add, fBetaN_add, fBetaN_smul, fBetaN_smul, fBetaN_smul, fBetaN_smul,
      fBetaN_Lof, fBetaN_Lof, fBetaN_Lof, fBetaN_Lof]
  simp

/-! ## Helper facts feeding the LF, FF and FL sectors -/

theorem sum_vN_smul_Lof (n : ℕ) (a : Fin (2 * n)) :
    ∑ t : Fin (2 * n), vN n t • eN (Lof a t) = -fBetaN n (eN (Fof a)) := by
  have h1 : ∀ t, (eN (Lof a t) : IndexedMod n) = eN (Lof t a) := by
    intro t; rw [Lof_comm]
  simp_rw [h1]
  rw [fBetaN_eN, fBetaBasis_alt_form]
  rw [neg_neg]

theorem sum_vN_mul_Jn_const_smul (n : ℕ) (s : Fin (2 * n)) (c : Pn n) (X : IndexedMod n) :
    ∑ t : Fin (2 * n), vN n t • ((c * Jn n s t) • X) = (c * -betaN n s) • X := by
  have step1 : ∀ t : Fin (2 * n), vN n t • ((c * Jn n s t) • X) = (vN n t * (c * Jn n s t)) • X := by
    intro t; rw [smul_smul]
  simp_rw [step1]
  rw [← Finset.sum_smul]
  congr 1
  calc ∑ t : Fin (2 * n), vN n t * (c * Jn n s t)
      = ∑ t : Fin (2 * n), c * (vN n t * Jn n s t) := by
        apply Finset.sum_congr rfl; intro t _; ring
    _ = c * ∑ t : Fin (2 * n), vN n t * Jn n s t := by rw [Finset.mul_sum]
    _ = c * (-betaN n s) := by rw [contraction_identity_neg]

theorem sum_vN_mul_Jn_const_smul' (n : ℕ) (s : Fin (2 * n)) (c : Pn n) (X : IndexedMod n) :
    ∑ t : Fin (2 * n), vN n t • ((c * Jn n t s) • X) = (c * betaN n s) • X := by
  have step1 : ∀ t : Fin (2 * n), vN n t • ((c * Jn n t s) • X) = (vN n t * (c * Jn n t s)) • X := by
    intro t; rw [smul_smul]
  simp_rw [step1]
  rw [← Finset.sum_smul]
  congr 1
  calc ∑ t : Fin (2 * n), vN n t * (c * Jn n t s)
      = ∑ t : Fin (2 * n), c * (vN n t * Jn n t s) := by
        apply Finset.sum_congr rfl; intro t _; ring
    _ = c * ∑ t : Fin (2 * n), vN n t * Jn n t s := by rw [Finset.mul_sum]
    _ = c * betaN n s := by rw [contraction_identity]

theorem sum_vN_const_smul_Lof (n : ℕ) (a : Fin (2 * n)) (c : Pn n) :
    ∑ t : Fin (2 * n), vN n t • (c • eN (Lof a t)) = c • (-fBetaN n (eN (Fof a))) := by
  have step1 : ∀ t : Fin (2 * n), vN n t • (c • eN (Lof a t)) = c • (vN n t • eN (Lof a t)) := by
    intro t; rw [smul_smul, smul_smul]; congr 1; ring
  simp_rw [step1]
  rw [← Finset.smul_sum, sum_vN_smul_Lof]

theorem sum_vN_const_smul_Lof' (n : ℕ) (a : Fin (2 * n)) (c : Pn n) :
    ∑ t : Fin (2 * n), vN n t • (c • eN (Lof t a)) = c • (-fBetaN n (eN (Fof a))) := by
  have h1 : ∀ t : Fin (2 * n), (eN (Lof t a) : IndexedMod n) = eN (Lof a t) := by
    intro t; rw [Lof_comm]
  simp_rw [h1]
  exact sum_vN_const_smul_Lof n a c

/-- The manuscript's `F(v) = sum_t v_t F_t`, notation-hazard-checked
(`T`'s second named hazard: `v` is a vector here, never an index). -/
noncomputable def FVecBeta (n : ℕ) : IndexedMod n := ∑ t : Fin (2 * n), vN n t • eN (Fof t)

theorem sum_vN_const_smul_Fof (n : ℕ) (c : Pn n) :
    ∑ t : Fin (2 * n), vN n t • (c • eN (Fof t)) = c • FVecBeta n := by
  have step1 : ∀ t : Fin (2 * n), vN n t • (c • eN (Fof t)) = c • (vN n t • eN (Fof t)) := by
    intro t; rw [smul_smul, smul_smul]; congr 1; ring
  simp_rw [step1]
  rw [← Finset.smul_sum]
  rfl

/-! ## G4, LF sector -/

theorem GammaBetaBasis_Lof_Fof (n : ℕ) (u v w : Fin (2 * n)) :
    GammaBetaBasis n (Lof u v) (Fof w) = gammaLFn n u v w := by
  rcases eq_or_ne u v with heq | hne
  · subst heq; rw [Lof, dif_pos (le_refl u)]; rfl
  · rcases lt_or_gt_of_ne hne with hlt | hgt
    · rw [Lof, dif_pos hlt.le]; rfl
    · rw [Lof, dif_neg (not_le.mpr hgt)]
      show gammaLFn n v u w = gammaLFn n u v w
      unfold gammaLFn; rw [add_comm]

/-- **LF sector**: expanding `f_beta(F_w) = -sum_t v_t L_{tw}` through the
accepted `LL` bracket formula and G0's contraction identity produces
`eq:gamma-lf` plus exactly `f_beta([L_uv,F_w]_0)`, which cancels against
`delta`'s own third term -- as Agent1's hand analysis predicted. -/
theorem G4_LF (n : ℕ) (u v w : Fin (2 * n)) :
    GammaBetaBasis n (Lof u v) (Fof w) = deltaFBasis n (fBetaN n) (Lof u v) (Fof w) := by
  unfold deltaFBasis
  rw [fBetaN_Lof, bracketN_zero_left, zero_add, parity_Lof, gsignN_01, one_smul,
      GammaBetaBasis_Lof_Fof]
  rw [fBetaN_eN, fBetaBasis_alt_form, bracketN_neg_right]
  rw [show bracketN n (eN (Lof u v : IndexedBasis n)) (∑ t : Fin (2 * n), vN n t • eN (Lof t w))
      = ∑ t : Fin (2 * n), vN n t • bracketBasisN n (Lof u v) (Lof t w) from by
    rw [bracketN_sum_right']
    apply Finset.sum_congr rfl; intro t _
    rw [bracketN_smul_right, bracketN_eN_eN]]
  simp_rw [bracketBasisN_Lof_Lof]
  unfold bracketLLn
  simp_rw [smul_add]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_add_distrib,
      sum_vN_mul_Jn_const_smul n v (cratN n (1 / 2)) (eN (Lof u w)),
      sum_vN_mul_Jn_const_smul n u (cratN n (1 / 2)) (eN (Lof v w)),
      sum_vN_const_smul_Lof n u (cratN n (1 / 2) * Jn n v w),
      sum_vN_const_smul_Lof n v (cratN n (1 / 2) * Jn n u w)]
  rw [bracketN_eN_eN, bracketBasisN_Lof_Fof]
  unfold bracketLFn
  rw [fBetaN_add, fBetaN_smul, fBetaN_smul]
  simp only [smul_neg, mul_neg]
  unfold gammaLFn
  simp only [neg_smul, smul_add, smul_smul]
  ring_nf

/-! ## G4, FF sector -/

/-- **FF sector**: `[f_beta(F_u),F_v]_0` and `[f_beta(F_v),F_u]_0`'s
`J`-parts cancel by `Jn_antisymm` ("interchange and add"), leaving
`eq:gamma-ff`; the third `delta`-term vanishes since `bracketFFn` lands
purely in the (`f_beta`-killed) even part -- as Agent1's hand analysis
predicted. -/
theorem G4_FF (n : ℕ) (u v : Fin (2 * n)) :
    GammaBetaBasis n (Fof u) (Fof v) = deltaFBasis n (fBetaN n) (Fof u) (Fof v) := by
  show gammaFFn n u v = deltaFBasis n (fBetaN n) (Fof u) (Fof v)
  unfold deltaFBasis
  rw [parity_Fof, gsignN_11]
  rw [fBetaN_eN, fBetaBasis_alt_form, fBetaN_eN, fBetaBasis_alt_form]
  rw [bracketN_neg_left, bracketN_neg_right]
  rw [show bracketN n (∑ t : Fin (2 * n), vN n t • eN (Lof t u)) (eN (Fof v))
      = ∑ t : Fin (2 * n), vN n t • bracketBasisN n (Lof t u) (Fof v) from by
    rw [bracketN_sum_left']
    apply Finset.sum_congr rfl; intro t _
    rw [bracketN_smul_left, bracketN_eN_eN]]
  rw [show bracketN n (eN (Fof u : IndexedBasis n)) (∑ t : Fin (2 * n), vN n t • eN (Lof t v))
      = ∑ t : Fin (2 * n), vN n t • bracketBasisN n (Fof u) (Lof t v) from by
    rw [bracketN_sum_right']
    apply Finset.sum_congr rfl; intro t _
    rw [bracketN_smul_right, bracketN_eN_eN]]
  simp_rw [bracketBasisN_Lof_Fof, bracketBasisN_Fof_Lof]
  simp only [smul_neg, neg_neg, neg_smul, one_smul]
  unfold bracketLFn
  simp_rw [smul_add, neg_add]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib,
      Finset.sum_neg_distrib, Finset.sum_neg_distrib,
      sum_vN_const_smul_Fof n (cratN n (1 / 2) * Jn n u v),
      sum_vN_mul_Jn_const_smul' n v (cratN n (1 / 2)) (eN (Fof u)),
      sum_vN_const_smul_Fof n (cratN n (1 / 2) * Jn n v u),
      sum_vN_mul_Jn_const_smul' n u (cratN n (1 / 2)) (eN (Fof v))]
  rw [bracketN_eN_eN, bracketBasisN_Fof_Fof]
  unfold bracketFFn
  rw [fBetaN_smul, fBetaN_Lof]
  rw [Jn_antisymm n u v]
  simp only [smul_zero, mul_neg, neg_smul, neg_neg, sub_zero]
  unfold gammaFFn
  simp only [smul_add, smul_smul]
  ring_nf

/-! ## G4, FL sector: corroboration of the declared convention, not an
independent check -/

theorem GammaBetaBasis_Fof_Lof (n : ℕ) (w u v : Fin (2 * n)) :
    GammaBetaBasis n (Fof w) (Lof u v) = -gammaLFn n u v w := by
  rcases eq_or_ne u v with heq | hne
  · subst heq; rw [Lof, dif_pos (le_refl u)]; rfl
  · rcases lt_or_gt_of_ne hne with hlt | hgt
    · rw [Lof, dif_pos hlt.le]; rfl
    · rw [Lof, dif_neg (not_le.mpr hgt)]
      show -gammaLFn n v u w = -gammaLFn n u v w
      unfold gammaLFn; rw [add_comm]

/-- **FL sector**: computed the same way as LF/LL's methods combined
(never derived from `-eq:gamma-lf` directly, which would be circular);
the result matches G2's declared convention `-gammaLFn`, corroborating
it. -/
theorem G4_FL (n : ℕ) (u v w : Fin (2 * n)) :
    GammaBetaBasis n (Fof w) (Lof u v) = deltaFBasis n (fBetaN n) (Fof w) (Lof u v) := by
  rw [GammaBetaBasis_Fof_Lof]
  unfold deltaFBasis
  rw [fBetaN_Lof, bracketN_zero_right, smul_zero, add_zero]
  rw [fBetaN_eN, fBetaBasis_alt_form, bracketN_neg_left]
  rw [show bracketN n (∑ t : Fin (2 * n), vN n t • eN (Lof t w)) (eN (Lof u v))
      = ∑ t : Fin (2 * n), vN n t • bracketBasisN n (Lof t w) (Lof u v) from by
    rw [bracketN_sum_left']
    apply Finset.sum_congr rfl; intro t _
    rw [bracketN_smul_left, bracketN_eN_eN]]
  simp_rw [bracketBasisN_Lof_Lof]
  unfold bracketLLn
  simp_rw [smul_add]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_add_distrib]
  rw [sum_vN_const_smul_Lof' n v (cratN n (1 / 2) * Jn n w u),
      sum_vN_mul_Jn_const_smul' n u (cratN n (1 / 2)) (eN (Lof w v)),
      sum_vN_const_smul_Lof' n u (cratN n (1 / 2) * Jn n w v),
      sum_vN_mul_Jn_const_smul' n v (cratN n (1 / 2)) (eN (Lof w u))]
  rw [bracketN_eN_eN, bracketBasisN_Fof_Lof, fBetaN_neg]
  unfold bracketLFn
  rw [fBetaN_add, fBetaN_smul, fBetaN_smul]
  rw [Jn_antisymm n u w, Jn_antisymm n v w]
  simp only [smul_neg]
  unfold gammaLFn
  rw [show Lof v w = Lof w v from Lof_comm v w, show Lof u w = Lof w u from Lof_comm u w]
  simp only [neg_smul, smul_add, smul_smul, mul_neg]
  ring_nf

/-! ## G4: assembled on all basis pairs -/

theorem G4 (n : ℕ) (i j : IndexedBasis n) : GammaBetaBasis n i j = deltaFBasis n (fBetaN n) i j := by
  match i, j with
  | .inl ⟨(a, b), h1⟩, .inl ⟨(c, d), h2⟩ =>
      have e1 : (Sum.inl ⟨(a, b), h1⟩ : IndexedBasis n) = Lof a b := by rw [Lof, dif_pos h1]
      have e2 : (Sum.inl ⟨(c, d), h2⟩ : IndexedBasis n) = Lof c d := by rw [Lof, dif_pos h2]
      rw [e1, e2]; exact G4_LL n a b c d
  | .inl ⟨(a, b), h1⟩, .inr w =>
      have e1 : (Sum.inl ⟨(a, b), h1⟩ : IndexedBasis n) = Lof a b := by rw [Lof, dif_pos h1]
      rw [e1]; exact G4_LF n a b w
  | .inr w, .inl ⟨(a, b), h1⟩ =>
      have e1 : (Sum.inl ⟨(a, b), h1⟩ : IndexedBasis n) = Lof a b := by rw [Lof, dif_pos h1]
      rw [e1]; exact G4_FL n a b w
  | .inr u, .inr v => exact G4_FF n u v

/-! ## G5: extension to arbitrary elements of `IndexedMod n`

Unconditional -- since `GammaBetaN` and `deltaFN` are both the
`bilinearExtend` of basis functions agreeing pointwise (`G4`), functional
equality on all of `IndexedMod n` follows directly (`Finset.sum_congr`
twice), with no separate homogeneous-decomposition argument needed. This
subsumes the homogeneous-hypothesis shape the design/packet describes
(`bracketN_super_skew_homog`/`jacobiN_homog`'s own shape), stated below as
a direct corollary. -/

theorem GammaBetaN_eq_deltaFN (n : ℕ) (x y : IndexedMod n) :
    GammaBetaN n x y = deltaFN n (fBetaN n) x y := by
  unfold GammaBetaN deltaFN bilinearExtend
  apply Finset.sum_congr rfl; intro i _
  apply Finset.sum_congr rfl; intro j _
  rw [G4 n i j]

theorem GammaBetaN_eq_deltaFN_homog (n : ℕ) (x y : IndexedMod n) (dx dy : ZMod 2)
    (hx : IsHomogN x dx) (hy : IsHomogN y dy) :
    GammaBetaN n x y = deltaFN n (fBetaN n) x y :=
  GammaBetaN_eq_deltaFN n x y

end Indexed
end InhomogeneousDeformations
