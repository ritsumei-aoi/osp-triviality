import InhomogeneousDeformations.IndexedKappa
import Mathlib.Tactic.Module

/-!
# I106 R1 — the U-conjugation identity (V0-V4)

Transcription (Section T of `IMPLEMENTATION_REQUIREMENTS.md`), from
`aoi2026_triviality_osp1_2n_revised.tex`:

* `h(L_uv) = beta_u F_v + beta_v F_u`, `h(F_u) = 0`, odd `P`-linear — lines 238-239.
* `U(L_uv) = L_uv + kappa h(L_uv)`, `U(F_u) = F_u`, even `R`-linear; inverse is the
  same formula with a minus sign — lines 240-243.
* `eq:recover-by-u`: `[X,Y]_beta = U^{-1}[UX,UY]_0` — lines 248-250.
* `f_beta = h - ad(2 F(v))` on `g_P` — lines 346-349.

Five-layer freeze (`Indexed.lean`, `IndexedLaws.lean`, `IndexedJacobi.lean`,
`IndexedCoboundary.lean`, `IndexedKappa.lean`): read only, never reopened. This is the
one new file.
-/

namespace InhomogeneousDeformations
namespace Indexed

/-! ## V0 — the map `h` -/

/-- `h` on the basis: `beta_u F_v + beta_v F_u` on `L_uv` (symmetric in `u,v`, matching
`Lof`'s sorting), `0` on `F_u` — the manuscript's `h`, transcribed directly (lines
238-239), not derived from `f_beta`. -/
noncomputable def hBasis (n : ℕ) : IndexedBasis n → IndexedMod n
  | .inl ⟨(u, v), _⟩ => betaN n u • eN (Fof v) + betaN n v • eN (Fof u)
  | .inr _ => 0

theorem hBasis_Lof (n : ℕ) (a b : Fin (2 * n)) :
    hBasis n (Lof a b) = betaN n a • eN (Fof b) + betaN n b • eN (Fof a) := by
  rcases eq_or_ne a b with hab | hab
  · subst hab; rw [Lof, dif_pos (le_refl a)]; rfl
  · rcases lt_or_gt_of_ne hab with hlt | hgt
    · rw [Lof, dif_pos hlt.le]; rfl
    · rw [Lof, dif_neg (not_le.mpr hgt)]
      show betaN n b • eN (Fof a) + betaN n a • eN (Fof b)
        = betaN n a • eN (Fof b) + betaN n b • eN (Fof a)
      abel

theorem hBasis_Fof (n : ℕ) (u : Fin (2 * n)) : hBasis n (Fof u) = 0 := rfl

/-- `h` extended `P`-linearly to all of `IndexedMod n`, exactly as `fBetaN` extends
`fBetaBasis` (`IndexedCoboundary.lean`, G1). -/
noncomputable def hMap (n : ℕ) (x : IndexedMod n) : IndexedMod n :=
  ∑ i : IndexedBasis n, x i • hBasis n i

theorem hMap_eN (n : ℕ) (i : IndexedBasis n) : hMap n (eN i) = hBasis n i := by
  unfold hMap
  rw [Finset.sum_eq_single i]
  · simp [eN]
  · intro b _ hb
    have : eN i b = 0 := by unfold eN; rw [if_neg hb]
    simp [this]
  · intro h; exact absurd (Finset.mem_univ i) h

theorem hMap_add (n : ℕ) (x1 x2 : IndexedMod n) :
    hMap n (x1 + x2) = hMap n x1 + hMap n x2 := by
  unfold hMap
  funext k
  simp only [Finset.sum_apply, Pi.add_apply]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro i _
  simp only [Pi.smul_apply, smul_eq_mul]; ring

theorem hMap_smul (n : ℕ) (c : Pn n) (x : IndexedMod n) :
    hMap n (c • x) = c • hMap n x := by
  unfold hMap
  funext k
  simp only [Finset.sum_apply, Pi.smul_apply]
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro i _
  simp only [smul_eq_mul]; ring

theorem hMap_zero (n : ℕ) : hMap n (0 : IndexedMod n) = 0 := by unfold hMap; simp

theorem hMap_neg (n : ℕ) (x : IndexedMod n) : hMap n (-x) = -hMap n x := by
  have h1 : hMap n (-x) + hMap n x = 0 := by
    rw [← hMap_add, neg_add_cancel, hMap_zero]
  exact eq_neg_of_add_eq_zero_left h1

theorem hMap_Lof (n : ℕ) (a b : Fin (2 * n)) :
    hMap n (eN (Lof a b)) = betaN n a • eN (Fof b) + betaN n b • eN (Fof a) := by
  rw [hMap_eN, hBasis_Lof]

theorem hMap_Fof (n : ℕ) (u : Fin (2 * n)) : hMap n (eN (Fof u)) = 0 := by
  rw [hMap_eN, hBasis_Fof]

/-- `h`'s degree law: it raises parity by exactly `1`, exactly as `fBetaBasis_degree`
does for `f_beta` (`IndexedCoboundary.lean`, G1). -/
theorem hBasis_degree (n : ℕ) (i k : IndexedBasis n) (h : hBasis n i k ≠ 0) :
    parity k = parity i + 1 := by
  match i with
  | .inl ⟨(u, v), _⟩ =>
    match k with
    | .inl _ => exfalso; apply h; simp [hBasis, eN, Fof]
    | .inr _ => simp [parity]
  | .inr _ => simp [hBasis] at h

theorem hMap_isHomogN (n : ℕ) (x : IndexedMod n) (dx : ZMod 2) (hx : IsHomogN x dx) :
    IsHomogN (hMap n x) (dx + 1) := by
  intro k hk
  by_contra hne
  apply hk
  unfold hMap
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  apply Finset.sum_eq_zero; intro i _
  by_cases hxi : x i = 0
  · simp [hxi]
  · have hpi := hx i hxi
    have hz : hBasis n i k = 0 := by
      by_contra hc
      exact hne (by rw [hBasis_degree n i k hc, hpi])
    simp [hz]

/-! ## V1 — `U` and its inverse

Reuses `TBeta`/`TBetaInv`'s own shape (`IndexedKappa.lean`, K3) with `hMap` in place of
`fBetaN`: the odd `R`-linear lift `hR` mirrors `fBetaR`, and `UMap := id + kappa • hR`
mirrors `TBeta := id + kappa • fBetaR`. -/

/-- The odd extension of `hMap`: on `r = x0 + kappa x1`, `h_R(r) = h(x0) - kappa h(x1)`,
matching `fBetaR`'s shape (K2) with `hMap` in place of `fBetaN`. -/
noncomputable def hR (n : ℕ) (r : RMod n) : RMod n :=
  iotaR (hMap n (falsePart r)) + kappaEmbed (-(hMap n (truePart r)))

theorem hR_iotaR (n : ℕ) (x : IndexedMod n) : hR n (iotaR x) = iotaR (hMap n x) := by
  unfold hR
  rw [falsePart_iotaR, truePart_iotaR, hMap_zero, neg_zero]
  have : kappaEmbed (0 : IndexedMod n) = (0 : RMod n) := by
    funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaEmbed]
  rw [this, add_zero]

theorem hR_kappaEmbed (n : ℕ) (x : IndexedMod n) :
    hR n (kappaEmbed x) = -(kappaEmbed (hMap n x)) := by
  unfold hR
  rw [falsePart_kappaEmbed, truePart_kappaEmbed, hMap_zero]
  have : iotaR (0 : IndexedMod n) = (0 : RMod n) := iotaR_zero n
  rw [this, zero_add, kappaEmbed_neg]

theorem hR_add (n : ℕ) (r1 r2 : RMod n) : hR n (r1 + r2) = hR n r1 + hR n r2 := by
  unfold hR
  rw [falsePart_add, truePart_add, hMap_add, hMap_add, iotaR_add, neg_add, kappaEmbed_add]
  abel

theorem hR_smul (n : ℕ) (c : Pn n) (r : RMod n) : hR n (c • r) = c • hR n r := by
  unfold hR
  rw [falsePart_smul, truePart_smul, hMap_smul, hMap_smul, iotaR_smul, ← neg_smul, kappaEmbed_smul,
    smul_add]
  congr 1
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaEmbed]

/-- `U = id + kappa h_R` — the manuscript's `U` on the basis, `U(L_uv)=L_uv+kappa
h(L_uv)`, `U(F_u)=F_u`, extended `R`-linearly, exactly `TBeta`'s shape. -/
noncomputable def UMap (n : ℕ) (r : RMod n) : RMod n := r + kappaMulR (hR n r)

/-- `U^{-1} = id - kappa h_R`, the manuscript's stated inverse (same formula, minus
sign). -/
noncomputable def UMapInv (n : ℕ) (r : RMod n) : RMod n := r - kappaMulR (hR n r)

theorem UMap_eq (n : ℕ) (r : RMod n) : UMap n r = r + kappaEmbed (hMap n (falsePart r)) := by
  unfold UMap hR
  rw [kappaMulR_add, kappaMulR_iotaR, kappaMulR_kappaEmbed, add_zero]

theorem UMapInv_eq (n : ℕ) (r : RMod n) : UMapInv n r = r - kappaEmbed (hMap n (falsePart r)) := by
  unfold UMapInv hR
  rw [kappaMulR_add, kappaMulR_iotaR, kappaMulR_kappaEmbed, add_zero]

theorem falsePart_UMap (n : ℕ) (r : RMod n) : falsePart (UMap n r) = falsePart r := by
  rw [UMap_eq, falsePart_add, falsePart_kappaEmbed, add_zero]

theorem falsePart_UMapInv (n : ℕ) (r : RMod n) : falsePart (UMapInv n r) = falsePart r := by
  rw [UMapInv_eq, sub_eq_add_neg, falsePart_add, falsePart_neg, falsePart_kappaEmbed, neg_zero,
    add_zero]

/-- **V1**: `U` and `U^{-1}` are mutually inverse, exactly, with no truncation. -/
theorem UMapInv_UMap (n : ℕ) (r : RMod n) : UMapInv n (UMap n r) = r := by
  rw [UMapInv_eq, falsePart_UMap, UMap_eq]
  abel

theorem UMap_UMapInv (n : ℕ) (r : RMod n) : UMap n (UMapInv n r) = r := by
  rw [UMap_eq, falsePart_UMapInv, UMapInv_eq]
  abel

/-- **V1**: `U` is additive and `Pn n`-linear (the "`P`-linear" half of `R`-linearity). -/
theorem UMap_add (n : ℕ) (r1 r2 : RMod n) : UMap n (r1 + r2) = UMap n r1 + UMap n r2 := by
  rw [UMap_eq, UMap_eq, UMap_eq, falsePart_add, hMap_add, kappaEmbed_add]
  abel

theorem UMap_smul (n : ℕ) (c : Pn n) (r : RMod n) : UMap n (c • r) = c • UMap n r := by
  rw [UMap_eq, UMap_eq, falsePart_smul, hMap_smul, kappaEmbed_smul, smul_add]

/-- **V1**: `U` commutes with multiplication by `kappa` (the other half of `R`-linearity). -/
theorem UMap_kappaMulR (n : ℕ) (r : RMod n) : UMap n (kappaMulR r) = kappaMulR (UMap n r) := by
  have hfalse : falsePart (kappaMulR r) = 0 := by funext i; simp [falsePart, kappaMulR]
  rw [UMap_eq, hfalse, hMap_zero]
  have hz : kappaEmbed (0 : IndexedMod n) = (0 : RMod n) := by
    funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaEmbed]
  rw [hz, add_zero, UMap_eq, kappaMulR_add, kappaMulR_kappaEmbed, add_zero]

/-- **V1**: `U` is even (parity-preserving) — same proof shape as `TBeta_isHomogR`,
with `hMap`'s degree law in place of `fBetaN`'s. -/
theorem UMap_isHomogR (n : ℕ) (r : RMod n) (d : ZMod 2) (hr : IsHomogR r d) :
    IsHomogR (UMap n r) d := by
  have hfp : IsHomogN (falsePart r) d := IsHomogR_falsePart hr
  have hhb : IsHomogN (hMap n (falsePart r)) (d + 1) := hMap_isHomogN n (falsePart r) d hfp
  have e11 : (1 : ZMod 2) + 1 = 0 := by decide
  intro p hp
  rcases p with ⟨i, b⟩
  cases b with
  | false =>
    have h1 : (UMap n r) (i, false) = r (i, false) := by
      rw [UMap_eq]; simp [kappaEmbed]
    exact hr (i, false) (by rw [← h1]; exact hp)
  | true =>
    have h1 : (UMap n r) (i, true) = r (i, true) + hMap n (falsePart r) i := by
      rw [UMap_eq]; simp [kappaEmbed]
    rw [parityR_true]
    by_cases h2 : r (i, true) = 0
    · have h3 : hMap n (falsePart r) i ≠ 0 := by
        intro h0; apply hp; rw [h1, h2, h0]; ring
      have hpar := hhb i h3
      calc parity i + 1 = (d + 1) + 1 := by rw [hpar]
        _ = d + (1 + 1) := by ring
        _ = d := by rw [e11, add_zero]
    · have hpar := hr (i, true) h2
      rwa [parityR_true] at hpar

/-! ## V2 — `delta h = Gamma_beta`, the round's one real obligation

Route chosen: **the manuscript's own remark** (lines 346-349), `f_beta = h -
ad(2 F(v))` on `g_P`, combined with `delta(ad Z) = 0` for the inner map `ad Z = [Z,-]_0`
-- reason: `delta f_beta = Gamma_beta` is *already accepted* (`GammaBetaN_eq_deltaFN`,
G5), so this route needs no fresh sector-by-sector computation for the LL sector
(where the manuscript itself says the identity needs Jacobi -- `eq:h-identity`'s own
proof); the LF/FF/FL sectors close directly, without Jacobi, since `hMap` vanishes on
every `F_u` (shown below). This is shorter than redoing `G4`'s method for `hMap`, and
it connects to P2 rather than re-deriving beside it, exactly as the scope document
recommends. -/

theorem eN_isHomogN {n : ℕ} (i : IndexedBasis n) : IsHomogN (eN i) (parity i) := by
  intro k hk
  by_contra hne
  apply hk
  unfold eN
  apply if_neg
  intro heq
  exact hne (by rw [heq])

theorem cratN_two_half (n : ℕ) : cratN n 2 * cratN n (1 / 2 : ℚ) = 1 := by
  rw [cratN_mul]; norm_num

/-- The manuscript's `2 F(v)`, as an element of `IndexedMod n`. -/
noncomputable def twoFVecBeta (n : ℕ) : IndexedMod n := cratN n 2 • FVecBeta n

/-- `[F(v), L_uv]_0 = (1/2) h(L_uv)` — the un-scaled companion of manuscript line
~261, isolating the sum/`Jn`-contraction bookkeeping from the scalar-`2` factor. -/
theorem bracket_FVecBeta_Lof (n : ℕ) (a b : Fin (2 * n)) :
    bracketN n (FVecBeta n) (eN (Lof a b)) = cratN n (1 / 2) • hBasis n (Lof a b) := by
  unfold FVecBeta
  have hstep : bracketN n (∑ t : Fin (2 * n), vN n t • eN (Fof t)) (eN (Lof a b))
      = ∑ t : Fin (2 * n), vN n t • bracketBasisN n (Fof t) (Lof a b) := by
    rw [bracketN_sum_left']
    apply Finset.sum_congr rfl; intro t _
    rw [bracketN_smul_left, bracketN_eN_eN]
  rw [hstep]
  simp_rw [bracketBasisN_Fof_Lof]
  unfold bracketLFn
  simp_rw [smul_neg, smul_add]
  rw [Finset.sum_neg_distrib, Finset.sum_add_distrib,
      sum_vN_mul_Jn_const_smul n b (cratN n (1 / 2)) (eN (Fof a)),
      sum_vN_mul_Jn_const_smul n a (cratN n (1 / 2)) (eN (Fof b))]
  rw [hBasis_Lof]
  simp only [mul_neg, neg_smul, neg_neg, neg_add_rev, smul_add, smul_smul]

/-- `[2F(v), L_uv]_0 = h(L_uv)` — manuscript line ~261, the identity feeding
`eq:h-identity`'s proof, from `bracket_FVecBeta_Lof` scaled by `2`. -/
theorem bracket_twoFVecBeta_Lof (n : ℕ) (a b : Fin (2 * n)) :
    bracketN n (twoFVecBeta n) (eN (Lof a b)) = hBasis n (Lof a b) := by
  unfold twoFVecBeta
  rw [bracketN_smul_left, bracket_FVecBeta_Lof, smul_smul, cratN_two_half, one_smul]

/-- `[F(v), F_u]_0 = (1/2)(-f_beta(F_u))` — the un-scaled companion. -/
theorem bracket_FVecBeta_Fof (n : ℕ) (u : Fin (2 * n)) :
    bracketN n (FVecBeta n) (eN (Fof u)) = cratN n (1 / 2) • (-fBetaBasis n (Fof u)) := by
  unfold FVecBeta
  have hstep : bracketN n (∑ t : Fin (2 * n), vN n t • eN (Fof t)) (eN (Fof u))
      = ∑ t : Fin (2 * n), vN n t • bracketBasisN n (Fof t) (Fof u) := by
    rw [bracketN_sum_left']
    apply Finset.sum_congr rfl; intro t _
    rw [bracketN_smul_left, bracketN_eN_eN]
  rw [hstep]
  simp_rw [bracketBasisN_Fof_Fof]
  unfold bracketFFn
  have heach : ∀ t : Fin (2 * n),
      vN n t • (cratN n (1 / 2) • eN (Lof t u)) = cratN n (1 / 2) • (vN n t • eN (Lof t u)) := by
    intro t; module
  simp_rw [heach]
  rw [← Finset.smul_sum]
  have hcomm : ∀ x : Fin (2 * n), (eN (Lof x u) : IndexedMod n) = eN (Lof u x) :=
    fun x => congrArg eN (Lof_comm x u)
  simp_rw [hcomm]
  rw [sum_vN_smul_Lof, fBetaN_eN]

/-- `[2F(v), F_u]_0 = -f_beta(F_u)` — the `F`-part companion identity, via
`fBetaBasis_alt_form` and the same `F(v)` sum, scaled by `2`. -/
theorem bracket_twoFVecBeta_Fof (n : ℕ) (u : Fin (2 * n)) :
    bracketN n (twoFVecBeta n) (eN (Fof u)) = -fBetaBasis n (Fof u) := by
  unfold twoFVecBeta
  rw [bracketN_smul_left, bracket_FVecBeta_Fof, smul_smul, cratN_two_half, one_smul]

/-- **The Jacobi step, isolated to the LL sector**: `2F(v)` acts as a derivation on
the bracket of two even (`L`-type) basis elements. Proved from `jacobiN_basis`
applied to a single `F_t` term of `2F(v)`'s sum, plus two super-skew flips (all
`gsignN` factors here are `+1`, since exactly one of the three inputs is odd) --
exactly the manuscript's own scoping of Jacobi to "two even inputs", isolated to
this one lemma rather than assumed globally. -/
theorem twoFVecBeta_deriv_Lof_Lof_term (n : ℕ) (t a b c d : Fin (2 * n)) :
    bracketN n (eN (Fof t)) (bracketN n (eN (Lof a b)) (eN (Lof c d)))
      = bracketN n (bracketN n (eN (Fof t)) (eN (Lof a b))) (eN (Lof c d))
        + bracketN n (eN (Lof a b)) (bracketN n (eN (Fof t)) (eN (Lof c d))) := by
  have hFt : IsHomogN (eN (Fof t : IndexedBasis n)) 1 := by
    have h := eN_isHomogN (Fof t : IndexedBasis n); rwa [parity_Fof] at h
  have hLab : IsHomogN (eN (Lof a b : IndexedBasis n)) 0 := by
    have h := eN_isHomogN (Lof a b : IndexedBasis n); rwa [parity_Lof] at h
  have hLcd : IsHomogN (eN (Lof c d : IndexedBasis n)) 0 := by
    have h := eN_isHomogN (Lof c d : IndexedBasis n); rwa [parity_Lof] at h
  have hj := jacobiN_basis (Fof t : IndexedBasis n) (Lof a b) (Lof c d)
  unfold jacobiSum at hj
  rw [parity_Fof, parity_Lof, parity_Lof, gsignN_10, gsignN_01, gsignN_00, one_smul, one_smul,
    one_smul] at hj
  have e1 : bracketN n (eN (Lof c d : IndexedBasis n)) (eN (Fof t))
      = -bracketN n (eN (Fof t)) (eN (Lof c d)) := by
    rw [bracketN_eN_eN, bracketN_eN_eN, bracketBasisN_Lof_Fof, bracketBasisN_Fof_Lof, neg_neg]
  rw [e1, bracketN_neg_right] at hj
  have hbFtLab : IsHomogN (bracketN n (eN (Fof t : IndexedBasis n)) (eN (Lof a b))) 1 := by
    have := bracketN_isHomogN (eN (Fof t : IndexedBasis n)) (eN (Lof a b)) 1 0 hFt hLab
    simpa using this
  have e2 : bracketN n (eN (Lof c d : IndexedBasis n)) (bracketN n (eN (Fof t)) (eN (Lof a b)))
      = -bracketN n (bracketN n (eN (Fof t)) (eN (Lof a b))) (eN (Lof c d)) := by
    rw [bracketN_super_skew_homog (eN (Lof c d : IndexedBasis n))
        (bracketN n (eN (Fof t)) (eN (Lof a b))) 0 1 hLcd hbFtLab,
      gsignN_01, neg_smul, one_smul]
  rw [e2] at hj
  rw [add_assoc, ← neg_add, ← sub_eq_add_neg, sub_eq_zero] at hj
  rw [hj]
  abel

/-- Lifted from a single `F_t` to all of `F(v)`: `F(v)` acts as a derivation on
the bracket of two even (`L`-type) basis elements, by summing
`twoFVecBeta_deriv_Lof_Lof_term` over `t` with weight `vN n t`. -/
theorem FVecBeta_deriv_Lof_Lof (n : ℕ) (a b c d : Fin (2 * n)) :
    bracketN n (FVecBeta n) (bracketN n (eN (Lof a b)) (eN (Lof c d)))
      = bracketN n (bracketN n (FVecBeta n) (eN (Lof a b))) (eN (Lof c d))
        + bracketN n (eN (Lof a b)) (bracketN n (FVecBeta n) (eN (Lof c d))) := by
  unfold FVecBeta
  have hL : bracketN n (∑ t : Fin (2 * n), vN n t • eN (Fof t))
        (bracketN n (eN (Lof a b)) (eN (Lof c d)))
      = ∑ t : Fin (2 * n),
          vN n t • bracketN n (eN (Fof t)) (bracketN n (eN (Lof a b)) (eN (Lof c d))) := by
    rw [bracketN_sum_left']
    apply Finset.sum_congr rfl; intro t _; rw [bracketN_smul_left]
  have hR1 : bracketN n (bracketN n (∑ t : Fin (2 * n), vN n t • eN (Fof t)) (eN (Lof a b)))
        (eN (Lof c d))
      = ∑ t : Fin (2 * n),
          vN n t • bracketN n (bracketN n (eN (Fof t)) (eN (Lof a b))) (eN (Lof c d)) := by
    have step1 : bracketN n (∑ t : Fin (2 * n), vN n t • eN (Fof t)) (eN (Lof a b))
        = ∑ t : Fin (2 * n), vN n t • bracketN n (eN (Fof t)) (eN (Lof a b)) := by
      rw [bracketN_sum_left']
      apply Finset.sum_congr rfl; intro t _; rw [bracketN_smul_left]
    rw [step1, bracketN_sum_left']
    apply Finset.sum_congr rfl; intro t _; rw [bracketN_smul_left]
  have hR2 : bracketN n (eN (Lof a b))
        (bracketN n (∑ t : Fin (2 * n), vN n t • eN (Fof t)) (eN (Lof c d)))
      = ∑ t : Fin (2 * n),
          vN n t • bracketN n (eN (Lof a b)) (bracketN n (eN (Fof t)) (eN (Lof c d))) := by
    have step2 : bracketN n (∑ t : Fin (2 * n), vN n t • eN (Fof t)) (eN (Lof c d))
        = ∑ t : Fin (2 * n), vN n t • bracketN n (eN (Fof t)) (eN (Lof c d)) := by
      rw [bracketN_sum_left']
      apply Finset.sum_congr rfl; intro t _; rw [bracketN_smul_left]
    rw [step2, bracketN_sum_right']
    apply Finset.sum_congr rfl; intro t _; rw [bracketN_smul_right]
  rw [hL, hR1, hR2, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro t _
  rw [← smul_add]
  congr 1
  exact twoFVecBeta_deriv_Lof_Lof_term n t a b c d

/-- The single-`F_t` derivation identity, lifted to `twoFVecBeta n` (with the
scalar `2` carried through by `bracketN_smul_left`/`smul_add`). -/
theorem twoFVecBeta_deriv_Lof_Lof (n : ℕ) (a b c d : Fin (2 * n)) :
    bracketN n (twoFVecBeta n) (bracketN n (eN (Lof a b)) (eN (Lof c d)))
      = bracketN n (bracketN n (twoFVecBeta n) (eN (Lof a b))) (eN (Lof c d))
        + bracketN n (eN (Lof a b)) (bracketN n (twoFVecBeta n) (eN (Lof c d))) := by
  unfold twoFVecBeta
  rw [bracketN_smul_left, FVecBeta_deriv_Lof_Lof, smul_add]
  congr 1
  · rw [← bracketN_smul_left, ← bracketN_smul_left]
  · rw [← bracketN_smul_right, ← bracketN_smul_left]

/-- `hMap` applied to `bracketLLn`'s output agrees with `bracket(twoFVecBeta n, -)`
applied to the same, termwise on `bracketLLn`'s four `Lof`-basis summands (via
`hMap_eN` and `bracket_twoFVecBeta_Lof`, both landing on `hBasis`). -/
theorem hMap_bracket_Lof_Lof (n : ℕ) (a b c d : Fin (2 * n)) :
    hMap n (bracketN n (eN (Lof a b)) (eN (Lof c d)))
      = bracketN n (twoFVecBeta n) (bracketN n (eN (Lof a b)) (eN (Lof c d))) := by
  rw [bracketN_eN_eN, bracketBasisN_Lof_Lof]
  unfold bracketLLn
  rw [hMap_add, hMap_add, hMap_add, hMap_smul, hMap_smul, hMap_smul, hMap_smul,
      hMap_eN, hMap_eN, hMap_eN, hMap_eN,
      bracketN_add_right, bracketN_add_right, bracketN_add_right,
      bracketN_smul_right, bracketN_smul_right, bracketN_smul_right, bracketN_smul_right,
      bracket_twoFVecBeta_Lof, bracket_twoFVecBeta_Lof, bracket_twoFVecBeta_Lof,
      bracket_twoFVecBeta_Lof]

/-- **V2, LL sector**: `delta h = Gamma_beta` on two even basis elements. The
one sector needing Jacobi (`eq:h-identity`'s own proof), discharged via
`twoFVecBeta_deriv_Lof_Lof` and `bracket_twoFVecBeta_Lof`; already-accepted P2
(`GammaBetaN_eq_deltaFN`/`G4_LL`) is **not** re-derived -- this sector closes on its
own, directly, and independently confirms `G4_LL`'s target (`Gamma_beta = 0` here)
by a different route. -/
theorem deltaH_LL (n : ℕ) (a b c d : Fin (2 * n)) :
    deltaFBasis n (hMap n) (Lof a b) (Lof c d) = GammaBetaBasis n (Lof a b) (Lof c d) := by
  rw [GammaBetaBasis_Lof_Lof]
  unfold deltaFBasis
  rw [parity_Lof, gsignN_01, one_smul, hMap_eN, hMap_eN, ← bracket_twoFVecBeta_Lof,
    ← bracket_twoFVecBeta_Lof, hMap_bracket_Lof_Lof, twoFVecBeta_deriv_Lof_Lof]
  abel

/-- **V2, LF sector**: closes directly, no Jacobi -- `hMap` vanishes on every `F_u`
(`hMap_Fof`), so the `delta`-correction and normalization terms both drop out. -/
theorem deltaH_LF (n : ℕ) (u v w : Fin (2 * n)) :
    deltaFBasis n (hMap n) (Lof u v) (Fof w) = GammaBetaBasis n (Lof u v) (Fof w) := by
  rw [GammaBetaBasis_Lof_Fof]
  unfold deltaFBasis
  rw [hMap_Lof, parity_Lof, gsignN_01, one_smul, hMap_Fof, bracketN_zero_right, add_zero]
  rw [bracketN_add_left, bracketN_smul_left, bracketN_smul_left,
    bracketN_eN_eN, bracketN_eN_eN, bracketBasisN_Fof_Fof, bracketBasisN_Fof_Fof]
  unfold bracketFFn
  rw [bracketN_eN_eN, bracketBasisN_Lof_Fof]
  unfold bracketLFn
  rw [hMap_add, hMap_smul, hMap_smul, hMap_Fof, hMap_Fof, smul_zero, smul_zero, add_zero, sub_zero]
  unfold gammaLFn
  module

/-- **V2, FF sector**: closes directly, no Jacobi. -/
theorem deltaH_FF (n : ℕ) (u v : Fin (2 * n)) :
    deltaFBasis n (hMap n) (Fof u) (Fof v) = GammaBetaBasis n (Fof u) (Fof v) := by
  show deltaFBasis n (hMap n) (Fof u) (Fof v) = gammaFFn n u v
  unfold deltaFBasis
  rw [hMap_Fof, bracketN_zero_left, parity_Fof, gsignN_11, hMap_Fof, bracketN_zero_right,
    smul_zero, zero_add, zero_sub]
  rw [bracketN_eN_eN, bracketBasisN_Fof_Fof]
  unfold bracketFFn
  rw [hMap_smul, hMap_Lof]
  unfold gammaFFn
  rfl

/-- **V2, FL sector**: closes directly, no Jacobi; corroborates the declared FL
convention (`GammaBetaBasis_Fof_Lof`, `-gammaLFn`), matching `G4_FL`'s own role for
`f_beta`. -/
theorem deltaH_FL (n : ℕ) (u v w : Fin (2 * n)) :
    deltaFBasis n (hMap n) (Fof w) (Lof u v) = GammaBetaBasis n (Fof w) (Lof u v) := by
  rw [GammaBetaBasis_Fof_Lof]
  unfold deltaFBasis
  rw [hMap_Fof, bracketN_zero_left, zero_add, parity_Fof, gsignN_11, hMap_Lof]
  rw [bracketN_eN_eN, bracketBasisN_Fof_Lof, hMap_neg]
  rw [bracketN_add_right, bracketN_smul_right, bracketN_smul_right,
    bracketN_eN_eN, bracketN_eN_eN, bracketBasisN_Fof_Fof, bracketBasisN_Fof_Fof]
  unfold bracketFFn
  unfold bracketLFn
  rw [hMap_add, hMap_smul, hMap_smul, hMap_Fof, hMap_Fof, smul_zero, smul_zero, add_zero]
  rw [show (Lof w v : IndexedBasis n) = Lof v w from Lof_comm w v,
      show (Lof w u : IndexedBasis n) = Lof u w from Lof_comm w u]
  unfold gammaLFn
  rw [sub_neg_eq_add, add_zero, neg_one_smul]
  congr 1
  module

/-- **V2, assembled**: `delta h = Gamma_beta` on all ordered basis pairs -- the
round's one real obligation, matching `G4`'s own shape for `f_beta`. -/
theorem deltaH (n : ℕ) (i j : IndexedBasis n) :
    deltaFBasis n (hMap n) i j = GammaBetaBasis n i j := by
  match i, j with
  | .inl ⟨(a, b), h1⟩, .inl ⟨(c, d), h2⟩ =>
      have e1 : (Sum.inl ⟨(a, b), h1⟩ : IndexedBasis n) = Lof a b := by rw [Lof, dif_pos h1]
      have e2 : (Sum.inl ⟨(c, d), h2⟩ : IndexedBasis n) = Lof c d := by rw [Lof, dif_pos h2]
      rw [e1, e2]; exact deltaH_LL n a b c d
  | .inl ⟨(a, b), h1⟩, .inr w =>
      have e1 : (Sum.inl ⟨(a, b), h1⟩ : IndexedBasis n) = Lof a b := by rw [Lof, dif_pos h1]
      rw [e1]; exact deltaH_LF n a b w
  | .inr w, .inl ⟨(a, b), h1⟩ =>
      have e1 : (Sum.inl ⟨(a, b), h1⟩ : IndexedBasis n) = Lof a b := by rw [Lof, dif_pos h1]
      rw [e1]; exact deltaH_FL n a b w
  | .inr u, .inr v => exact deltaH_FF n u v

/-- **V2, extended to arbitrary elements**: unconditional, since `GammaBetaN` and
`deltaFN` are both the `bilinearExtend` of basis functions agreeing pointwise
(`deltaH`) -- the exact shape of `GammaBetaN_eq_deltaFN` (G5). -/
theorem GammaBetaN_eq_deltaHN (n : ℕ) (x y : IndexedMod n) :
    GammaBetaN n x y = deltaFN n (hMap n) x y := by
  unfold GammaBetaN deltaFN bilinearExtend
  apply Finset.sum_congr rfl; intro i _
  apply Finset.sum_congr rfl; intro j _
  rw [deltaH n i j]

/-! ## V3 — the gate statement, proved

Transposed from R2-E's own `intertwining_basis`/`intertwining` (K5,
`eq:intertwining`), with `UMap` in place of `TBeta`, `hMap` in place of `fBetaN`,
and `deltaH` (V2, above) in place of `G4` in the one substantive sector. -/

/-- `U` on a `g_P` coordinate basis vector: `h`'s manuscript formula enters here. -/
theorem UMap_eR_false (n : ℕ) (i : IndexedBasis n) :
    UMap n (eR ((i, false) : RBasis n)) = eR ((i, false) : RBasis n) + kappaEmbed (hBasis n i) := by
  rw [UMap_eq, falsePart_eR_false, hMap_eN]

/-- `U` fixes any `kappa`-basis vector exactly. -/
theorem UMap_eR_true (n : ℕ) (i : IndexedBasis n) :
    UMap n (eR ((i, true) : RBasis n)) = eR ((i, true) : RBasis n) := by
  rw [UMap_eq, falsePart_eR_true, hMap_zero, kappaEmbed_zero, add_zero]

/-- **V3, basis level**: the conjugation identity holds on every basis pair,
citing `deltaH` in the substantive case -- the only nontrivial one, exactly
mirroring `intertwining_basis`'s own structure. -/
theorem intertwiningU_basis (n : ℕ) (p q : RBasis n) :
    bracketR n (UMap n (eR p)) (UMap n (eR q)) = UMap n (bracketRBeta n (eR p) (eR q)) := by
  rcases p with ⟨i, bi⟩; rcases q with ⟨j, bj⟩
  cases bi <;> cases bj
  · -- (i,false),(j,false) -- the substantive case
    rw [UMap_eR_false, UMap_eR_false, bracketRBeta_eR_eR]
    show bracketR n (eR ((i, false) : RBasis n) + kappaEmbed (hBasis n i))
        (eR ((j, false) : RBasis n) + kappaEmbed (hBasis n j))
      = UMap n (bracketRBetaBasis n (i, false) (j, false))
    rw [bracketR_add_left, bracketR_add_right, bracketR_add_right]
    rw [bracketR_eR_eR, bracketR_eR_false_kappaEmbed, bracketR_kappaEmbed_eR_false,
        bracketR_kappaEmbed_kappaEmbed, add_zero]
    simp only [bracketRBasis]
    set A := gsignN n (parity i) 1 • bracketN n (eN i) (hBasis n j) with hA
    set B := bracketN n (hBasis n i) (eN j) with hB
    have hfix : UMap n (kappaEmbed (GammaBetaBasis n i j)) = kappaEmbed (GammaBetaBasis n i j) := by
      rw [UMap_eq, falsePart_kappaEmbed, hMap_zero, kappaEmbed_zero, add_zero]
    simp only [bracketRBetaBasis]
    rw [UMap_add, UMap_eq, falsePart_iotaR, hfix]
    set Z := hMap n (bracketBasisN n i j) with hZ
    rw [← deltaH n i j]
    unfold deltaFBasis
    rw [bracketN_eN_eN, hMap_eN, hMap_eN, ← hB, ← hA, ← hZ]
    have key : kappaEmbed A + kappaEmbed B = kappaEmbed Z + kappaEmbed (B + A - Z) := by
      rw [← kappaEmbed_add, ← kappaEmbed_add]
      congr 1
      abel
    calc iotaR (bracketBasisN n i j) + kappaEmbed A + kappaEmbed B
        = iotaR (bracketBasisN n i j) + (kappaEmbed A + kappaEmbed B) := by abel
      _ = iotaR (bracketBasisN n i j) + (kappaEmbed Z + kappaEmbed (B + A - Z)) := by rw [key]
      _ = iotaR (bracketBasisN n i j) + kappaEmbed Z + kappaEmbed (B + A - Z) := by abel
  · -- (i,false),(j,true)
    rw [UMap_eR_false, UMap_eR_true, bracketRBeta_eR_eR]
    show bracketR n (eR ((i, false) : RBasis n) + kappaEmbed (hBasis n i)) (eR ((j, true) : RBasis n))
      = UMap n (bracketRBetaBasis n (i, false) (j, true))
    rw [bracketR_add_left, bracketR_eR_eR, bracketR_kappaEmbed_eR_true, add_zero]
    show bracketRBasis n (i, false) (j, true) = UMap n (bracketRBetaBasis n (i, false) (j, true))
    show kappaEmbed (gsignN n (parity i) 1 • bracketBasisN n i j)
      = UMap n (kappaEmbed (gsignN n (parity i) 1 • bracketBasisN n i j))
    rw [UMap_eq, falsePart_kappaEmbed, hMap_zero]
    have : kappaEmbed (0 : IndexedMod n) = (0 : RMod n) := kappaEmbed_zero n
    rw [this, add_zero]
  · -- (i,true),(j,false)
    rw [UMap_eR_true, UMap_eR_false, bracketRBeta_eR_eR]
    show bracketR n (eR ((i, true) : RBasis n)) (eR ((j, false) : RBasis n) + kappaEmbed (hBasis n j))
      = UMap n (bracketRBetaBasis n (i, true) (j, false))
    rw [bracketR_add_right, bracketR_eR_eR, bracketR_eR_true_kappaEmbed, add_zero]
    show bracketRBasis n (i, true) (j, false) = UMap n (bracketRBetaBasis n (i, true) (j, false))
    show kappaEmbed (bracketBasisN n i j) = UMap n (kappaEmbed (bracketBasisN n i j))
    rw [UMap_eq, falsePart_kappaEmbed, hMap_zero]
    have : kappaEmbed (0 : IndexedMod n) = (0 : RMod n) := kappaEmbed_zero n
    rw [this, add_zero]
  · -- (i,true),(j,true)
    rw [UMap_eR_true, UMap_eR_true, bracketRBeta_eR_eR, bracketR_eR_eR]
    show (0 : RMod n) = UMap n (0 : RMod n)
    have hfp0 : falsePart (0 : RMod n) = 0 := by funext k; simp [falsePart]
    rw [UMap_eq, hfp0, hMap_zero]
    have : kappaEmbed (0 : IndexedMod n) = (0 : RMod n) := kappaEmbed_zero n
    rw [this, add_zero]

theorem UMap_zero (n : ℕ) : UMap n (0 : RMod n) = 0 := by
  have hfp0 : falsePart (0 : RMod n) = 0 := by funext k; simp [falsePart]
  rw [UMap_eq, hfp0, hMap_zero, kappaEmbed_zero, add_zero]

theorem UMap_sum' {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (c : ι → Pn n)
    (f : ι → RMod n) : UMap n (∑ i ∈ s, c i • f i) = ∑ i ∈ s, c i • UMap n (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [UMap_zero]
  | @insert a s ha ih =>
    rw [Finset.sum_insert ha, UMap_add, UMap_smul, ih, Finset.sum_insert ha]

theorem UMap_sum {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (f : ι → RMod n) :
    UMap n (∑ i ∈ s, f i) = ∑ i ∈ s, UMap n (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [UMap_zero]
  | @insert a s ha ih => rw [Finset.sum_insert ha, UMap_add, ih, Finset.sum_insert ha]

/-- **V3**: the conjugation identity for **arbitrary** elements of `g_R` --
unconditional, exactly as `intertwining` (K5) was, transposed with `UMap`/`UMapInv`
in place of `TBeta`/`TBetaInv`. -/
theorem intertwiningU (n : ℕ) (x y : RMod n) :
    bracketR n (UMap n x) (UMap n y) = UMap n (bracketRBeta n x y) := by
  have hLHS : bracketR n (UMap n x) (UMap n y)
      = ∑ p : RBasis n, ∑ q : RBasis n, (x p * y q) • bracketR n (UMap n (eR p)) (UMap n (eR q)) := by
    conv_lhs => rw [eR_decompose n x, eR_decompose n y]
    rw [UMap_sum', UMap_sum', bracketR_sum_left'']
    apply Finset.sum_congr rfl; intro p _
    rw [bracketR_sum_right'', Finset.smul_sum]
    apply Finset.sum_congr rfl; intro q _
    rw [smul_smul]
  have hRHS : UMap n (bracketRBeta n x y)
      = ∑ p : RBasis n, ∑ q : RBasis n, (x p * y q) • bracketR n (UMap n (eR p)) (UMap n (eR q)) := by
    unfold bracketRBeta
    rw [UMap_sum]
    apply Finset.sum_congr rfl; intro p _
    rw [UMap_sum']
    apply Finset.sum_congr rfl; intro q _
    rw [← bracketRBeta_eR_eR n p q]
    exact congrArg (fun z => (x p * y q) • z) (intertwiningU_basis n p q).symm
  rw [hLHS, hRHS]

/-- **V3, the gate statement**: `U^{-1} [Ux,Uy]_0 = [x,y]_beta`, with `GammaBetaN`
appearing literally once `bracketRBeta` and `bracketR` are read on the `iotaR`
(coordinate) part -- see `intertwiningU_coordinate_corollary` below for the
explicit form. Follows from `intertwiningU` by applying `UMapInv` to both sides
and cancelling via `UMapInv_UMap`. -/
theorem intertwiningU_inv (n : ℕ) (x y : RMod n) :
    UMapInv n (bracketR n (UMap n x) (UMap n y)) = bracketRBeta n x y := by
  rw [intertwiningU, UMapInv_UMap]

/-- **V3, the coordinate corollary**: on coordinate (`g_P`) elements, the gate
statement reads exactly `U^{-1}[UX,UY]_0 = X,Y bracket + kappa GammaBetaN X Y`,
with **`GammaBetaN` appearing literally** -- via the already-accepted
`bracketRBeta_iotaR_iotaR` (K4). This is the statement V3 exists to produce. -/
theorem intertwiningU_coordinate_corollary (n : ℕ) (X Y : IndexedMod n) :
    UMapInv n (bracketR n (UMap n (iotaR X)) (UMap n (iotaR Y)))
      = iotaR (bracketN n X Y) + kappaEmbed (GammaBetaN n X Y) := by
  rw [intertwiningU_inv, bracketRBeta_iotaR_iotaR]

end Indexed
end InhomogeneousDeformations
