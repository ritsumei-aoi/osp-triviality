import InhomogeneousDeformations.IndexedCoboundary

/-!
# R2-E -- the `kappa` extension and the trivializing map `T_beta` (P3)

For every rank `n >= 1`, on `g_R = g_P + kappa g_P` (`kappa` odd, `kappa^2 =
0`), with the undeformed bracket extended by `eq:scalar-rule` and the
deformed bracket `[X,Y]_beta = [X,Y]_0 + kappa Gamma_beta(X,Y)` extended the
same way, the map `T_beta = id + kappa (f_beta)_R` and its inverse
`T_beta^{-1} = id - kappa (f_beta)_R` are exactly (no truncation) mutually
inverse, even, `R`-linear maps satisfying the intertwining
`[T_beta X, T_beta Y]_0 = T_beta([X,Y]_beta)` -- `eq:intertwining` --
**unconditionally**, for arbitrary (not just homogeneous) `X,Y : g_R`.
Together with the frozen, accepted P2 (`Gamma_beta = delta f_beta`, in
`IndexedCoboundary.lean`), this is `thm:main` **for the coordinate-defined
`Gamma_beta`**: the deformation is trivialized by an explicit even change of
generators, exactly, at every rank.

**H1.3 (verbatim), what this establishes and what it does not.** This
completes **P3**. It does NOT establish the identification of this
`Gamma_beta` with the coefficient recovered from the oscillator source
(`prop:recovery`, P4) -- the single remaining declared gap to `thm:main` as
the manuscript states it. It does NOT establish `lem:source-isomorphism`,
the maps `Phi,Psi,U`, anything about `A_beta`/`A_B`, `cor:complex`, or
`f_beta = h - ad(2F(v))` (recovery machinery). It is NOT a formalization of
the manuscript's own proof, which derives `Gamma_beta` from the source
rather than defining it; the Lean development remains a deliberately
different route to the same theorem.

## Section T -- the transcription, term by term

Every sign below is transcribed directly from
`aoi2026_triviality_osp1_2n_revised.tex` (20137 bytes, SHA256
`68e8575797f041f25edf9c8a37c3d82c0e06f918f1acecebdb585e11459e6142`), the
same identity R2-D transcribed from. `Gamma_beta` and `f_beta` are NOT
re-transcribed here: they are the frozen, accepted `GammaBetaN`/`fBetaN`.

| manuscript | lines | Lean |
|---|---|---|
| `R = P + kappa P`, `kappa^2 = 0`, `kappa` odd, `R` supercommutative | 57-62 | the `RBasis n = IndexedBasis n x Bool`/`RMod n` model; `true` is the `kappa` slot |
| coefficients written on the left | 63 | `eR`, `bracketRBasis`'s four-row table below |
| `eq:scalar-rule`: `[rX,sY]_0 = (-1)^{p(X)p(s)} rs [X,Y]_0` | 205-206 | `bracketRBasis`'s rows 2/3: sign `gsignN n (parity i) 1` when the RIGHT slot carries `kappa`, unsigned when the LEFT slot does |
| `eq:deformed-bracket`: `[X,Y]_beta = [X,Y]_0 + kappa Gamma_beta(X,Y)` | 216-217 | `bracketRBetaBasis`'s row 1 (`bracketRBasis`'s row 1 plus `kappaEmbed (GammaBetaBasis n i j)`) |
| `g_R = g_P + kappa g_P`; no `kappa`-injectivity claimed | 270-272 | `iotaR`/`kappaEmbed` decomposition (`decompose`); `kappaMulR_not_injective` (K0c) |
| `(f_beta)_R(rX) = (-1)^{p(r)} r f_beta(X)` | 298-299 | `fBetaR`; `fBetaR_iotaR` (`r=1`) and `fBetaR_kappaEmbed` (`r=kappa`, sign `-1`) both proved, not assumed |
| `eq:splitting`: `T_beta = id + kappa (f_beta)_R`, `T_beta^{-1} = id - kappa (f_beta)_R` | 300-302 | `TBeta`, `TBetaInv`, literal; `TBetaInv_TBeta`/`TBeta_TBetaInv` exact |
| `eq:intertwining`: `[T_beta X, T_beta Y]_0 = T_beta([X,Y]_beta)` | 305-306 | `intertwining_basis` (basis level, cites `G4`), `intertwining` (arbitrary elements, unconditional) |

**Two named hazards, both resolved by direct computation, not by tuning
towards the target.** (i) `eq:scalar-rule`'s sign depends on the LEFT
argument and the RIGHT coefficient -- `bracketRBasis`'s rows 2/3 are
asymmetric exactly this way (`K1c`, `bracketRBasis_super_skew`, verifies
this pins super-skew symmetry and that the two named wrong variants break
it). (ii) the odd extension's sign depends on the COEFFICIENT's parity, not
the argument's -- `fBetaR`'s definition (via `falsePart`/`truePart`) signs
only the `kappa`-slot contribution, proved (not assumed) in `fBetaR_kappaEmbed`.

## Deliverable table

- **K0** (the model): `RBasis`, `RMod`, `parityR`, `IsHomogR`; `iotaR`,
  `kappaEmbed`, `decompose` (the embedding and the `R`-coefficient
  decomposition); `kappaMulR`, **K0c**: `kappaMulR_sq` (`kappa^2=0`) and
  `kappaMulR_not_injective` (the manuscript's own disclaimer, not an
  oversight).
- **K1** (the extended undeformed bracket): `bracketRBasis`/`bracketR`,
  transcribed (`eq:scalar-rule`); **K1a** `bracketR_iotaR_iotaR` (restricts
  to `bracketN`); **K1b** `bracketR_add_left/right`,
  `bracketR_smul_left/right`; **K1c** `bracketRBasis_super_skew` (basis
  level, the convention-pinning check, proved BEFORE K5 is attempted) and
  `bracketR_super_skew_homog` (homogeneous elements).
- **K2** (the odd extension): `fBetaR`, with `fBetaR_iotaR`/
  `fBetaR_kappaEmbed` (the two extension-rule cases, proved) and
  `fBetaR_add`/`fBetaR_smul` (`Pn n`-linearity).
- **K3** (`T_beta`): `TBeta`/`TBetaInv` (literal `eq:splitting`);
  `TBetaInv_TBeta`/`TBeta_TBetaInv` (exact inverses, no truncation);
  `TBeta_add`/`TBeta_smul` (`P`-linear) and `TBeta_kappaMulR` (the other
  half of `R`-linearity); `TBeta_isHomogR` (even).
- **K4** (the deformed bracket): `bracketRBetaBasis`/`bracketRBeta`,
  transcribed; `bracketRBeta_iotaR_iotaR` restricts to
  `bracketN + kappa GammaBetaN`, as required.
- **K5** (the intertwining, the stage's theorem): `intertwining_basis`
  (every basis pair; the substantive `(false,false)` case cites `G4` by
  name -- the residue after expanding both sides is EXACTLY `G4`'s
  statement, `GammaBetaBasis n i j = deltaFBasis n (fBetaN n) i j`);
  `intertwining` (arbitrary elements of `g_R`, **unconditional**, no
  homogeneity hypothesis needed -- `T_beta` is `Pn n`-linear so it commutes
  with the basis expansion, and `bracketR`/`bracketRBeta` are both
  bilinear, exactly as `GammaBetaN_eq_deltaFN` was unconditional for G5).
- **K6** (optional, not attempted): transport of the Lie superalgebra
  structure along `T_beta`. Not required for P3 and not held against this
  round.

Agent1c's hand analysis (design section 4) held in every substantive point:
the mixed basis cases reduce to a single term or `0`, and the coordinate
case's residue is literally `G4`.
-/

namespace InhomogeneousDeformations
namespace Indexed

/-! ## K0 -/

abbrev RBasis (n : ℕ) : Type := IndexedBasis n × Bool

def parityR {n : ℕ} (p : RBasis n) : ZMod 2 := parity p.1 + (if p.2 then 1 else 0)

abbrev RMod (n : ℕ) : Type := RBasis n → Pn n

@[simp] lemma rMod_zero_apply {n : ℕ} (k : RBasis n) : (0 : RMod n) k = 0 := rfl
@[simp] lemma rMod_add_apply {n : ℕ} (x y : RMod n) (k : RBasis n) : (x + y) k = x k + y k := rfl
@[simp] lemma rMod_smul_apply {n : ℕ} (c : Pn n) (x : RMod n) (k : RBasis n) :
    (c • x) k = c * x k := rfl
@[simp] lemma rMod_neg_apply {n : ℕ} (x : RMod n) (k : RBasis n) : (-x) k = -x k := rfl

noncomputable def eR {n : ℕ} (b : RBasis n) : RMod n := fun k => if k = b then 1 else 0

/-- The embedding `iota_R : g_P -> g_R` into the `false` (un-scaled) slot. -/
noncomputable def iotaR {n : ℕ} (x : IndexedMod n) : RMod n := fun p => if p.2 then 0 else x p.1

/-- Embeds `x` scaled by `kappa`, directly into the `true` slot. -/
noncomputable def kappaEmbed {n : ℕ} (x : IndexedMod n) : RMod n := fun p => if p.2 then x p.1 else 0

/-- The `false`-slot coordinate function of an `R`-module element. -/
def falsePart {n : ℕ} (r : RMod n) : IndexedMod n := fun i => r (i, false)

/-- The `true`-slot coordinate function of an `R`-module element. -/
def truePart {n : ℕ} (r : RMod n) : IndexedMod n := fun i => r (i, true)

@[simp] theorem falsePart_iotaR {n : ℕ} (x : IndexedMod n) : falsePart (iotaR x) = x := by
  funext i; simp [falsePart, iotaR]

@[simp] theorem truePart_iotaR {n : ℕ} (x : IndexedMod n) : truePart (iotaR x) = 0 := by
  funext i; simp [truePart, iotaR]

@[simp] theorem falsePart_kappaEmbed {n : ℕ} (x : IndexedMod n) : falsePart (kappaEmbed x) = 0 := by
  funext i; simp [falsePart, kappaEmbed]

@[simp] theorem truePart_kappaEmbed {n : ℕ} (x : IndexedMod n) : truePart (kappaEmbed x) = x := by
  funext i; simp [truePart, kappaEmbed]

/-- Every `r : RMod n` decomposes exactly as its `false` part plus `kappa`
times its `true` part. -/
theorem decompose {n : ℕ} (r : RMod n) : r = iotaR (falsePart r) + kappaEmbed (truePart r) := by
  funext p
  rcases p with ⟨i, b⟩
  cases b <;> simp [iotaR, kappaEmbed, falsePart, truePart]

theorem iotaR_add {n : ℕ} (x y : IndexedMod n) : iotaR (x + y) = iotaR x + iotaR y := by
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [iotaR]

theorem iotaR_smul {n : ℕ} (c : Pn n) (x : IndexedMod n) : iotaR (c • x) = c • iotaR x := by
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [iotaR]

theorem iotaR_neg {n : ℕ} (x : IndexedMod n) : iotaR (-x) = -iotaR x := by
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [iotaR]

theorem kappaEmbed_add {n : ℕ} (x y : IndexedMod n) :
    kappaEmbed (x + y) = kappaEmbed x + kappaEmbed y := by
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaEmbed]

theorem kappaEmbed_smul {n : ℕ} (c : Pn n) (x : IndexedMod n) :
    kappaEmbed (c • x) = c • kappaEmbed x := by
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaEmbed]

theorem kappaEmbed_neg {n : ℕ} (x : IndexedMod n) : kappaEmbed (-x) = -kappaEmbed x := by
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaEmbed]

theorem iotaR_injective {n : ℕ} : Function.Injective (iotaR (n := n)) := by
  intro x y hxy
  funext i
  have := congrFun hxy (i, false)
  simpa [iotaR] using this

theorem kappaEmbed_injective {n : ℕ} : Function.Injective (kappaEmbed (n := n)) := by
  intro x y hxy
  funext i
  have := congrFun hxy (i, true)
  simpa [kappaEmbed] using this

theorem iotaR_ne_zero_iff {n : ℕ} (x : IndexedMod n) : iotaR x ≠ 0 ↔ x ≠ 0 := by
  constructor
  · intro h heq; apply h; rw [heq]; funext p; rcases p with ⟨i, b⟩; cases b <;> simp [iotaR]
  · intro h heq; apply h
    have := congrFun heq
    funext i
    have h2 := this (i, false)
    simpa [iotaR] using h2

/-- `kappa`-multiplication on `g_R`: `kappa * (x0 + kappa x1) = kappa x0`. -/
noncomputable def kappaMulR {n : ℕ} (r : RMod n) : RMod n := fun p => if p.2 then r (p.1, false) else 0

theorem kappaMulR_eq {n : ℕ} (r : RMod n) : kappaMulR r = kappaEmbed (falsePart r) := by
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaMulR, kappaEmbed, falsePart]

/-- **K0c**: `kappa^2 = 0` on `g_R`, as a stated theorem. -/
theorem kappaMulR_sq (n : ℕ) (r : RMod n) : kappaMulR (kappaMulR r) = 0 := by
  funext p; rcases p with ⟨i, b⟩
  cases b <;> simp [kappaMulR]

/-- **K0c**: multiplication by `kappa` is not claimed, and is not, injective
on `g_R` -- the manuscript's own explicit disclaimer (lines 270-272),
witnessed concretely: `0` and `kappa e_i` (for any fixed even-pair basis
element, hence nonzero) have the same image `0`. -/
theorem kappaMulR_not_injective (n : ℕ) (hn : 0 < n) :
    ¬ Function.Injective (kappaMulR (n := n)) := by
  intro hinj
  have h0 : ∃ v : Fin (2 * n), True := ⟨⟨0, by omega⟩, trivial⟩
  obtain ⟨v, -⟩ := h0
  have key : kappaMulR (n := n) (eR (Fof v, true)) = kappaMulR (n := n) (0 : RMod n) := by
    funext p; rcases p with ⟨i, b⟩
    cases b with
    | false => simp [kappaMulR]
    | true =>
      have hc : ((i, false) : RBasis n) ≠ (Fof v, true) :=
        fun h => Bool.false_ne_true (congrArg Prod.snd h)
      show eR (Fof v, true) (i, false) = kappaMulR (0 : RMod n) (i, true)
      simp [eR, kappaMulR, hc]
  have := hinj key
  have hne : (eR (n := n) (Fof v, true)) ≠ (0 : RMod n) := by
    intro h
    have := congrFun h (Fof v, true)
    simp [eR] at this
  exact hne this

def IsHomogR {n : ℕ} (r : RMod n) (d : ZMod 2) : Prop := ∀ p : RBasis n, r p ≠ 0 → parityR p = d

/-! ## K1 -- the extended undeformed bracket, `eq:scalar-rule` -/

/-- The four-row table of `eq:scalar-rule` applied to basis pairs of `g_R`:
`(e_i,e_j)` unsigned in the `false` slot; `(e_i,kappa e_j)` signed
`(-1)^{p(i)}`, `kappa` in the `true` slot; `(kappa e_i,e_j)` unsigned,
`kappa` in the `true` slot; `(kappa e_i,kappa e_j)` `0` (`kappa^2=0`). -/
noncomputable def bracketRBasis (n : ℕ) : RBasis n → RBasis n → RMod n
  | (i, false), (j, false) => iotaR (bracketBasisN n i j)
  | (i, false), (j, true) => kappaEmbed (gsignN n (parity i) 1 • bracketBasisN n i j)
  | (i, true), (j, false) => kappaEmbed (bracketBasisN n i j)
  | (_, true), (_, true) => 0

noncomputable def bracketR (n : ℕ) (x y : RMod n) : RMod n :=
  ∑ p : RBasis n, ∑ q : RBasis n, (x p * y q) • bracketRBasis n p q

theorem bracketR_eR_eR (n : ℕ) (p q : RBasis n) :
    bracketR n (eR p) (eR q) = bracketRBasis n p q := by
  unfold bracketR
  rw [Finset.sum_eq_single p]
  · rw [Finset.sum_eq_single q]
    · simp [eR]
    · intro b _ hb
      have : eR q b = 0 := by unfold eR; rw [if_neg hb]
      simp [this]
    · intro h; exact absurd (Finset.mem_univ q) h
  · intro b _ hb
    have : eR p b = 0 := by unfold eR; rw [if_neg hb]
    simp [this]
  · intro h; exact absurd (Finset.mem_univ p) h

theorem bracketR_zero_left (n : ℕ) (y : RMod n) : bracketR n 0 y = 0 := by unfold bracketR; simp
theorem bracketR_zero_right (n : ℕ) (x : RMod n) : bracketR n x 0 = 0 := by unfold bracketR; simp

theorem bracketR_add_left (n : ℕ) (x1 x2 y : RMod n) :
    bracketR n (x1 + x2) y = bracketR n x1 y + bracketR n x2 y := by
  unfold bracketR
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro p _
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro q _
  simp only [rMod_add_apply]
  rw [add_mul, add_smul]

theorem bracketR_add_right (n : ℕ) (x y1 y2 : RMod n) :
    bracketR n x (y1 + y2) = bracketR n x y1 + bracketR n x y2 := by
  unfold bracketR
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro p _
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro q _
  simp only [rMod_add_apply]
  rw [mul_add, add_smul]

theorem bracketR_smul_left (n : ℕ) (c : Pn n) (x y : RMod n) :
    bracketR n (c • x) y = c • bracketR n x y := by
  unfold bracketR
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro p _
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro q _
  simp only [rMod_smul_apply, smul_smul]
  congr 1; ring

theorem bracketR_smul_right (n : ℕ) (c : Pn n) (x y : RMod n) :
    bracketR n x (c • y) = c • bracketR n x y := by
  unfold bracketR
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro p _
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro q _
  simp only [rMod_smul_apply, smul_smul]
  congr 1; ring

theorem bracketR_neg_left (n : ℕ) (x y : RMod n) : bracketR n (-x) y = -bracketR n x y := by
  have h1 : bracketR n (-x) y + bracketR n x y = 0 := by
    rw [← bracketR_add_left, neg_add_cancel, bracketR_zero_left]
  exact eq_neg_of_add_eq_zero_left h1

theorem sum_bool_eq {M : Type*} [AddCommMonoid M] (f : Bool → M) :
    ∑ b : Bool, f b = f true + f false := by
  rw [Fintype.univ_bool, Finset.sum_insert (by decide), Finset.sum_singleton]

theorem iotaR_zero (n : ℕ) : iotaR (0 : IndexedMod n) = 0 := by
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [iotaR]

/-- `iotaR` distributes over a finite sum of `Pn n`-scaled terms. -/
theorem iotaR_sum_smul {n : ℕ} {ι : Type*} (s : Finset ι) (c : ι → Pn n) (f : ι → IndexedMod n) :
    iotaR (∑ i ∈ s, c i • f i) = ∑ i ∈ s, c i • iotaR (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [iotaR_zero]
  | @insert a s ha ih =>
    rw [Finset.sum_insert ha, iotaR_add, iotaR_smul, ih, Finset.sum_insert ha]

theorem iotaR_sum {n : ℕ} {ι : Type*} (s : Finset ι) (f : ι → IndexedMod n) :
    iotaR (∑ i ∈ s, f i) = ∑ i ∈ s, iotaR (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [iotaR_zero]
  | @insert a s ha ih => rw [Finset.sum_insert ha, iotaR_add, ih, Finset.sum_insert ha]

/-- **K1a**: on the image of `g_P -> g_R`, `bracketR` restricts exactly to
`bracketN`. -/
theorem bracketR_iotaR_iotaR (n : ℕ) (x y : IndexedMod n) :
    bracketR n (iotaR x) (iotaR y) = iotaR (bracketN n x y) := by
  unfold bracketR bracketN
  rw [iotaR_sum]
  simp only [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl; intro i _
  rw [iotaR_sum_smul, sum_bool_eq]
  have h1 : (∑ j : IndexedBasis n, ∑ b' : Bool,
      (iotaR x (i, true) * iotaR y (j, b')) • bracketRBasis n (i, true) (j, b')) = 0 := by
    apply Finset.sum_eq_zero; intro j _
    apply Finset.sum_eq_zero; intro b' _
    simp [iotaR]
  rw [h1, zero_add]
  apply Finset.sum_congr rfl; intro j _
  rw [sum_bool_eq]
  have h2 : (iotaR x (i, false) * iotaR y (j, true)) • bracketRBasis n (i, false) (j, true) = 0 := by
    simp [iotaR]
  rw [h2, zero_add]
  simp [iotaR, bracketRBasis]

/-! ## K1c -- super-skew symmetry on `g_R`, the convention-pinning check -/

theorem zmod2_cases : ∀ q : ZMod 2, q = 0 ∨ q = 1 := by decide

theorem gsignN_comm (n : ℕ) (a b : ZMod 2) : gsignN n a b = gsignN n b a := by
  unfold gsignN; congr 1; rw [and_comm]

theorem gsignN_mul_one_left (n : ℕ) (p q : ZMod 2) :
    gsignN n p 1 * gsignN n p q = gsignN n p (q + 1) := by
  have e01 : (0 : ZMod 2) + 1 = 1 := by decide
  have e11 : (1 : ZMod 2) + 1 = 0 := by decide
  rcases zmod2_cases p with hp | hp <;> rcases zmod2_cases q with hq | hq <;>
    subst hp <;> subst hq <;>
    simp [e01, e11, gsignN_00, gsignN_01, gsignN_10, gsignN_11]

theorem gsignN_succ_left_mul (n : ℕ) (p q : ZMod 2) :
    gsignN n (p + 1) q * gsignN n q 1 = gsignN n p q := by
  have e01 : (0 : ZMod 2) + 1 = 1 := by decide
  have e11 : (1 : ZMod 2) + 1 = 0 := by decide
  rcases zmod2_cases p with hp | hp <;> rcases zmod2_cases q with hq | hq <;>
    subst hp <;> subst hq <;>
    simp [e01, e11, gsignN_00, gsignN_01, gsignN_10, gsignN_11]

@[simp] theorem parityR_false {n : ℕ} (i : IndexedBasis n) : parityR ((i, false) : RBasis n) = parity i := by
  unfold parityR; simp

@[simp] theorem parityR_true {n : ℕ} (i : IndexedBasis n) : parityR ((i, true) : RBasis n) = parity i + 1 := by
  unfold parityR; simp

/-- **K1c**: super-skew symmetry for `bracketRBasis`, over all four sectors
of `g_R` -- the convention-pinning check. Never mentions `GammaBetaN`,
`fBetaN` or `TBeta`. -/
theorem bracketRBasis_super_skew (n : ℕ) (p q : RBasis n) :
    bracketRBasis n p q = -(gsignN n (parityR p) (parityR q)) • bracketRBasis n q p := by
  rcases p with ⟨i, bi⟩; rcases q with ⟨j, bj⟩
  cases bi <;> cases bj <;> simp only [parityR_false, parityR_true, bracketRBasis]
  · -- (i,false),(j,false)
    rw [bracketBasisN_super_skew i j, iotaR_smul]
  · -- (i,false),(j,true)
    rw [kappaEmbed_smul, bracketBasisN_super_skew i j, kappaEmbed_smul, smul_smul]
    congr 1
    rw [mul_neg, gsignN_mul_one_left]
  · -- (i,true),(j,false)
    rw [bracketBasisN_super_skew i j, kappaEmbed_smul, kappaEmbed_smul, smul_smul]
    congr 1
    rw [neg_mul, gsignN_succ_left_mul]
  · -- (i,true),(j,true)
    simp

/-- **K1c**, extended to arbitrary homogeneous elements of `RMod n` -- the
same finite-sum bilinear argument as the accepted `bracketN_super_skew_homog`,
transposed to `RBasis n`. -/
theorem bracketR_super_skew_homog (n : ℕ) (x y : RMod n) (dx dy : ZMod 2)
    (hx : IsHomogR x dx) (hy : IsHomogR y dy) :
    bracketR n x y = -(gsignN n dx dy) • bracketR n y x := by
  have hswap : bracketR n y x
      = ∑ i : RBasis n, ∑ j : RBasis n, (x i * y j) • bracketRBasis n j i := by
    unfold bracketR
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl; intro j _
    apply Finset.sum_congr rfl; intro i _
    congr 1; ring
  rw [hswap]
  unfold bracketR
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro i _
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl; intro j _
  by_cases hxy : x i * y j = 0
  · simp [hxy]
  · have hxi : x i ≠ 0 := fun h0 => hxy (by rw [h0]; ring)
    have hyj : y j ≠ 0 := fun h0 => hxy (by rw [h0]; ring)
    rw [bracketRBasis_super_skew n i j, hx i hxi, hy j hyj, smul_smul, smul_smul, neg_mul]
    congr 1
    ring

/-! ## K2 -- the odd extension `(f_beta)_R` -/

theorem falsePart_add {n : ℕ} (r1 r2 : RMod n) :
    falsePart (r1 + r2) = falsePart r1 + falsePart r2 := by
  funext i; simp [falsePart]

theorem truePart_add {n : ℕ} (r1 r2 : RMod n) :
    truePart (r1 + r2) = truePart r1 + truePart r2 := by
  funext i; simp [truePart]

theorem falsePart_smul {n : ℕ} (c : Pn n) (r : RMod n) :
    falsePart (c • r) = c • falsePart r := by
  funext i; simp [falsePart]

theorem truePart_smul {n : ℕ} (c : Pn n) (r : RMod n) :
    truePart (c • r) = c • truePart r := by
  funext i; simp [truePart]

/-- The odd extension of `fBetaN`: on `r = x0 + kappa x1`
(`x0 = falsePart r`, `x1 = truePart r`), `(f_beta)_R(r) = f_beta(x0) -
kappa f_beta(x1)`, matching `(f_beta)_R(rX) = (-1)^{p(r)} r f_beta(X)` for
`r in {1,kappa}` (K2). -/
noncomputable def fBetaR (n : ℕ) (r : RMod n) : RMod n :=
  iotaR (fBetaN n (falsePart r)) + kappaEmbed (-(fBetaN n (truePart r)))

/-- **K2**: `(f_beta)_R` restricted to `g_P` (the `r = 1` case) is exactly
`f_beta` in the `false` slot -- not assumed, proved. -/
theorem fBetaR_iotaR (n : ℕ) (x : IndexedMod n) : fBetaR n (iotaR x) = iotaR (fBetaN n x) := by
  unfold fBetaR
  rw [falsePart_iotaR, truePart_iotaR, fBetaN_zero, neg_zero]
  have : kappaEmbed (0 : IndexedMod n) = (0 : RMod n) := by funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaEmbed]
  rw [this, add_zero]

/-- **K2**: `(f_beta)_R(kappa X) = -kappa f_beta(X)` -- not assumed, proved. -/
theorem fBetaR_kappaEmbed (n : ℕ) (x : IndexedMod n) :
    fBetaR n (kappaEmbed x) = -(kappaEmbed (fBetaN n x)) := by
  unfold fBetaR
  rw [falsePart_kappaEmbed, truePart_kappaEmbed, fBetaN_zero]
  have : iotaR (0 : IndexedMod n) = (0 : RMod n) := iotaR_zero n
  rw [this, zero_add, kappaEmbed_neg]

theorem fBetaR_add (n : ℕ) (r1 r2 : RMod n) : fBetaR n (r1 + r2) = fBetaR n r1 + fBetaR n r2 := by
  unfold fBetaR
  rw [falsePart_add, truePart_add, fBetaN_add, fBetaN_add, iotaR_add, neg_add, kappaEmbed_add]
  abel

theorem fBetaR_smul (n : ℕ) (c : Pn n) (r : RMod n) : fBetaR n (c • r) = c • fBetaR n r := by
  unfold fBetaR
  rw [falsePart_smul, truePart_smul, fBetaN_smul, fBetaN_smul, iotaR_smul, ← neg_smul, kappaEmbed_smul, smul_add]
  congr 1
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaEmbed]

/-! ## K3 -- `TBeta` and its inverse, `eq:splitting` -/

theorem falsePart_neg {n : ℕ} (r : RMod n) : falsePart (-r) = -falsePart r := by
  funext i; simp [falsePart]

theorem truePart_neg {n : ℕ} (r : RMod n) : truePart (-r) = -truePart r := by
  funext i; simp [truePart]

theorem kappaMulR_add (n : ℕ) (r1 r2 : RMod n) :
    kappaMulR (r1 + r2) = kappaMulR r1 + kappaMulR r2 := by
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaMulR]

theorem kappaMulR_iotaR (n : ℕ) (x : IndexedMod n) : kappaMulR (iotaR x) = kappaEmbed x := by
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaMulR, iotaR, kappaEmbed]

theorem kappaMulR_kappaEmbed (n : ℕ) (x : IndexedMod n) : kappaMulR (kappaEmbed x) = 0 := by
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaMulR, kappaEmbed]

/-- `T_beta = id + kappa (f_beta)_R` -- `eq:splitting`, literal. -/
noncomputable def TBeta (n : ℕ) (r : RMod n) : RMod n := r + kappaMulR (fBetaR n r)

/-- `T_beta^{-1} = id - kappa (f_beta)_R` -- `eq:splitting`, literal. -/
noncomputable def TBetaInv (n : ℕ) (r : RMod n) : RMod n := r - kappaMulR (fBetaR n r)

theorem TBeta_eq (n : ℕ) (r : RMod n) : TBeta n r = r + kappaEmbed (fBetaN n (falsePart r)) := by
  unfold TBeta fBetaR
  rw [kappaMulR_add, kappaMulR_iotaR, kappaMulR_kappaEmbed, add_zero]

theorem TBetaInv_eq (n : ℕ) (r : RMod n) : TBetaInv n r = r - kappaEmbed (fBetaN n (falsePart r)) := by
  unfold TBetaInv fBetaR
  rw [kappaMulR_add, kappaMulR_iotaR, kappaMulR_kappaEmbed, add_zero]

theorem falsePart_TBeta (n : ℕ) (r : RMod n) : falsePart (TBeta n r) = falsePart r := by
  rw [TBeta_eq, falsePart_add, falsePart_kappaEmbed, add_zero]

theorem falsePart_TBetaInv (n : ℕ) (r : RMod n) : falsePart (TBetaInv n r) = falsePart r := by
  rw [TBetaInv_eq, sub_eq_add_neg, falsePart_add, falsePart_neg, falsePart_kappaEmbed, neg_zero, add_zero]

/-- **K3**: `T_beta` and `T_beta^{-1}` are mutually inverse, exactly, with
no truncation -- `eq:splitting`. -/
theorem TBetaInv_TBeta (n : ℕ) (r : RMod n) : TBetaInv n (TBeta n r) = r := by
  rw [TBetaInv_eq, falsePart_TBeta, TBeta_eq]
  abel

theorem TBeta_TBetaInv (n : ℕ) (r : RMod n) : TBeta n (TBetaInv n r) = r := by
  rw [TBeta_eq, falsePart_TBetaInv, TBetaInv_eq]
  abel

/-- **K3**: `T_beta` is additive and `Pn n`-linear ("`P`-linear" half of
`R`-linearity). -/
theorem TBeta_add (n : ℕ) (r1 r2 : RMod n) : TBeta n (r1 + r2) = TBeta n r1 + TBeta n r2 := by
  rw [TBeta_eq, TBeta_eq, TBeta_eq, falsePart_add, fBetaN_add, kappaEmbed_add]
  abel

theorem TBeta_smul (n : ℕ) (c : Pn n) (r : RMod n) : TBeta n (c • r) = c • TBeta n r := by
  rw [TBeta_eq, TBeta_eq, falsePart_smul, fBetaN_smul, kappaEmbed_smul, smul_add]

/-- **K3**: `T_beta` commutes with multiplication by `kappa` -- the other
half of `R`-linearity (an even map needs no extra sign here). -/
theorem TBeta_kappaMulR (n : ℕ) (r : RMod n) : TBeta n (kappaMulR r) = kappaMulR (TBeta n r) := by
  have hfalse : falsePart (kappaMulR r) = 0 := by funext i; simp [falsePart, kappaMulR]
  rw [TBeta_eq, hfalse, fBetaN_zero]
  have hz : kappaEmbed (0 : IndexedMod n) = (0 : RMod n) := by
    funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaEmbed]
  rw [hz, add_zero, TBeta_eq, kappaMulR_add, kappaMulR_kappaEmbed, add_zero]

theorem IsHomogR_falsePart {n : ℕ} {r : RMod n} {d : ZMod 2} (hr : IsHomogR r d) :
    IsHomogN (falsePart r) d := by
  intro i hi
  rw [← parityR_false]
  exact hr (i, false) (by simpa [falsePart] using hi)

/-- **K3**: `T_beta` is even (parity-preserving). -/
theorem TBeta_isHomogR (n : ℕ) (r : RMod n) (d : ZMod 2) (hr : IsHomogR r d) :
    IsHomogR (TBeta n r) d := by
  have hfp : IsHomogN (falsePart r) d := IsHomogR_falsePart hr
  have hfb : IsHomogN (fBetaN n (falsePart r)) (d + 1) := fBetaN_isHomogN n (falsePart r) d hfp
  have e11 : (1 : ZMod 2) + 1 = 0 := by decide
  intro p hp
  rcases p with ⟨i, b⟩
  cases b with
  | false =>
    have h1 : (TBeta n r) (i, false) = r (i, false) := by
      rw [TBeta_eq]; simp [kappaEmbed]
    exact hr (i, false) (by rw [← h1]; exact hp)
  | true =>
    have h1 : (TBeta n r) (i, true) = r (i, true) + fBetaN n (falsePart r) i := by
      rw [TBeta_eq]; simp [kappaEmbed]
    rw [parityR_true]
    by_cases h2 : r (i, true) = 0
    · have h3 : fBetaN n (falsePart r) i ≠ 0 := by
        intro h0; apply hp; rw [h1, h2, h0]; ring
      have hpar := hfb i h3
      calc parity i + 1 = (d + 1) + 1 := by rw [hpar]
        _ = d + (1 + 1) := by ring
        _ = d := by rw [e11, add_zero]
    · have hpar := hr (i, true) h2
      rwa [parityR_true] at hpar

/-! ## K4 -- the deformed bracket, `eq:deformed-bracket` -/

/-- The four-row table, identical to `bracketRBasis` except row 1 (both
coordinate slots) also carries `kappa Gamma_beta` -- `eq:deformed-bracket`
on coordinate elements, extended by the same scalar rule. -/
noncomputable def bracketRBetaBasis (n : ℕ) : RBasis n → RBasis n → RMod n
  | (i, false), (j, false) => iotaR (bracketBasisN n i j) + kappaEmbed (GammaBetaBasis n i j)
  | (i, false), (j, true) => kappaEmbed (gsignN n (parity i) 1 • bracketBasisN n i j)
  | (i, true), (j, false) => kappaEmbed (bracketBasisN n i j)
  | (_, true), (_, true) => 0

noncomputable def bracketRBeta (n : ℕ) (x y : RMod n) : RMod n :=
  ∑ p : RBasis n, ∑ q : RBasis n, (x p * y q) • bracketRBetaBasis n p q

theorem bracketRBeta_eR_eR (n : ℕ) (p q : RBasis n) :
    bracketRBeta n (eR p) (eR q) = bracketRBetaBasis n p q := by
  unfold bracketRBeta
  rw [Finset.sum_eq_single p]
  · rw [Finset.sum_eq_single q]
    · simp [eR]
    · intro b _ hb
      have : eR q b = 0 := by unfold eR; rw [if_neg hb]
      simp [this]
    · intro h; exact absurd (Finset.mem_univ q) h
  · intro b _ hb
    have : eR p b = 0 := by unfold eR; rw [if_neg hb]
    simp [this]
  · intro h; exact absurd (Finset.mem_univ p) h

theorem kappaEmbed_zero (n : ℕ) : kappaEmbed (0 : IndexedMod n) = 0 := by
  funext p; rcases p with ⟨i, b⟩; cases b <;> simp [kappaEmbed]

theorem kappaEmbed_sum {n : ℕ} {ι : Type*} (s : Finset ι) (f : ι → IndexedMod n) :
    kappaEmbed (∑ i ∈ s, f i) = ∑ i ∈ s, kappaEmbed (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [kappaEmbed_zero]
  | @insert a s ha ih => rw [Finset.sum_insert ha, kappaEmbed_add, ih, Finset.sum_insert ha]

theorem kappaEmbed_sum_smul {n : ℕ} {ι : Type*} (s : Finset ι) (c : ι → Pn n) (f : ι → IndexedMod n) :
    kappaEmbed (∑ i ∈ s, c i • f i) = ∑ i ∈ s, c i • kappaEmbed (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [kappaEmbed_zero]
  | @insert a s ha ih =>
    rw [Finset.sum_insert ha, kappaEmbed_add, kappaEmbed_smul, ih, Finset.sum_insert ha]

/-- **K4**: on the image of `g_P -> g_R`, `bracketRBeta` restricts exactly
to `bracketN + kappa GammaBetaN`. -/
theorem bracketRBeta_iotaR_iotaR (n : ℕ) (x y : IndexedMod n) :
    bracketRBeta n (iotaR x) (iotaR y) = iotaR (bracketN n x y) + kappaEmbed (GammaBetaN n x y) := by
  have hlhs : bracketRBeta n (iotaR x) (iotaR y)
      = ∑ i : IndexedBasis n, ∑ j : IndexedBasis n, (x i * y j) • bracketRBetaBasis n (i, false) (j, false) := by
    unfold bracketRBeta
    simp only [Fintype.sum_prod_type]
    apply Finset.sum_congr rfl; intro i _
    rw [sum_bool_eq]
    have h1 : (∑ j : IndexedBasis n, ∑ b' : Bool,
        (iotaR x (i, true) * iotaR y (j, b')) • bracketRBetaBasis n (i, true) (j, b')) = 0 := by
      apply Finset.sum_eq_zero; intro j _
      apply Finset.sum_eq_zero; intro b' _; simp [iotaR]
    rw [h1, zero_add]
    apply Finset.sum_congr rfl; intro j _
    rw [sum_bool_eq]
    have h2 : (iotaR x (i, false) * iotaR y (j, true)) • bracketRBetaBasis n (i, false) (j, true) = 0 := by
      simp [iotaR]
    rw [h2, zero_add]
    simp [iotaR]
  rw [hlhs]
  unfold bracketRBetaBasis
  simp only [smul_add]
  rw [Finset.sum_congr rfl (fun i _ => Finset.sum_add_distrib (s := (Finset.univ : Finset (IndexedBasis n)))
      (f := fun j => (x i * y j) • iotaR (bracketBasisN n i j))
      (g := fun j => (x i * y j) • kappaEmbed (GammaBetaBasis n i j)))]
  rw [Finset.sum_add_distrib]
  unfold bracketN GammaBetaN bilinearExtend
  congr 1
  · symm; rw [iotaR_sum]; apply Finset.sum_congr rfl; intro i _; rw [iotaR_sum_smul]
  · symm; rw [kappaEmbed_sum]; apply Finset.sum_congr rfl; intro i _; rw [kappaEmbed_sum_smul]

/-! ## K5 -- the intertwining, `eq:intertwining` -/

theorem bracketR_eR_left (n : ℕ) (p : RBasis n) (x : RMod n) :
    bracketR n (eR p) x = ∑ q : RBasis n, x q • bracketRBasis n p q := by
  unfold bracketR
  rw [Finset.sum_eq_single p]
  · apply Finset.sum_congr rfl; intro q _; simp [eR]
  · intro b _ hb
    have : eR p b = 0 := by unfold eR; rw [if_neg hb]
    simp [this]
  · intro h; exact absurd (Finset.mem_univ p) h

theorem bracketR_eR_right (n : ℕ) (x : RMod n) (q : RBasis n) :
    bracketR n x (eR q) = ∑ p : RBasis n, x p • bracketRBasis n p q := by
  unfold bracketR
  apply Finset.sum_congr rfl; intro p _
  rw [Finset.sum_eq_single q]
  · simp [eR]
  · intro b _ hb
    have : eR q b = 0 := by unfold eR; rw [if_neg hb]
    simp [this]
  · intro h; exact absurd (Finset.mem_univ q) h

theorem kappaEmbed_eq_sum (n : ℕ) (y : IndexedMod n) :
    kappaEmbed y = ∑ k : IndexedBasis n, y k • eR ((k, true) : RBasis n) := by
  funext p; rcases p with ⟨l, b⟩
  rw [Finset.sum_apply]
  simp only [rMod_smul_apply]
  rw [Finset.sum_eq_single l]
  · simp [eR, kappaEmbed]
  · intro k _ hk
    have : eR ((k, true) : RBasis n) (l, b) = 0 := by
      unfold eR
      rw [if_neg (fun h => hk (congrArg Prod.fst h).symm)]
    simp [this]
  · intro h; exact absurd (Finset.mem_univ l) h

theorem bracketN_eN_left_eq_sum (n : ℕ) (i : IndexedBasis n) (y : IndexedMod n) :
    bracketN n (eN i) y = ∑ k : IndexedBasis n, y k • bracketBasisN n i k := by
  unfold bracketN
  rw [Finset.sum_eq_single i]
  · apply Finset.sum_congr rfl; intro k _; simp [eN]
  · intro a _ ha
    apply Finset.sum_eq_zero; intro k _
    have : eN i a = 0 := by unfold eN; rw [if_neg ha]
    simp [this]
  · intro h; exact absurd (Finset.mem_univ i) h

theorem bracketN_eN_right_eq_sum (n : ℕ) (y : IndexedMod n) (j : IndexedBasis n) :
    bracketN n y (eN j) = ∑ k : IndexedBasis n, y k • bracketBasisN n k j := by
  unfold bracketN
  apply Finset.sum_congr rfl; intro k _
  rw [Finset.sum_eq_single j]
  · simp [eN]
  · intro a _ ha
    have : eN j a = 0 := by unfold eN; rw [if_neg ha]
    simp [this]
  · intro h; exact absurd (Finset.mem_univ j) h

theorem bracketR_eR_false_kappaEmbed (n : ℕ) (i : IndexedBasis n) (y : IndexedMod n) :
    bracketR n (eR ((i, false) : RBasis n)) (kappaEmbed y)
      = kappaEmbed (gsignN n (parity i) 1 • bracketN n (eN i) y) := by
  have hlhs : bracketR n (eR ((i, false) : RBasis n)) (kappaEmbed y)
      = ∑ k : IndexedBasis n, y k • kappaEmbed (gsignN n (parity i) 1 • bracketBasisN n i k) := by
    rw [bracketR_eR_left]
    simp only [Fintype.sum_prod_type]
    apply Finset.sum_congr rfl; intro k _
    rw [sum_bool_eq]
    have h1 : (kappaEmbed y (k, false)) • bracketRBasis n (i, false) (k, false) = 0 := by
      simp [kappaEmbed]
    rw [h1, add_zero]
    show (kappaEmbed y (k, true)) • bracketRBasis n (i, false) (k, true)
      = (y k) • kappaEmbed (gsignN n (parity i) 1 • bracketBasisN n i k)
    simp [kappaEmbed, bracketRBasis]
  rw [hlhs, bracketN_eN_left_eq_sum, Finset.smul_sum, kappaEmbed_sum_smul]
  apply Finset.sum_congr rfl; intro k _
  rw [kappaEmbed_smul, kappaEmbed_smul, smul_smul, smul_smul]
  congr 1
  ring

theorem bracketR_sum_left' {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (f : ι → RMod n)
    (y : RMod n) : bracketR n (∑ i ∈ s, f i) y = ∑ i ∈ s, bracketR n (f i) y := by
  induction s using Finset.induction with
  | empty => simp [bracketR_zero_left]
  | @insert a s ha ih => rw [Finset.sum_insert ha, bracketR_add_left, ih, Finset.sum_insert ha]

theorem bracketR_sum_right' {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (f : ι → RMod n)
    (x : RMod n) : bracketR n x (∑ i ∈ s, f i) = ∑ i ∈ s, bracketR n x (f i) := by
  induction s using Finset.induction with
  | empty => simp [bracketR_zero_right]
  | @insert a s ha ih => rw [Finset.sum_insert ha, bracketR_add_right, ih, Finset.sum_insert ha]

theorem bracketR_kappaEmbed_eR_false (n : ℕ) (y : IndexedMod n) (j : IndexedBasis n) :
    bracketR n (kappaEmbed y) (eR ((j, false) : RBasis n)) = kappaEmbed (bracketN n y (eN j)) := by
  rw [kappaEmbed_eq_sum, bracketR_sum_left']
  have : ∀ k : IndexedBasis n, bracketR n (y k • eR ((k, true) : RBasis n)) (eR ((j, false) : RBasis n))
      = y k • kappaEmbed (bracketBasisN n k j) := by
    intro k
    rw [bracketR_smul_left, bracketR_eR_eR]
    rfl
  rw [Finset.sum_congr rfl (fun k _ => this k), bracketN_eN_right_eq_sum, kappaEmbed_sum_smul]

theorem bracketR_eR_true_kappaEmbed (n : ℕ) (i : IndexedBasis n) (y : IndexedMod n) :
    bracketR n (eR ((i, true) : RBasis n)) (kappaEmbed y) = 0 := by
  rw [kappaEmbed_eq_sum, bracketR_sum_right']
  apply Finset.sum_eq_zero; intro k _
  rw [bracketR_smul_right, bracketR_eR_eR]
  simp [bracketRBasis]

theorem bracketR_kappaEmbed_eR_true (n : ℕ) (y : IndexedMod n) (j : IndexedBasis n) :
    bracketR n (kappaEmbed y) (eR ((j, true) : RBasis n)) = 0 := by
  rw [kappaEmbed_eq_sum, bracketR_sum_left']
  apply Finset.sum_eq_zero; intro k _
  rw [bracketR_smul_left, bracketR_eR_eR]
  simp [bracketRBasis]

theorem bracketR_kappaEmbed_kappaEmbed (n : ℕ) (y1 y2 : IndexedMod n) :
    bracketR n (kappaEmbed y1) (kappaEmbed y2) = 0 := by
  rw [kappaEmbed_eq_sum y1 (n := n), bracketR_sum_left']
  apply Finset.sum_eq_zero; intro k _
  rw [bracketR_smul_left, bracketR_eR_true_kappaEmbed]
  simp

theorem falsePart_eR_false (n : ℕ) (i : IndexedBasis n) :
    falsePart ((eR ((i, false) : RBasis n))) = eN i := by
  funext k
  unfold falsePart eR eN
  simp [Prod.ext_iff]

theorem falsePart_eR_true (n : ℕ) (i : IndexedBasis n) :
    falsePart ((eR ((i, true) : RBasis n))) = 0 := by
  funext k
  unfold falsePart eR
  simp

/-- `T_beta` on a `g_P` coordinate basis vector: `eq:primitive`'s `f_beta`
enters exactly here. -/
theorem TBeta_eR_false (n : ℕ) (i : IndexedBasis n) :
    TBeta n (eR ((i, false) : RBasis n)) = eR ((i, false) : RBasis n) + kappaEmbed (fBetaBasis n i) := by
  rw [TBeta_eq, falsePart_eR_false, fBetaN_eN]

/-- `T_beta` fixes any `kappa`-basis vector exactly. -/
theorem TBeta_eR_true (n : ℕ) (i : IndexedBasis n) :
    TBeta n (eR ((i, true) : RBasis n)) = eR ((i, true) : RBasis n) := by
  rw [TBeta_eq, falsePart_eR_true, fBetaN_zero]
  have : kappaEmbed (0 : IndexedMod n) = (0 : RMod n) := kappaEmbed_zero n
  rw [this, add_zero]

/-- **K5**, basis level: the intertwining holds on every basis pair,
citing `G4` in the substantive case -- the only nontrivial one. -/
theorem intertwining_basis (n : ℕ) (p q : RBasis n) :
    bracketR n (TBeta n (eR p)) (TBeta n (eR q)) = TBeta n (bracketRBeta n (eR p) (eR q)) := by
  rcases p with ⟨i, bi⟩; rcases q with ⟨j, bj⟩
  cases bi <;> cases bj
  · -- (i,false),(j,false) -- the substantive case
    rw [TBeta_eR_false, TBeta_eR_false, bracketRBeta_eR_eR]
    show bracketR n (eR ((i, false) : RBasis n) + kappaEmbed (fBetaBasis n i))
        (eR ((j, false) : RBasis n) + kappaEmbed (fBetaBasis n j))
      = TBeta n (bracketRBetaBasis n (i, false) (j, false))
    rw [bracketR_add_left, bracketR_add_right, bracketR_add_right]
    rw [bracketR_eR_eR, bracketR_eR_false_kappaEmbed, bracketR_kappaEmbed_eR_false,
        bracketR_kappaEmbed_kappaEmbed, add_zero]
    simp only [bracketRBasis]
    set A := gsignN n (parity i) 1 • bracketN n (eN i) (fBetaBasis n j) with hA
    set B := bracketN n (fBetaBasis n i) (eN j) with hB
    -- LHS is now `iotaR (bracketBasisN n i j) + kappaEmbed A + kappaEmbed B`.
    have hfix : TBeta n (kappaEmbed (GammaBetaBasis n i j)) = kappaEmbed (GammaBetaBasis n i j) := by
      rw [TBeta_eq, falsePart_kappaEmbed, fBetaN_zero, kappaEmbed_zero, add_zero]
    simp only [bracketRBetaBasis]
    rw [TBeta_add, TBeta_eq, falsePart_iotaR, hfix]
    set Z := fBetaN n (bracketBasisN n i j) with hZ
    -- RHS is now `iotaR (bracketBasisN n i j) + kappaEmbed Z + kappaEmbed (GammaBetaBasis n i j)`.
    rw [G4 n i j]
    unfold deltaFBasis
    rw [bracketN_eN_eN, fBetaN_eN, fBetaN_eN, ← hB, ← hA, ← hZ]
    -- RHS's third term is now `kappaEmbed (B + A - Z)`.
    have key : kappaEmbed A + kappaEmbed B = kappaEmbed Z + kappaEmbed (B + A - Z) := by
      rw [← kappaEmbed_add, ← kappaEmbed_add]
      congr 1
      abel
    calc iotaR (bracketBasisN n i j) + kappaEmbed A + kappaEmbed B
        = iotaR (bracketBasisN n i j) + (kappaEmbed A + kappaEmbed B) := by abel
      _ = iotaR (bracketBasisN n i j) + (kappaEmbed Z + kappaEmbed (B + A - Z)) := by rw [key]
      _ = iotaR (bracketBasisN n i j) + kappaEmbed Z + kappaEmbed (B + A - Z) := by abel
  · -- (i,false),(j,true)
    rw [TBeta_eR_false, TBeta_eR_true, bracketRBeta_eR_eR]
    show bracketR n (eR ((i, false) : RBasis n) + kappaEmbed (fBetaBasis n i)) (eR ((j, true) : RBasis n))
      = TBeta n (bracketRBetaBasis n (i, false) (j, true))
    rw [bracketR_add_left, bracketR_eR_eR, bracketR_kappaEmbed_eR_true, add_zero]
    show bracketRBasis n (i, false) (j, true) = TBeta n (bracketRBetaBasis n (i, false) (j, true))
    show kappaEmbed (gsignN n (parity i) 1 • bracketBasisN n i j)
      = TBeta n (kappaEmbed (gsignN n (parity i) 1 • bracketBasisN n i j))
    rw [TBeta_eq, falsePart_kappaEmbed, fBetaN_zero]
    have : kappaEmbed (0 : IndexedMod n) = (0 : RMod n) := kappaEmbed_zero n
    rw [this, add_zero]
  · -- (i,true),(j,false)
    rw [TBeta_eR_true, TBeta_eR_false, bracketRBeta_eR_eR]
    show bracketR n (eR ((i, true) : RBasis n)) (eR ((j, false) : RBasis n) + kappaEmbed (fBetaBasis n j))
      = TBeta n (bracketRBetaBasis n (i, true) (j, false))
    rw [bracketR_add_right, bracketR_eR_eR, bracketR_eR_true_kappaEmbed, add_zero]
    show bracketRBasis n (i, true) (j, false) = TBeta n (bracketRBetaBasis n (i, true) (j, false))
    show kappaEmbed (bracketBasisN n i j) = TBeta n (kappaEmbed (bracketBasisN n i j))
    rw [TBeta_eq, falsePart_kappaEmbed, fBetaN_zero]
    have : kappaEmbed (0 : IndexedMod n) = (0 : RMod n) := kappaEmbed_zero n
    rw [this, add_zero]
  · -- (i,true),(j,true)
    rw [TBeta_eR_true, TBeta_eR_true, bracketRBeta_eR_eR, bracketR_eR_eR]
    show (0 : RMod n) = TBeta n (0 : RMod n)
    have hfp0 : falsePart (0 : RMod n) = 0 := by funext k; simp [falsePart]
    rw [TBeta_eq, hfp0, fBetaN_zero]
    have : kappaEmbed (0 : IndexedMod n) = (0 : RMod n) := kappaEmbed_zero n
    rw [this, add_zero]

theorem eR_decompose (n : ℕ) (r : RMod n) : r = ∑ p : RBasis n, r p • eR p := by
  funext l
  rw [Finset.sum_apply]
  simp only [rMod_smul_apply]
  rw [Finset.sum_eq_single l]
  · simp [eR]
  · intro k _ hk
    have : eR k l = 0 := by unfold eR; rw [if_neg (Ne.symm hk)]
    simp [this]
  · intro h; exact absurd (Finset.mem_univ l) h

theorem TBeta_zero (n : ℕ) : TBeta n (0 : RMod n) = 0 := by
  have hfp0 : falsePart (0 : RMod n) = 0 := by funext k; simp [falsePart]
  rw [TBeta_eq, hfp0, fBetaN_zero, kappaEmbed_zero, add_zero]

theorem TBeta_sum' {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (c : ι → Pn n) (f : ι → RMod n) :
    TBeta n (∑ i ∈ s, c i • f i) = ∑ i ∈ s, c i • TBeta n (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [TBeta_zero]
  | @insert a s ha ih =>
    rw [Finset.sum_insert ha, TBeta_add, TBeta_smul, ih, Finset.sum_insert ha]

theorem bracketR_sum_left'' {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (c : ι → Pn n)
    (f : ι → RMod n) (y : RMod n) :
    bracketR n (∑ i ∈ s, c i • f i) y = ∑ i ∈ s, c i • bracketR n (f i) y := by
  classical
  induction s using Finset.induction with
  | empty => simp [bracketR_zero_left]
  | @insert a s ha ih =>
    rw [Finset.sum_insert ha, bracketR_add_left, bracketR_smul_left, ih, Finset.sum_insert ha]

theorem bracketR_sum_right'' {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (c : ι → Pn n)
    (f : ι → RMod n) (x : RMod n) :
    bracketR n x (∑ i ∈ s, c i • f i) = ∑ i ∈ s, c i • bracketR n x (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [bracketR_zero_right]
  | @insert a s ha ih =>
    rw [Finset.sum_insert ha, bracketR_add_right, bracketR_smul_right, ih, Finset.sum_insert ha]

theorem TBeta_sum {n : ℕ} {ι : Type*} [DecidableEq ι] (s : Finset ι) (f : ι → RMod n) :
    TBeta n (∑ i ∈ s, f i) = ∑ i ∈ s, TBeta n (f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp [TBeta_zero]
  | @insert a s ha ih => rw [Finset.sum_insert ha, TBeta_add, ih, Finset.sum_insert ha]

/-- **K5**: the intertwining `eq:intertwining`, for **arbitrary** elements
of `g_R` -- **unconditional**, with no homogeneity hypothesis, exactly as
`GammaBetaN_eq_deltaFN` was for P2: `T_beta` is `Pn n`-linear
(`TBeta_add`/`TBeta_smul`), so it commutes with the basis expansion, and
`bracketR`/`bracketRBeta` are both bilinear, so the whole identity reduces
to `intertwining_basis` pointwise via two nested `Finset.sum_congr`s. -/
theorem intertwining (n : ℕ) (x y : RMod n) :
    bracketR n (TBeta n x) (TBeta n y) = TBeta n (bracketRBeta n x y) := by
  have hLHS : bracketR n (TBeta n x) (TBeta n y)
      = ∑ p : RBasis n, ∑ q : RBasis n, (x p * y q) • bracketR n (TBeta n (eR p)) (TBeta n (eR q)) := by
    conv_lhs => rw [eR_decompose n x, eR_decompose n y]
    rw [TBeta_sum', TBeta_sum', bracketR_sum_left'']
    apply Finset.sum_congr rfl; intro p _
    rw [bracketR_sum_right'', Finset.smul_sum]
    apply Finset.sum_congr rfl; intro q _
    rw [smul_smul]
  have hRHS : TBeta n (bracketRBeta n x y)
      = ∑ p : RBasis n, ∑ q : RBasis n, (x p * y q) • bracketR n (TBeta n (eR p)) (TBeta n (eR q)) := by
    unfold bracketRBeta
    rw [TBeta_sum]
    apply Finset.sum_congr rfl; intro p _
    rw [TBeta_sum']
    apply Finset.sum_congr rfl; intro q _
    rw [← bracketRBeta_eR_eR n p q]
    exact congrArg (fun z => (x p * y q) • z) (intertwining_basis n p q).symm
  rw [hLHS, hRHS]

end Indexed
end InhomogeneousDeformations
