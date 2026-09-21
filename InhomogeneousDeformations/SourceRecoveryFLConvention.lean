import InhomogeneousDeformations.SourceRecoveryClosedImage

/-!
# I106 R8b — U0-U5: the FL sector, re-derived without the circular link

Twelve-layer freeze (R7's eleven layers plus `SourceRecoveryClosedImage.lean`) is read-only; this
file only reads their already-accepted definitions/theorems. `GammaBetaBasis`
(`IndexedCoboundary.lean`, frozen) is NEVER edited.

## U4 — what went wrong with R8, corrected here

R8's file claimed the chain from `iotaBetaRBracket_super_skew` down to `GammaBetaBasis_Fof_Lof_forced`
never used `GammaBetaBasis`'s declared FL row. That claim was **true of the new file's own steps**
and **false of the chain**: `bracketRBetaBasis_super_skew` cites `iotaBetaR_bracketRBetaBasis`,
which for the `(false, false)` case cites R7's `iotaBetaRBracket_false_false`, which is proved via
R5's `liftsBracket_eq_bridge`, which internally branches on `F0hatBeta_bracket_FL` — a theorem
*stated* through `GammaBetaBasis` whose own *proof* reads `Indexed.GammaBetaBasis_Fof_Lof` (the
declared FL row) directly. Lean's dependency tracking is per compiled constant: any theorem
depending on `liftsBracket_eq_bridge`, even instantiated at unrelated index values, carries that
transitive dependency. Agent1c's mechanical checker (BFS over `ConstantInfo.getUsedConstantsAsSet`
from each claimed theorem's compiled proof term to `InhomogeneousDeformations.Indexed.GammaBetaBasis_Fof_Lof`)
confirmed the path

`GammaBetaBasis_Fof_Lof_forced → bracketRBetaBasis_super_skew → iotaBetaR_bracketRBetaBasis
  → iotaBetaRBracket_false_false → liftsBracket_eq_bridge → GammaBetaBasis_Fof_Lof`.

**This file drops that entire chain.** `bracketRBetaBasis_super_skew`,
`GammaBetaBasis_Fof_Lof_forced`, and `GammaBetaBasis_Fof_Lof_unique` are not reproduced here at
all (not even relabelled) — the round's own instruction allows dropping them outright, which is
the choice made. Only `gsignNQ_symm`, `gsignNQ_sq`, and `iotaBetaRBracket_super_skew` are kept
verbatim (U1): Agent1c's checker found no dependency path from any of the three to the FL row, and
that result is unaffected by anything below.

In their place, U2/U3 give a second, independent route to the same FL value, built from R5's
`L0hatBeta_bracket_LF` (the LF sector's *own* theorem, itself checked FL-free) and a sign flip in
the concrete algebra `A_B`, never mentioning `GammaBetaBasis` and never citing
`F0hatBeta_bracket_FL` (the contaminated theorem) or `liftsBracket_eq_bridge`.

## U0 — the mechanical check, run during the work

Agent1c's checker (`agent1c_checker/Agent1cCheck4.lean`) was copied to a throwaway scratch file
inside `InhomogeneousDeformations/`, edited after each theorem below was added, run via
`lake env lean InhomogeneousDeformations/Agent1cCheck4Scratch.lean`, and deleted before delivery (it is not
part of this payload). Its exact output, per theorem:

* `InhomogeneousDeformations.Source.gsignNQ_symm` — `NO PATH from InhomogeneousDeformations.Source.gsignNQ_symm to the FL
  row lemma` (R8's own result, unaffected; re-run here as a control, unchanged).
* `InhomogeneousDeformations.Source.gsignNQ_sq` — `NO PATH from InhomogeneousDeformations.Source.gsignNQ_sq to the FL row
  lemma` (control, unchanged).
* `InhomogeneousDeformations.Source.iotaBetaRBracket_super_skew` — `NO PATH from
  InhomogeneousDeformations.Source.iotaBetaRBracket_super_skew to the FL row lemma` (control, unchanged).
* `InhomogeneousDeformations.Source.F0hatBeta_comm_L0hatBeta_FL_free` — `NO PATH from
  InhomogeneousDeformations.Source.F0hatBeta_comm_L0hatBeta_FL_free to the FL row lemma`.
* `InhomogeneousDeformations.Source.GammaBetaBasis_Fof_Lof_forced_FL_free` — `NO PATH from
  InhomogeneousDeformations.Source.GammaBetaBasis_Fof_Lof_forced_FL_free to the FL row lemma`.

(Exact `run_cmd`/`logInfo` output as produced by the scratch run is quoted verbatim in the round's
report; the lines above match it.)

## U5 — for the appendix

U0's mechanical check shows both `F0hatBeta_comm_L0hatBeta_FL_free` (U2) and
`GammaBetaBasis_Fof_Lof_forced_FL_free` (U3) have no dependency path to
`InhomogeneousDeformations.Indexed.GammaBetaBasis_Fof_Lof`. U3 states that any coefficient `c` the concrete
algebra `A_B` admits on the FL sector (`F0hatBeta n w * L0hatBeta n u v - L0hatBeta n u v *
F0hatBeta n w = iota0 n (bracketBasisN n (Fof w) (Lof u v)) + kappaAB n * iota0 n c`) is forced to
equal `-(gammaLFn n u v w)` — derived from R5's LF theorem (`L0hatBeta_bracket_LF`, itself
FL-free) via a sign flip in `A_B` and R6's `kappaAB_iota0_eq_zero_imp`, never inspecting
`GammaBetaBasis`'s declared FL row. So the FL sector's value, previously a declared convention
corroborated only by its own row definition, is forced by the concrete algebra plus the
independently-known LF value, through a route mechanically confirmed clean end to end.
-/

namespace InhomogeneousDeformations
namespace Source

open scoped TensorProduct

/-! ## U1 — kept verbatim from the rejected R8 file (confirmed FL-free by Agent1c and Agent2c) -/

theorem gsignNQ_symm (n : ℕ) (a b : ZMod 2) : gsignNQ n a b = gsignNQ n b a := by
  match a, b with
  | (0 : ZMod 2), (0 : ZMod 2) => rw [gsignNQ_00]
  | (0 : ZMod 2), (1 : ZMod 2) => rw [gsignNQ_01, gsignNQ_10]
  | (1 : ZMod 2), (0 : ZMod 2) => rw [gsignNQ_10, gsignNQ_01]
  | (1 : ZMod 2), (1 : ZMod 2) => rw [gsignNQ_11]

theorem gsignNQ_sq (n : ℕ) (a b : ZMod 2) : gsignNQ n a b * gsignNQ n a b = 1 := by
  match a, b with
  | (0 : ZMod 2), (0 : ZMod 2) => rw [gsignNQ_00]; ring
  | (0 : ZMod 2), (1 : ZMod 2) => rw [gsignNQ_01]; ring
  | (1 : ZMod 2), (0 : ZMod 2) => rw [gsignNQ_10]; ring
  | (1 : ZMod 2), (1 : ZMod 2) => rw [gsignNQ_11]; ring

theorem iotaBetaRBracket_super_skew (n : ℕ) (p q : Indexed.RBasis n) :
    iotaBetaRBracket n p q
      = -(gsignNQ n (Indexed.parityR p) (Indexed.parityR q)) • iotaBetaRBracket n q p := by
  unfold iotaBetaRBracket
  rw [show gsignNQ n (Indexed.parityR q) (Indexed.parityR p)
      = gsignNQ n (Indexed.parityR p) (Indexed.parityR q) from
        gsignNQ_symm n (Indexed.parityR q) (Indexed.parityR p)]
  set g := gsignNQ n (Indexed.parityR p) (Indexed.parityR q) with hgdef
  set X := iotaBetaR n (Indexed.eR p) with hXdef
  set Y := iotaBetaR n (Indexed.eR q) with hYdef
  have hsq : g * g = 1 := gsignNQ_sq n (Indexed.parityR p) (Indexed.parityR q)
  rw [smul_sub, smul_smul, neg_mul, hsq, neg_one_smul, sub_neg_eq_add, neg_smul]
  abel

/-! ## U2 — the FL computation, stated against `-(gammaLFn)`, `GammaBetaBasis` unmentioned

The route: `F0hatBeta n w * L0hatBeta n u v - L0hatBeta n u v * F0hatBeta n w` is exactly the
negative of R5's frozen `L0hatBeta_bracket_LF n u v w`'s own left side. Negating that theorem and
pushing the negation through `iota0` (`iota0_neg`, frozen, `SourceRecoveryBridge.lean`) and through
the sign in `A_B` (`mul_neg`), then converting `bracketBasisN n (Lof u v) (Fof w)` /
`bracketBasisN n (Fof w) (Lof u v)` into each other via `bracketBasisN_Lof_Fof` /
`bracketBasisN_Fof_Lof` (frozen, `IndexedJacobi.lean`, structural — unrelated to `GammaBetaBasis`)
gives the FL counterpart directly. `F0hatBeta_bracket_FL` (R5's contaminated theorem) is not
cited anywhere. -/

theorem F0hatBeta_comm_L0hatBeta_FL_free (n : ℕ) (u v w : Fin (2 * n)) :
    F0hatBeta n w * L0hatBeta n u v - L0hatBeta n u v * F0hatBeta n w
      = iota0 n (Indexed.bracketBasisN n (Indexed.Fof w) (Indexed.Lof u v))
        + kappaAB n * iota0 n (-(Indexed.gammaLFn n u v w)) := by
  have h := L0hatBeta_bracket_LF n u v w
  have hLHS : F0hatBeta n w * L0hatBeta n u v - L0hatBeta n u v * F0hatBeta n w
      = -(L0hatBeta n u v * F0hatBeta n w - F0hatBeta n w * L0hatBeta n u v) :=
    (neg_sub (L0hatBeta n u v * F0hatBeta n w) (F0hatBeta n w * L0hatBeta n u v)).symm
  rw [hLHS, h, Indexed.bracketBasisN_Lof_Fof, Indexed.bracketBasisN_Fof_Lof]
  simp only [iota0_neg, mul_neg]
  abel

/-! ## U3 — the FL value pinned, FL-free transitively

Working directly with the raw commutator and `iota0`/`kappaAB` (not via `iotaBetaR`/
`iotaBetaRBracket`/`bracketRBetaBasis`): given any coefficient `c` for which the concrete algebra
satisfies the same commutator equation as `F0hatBeta_comm_L0hatBeta_FL_free`, cancel the shared
`iota0 n (bracketBasisN n (Fof w) (Lof u v))` term, then apply R6's frozen
`kappaAB_iota0_eq_zero_imp` (`SourceRecoveryClosure.lean`, confirmed FL-free) to
`c + gammaLFn n u v w`. -/

theorem GammaBetaBasis_Fof_Lof_forced_FL_free (n : ℕ) (u v w : Fin (2 * n)) (c : Indexed.IndexedMod n)
    (h : F0hatBeta n w * L0hatBeta n u v - L0hatBeta n u v * F0hatBeta n w
        = iota0 n (Indexed.bracketBasisN n (Indexed.Fof w) (Indexed.Lof u v))
          + kappaAB n * iota0 n c) :
    c = -(Indexed.gammaLFn n u v w) := by
  have h2 := F0hatBeta_comm_L0hatBeta_FL_free n u v w
  rw [h2] at h
  have hBC : kappaAB n * iota0 n c - kappaAB n * iota0 n (-(Indexed.gammaLFn n u v w)) = 0 := by
    have heq : kappaAB n * iota0 n c - kappaAB n * iota0 n (-(Indexed.gammaLFn n u v w))
        = (iota0 n (Indexed.bracketBasisN n (Indexed.Fof w) (Indexed.Lof u v)) + kappaAB n * iota0 n c)
          - (iota0 n (Indexed.bracketBasisN n (Indexed.Fof w) (Indexed.Lof u v))
              + kappaAB n * iota0 n (-(Indexed.gammaLFn n u v w))) := by abel
    rw [heq, h]; abel
  have hcancel : kappaAB n * iota0 n c = kappaAB n * iota0 n (-(Indexed.gammaLFn n u v w)) :=
    sub_eq_zero.mp hBC
  have hexpand : kappaAB n * iota0 n (c + Indexed.gammaLFn n u v w)
      = kappaAB n * iota0 n c + kappaAB n * iota0 n (Indexed.gammaLFn n u v w) := by
    rw [iota0_add, mul_add]
  have hzero : kappaAB n * iota0 n (c + Indexed.gammaLFn n u v w) = 0 := by
    rw [hexpand, hcancel, iota0_neg, mul_neg]
    abel
  have hsum := kappaAB_iota0_eq_zero_imp n (c + Indexed.gammaLFn n u v w) hzero
  exact eq_neg_iff_add_eq_zero.mpr hsum

end Source
end InhomogeneousDeformations
