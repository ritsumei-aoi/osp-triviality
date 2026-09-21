import InhomogeneousDeformations.FixtureData

/-!
# T5-T7 — decode success, native correspondence, examples

`H1.3` (verbatim, do not upgrade): a successful T5-T7 establishes P0 and P5
**at n=1, for this one datum**: that a structured input in the declared
shape decodes, and that the decoded bracket is the independently defined
native bracket. It does NOT establish correctness of a JSON parser, of the
Python extractor, of any other document, of any other profile, or of a
general generator/parser correspondence.
-/

namespace InhomogeneousDeformations
namespace Bridge

open Decode Wire

/-! ## T5: decoding the actual generated fixture datum succeeds -/

theorem T5_decode_succeeds :
    Decode.decode fixtureRawInput = Except.ok fixtureRawInput.rows := rfl

theorem T5_module_id : fixtureRawInput.moduleId = "g_n1" := rfl
theorem T5_basis_ids : fixtureRawInput.basisIds = Decode.n1BasisIds := rfl
theorem T5_basis_degrees : fixtureRawInput.basisDegrees = [0, 0, 0, 1, 1] := rfl
theorem T5_op_id : fixtureRawInput.opId = "bracket_n1" := rfl
theorem T5_op_kind : fixtureRawInput.opKind = "lie-bracket" := rfl
theorem T5_op_degree : fixtureRawInput.opDegree = 0 := rfl
theorem T5_op_scalar_behavior : fixtureRawInput.opScalarBehavior = "bilinear-even-scalars" := rfl
theorem T5_def_kind : fixtureRawInput.defKind = "canonical-pair-table" := rfl
theorem T5_coverage : fixtureRawInput.coverage = "total" := rfl
theorem T5_row_count : fixtureRawInput.rows.length = 15 := rfl

/-! ## T6: the decoded bracket equals the independent native bracket

Fifteen forward lemmas (one per declared-order pair `p ≤ q`), each proved
by finding the row (a pure computation, closed by `rfl`) and then
reducing the resulting `Coeff` equation. -/

theorem forward_L11_L11 : Decode.forwardValue fixtureRawInput.rows 0 0 = bracketBasis .L11 .L11 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (0, 0)) =
      some { inputs := ["L11", "L11"], output := ([] : Wire.WVector) } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod
  funext k; fin_cases k <;> simp [bracketBasis, bracketLL, Jmat]

theorem forward_L11_L12 : Decode.forwardValue fixtureRawInput.rows 0 1 = bracketBasis .L11 .L12 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (0, 1)) =
      some { inputs := ["L11", "L12"],
             output := [{ basisId := "L11", coeff := [{ coeff := { num := 1, den := 1 }, exponents := [0, 0] }] }] } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod Decode.polyAsCoeff Decode.termAsCoeff
  funext k; fin_cases k <;> simp [basisOfName, bracketBasis, bracketLL, Jmat, Lof, Wire.Rat.toRat] <;> norm_num

theorem forward_L11_L22 : Decode.forwardValue fixtureRawInput.rows 0 2 = bracketBasis .L11 .L22 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (0, 2)) =
      some { inputs := ["L11", "L22"],
             output := [{ basisId := "L12", coeff := [{ coeff := { num := 2, den := 1 }, exponents := [0, 0] }] }] } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod Decode.polyAsCoeff Decode.termAsCoeff
  funext k; fin_cases k <;> simp [basisOfName, bracketBasis, bracketLL, Jmat, Lof, Wire.Rat.toRat] <;> norm_num

theorem forward_L11_F1 : Decode.forwardValue fixtureRawInput.rows 0 3 = bracketBasis .L11 .F1 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (0, 3)) =
      some { inputs := ["L11", "F1"], output := ([] : Wire.WVector) } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod
  funext k; fin_cases k <;> simp [bracketBasis, bracketLF, Jmat]

theorem forward_L11_F2 : Decode.forwardValue fixtureRawInput.rows 0 4 = bracketBasis .L11 .F2 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (0, 4)) =
      some { inputs := ["L11", "F2"],
             output := [{ basisId := "F1", coeff := [{ coeff := { num := 1, den := 1 }, exponents := [0, 0] }] }] } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod Decode.polyAsCoeff Decode.termAsCoeff
  funext k; fin_cases k <;> simp [basisOfName, bracketBasis, bracketLF, Jmat, Fof, Wire.Rat.toRat] <;> norm_num

theorem forward_L12_L12 : Decode.forwardValue fixtureRawInput.rows 1 1 = bracketBasis .L12 .L12 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (1, 1)) =
      some { inputs := ["L12", "L12"], output := ([] : Wire.WVector) } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod
  funext k; fin_cases k <;> simp [bracketBasis, bracketLL, Jmat, Lof] <;> norm_num

theorem forward_L12_L22 : Decode.forwardValue fixtureRawInput.rows 1 2 = bracketBasis .L12 .L22 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (1, 2)) =
      some { inputs := ["L12", "L22"],
             output := [{ basisId := "L22", coeff := [{ coeff := { num := 1, den := 1 }, exponents := [0, 0] }] }] } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod Decode.polyAsCoeff Decode.termAsCoeff
  funext k; fin_cases k <;> simp [basisOfName, bracketBasis, bracketLL, Jmat, Lof, Wire.Rat.toRat] <;> norm_num

theorem forward_L12_F1 : Decode.forwardValue fixtureRawInput.rows 1 3 = bracketBasis .L12 .F1 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (1, 3)) =
      some { inputs := ["L12", "F1"],
             output := [{ basisId := "F1", coeff := [{ coeff := { num := -1, den := 2 }, exponents := [0, 0] }] }] } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod Decode.polyAsCoeff Decode.termAsCoeff
  funext k; fin_cases k <;> simp [basisOfName, bracketBasis, bracketLF, Jmat, Fof, Wire.Rat.toRat] <;> norm_num

theorem forward_L12_F2 : Decode.forwardValue fixtureRawInput.rows 1 4 = bracketBasis .L12 .F2 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (1, 4)) =
      some { inputs := ["L12", "F2"],
             output := [{ basisId := "F2", coeff := [{ coeff := { num := 1, den := 2 }, exponents := [0, 0] }] }] } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod Decode.polyAsCoeff Decode.termAsCoeff
  funext k; fin_cases k <;> simp [basisOfName, bracketBasis, bracketLF, Jmat, Fof, Wire.Rat.toRat] <;> norm_num

theorem forward_L22_L22 : Decode.forwardValue fixtureRawInput.rows 2 2 = bracketBasis .L22 .L22 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (2, 2)) =
      some { inputs := ["L22", "L22"], output := ([] : Wire.WVector) } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod
  funext k; fin_cases k <;> simp [bracketBasis, bracketLL, Jmat] <;> norm_num

theorem forward_L22_F1 : Decode.forwardValue fixtureRawInput.rows 2 3 = bracketBasis .L22 .F1 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (2, 3)) =
      some { inputs := ["L22", "F1"],
             output := [{ basisId := "F2", coeff := [{ coeff := { num := -1, den := 1 }, exponents := [0, 0] }] }] } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod Decode.polyAsCoeff Decode.termAsCoeff
  funext k; fin_cases k <;> simp [basisOfName, bracketBasis, bracketLF, Jmat, Fof, Wire.Rat.toRat] <;> norm_num

theorem forward_L22_F2 : Decode.forwardValue fixtureRawInput.rows 2 4 = bracketBasis .L22 .F2 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (2, 4)) =
      some { inputs := ["L22", "F2"], output := ([] : Wire.WVector) } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod
  funext k; fin_cases k <;> simp [bracketBasis, bracketLF, Jmat] <;> norm_num

theorem forward_F1_F1 : Decode.forwardValue fixtureRawInput.rows 3 3 = bracketBasis .F1 .F1 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (3, 3)) =
      some { inputs := ["F1", "F1"],
             output := [{ basisId := "L11", coeff := [{ coeff := { num := 1, den := 2 }, exponents := [0, 0] }] }] } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod Decode.polyAsCoeff Decode.termAsCoeff
  funext k; fin_cases k <;> simp [basisOfName, bracketBasis, bracketFF, Lof, Wire.Rat.toRat] <;> norm_num

theorem forward_F1_F2 : Decode.forwardValue fixtureRawInput.rows 3 4 = bracketBasis .F1 .F2 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (3, 4)) =
      some { inputs := ["F1", "F2"],
             output := [{ basisId := "L12", coeff := [{ coeff := { num := 1, den := 2 }, exponents := [0, 0] }] }] } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod Decode.polyAsCoeff Decode.termAsCoeff
  funext k; fin_cases k <;> simp [basisOfName, bracketBasis, bracketFF, Lof, Wire.Rat.toRat] <;> norm_num

theorem forward_F2_F2 : Decode.forwardValue fixtureRawInput.rows 4 4 = bracketBasis .F2 .F2 := by
  have h : fixtureRawInput.rows.find? (fun r => r.pairIdx n1BasisIds = some (4, 4)) =
      some { inputs := ["F2", "F2"],
             output := [{ basisId := "L22", coeff := [{ coeff := { num := 1, den := 2 }, exponents := [0, 0] }] }] } := rfl
  unfold Decode.forwardValue; rw [h]; unfold Decode.rowOutputAsMod Decode.polyAsCoeff Decode.termAsCoeff
  funext k; fin_cases k <;> simp [basisOfName, bracketBasis, bracketFF, Lof, Wire.Rat.toRat] <;> norm_num

/-- The reverse-pair value, obtained from the already-proved
`bracket_super_skew` (T2) rather than a sixteenth serialized row. -/
theorem gsign_reverse (i j : Basis5) :
    -(gsign (Basis5.parity i) (Basis5.parity j)) • bracketBasis j i = bracketBasis i j := by
  rw [← bracket_e_e i j, ← bracket_e_e j i]
  exact (bracket_super_skew i j).symm

set_option maxHeartbeats 4000000 in
/-- `T6` on basis pairs: the decoded bracket table, built from the wire
rows alone (via `Decode.forwardValue`) plus super-skew for the reverse
direction, agrees with the independent native `bracketBasis` on ALL 25
ordered `n=1` basis pairs. -/
theorem decodedBracketBasis_eq (i j : Basis5) :
    (if Indexed.basis5Rank i ≤ Indexed.basis5Rank j then
        Decode.forwardValue fixtureRawInput.rows (Indexed.basis5Rank i) (Indexed.basis5Rank j)
      else
        -(gsign (Basis5.parity i) (Basis5.parity j)) •
          Decode.forwardValue fixtureRawInput.rows (Indexed.basis5Rank j) (Indexed.basis5Rank i))
      = bracketBasis i j := by
  cases i <;> cases j <;> simp only [Indexed.basis5Rank] <;>
    first
    | (rw [if_pos (by decide)]; exact forward_L11_L11)
    | (rw [if_pos (by decide)]; exact forward_L11_L12)
    | (rw [if_pos (by decide)]; exact forward_L11_L22)
    | (rw [if_pos (by decide)]; exact forward_L11_F1)
    | (rw [if_pos (by decide)]; exact forward_L11_F2)
    | (rw [if_pos (by decide)]; exact forward_L12_L12)
    | (rw [if_pos (by decide)]; exact forward_L12_L22)
    | (rw [if_pos (by decide)]; exact forward_L12_F1)
    | (rw [if_pos (by decide)]; exact forward_L12_F2)
    | (rw [if_pos (by decide)]; exact forward_L22_L22)
    | (rw [if_pos (by decide)]; exact forward_L22_F1)
    | (rw [if_pos (by decide)]; exact forward_L22_F2)
    | (rw [if_pos (by decide)]; exact forward_F1_F1)
    | (rw [if_pos (by decide)]; exact forward_F1_F2)
    | (rw [if_pos (by decide)]; exact forward_F2_F2)
    | (rw [if_neg (by decide), forward_L11_L12]; exact gsign_reverse _ _)
    | (rw [if_neg (by decide), forward_L11_L22]; exact gsign_reverse _ _)
    | (rw [if_neg (by decide), forward_L12_L22]; exact gsign_reverse _ _)
    | (rw [if_neg (by decide), forward_L11_F1]; exact gsign_reverse _ _)
    | (rw [if_neg (by decide), forward_L12_F1]; exact gsign_reverse _ _)
    | (rw [if_neg (by decide), forward_L22_F1]; exact gsign_reverse _ _)
    | (rw [if_neg (by decide), forward_L11_F2]; exact gsign_reverse _ _)
    | (rw [if_neg (by decide), forward_L12_F2]; exact gsign_reverse _ _)
    | (rw [if_neg (by decide), forward_L22_F2]; exact gsign_reverse _ _)
    | (rw [if_neg (by decide), forward_F1_F2]; exact gsign_reverse _ _)

/-- `T6`: the decoded bracket table as a plain function, built from the
wire rows alone -- the native `bracketBasis`/`Native.lean` formulas are
never consulted in this DEFINITION, only in the separate equality theorem
above. -/
noncomputable def decodedBracketBasis (i j : Basis5) : Mod :=
  if Indexed.basis5Rank i ≤ Indexed.basis5Rank j then
    Decode.forwardValue fixtureRawInput.rows (Indexed.basis5Rank i) (Indexed.basis5Rank j)
  else
    -(gsign (Basis5.parity i) (Basis5.parity j)) •
      Decode.forwardValue fixtureRawInput.rows (Indexed.basis5Rank j) (Indexed.basis5Rank i)

theorem decodedBracketBasis_eq' (i j : Basis5) : decodedBracketBasis i j = bracketBasis i j :=
  decodedBracketBasis_eq i j

/-- `T6`, extended to the whole module: literally the same finite-sum
bilinear-extension SHAPE as `bracket` itself (`Native.lean`'s `T0`), so
the extension is immediate from the basis-pair equality by
`Finset.sum_congr` -- no separate bilinearity argument is restated. -/
noncomputable def decodedBracket (x y : Mod) : Mod :=
  ∑ i : Basis5, ∑ j : Basis5, (x i * y j) • decodedBracketBasis i j

theorem decodedBracket_eq_bracket (x y : Mod) : decodedBracket x y = bracket x y := by
  unfold decodedBracket bracket
  apply Finset.sum_congr rfl; intro i _
  apply Finset.sum_congr rfl; intro j _
  rw [decodedBracketBasis_eq']

/-! ## T7: positive coefficient examples and negative decoder examples -/

/-- Positive: `-1/2`, in reduced form with positive denominator. -/
def posRatExample : Wire.Rat := { num := -1, den := 2 }
theorem posRatExample_valid : posRatExample.validB = true := by decide
theorem posRatExample_value : posRatExample.toRat = -1 / 2 := by
  unfold Wire.Rat.toRat posRatExample; norm_num

/-- Positive: the variable-dependent polynomial `beta1 + beta2^2`, whose
canonical exponent order is `(0,2)` then `(1,0)`. -/
def posPolyExample : Wire.Poly :=
  [{ coeff := { num := 1, den := 1 }, exponents := [0, 2] },
   { coeff := { num := 1, den := 1 }, exponents := [1, 0] }]
theorem posPolyExample_valid : posPolyExample.validB = true := by decide
theorem posPolyExample_value : Decode.polyAsCoeff posPolyExample = beta2 ^ 2 + beta1 := by
  unfold Decode.polyAsCoeff Decode.termAsCoeff posPolyExample Wire.Rat.toRat
  simp <;> ring

/-- Negative: non-positive denominator. -/
def negBadDen : Wire.PolyTerm := { coeff := { num := 1, den := 0 }, exponents := [0, 0] }
theorem negBadDen_rejected : Decode.checkTerm 2 negBadDen = some .badDenominator := by decide

/-- Negative: unreduced `2/4`. -/
def negUnreduced : Wire.PolyTerm := { coeff := { num := 2, den := 4 }, exponents := [0, 0] }
theorem negUnreduced_rejected : Decode.checkTerm 2 negUnreduced = some .unreducedFraction := by decide

/-- Negative: duplicate exponent vector. -/
def negDupExp : Wire.Poly :=
  [{ coeff := { num := 1, den := 1 }, exponents := [0, 0] },
   { coeff := { num := 1, den := 1 }, exponents := [0, 0] }]
theorem negDupExp_rejected : Decode.firstPolyOrderDefect negDupExp = some .duplicateExponent := by decide

/-- Negative: unsorted exponent vectors. -/
def negUnsorted : Wire.Poly :=
  [{ coeff := { num := 1, den := 1 }, exponents := [1, 0] },
   { coeff := { num := 1, den := 1 }, exponents := [0, 0] }]
theorem negUnsorted_rejected : Decode.firstPolyOrderDefect negUnsorted = some .unsortedExponents := by decide

/-- Negative: wrong exponent-vector length. -/
def negWrongLen : Wire.PolyTerm := { coeff := { num := 1, den := 1 }, exponents := [0] }
theorem negWrongLen_rejected : Decode.checkTerm 2 negWrongLen = some .wrongExponentLength := by decide

/-- Negative: duplicate vector basis position. -/
def negDupVecBasis : Wire.WVector :=
  [{ basisId := "L11", coeff := [{ coeff := { num := 1, den := 1 }, exponents := [0, 0] }] },
   { basisId := "L11", coeff := [{ coeff := { num := 1, den := 1 }, exponents := [0, 0] }] }]
theorem negDupVecBasis_rejected :
    Decode.checkVector Decode.n1BasisIds 2 negDupVecBasis = some .duplicateVectorBasis := by decide

/-- Negative: unknown basis or role reference. -/
def negUnknownBasis : Wire.WVector :=
  [{ basisId := "L99", coeff := [{ coeff := { num := 1, den := 1 }, exponents := [0, 0] }] }]
theorem negUnknownBasis_rejected :
    Decode.checkVector Decode.n1BasisIds 2 negUnknownBasis = some .unknownBasisOrRole := by decide

/-- Negative: missing zero row (drop the `(L11,L11)` row entirely). -/
def negMissingRow : List Wire.Row := fixtureRawInput.rows.drop 1
theorem negMissingRow_rejected :
    Decode.checkRows Decode.n1BasisIds 2 negMissingRow = some .missingZeroRow := by decide

/-- Negative: duplicate row (the same row twice). -/
def negDuplicateRow : List Wire.Row :=
  (fixtureRawInput.rows.take 1) ++ (fixtureRawInput.rows.take 1) ++ (fixtureRawInput.rows.drop 1)
theorem negDuplicateRow_rejected :
    Decode.checkRows Decode.n1BasisIds 2 negDuplicateRow = some .duplicateRow := by decide

/-- Negative: reversed row (`(L12,L11)` instead of `(L11,L12)`). -/
def negReversedRow : Wire.Row := { inputs := ["L12", "L11"], output := [] }
theorem negReversedRow_rejected :
    Decode.checkRow Decode.n1BasisIds 2 negReversedRow = some .reversedRow := by decide

/-- Negative: `partial` coverage. -/
def negPartialInput : Decode.RawInput := { fixtureRawInput with coverage := "partial" }
theorem negPartialInput_rejected : Decode.decode negPartialInput = Except.error .partialCoverage := by decide

/-- Negative: `sparse-total` coverage. -/
def negSparseInput : Decode.RawInput := { fixtureRawInput with coverage := "sparse-total" }
theorem negSparseInput_rejected : Decode.decode negSparseInput = Except.error .sparseTotalCoverage := by decide

/-! Own adversarial additions, beyond Agent1's negative list. -/

/-- Own: variables declared in the wrong order (`beta2` before `beta1`). -/
def ownWrongVarOrderInput : Decode.RawInput := { fixtureRawInput with pVariables := ["beta2", "beta1"] }
theorem ownWrongVarOrderInput_rejected :
    Decode.decode ownWrongVarOrderInput = Except.error .ownWrongVariableOrder := by decide

/-- Own: a basis degree that disagrees with the role table (`L11` claimed
degree `1` instead of `0`). -/
def ownDegreeMismatchInput : Decode.RawInput := { fixtureRawInput with basisDegrees := [1, 0, 0, 1, 1] }
theorem ownDegreeMismatchInput_rejected :
    Decode.decode ownDegreeMismatchInput = Except.error .ownDegreeMismatch := by decide

end Bridge
end InhomogeneousDeformations
