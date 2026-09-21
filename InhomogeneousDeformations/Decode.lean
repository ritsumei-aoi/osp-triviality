import InhomogeneousDeformations.Wire
import InhomogeneousDeformations.Carrier
import InhomogeneousDeformations.Basis
import InhomogeneousDeformations.N1Specialization

/-!
# T5 — the `n=1` structured-input decision procedure

`H1.2`: this file specializes `Wire.lean`'s general primitives to the one
`n=1` projection this pilot checks -- five basis positions, fifteen
canonical rows, `beta1`/`beta2`, the `s-model-n1` role table -- exactly the
material the accepted correction002 round's own H1.2 disposition
classifies as genuinely rank-specific. Nothing here is a public validator
(V table, "validation level" row): `decode`'s only intended consumer is
`Bridge.lean`'s T6 correspondence theorem.
-/

namespace InhomogeneousDeformations
namespace Decode

open Wire

/-- Distinct, stable rejection-reason identifiers, one per checked
invariant in `IMPLEMENTATION_REQUIREMENTS.md`'s V table "rejected input"
row, plus a small number of Agent2's own adversarial additions (the
`own*` cases), per the Assurance Protocol. -/
inductive Reason : Type
  | badDenominator
  | unreducedFraction
  | zeroCoefficientTerm
  | duplicateExponent
  | unsortedExponents
  | wrongExponentLength
  | duplicateVectorBasis
  | unknownBasisOrRole
  | missingZeroRow
  | duplicateRow
  | reversedRow
  | partialCoverage
  | sparseTotalCoverage
  | ownWrongVariableOrder
  | ownDegreeMismatch
  | ownWrongOperationShape
  deriving DecidableEq, Repr

/-- The declared `n=1` basis-id order, in exactly `basis5Rank`'s order
(`L11,L12,L22,F1,F2` at ranks `0,1,2,3,4`). Fixed here, not in `Wire.lean`,
because this specific list is genuinely `n=1`-only. -/
def n1BasisIds : List String := ["L11", "L12", "L22", "F1", "F2"]

def basisOfName : String → Option Basis5
  | "L11" => some .L11
  | "L12" => some .L12
  | "L22" => some .L22
  | "F1"  => some .F1
  | "F2"  => some .F2
  | _     => none

/-- The raw, possibly-invalid `n=1` structured input -- exactly the
declared-shape fields named in the V table's "accepted input" row, at the
raw (unchecked) level. Every field is a plain, unconstrained value so that
every listed violation is representable. -/
structure RawInput where
  schemaVersion : String
  qId : String
  pId : String
  pBase : String
  pVariables : List String
  moduleId : String
  moduleCoeffDomain : String
  moduleGrading : String
  basisIds : List String
  basisDegrees : List ℕ
  opId : String
  opKind : String
  opModule : String
  opDegree : ℕ
  opScalarBehavior : String
  defKind : String
  coverage : String
  rows : List Wire.Row
  smBracket : String
  smModule : String
  smNormalization : String
  smRank : ℕ
  smRoles : List (String × String)
  deriving Repr

/-- The first adjacent-pair defect in a `Wire.Poly`'s exponent lists, if
any: `duplicateExponent` for an equal adjacent pair, `unsortedExponents`
for an out-of-order (but unequal) adjacent pair. Distinguishing these
(rather than one generic "not sorted" reason) is what the Assurance
Protocol's "distinct, stable reason" requirement needs. -/
def firstPolyOrderDefect (p : Wire.Poly) : Option Reason :=
  let go : List (List ℕ) → Option Reason
    | [] => none
    | [_] => none
    | a :: b :: rest =>
        if a = b then some .duplicateExponent
        else if Wire.natListLtB a b then
          match rest with
          | [] => none
          | c :: cs => if b = c then some .duplicateExponent
                       else if Wire.natListLtB b c then none
                       else some .unsortedExponents
        else some .unsortedExponents
  -- single top-level pass over the mapped exponent list; recursion inlined
  -- via `List.foldl`-free direct pattern match kept small since `p` has at
  -- most a handful of terms in every case this pilot checks.
  match p.map Wire.PolyTerm.exponents with
  | [] => none
  | [_] => none
  | l => go l

/-- Check one `Wire.PolyTerm` against the `n=1` requirement (coefficient
valid and nonzero, exponent list of length exactly `expectedLen`). -/
def checkTerm (expectedLen : ℕ) (t : Wire.PolyTerm) : Option Reason :=
  if !(0 < t.coeff.den) then some .badDenominator
  else if !(Int.gcd t.coeff.num t.coeff.den = 1) then some .unreducedFraction
  else if t.coeff.num = 0 then some .zeroCoefficientTerm
  else if t.exponents.length ≠ expectedLen then some .wrongExponentLength
  else none

/-- Check one `Wire.Poly` (all terms individually, then the exponent
order/duplicate defect). -/
def checkPoly (expectedLen : ℕ) (p : Wire.Poly) : Option Reason :=
  (p.findSome? (checkTerm expectedLen)).orElse (fun _ => firstPolyOrderDefect p)

/-- Check one `Wire.WVector` against the declared basis list: unknown
basis id first, then duplicate id (adjacent-equal declared index), then
each entry's own polynomial. -/
def checkVector (declared : List String) (expectedLen : ℕ) (v : Wire.WVector) : Option Reason :=
  if h : v.any (fun e => !Wire.knownIn declared e.basisId) then some .unknownBasisOrRole
  else
    let idxs := v.map (fun e => Wire.indexIn declared e.basisId)
    let rec dup : List ℕ → Bool
      | [] => false
      | [_] => false
      | a :: b :: rest => a = b || dup (b :: rest)
    if dup idxs then some .duplicateVectorBasis
    else (v.findSome? (fun e => checkPoly expectedLen e.coeff))

/-- Check one `Wire.Row`: the shape (`inputs` a two-element list resolving
to the given `declared` order, forward `p ≤ q`) then its output vector. A
row given with `inputs` reversed relative to the declared order produces
`p > q`, distinguished here from a genuinely unknown/malformed pair. -/
def checkRow (declared : List String) (expectedLen : ℕ) (r : Wire.Row) : Option Reason :=
  match r.inputs with
  | [a, b] =>
      if !Wire.knownIn declared a || !Wire.knownIn declared b then some .unknownBasisOrRole
      else
        let pa := Wire.indexIn declared a
        let pb := Wire.indexIn declared b
        if pa > pb then some .reversedRow
        else checkVector declared expectedLen r.output
  | _ => some .unknownBasisOrRole

/-- Check the full row LIST against total canonical-pair coverage: every
row individually valid first, then no duplicate row (two rows resolving
to the same declared-index pair), then that every canonical pair is
present (a genuinely absent pair -- including an absent explicit-zero row
-- is what `missingZeroRow`/coverage failure below catches). -/
def checkRows (declared : List String) (expectedLen : ℕ) (rows : List Wire.Row) : Option Reason :=
  match rows.findSome? (checkRow declared expectedLen) with
  | some reason => some reason
  | none =>
      let pairs := rows.map (fun r => (Wire.indexIn declared r.inputs.head!, Wire.indexIn declared (r.inputs.getD 1 "")))
      let rec dup : List (ℕ × ℕ) → Bool
        | [] => false
        | [_] => false
        | a :: b :: rest => a = b || dup (b :: rest)
      if dup pairs then some .duplicateRow
      else if pairs = Wire.canonicalPairs declared.length then none
      else some .missingZeroRow

/-- `T5`: the full `n=1` decision procedure. On success, returns the
input's own (now-validated) row list -- the decoded canonical-pair table
`Bridge.lean` builds the decoded bracket from. -/
def decode (raw : RawInput) : Except Reason (List Wire.Row) := do
  if raw.pVariables ≠ ["beta1", "beta2"] then throw .ownWrongVariableOrder
  if raw.qId ≠ "Q" ∨ raw.pId ≠ "P" ∨ raw.pBase ≠ "Q" then throw .ownWrongVariableOrder
  if raw.moduleCoeffDomain ≠ "P" ∨ raw.moduleGrading ≠ "z2" then throw .ownWrongOperationShape
  if raw.basisIds ≠ n1BasisIds then throw .ownWrongOperationShape
  if raw.basisDegrees ≠ [0, 0, 0, 1, 1] then throw .ownDegreeMismatch
  if raw.opKind ≠ "lie-bracket" ∨ raw.opModule ≠ raw.moduleId ∨ raw.opDegree ≠ 0 ∨
      raw.opScalarBehavior ≠ "bilinear-even-scalars" ∨ raw.defKind ≠ "canonical-pair-table" then
    throw .ownWrongOperationShape
  if raw.coverage = "partial" then throw .partialCoverage
  if raw.coverage = "sparse-total" then throw .sparseTotalCoverage
  if raw.coverage ≠ "total" then throw .ownWrongOperationShape
  if raw.smRank ≠ 1 ∨ raw.smModule ≠ raw.moduleId ∨ raw.smBracket ≠ raw.opId then
    throw .ownWrongOperationShape
  match checkRows n1BasisIds 2 raw.rows with
  | some reason => throw reason
  | none => pure raw.rows

/-- Interpret one already-validated `Wire.PolyTerm` (exponents of length
2, `[e0, e1]`) as a `Coeff` value: `crat coeff * beta1 ^ e0 * beta2 ^ e1`. -/
noncomputable def termAsCoeff (t : Wire.PolyTerm) : Coeff :=
  crat t.coeff.toRat * beta1 ^ (t.exponents.getD 0 0) * beta2 ^ (t.exponents.getD 1 0)

noncomputable def polyAsCoeff (p : Wire.Poly) : Coeff := (p.map termAsCoeff).sum

/-- The `Mod` value a single row's output vector denotes: for each basis
element `b`, sum `polyAsCoeff` over every vector entry whose id resolves
to `b` (validity guarantees at most one). -/
noncomputable def rowOutputAsMod (r : Wire.Row) : Mod :=
  fun b => ((r.output.filter (fun e => basisOfName e.basisId = some b)).map (fun e => polyAsCoeff e.coeff)).sum

/-- The forward-direction value for the declared-order pair `(p, q)`
(`p ≤ q`), read off the validated row list. -/
noncomputable def forwardValue (rows : List Wire.Row) (p q : ℕ) : Mod :=
  match rows.find? (fun r => r.pairIdx n1BasisIds = some (p, q)) with
  | some r => rowOutputAsMod r
  | none => 0

end Decode
end InhomogeneousDeformations
