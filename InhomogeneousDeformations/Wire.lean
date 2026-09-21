import Mathlib.Data.Int.GCD
import Mathlib.Data.Rat.Defs

/-!
# T5-T7 wire types — rank-independent structured-input primitives

`H1.2`: these types and their validity predicates are stated at the
greatest generality that costs no extra proof obligation. Nothing here
hard-codes a variable count, a basis size, or a row count; those are
genuinely `n=1`-specific and belong in `Decode.lean`, which specializes
these general primitives.

Every raw type here deliberately ADMITS violating values (a non-positive
denominator, an unsorted exponent list, an unknown basis reference, ...):
if the type could not represent a bad state, rejecting it would be a
vacuous check on an uninhabited case, not a real one (`IMPLEMENTATION_REQUIREMENTS.md`
§T5: "The types must admit the violating values, or the checks are
vacuous."). Validity is always a separate `Bool`-valued function over the
raw type, never baked into the type itself, and every function here
reduces by the kernel (`decide`); `native_decide` is never used.
-/

namespace InhomogeneousDeformations
namespace Wire

/-- A raw wire rational: a signed integer numerator over a signed integer
denominator, with NO positivity or reduced-form constraint in the type
itself -- a non-positive or unreduced denominator is a real, representable
value, checked by `Rat.validB` below. -/
structure Rat where
  num : ℤ
  den : ℤ
  deriving DecidableEq, Repr

/-- Wire-rational validity: strictly positive denominator, and
`numerator`/`denominator` coprime (so `0` is exactly `0/1`, per the V
table's "accepted input" row). -/
def Rat.validB (r : Rat) : Bool :=
  decide (0 < r.den) && decide (Int.gcd r.num r.den = 1)

/-- Interpret a wire rational into `ℚ`. Total (division by a non-positive
`den` is `ℚ`'s own convention), but only meaningful where `Rat.validB` has
already been checked; no proof step relies on its value otherwise. -/
noncomputable def Rat.toRat (r : Rat) : ℚ := (r.num : ℚ) / (r.den : ℚ)

/-- A general strict lexicographic order on `List ℕ`, self-contained
(never relying on `List`'s own possibly-differently-conventioned order
instance): shorter is smaller when a strict prefix, otherwise the first
differing entry decides. This is exactly "natural-tuple lexicographic
order", general in the list's length. -/
def natListLtB : List ℕ → List ℕ → Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | a :: as, b :: bs => if a < b then true else if a = b then natListLtB as bs else false

/-- A general list is strictly increasing under a decidable `<` (here
always `natListLtB` or `Nat.blt`-style comparisons, but kept generic in the
element type). -/
def strictlyIncreasingB {α : Type} (ltB : α → α → Bool) : List α → Bool
  | [] => true
  | [_] => true
  | a :: b :: rest => ltB a b && strictlyIncreasingB ltB (b :: rest)

/-- A single monomial term: a wire-rational coefficient and a raw exponent
list of ARBITRARY length -- the `n=1` requirement of "length exactly 2" is
a validity condition checked in `Decode.lean`, not a type-level constraint
here (an exponent list of the wrong length is exactly one of the required
negative cases, and must be representable to be rejectable). -/
structure PolyTerm where
  coeff : Rat
  exponents : List ℕ
  deriving DecidableEq, Repr

/-- A raw polynomial: an arbitrary list of terms; `[]` denotes zero. -/
abbrev Poly := List PolyTerm

/-- A single term is valid: its coefficient is a valid, nonzero wire
rational (a zero-coefficient term is never itself absent from a raw list --
excluding it is exactly the "no term has a zero coefficient" check). -/
def PolyTerm.validB (t : PolyTerm) : Bool := t.coeff.validB && decide (t.coeff.num ≠ 0)

/-- Polynomial validity: every term valid, and the terms' exponent lists
strictly increasing in natural-tuple lexicographic order -- this
simultaneously forbids duplicate exponent vectors (a repeat cannot be
strictly greater than its predecessor) and enforces the required
canonical order, at whatever exponent-list length the terms actually
have (general in the number of variables). -/
def Poly.validB (p : Poly) : Bool :=
  p.all PolyTerm.validB && strictlyIncreasingB natListLtB (p.map PolyTerm.exponents)

/-- A raw output-vector entry: a basis-id reference (a general `String`
identifier, resolved against a caller-supplied declared basis list -- see
`WVector.validB`) with a raw polynomial coefficient. -/
structure VectorEntry where
  basisId : String
  coeff : Poly
  deriving DecidableEq, Repr

abbrev WVector := List VectorEntry

/-- `declared`'s index of `s`, or `declared.length` (an out-of-range
sentinel) if `s` does not occur -- used to state "strictly increasing
declared positions" and "no unknown id" against an arbitrary declared
basis list, general in that list's size and contents. -/
def indexIn (declared : List String) (s : String) : ℕ :=
  (declared.findIdx? (· = s)).getD declared.length

def knownIn (declared : List String) (s : String) : Bool :=
  (declared.findIdx? (· = s)).isSome

/-- Vector validity against a declared basis list `declared`: every entry's
`basisId` is a known member of `declared`, the entries are strictly
increasing in `declared`'s order (which also forbids a duplicate id), and
no entry carries a zero (`[]`) or otherwise-invalid coefficient. General in
`declared` -- it is never fixed to the five `n=1` positions here. -/
def WVector.validB (declared : List String) (v : WVector) : Bool :=
  v.all (fun e => knownIn declared e.basisId) &&
  strictlyIncreasingB (fun a b => decide (a < b)) (v.map (fun e => indexIn declared e.basisId)) &&
  v.all (fun e => !e.coeff.isEmpty && e.coeff.validB)

/-- A raw canonical-table row: the two (unordered on the wire, checked
below) basis-id inputs, and the resulting output vector. -/
structure Row where
  inputs : List String
  output : WVector
  deriving DecidableEq, Repr

/-- The declared-order index pair of a row's `inputs`, if it is exactly a
two-element list; `none` (never matched as a valid canonical pair) if it
is any other length -- this is exactly the "wrong argument-count shape"
defect made representable at the row level. -/
def Row.pairIdx (declared : List String) (r : Row) : Option (ℕ × ℕ) :=
  match r.inputs with
  | [a, b] => some (indexIn declared a, indexIn declared b)
  | _ => none

/-- All ordered index pairs `(p, q)` with `p ≤ q < len`, in strict
lexicographic order -- the canonical "total coverage" pair list for a
declared basis of size `len`, general in `len`. -/
def canonicalPairs (len : ℕ) : List (ℕ × ℕ) :=
  (List.range len).flatMap (fun p => (List.range len).filterMap (fun q => if p ≤ q then some (p, q) else none))

/-- A raw canonical-pair-table operation definition is valid against a
declared basis list exactly when its rows' declared-index pairs, read off
in the rows' own list order, are literally the canonical total-coverage
list -- this one equality simultaneously forces: every row's own inputs
are already in forward (`p ≤ q`) order (a "reversed row" produces a pair
with `p > q`, which never appears in `canonicalPairs`), no duplicate row,
no missing row, and exactly total coverage (a `coverage` field claiming
anything other than the literal string `"total"`, e.g. `"partial"` or
`"sparse-total"`, is checked separately by the caller, since coverage is a
label on the wire, not derived from the row list). -/
def RowsValid (declared : List String) (rows : List Row) : Bool :=
  (rows.all (fun r => (r.pairIdx declared).isSome && r.output.validB declared)) &&
  decide ((rows.map (fun r => (r.pairIdx declared).getD (0, 0))) = canonicalPairs declared.length)

end Wire
end InhomogeneousDeformations
