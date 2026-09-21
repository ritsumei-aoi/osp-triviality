# `algebra-data/1` — a JSON format for finite algebra data

This document specifies `algebra-data/1`, the JSON format used by the data file in this
repository. It is written so that an instance can be produced or checked without reading the
Lean development, and so that the Lean development's own reader cannot silently accept something
this document forbids.

**What makes this format worth publishing is narrow and concrete**: the instance shipped here is
not merely *described* by this specification, it is **decoded inside a proof assistant and proved
to agree with the algebra defined independently there**. See §10.

## 1. Scope, and what version 1 is not

Version 1 covers **finite, explicitly tabulated** algebra data: a basis, exact coefficients, a
bracket given as a table, and optional linear maps. It is deliberately small.

It does **not** cover: indexed or generated families (a document describing one is recognised and
rejected as `unsupported`, never silently read as a finite table); relations or presentations;
expression languages of any kind. Those are absent from version 1 rather than optional in it.

## 2. Envelope

The root object has exactly these fields; unknown fields are rejected.

| field | value |
|---|---|
| `schema_version` | the literal `algebra-data/1` |
| `kind` | `finite-instance` |
| `model` | `{id, revision}`, both non-empty strings |
| `required_profiles` | distinct profile names (§3) |
| `coefficient_domains`, `modules`, `operations`, `maps`, `claims`, `provenance` | the collections below |

IDs are globally unique across domains, modules, operations, maps and claims. References are to
locally declared IDs only. There is **no `verified` input field**: an instance cannot assert its
own correctness.

A profile or version outside the implemented set must return `unsupported` — never success.

## 3. Profiles

Profiles name **capabilities**, not a particular algebra: `core/1`, `exact-coefficients/1`,
`free-module/1`, `z2-graded/1`, `lie-super/1`, and — for the model in §9 —
`s-model-bs17-normalized-n1/1`. `lie-super/1` requires the five preceding it.

This is what lets the format outlive the one algebra it currently carries: another family adds a
profile rather than changing the envelope.

## 4. Coefficient domains and values

Domains are discriminated by `kind`:

- **`rational`** — `id`, `kind`.
- **`polynomial`** — `id`, `kind`, `base` (a rational domain), `variables` (ordered, distinct; may
  be empty).
- **`square-zero`** — `id`, `kind`, `base` (a polynomial domain), `odd_symbol`.

**Rational value**: `{numerator, denominator}` as canonical decimal strings — numerator `0` or an
optionally negative value with no leading zero, denominator positive with no leading zero, always
in reduced form, zero exactly `0/1`. No floats, no `NaN`/`Infinity`, no numeric expressions, and
booleans are not integers.

**Polynomial value**: an array of `{coefficient, exponents}` with non-zero canonical rational
coefficients and an exponent vector of the domain's variable count. Terms are strictly increasing
lexicographically, with no duplicate exponent vectors and no zero terms. Zero is `[]`; a constant
uses the all-zero exponent vector.

**Square-zero value**: `{even, odd}`, both polynomials over the base, representing \(p+\kappa q\)
with

\[(p+\kappa q)(r+\kappa s)=pr+\kappa(ps+qr),\qquad \kappa^2=0,\]

the odd tag having degree one. This is a **codec for such scalars only**: it is not an
implementation of an ambient relation \(a\kappa=-\kappa a\), nor of a graded extension of a
bracket.

**A non-canonical value is rejected, not repaired.** An implementation may normalise its own
intermediate arithmetic, but never incoming data. Arithmetic is exact.

## 5. Modules and vectors

A module has `id`, `coefficient_domain`, `grading` (`ordinary` or `z2`), and an ordered `basis` of
records with `id`; under `z2` each also carries an integer `degree` in \(\{0,1\}\), and under
`ordinary` no degree field appears.

A vector is an array of `{basis, coefficient}` in strictly increasing declared basis order, with
no duplicate basis IDs and no zero coefficients. Coefficients use the module's own domain.

## 6. Operations: the bracket

An operation has `id`, `kind` = `lie-bracket`, `module`, `degree` = 0, `scalar_behavior` =
`bilinear-even-scalars`, and a `definition` of `kind` = `canonical-pair-table` with `coverage` and
`entries`. Each entry is `{inputs: [basisId, basisId], output: vector}`, the pair strictly
increasing by basis position with the first at most the second; duplicate or reversed entries are
invalid.

**Reverse rows are reconstructed only by the graded skew rule** — never by ordinary antisymmetry
on an odd pair.

`coverage` is:

- **`total`** — every canonical pair present, including explicit zero rows;
- **`sparse-total`** — an omitted row means zero;
- **`partial`** — an omitted row means *unknown*, and no global skew or Jacobi verdict may be
  reported. A checker returns `incomplete`, even when every supplied row is consistent.

Non-zero even diagonals and wrong output parity are invalid. Bracket execution requires purely
even scalars — a rational or polynomial domain.

## 7. Maps

A map has `id`, `source`, `target`, `degree` (0 or 1), `scalar_behavior` = `linear-even-scalars`,
and a `basis-table` definition with one entry per source basis ID **including zeros**. Source and
target share a domain. On a graded module, every non-zero output term has degree
\(p(\text{input})+\deg\); a zero output satisfies either.

This checks **typing and homogeneity only**. That a map is a cocycle, a morphism, or an inverse is
not asserted or checked here.

## 8. Claims and provenance

`claims` are descriptive: `{id, statement, evidence_refs}` with `statement` free text and
`evidence_refs` resolving to `provenance` IDs. `provenance` records are `{id, description}` with
optional `location`.

**Neither ever discharges a check.** Statements are not evaluated, and a claim is not a proof.
They record where data came from, which is worth recording and is not the same thing.

## 9. The S-model profile

`s-model-bs17-normalized-n1/1` adds a required root field `s_model`, the sole extension the core
permits: `rank` = 1, `normalization` = `bs17-LF`, `module`, `bracket`, and `roles` mapping
`{L11, L12, L22, F1, F2}` to distinct basis IDs. The module's basis order is exactly that role
order, with degrees \(0,0,0,1,1\), over a polynomial domain in exactly `beta1`, `beta2`; the
bracket is a total canonical table on that module.

The profile's checker verifies the table **against the index formulas**, not merely that it
satisfies Jacobi — so relabelling an abelian table cannot make it this model. Renamed basis IDs
are supported through the role map. A different rank or normalisation returns `unsupported`.

## 10. What the formalization proves about the instance

The data file `fixture/s_model_n1.json` is decoded inside Lean 4 and the result is compared with
the bracket defined independently in the development. The theorem
`InhomogeneousDeformations.Bridge.decodedBracket_eq_bracket` states that the decoded bracket and
the internally defined bracket **agree on all inputs**.

The scope of that guarantee, stated exactly: it concerns **this one instance**, at rank 1. It is
not a statement about the format in general, about a parser, or about any other file. What it does
establish is that for the instance published here, the external description and the verified
mathematics are the same object — which is the reason this format is published alongside a proof
development rather than on its own.

## 11. Conformance

An instance conforms when: the envelope is exactly §2; every value is canonical per §4; every
collection satisfies §5-§8; and every name in `required_profiles` is implemented. A conforming
reader **rejects** rather than repairs, and returns `unsupported` rather than success for anything
outside its implemented set.
