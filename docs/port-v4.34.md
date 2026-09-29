# Port notes: Lean v4.34.1

This branch, `lean-v4.34`, is a **reference port** of the formalization to Lean v4.34.1 and
mathlib `v4.34.1`. The paper (arXiv:2604.05252v2) cites the tag `v2-lean-formalization`; `main` is
`v2.1-lint-clean` at Lean v4.29.1 (mathlib `5e932f97`). Neither moved.

## What is the same

Measured from a fresh clone of this branch (`lake exe cache get`, then `lake build`):

- **0 errors and 0 warnings** in `InhomogeneousDeformations/`, under the same Mathlib standard linter
  set as `v2.1-lint-clean`.
- **The audit covers the same 649 declarations**, 0 depending on `sorryAx`. Their axiom sets are
  those of `v2.1-lint-clean`, declaration by declaration, with one exception (below).
- **The same 884 declarations**, none added or removed. Their statements and definitions are those
  of `v2.1-lint-clean`, with the exceptions below. Every one is either certified in Lean or confined
  to a proof.
- `tools/extract_fixture.py` regenerates `FixtureData.lean` byte for byte.

## What changed, and why

**1. Seven statements, certified.** mathlib `631b214f7c` (#41427) deleted `MvPolynomial.coeff m p`,
with no deprecated alias. The coefficient is now `p.coeff m` (`AddMonoidAlgebra.coeff`), with the
arguments in the other order. Seven statements of `SourceQuadraticIndependence.lean` use it:
`coeff_smul_one_eq_zero`, `coeff_single_add_single_X_mul_X` and `hval0`–`hval4`.

[`tools/port/PortCertificate.lean`](../tools/port/PortCertificate.lean):
- re-defines the old `MvPolynomial.coeff m p` as the new `AddMonoidAlgebra.coeff p m`;
- re-states each of the seven statements of `v2.1-lint-clean` verbatim;
- closes each by the ported theorem.

That it compiles shows each ported statement is the old one, reading the old `coeff` as the new
projection. Run it after building:

```bash
tools/port/run_certificate.sh
# good: exit 0; errors 0
# axiom entries: 7 of 7; non-standard: 0
# known-bad: exit 1 (must be nonzero); ... error: Type mismatch
# CERTIFICATE PASS
```

The known-bad changes one constant of `hval1` and must fail to compile.

**2. Four definitions, changed only inside proofs** ([`tools/port/proof_only.txt`](../tools/port/proof_only.txt)):
`A0Grading_setLike`, `ABGrading_setLike`, `WGradedAlgebra` and `RRingToTriv`. Each has the same type.
The changed lines lie in proof positions:
- `TensorProduct.induction_on` → `TensorProduct.inductionOn` in two membership proofs;
- a trailing `rfl` in a `(by …)`;
- the proof component of the pair in `RRingToTriv`, whose data `TrivSqZeroExt.inrHom` is unchanged.

**3. One axiom set.** `Indexed.sum_bool_eq` (`∑ b : Bool, f b = f true + f false`) now depends on
`propext`, `Classical.choice` and `Quot.sound`; it depended on `propext` and `Quot.sound`.
- The cause is upstream. At mathlib `v4.34.1` the instance `Bool.fintype`
  (`Mathlib/Data/Fintype/Defs.lean`) is built as `⟨⟨{true, false}, by simp⟩, …⟩` and itself depends
  on `Classical.choice`.
- The statement uses that instance, so no proof avoids it. The statement is unchanged.
- The audit's profile becomes 604 / 15 / 15 / 15 (standard three / `propext, Quot.sound` / `propext`
  / none), from 603 / 16 / 15 / 15.

**4. The dependency graph.** `tools/extract.lean` gives `nodes 1079  edges 11440`, against
`1081 / 11450`.
- The two nodes gone are `Basis5.toCtorIdx` and `Decode.Reason.toCtorIdx`, which Lean generated
  for enumerations and no longer does. No declaration of the development is missing.
- 11 edges are gone and 1 is new.
  - The 4 edges of those two nodes are gone.
  - So are the 7 edges from `RGradedAlgebraQ`, `RGradingQ`, `RRingRetract`, `RRingRetractKappa`,
    `algebraMap_Jn`, `algebraMap_cratN` and `kappa_mul_algebraMap_mem_RGradingQ_one` to the instance
    `Source.RRing_isScalarTower`, which instance search no longer selects there.
  - `Indexed.rho_Jn → crat` is new.

**5. Settings.**
- `lakefile.toml` sets `weak.linter.style.header = false`, with a comment. mathlib's header
  linter is new since v4.29.1 and expects mathlib's Apache header, while this repository is
  MIT-licensed.
- `IndexedU.lean` and `IndexedCoboundary.lean` set
  `backward.isDefEq.respectTransparency.types false`, each with a comment. v4.34 changed how
  `isDefEq` unfolds types, and `IndexedBasis` stays a `def`. The other files were fixed in their
  proofs.
- `maxHeartbeats` is not raised anywhere.

**6. Renames.** No mathlib identifier in a statement was merely renamed. Renames inside proofs
(for example `induction_on` → `inductionOn`) are proof changes and are not listed.

## The blueprint

[`blueprint/`](../blueprint/) is a separate Lake package, a
[verso-blueprint](https://github.com/leanprover/verso-blueprint) (`v4.34.0`) of arXiv:2604.05252v2.
It requires this formalization by path and changes nothing in it.
- Each node names the declarations that the paper's Appendix A.3 assigns to it. Its status is
  computed from them.
- The rendered site is committed in [`docs/blueprint/`](blueprint/) and published with GitHub
  Pages.
- How to rebuild and check it is in the README.
