# osp-triviality

Machine-checked formalization, in Lean 4, of the triviality of inhomogeneous deformations of the
oscillator Lie superalgebra \(B(0,n)=\mathfrak{osp}(1|2n)\).

## What is established

For the symmetrized family studied in the accompanying paper, and **for every \(n\geq1\)** with no
restriction on the rank:

- the coordinate presentation is a Lie superalgebra of the stated shape — the \(\mathbb Z/2\)
  grading, super-skew symmetry, and the super-Jacobi identity;
- the deformation coefficient is a coboundary, \(\Gamma_\beta=\delta f_\beta\);
- the change of generators \(T_\beta=\mathrm{id}+\kappa(f_\beta)_R\) and its inverse are even,
  \(R\)-linear and **exactly** mutually inverse, and intertwine the two brackets;
- the source recovery holds: the map \(\iota_\beta\) is injective with closed image, the recovered
  bracket is \([X,Y]_0+\kappa\Gamma_\beta(X,Y)\), and that coefficient is unique as an element of
  \(\mathfrak g_P\).

Together these are the paper's main theorem **for the coefficient recovered from the oscillator
source**, not merely for the coefficient written down in coordinates.

**Zero declarations depend on `sorryAx`.** The build audits 649 declarations: 634 depend on axioms
and 15 on none, every one on at most `propext`, `Classical.choice` and `Quot.sound` — the standard
axioms of Lean's logic. No additional axiom is introduced, and no `native_decide` or other
compiler-trusted evaluation is used.

## What is *not* established

Stated so that nothing is inferred beyond what is checked.

- **The source isomorphism as an abstract statement is not formalized.** The deformed source is
  *realized concretely* — generators are exhibited inside an explicit algebra and proved to satisfy
  the defining relations — rather than the algebra presented by those relations being constructed
  and identified with it. The corollary about the complex, and the identity
  \(f_\beta=h-\mathrm{ad}(2F(v))\), are outside for the same reason.
- **The development does not follow the paper's proof line by line.** It does not construct the
  isomorphism \(\Phi\); both algebras are realized in one ambient algebra, so the change of lifts
  is a computation there. The conclusions agree; the route does not.
- **One sector of \(\Gamma_\beta\) is forced, but remains a definition.** The concrete algebra
  admits exactly one coefficient on the \((F,L)\) sector; the declaration equating the definition
  with that value proves it by unfolding the definition, so it records the choice rather than
  deriving it.
- The decoding check described below concerns **one data instance** at rank 1.
- Nothing is established about other models, other families, or the full \(H^2\).

## Contents

```
InhomogeneousDeformations/        the Lean 4 development
InhomogeneousDeformations.lean    the root module
fixture/s_model_n1.json           the data instance (see below)
tools/extract_fixture.py          regenerates the fixture's Lean encoding
docs/schema/algebra-data-1.md     specification of the data format
aoi2026_triviality_osp1_2n.pdf    the paper
```

The PDF is the version currently on arXiv. It is replaced, together with its machine-checked
appendix, when the arXiv replacement is posted; until then this repository is **ahead of** the
published paper, and the section below says in what respect.

## Building and checking

Requires the pinned toolchain; both are fixed in the project configuration.

```bash
lake build InhomogeneousDeformations
```

The build prints the axiom dependencies of every audited declaration, so the figures quoted above
can be reproduced rather than taken on trust. A reader who wants only the headline can grep the
output for `sorryAx` and find nothing.

- Lean: `leanprover/lean4:v4.29.1`
- mathlib: `5e932f97dd25535344f80f9dd8da3aab83df0fe6`

## The data instance, and what is proved about it

`fixture/s_model_n1.json` is an instance of the `algebra-data/1` format, specified in
[`docs/schema/algebra-data-1.md`](docs/schema/algebra-data-1.md).

It is not merely described by that specification: it is **decoded inside Lean and proved to agree
with the algebra defined independently there** (`Bridge.decodedBracket_eq_bracket`). The guarantee
is exactly scoped — it concerns this one instance at rank 1, and is not a statement about the
format in general or about any parser.

## Relation to the earlier version of this repository

An earlier version of this repository provided Python code that verified triviality **for small
\(n\) numerically**, by solving the cohomology equation by least squares and checking a residual
against a tolerance. That work is superseded in strength rather than corrected: what it checked
for small \(n\) is now proved for **every** \(n\), exactly and machine-checked.

The previous state remains available at the tag
[`v1-python-arXiv-2604.05252`](../../releases/tag/v1-python-arXiv-2604.05252), so that references
from the earlier version of the paper continue to resolve.

## References

- H. Aoi, *On the triviality of inhomogeneous deformations of osp(1|2n)*, arXiv:2604.05252
  [math.RT], 2026. \[[arXiv](https://arxiv.org/abs/2604.05252)\]
- H. Aoi, *A collaborative workflow for human-AI research in pure mathematics*, Preprint, 2026.
  \[[PDF](https://github.com/ritsumei-aoi/ai-research-workflow-template/blob/main/aoi2026_collaborative_workflow.pdf)\]
- Workflow template: [ai-research-workflow-template](https://github.com/ritsumei-aoi/ai-research-workflow-template)

## License

See [LICENSE](LICENSE).
