# osp-triviality

Machine-checked formalization, in Lean 4, of the triviality of inhomogeneous deformations of the
oscillator Lie superalgebra $B(0,n)=\mathfrak{osp}(1|2n)$.

## What is established

For the symmetrized family studied in the accompanying paper, and **for every $n\geq1$** with no
restriction on the rank:

- the coordinate presentation is a Lie superalgebra of the stated shape — the $\mathbb Z/2$
  grading, super-skew symmetry, and the super-Jacobi identity;
- the deformation coefficient is a coboundary, $\Gamma_\beta=\delta f_\beta$;
- the change of generators $T_\beta=\mathrm{id}+\kappa(f_\beta)_R$ and its inverse are even,
  $R$-linear and **exactly** mutually inverse, and intertwine the two brackets;
- the source recovery holds: the map $\iota_\beta$ is injective with closed image, the recovered
  bracket is $\lbrack X,Y\rbrack_0+\kappa\Gamma_\beta(X,Y)$, and that coefficient is unique as an element of
  $\mathfrak g_P$.

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
  $f_\beta=h-\mathrm{ad}(2F(v))$, are outside for the same reason.
- **The development does not follow the paper's proof line by line.** It does not construct the
  isomorphism $\Phi$; both algebras are realized in one ambient algebra, so the change of lifts
  is a computation there. The conclusions agree; the route does not.
- **One sector of $\Gamma_\beta$ is forced, but remains a definition.** The concrete algebra
  admits exactly one coefficient on the $(F,L)$ sector; the declaration equating the definition
  with that value proves it by unfolding the definition, so it records the choice rather than
  deriving it.
- The decoding check described below concerns **one data instance** at rank 1.
- Nothing is established about other models, other families, or the full $H^2$.

## Contents

```
InhomogeneousDeformations/        the Lean 4 development
InhomogeneousDeformations.lean    the root module
fixture/s_model_n1.json           the data instance (see below)
tools/extract_fixture.py          regenerates the fixture's Lean encoding
tools/extract.lean                the dependency-graph extractor (see below)
docs/schema/algebra-data-1.md     specification of the data format
docs/what-is-proved.md            what the audit and the dependency graph do and do not show
aoi2026_triviality_osp1_2n.pdf    the paper
```

This repository's state and the paper are the same: the PDF above is the version the paper's own
appendix describes, and the tag [`v2-lean-formalization`](../../releases/tag/v2-lean-formalization)
is that same state — `main` may move past it, but the tag will not.

## Building and checking

Everything below is pinned: the toolchain in [`lean-toolchain`](lean-toolchain), every dependency
revision in [`lake-manifest.json`](lake-manifest.json). Nothing needs to be chosen.

**Prerequisites.** `git`, and [`elan`](https://github.com/leanprover/elan). You do not need to
install Lean yourself — `elan` reads `lean-toolchain` and fetches `leanprover/lean4:v4.29.1` the
first time you run a Lean or Lake command here.

```bash
git clone --branch v2-lean-formalization \
    https://github.com/ritsumei-aoi/osp-triviality.git
cd osp-triviality

lake exe cache get        # fetch mathlib's prebuilt .olean files — do this first
lake build InhomogeneousDeformations
```

**Do not run `lake update`.** It would move the dependency revisions away from the ones pinned in
`lake-manifest.json`, and the build would no longer be the one this repository and the paper
describe.

**Do not skip `lake exe cache get`.** Skipping it does not fail — it silently falls back to
*compiling mathlib from source*, which is hours of CPU on a laptop where the cache step is
minutes. Measured with mathlib's archives already in the local cache store: `lake exe cache get`
took 1 min 35 s to decompress them; a machine fetching that revision for the first time also
downloads roughly 400 MB before that step. The project's own `lake build` afterward took 2 min
25 s for its own files and did not recompile anything upstream. A second, fully cached
`lake build` took about a second and printed the identical report.

Approximate sizes, measured on a clean run:

| | |
|---|---|
| the clone | ~2.7 MB |
| mathlib and the other dependencies, cached (in the working copy, `.lake/`) | ~7.0 GB |
| this project's own build (in `.lake/`) | ~56 MB |
| the Lean toolchain (`elan`, one-time, under `~/.elan`, shared across projects) | ~2.5 GB |

The `.lake/` total lands in the working copy (`.gitignore` excludes it); the toolchain lives
elsewhere on your machine and is not part of this repository at all.

### Reading the result

The build prints the axiom dependencies of every audited declaration —
`InhomogeneousDeformations/AxiomAudit.lean` is imported by the root module, so this runs as part
of the ordinary build; no separate command is needed. To check the figures quoted above rather
than take them on trust:

```bash
lake build InhomogeneousDeformations 2>&1 | tee audit.txt

grep -c 'sorryAx'                           audit.txt   # 0   — nothing is assumed
grep -cE 'depends on axioms|does not depend on any axioms' \
                                             audit.txt   # 649 — declarations audited
grep -c 'depends on axioms'                 audit.txt   # 634 — depend on axioms
grep -c 'does not depend on any axioms'     audit.txt   # 15  — depend on none
grep -c '^error'                            audit.txt   # 0
```

And the axiom profiles — which axioms, not just how many. Lean wraps long lines, so join them
first:

```bash
tr '\n' ' ' < audit.txt | grep -o 'depends on axioms: \[[^]]*\]' \
  | tr -d ' ' | sort | uniq -c | sort -rn
#  603 dependsonaxioms:[propext,Classical.choice,Quot.sound]
#   16 dependsonaxioms:[propext,Quot.sound]
#   15 dependsonaxioms:[propext]
```

These three are the standard axioms of Lean's own logic; nothing else appears anywhere in the
audit. A build with nothing changed replays the same report from cache in about a second, so this
check can be repeated at any time without forcing a rebuild.

Warnings from mathlib's own style linters (unused `simp` arguments, line-length, and similar) may
appear during the build. They are not errors and do not affect the audit;
`grep -c '^error' audit.txt` is the number that should be `0`.

See [`docs/what-is-proved.md`](docs/what-is-proved.md) for what the audit does and does not tell
you, and how Appendix A.3's declaration table binds it to the paper's claims.

### The dependency graph

`tools/extract.lean` (1999 bytes) walks the compiled proof terms and writes `nodes.tsv` and
`edges.tsv` — every declaration in the development, and every real dependency between them, with
compiler-generated auxiliaries passed through rather than counted. It sits outside
`InhomogeneousDeformations/` so an ordinary `lake build` does not elaborate it.

Run it deliberately, **from the repository root**, after building:

```bash
lake env lean --run tools/extract.lean
# nodes 1081  edges 11450
```

`nodes.tsv` and `edges.tsv` are written into the directory you ran the command from, not next to
the script. `lean --run` also prints a harmless `(interpreter) unknown declaration 'main'`
trailer afterward — the file has no `def main`, only a top-level `#eval`; this is not an error.
`docs/what-is-proved.md` §3 explains what the graph is used to check.

## The data instance, and what is proved about it

`fixture/s_model_n1.json` is an instance of the `algebra-data/1` format, specified in
[`docs/schema/algebra-data-1.md`](docs/schema/algebra-data-1.md).

It is not merely described by that specification: it is **decoded inside Lean and proved to agree
with the algebra defined independently there** (`Bridge.decodedBracket_eq_bracket`). The guarantee
is exactly scoped — it concerns this one instance at rank 1, and is not a statement about the
format in general or about any parser.

## Relation to the earlier version of this repository

An earlier version of this repository provided Python code that verified triviality **for small
$n$ numerically**, by solving the cohomology equation by least squares and checking a residual
against a tolerance. That work is superseded in strength rather than corrected: what it checked
for small $n$ is now proved for **every** $n$, exactly and machine-checked.

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
