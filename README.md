# osp-triviality

> **This branch is a reference port to Lean v4.34.1 (mathlib v4.34.1).** The paper cites the tag
> [`v2-lean-formalization`](../../releases/tag/v2-lean-formalization), and `main`
> (= [`v2.1-lint-clean`](../../releases/tag/v2.1-lint-clean)) stays at Lean v4.29.1. Here the
> statements and definitions are those of `main`, except for the mathlib changes listed in the
> port notes ([`docs/port-v4.34.md`](docs/port-v4.34.md)), and the audit covers the same 649
> declarations. Nothing on `main` or on the older tags has moved. This branch also carries a
> **blueprint** of the paper, published at <https://ritsumei-aoi.github.io/osp-triviality/blueprint/>.

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
tools/extract_fixture.py          regenerates the fixture's Lean encoding (byte for byte; `--report PATH`
                                  writes its field-by-field report, otherwise printed)
tools/extract.lean                the dependency-graph extractor (see below)
docs/schema/algebra-data-1.md     specification of the data format
docs/what-is-proved.md            what the audit and the dependency graph do and do not show
docs/port-v4.34.md                the port notes: what changed from v2.1-lint-clean, and why
tools/port/                       the port certificate and its lists (see the port notes)
blueprint/                        the blueprint, a separate Lake package, with check_a3.py and
                                  publish_site.py (see below)
docs/blueprint/                   the rendered blueprint, served by GitHub Pages
aoi2026_triviality_osp1_2n.pdf    the paper
```

This repository's state and the paper are the same: the PDF above is the version the paper's own
appendix describes, and the tag [`v2-lean-formalization`](../../releases/tag/v2-lean-formalization)
is that same state — `main` may move past it, but the tag will not.

**Two tags on `main`.** `v2-lean-formalization` is the state of arXiv v2, and it does not move.
[`v2.1-lint-clean`](../../releases/tag/v2.1-lint-clean) has **identical statements and an identical
audit** — the same 884 declarations with the same statements and definitions, and the same 649
audited declarations, each with the same axioms (634 depending only on the standard axioms, 15 depending on none) —
and the warnings are fixed: `lake build` finishes with 0 warnings under the Mathlib standard linter
set. Only proofs and formatting differ. Where a linter asked for a change of a statement or a
definition, that one declaration keeps its text and carries a `set_option linter.… false in` line
with a one-line reason.

**A third tag, on this branch.** `v2.2-lean4.34` is the port to Lean v4.34.1: the same 884
declarations and the same 649 audited, with the differences stated in the
[port notes](docs/port-v4.34.md) — seven statements that mathlib's removal of `MvPolynomial.coeff`
forced to change, each certified in Lean to be the old statement; four definitions changed only
inside proofs; and one declaration whose axiom set gains `Classical.choice` from mathlib.

## Building and checking

Everything below is pinned: the toolchain in [`lean-toolchain`](lean-toolchain), every dependency
revision in [`lake-manifest.json`](lake-manifest.json). Nothing needs to be chosen.

**Prerequisites.** `git`, and [`elan`](https://github.com/leanprover/elan). You do not need to
install Lean yourself — `elan` reads `lean-toolchain` and fetches `leanprover/lean4:v4.34.1` the
first time you run a Lean or Lake command here.

```bash
git clone --branch v2.2-lean4.34 \
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
minutes. On two machines with mathlib's archives already in the local cache store,
`lake exe cache get` followed by `lake build` took between about 3 and 5 minutes, and recompiled
nothing upstream; the time depends on the machine and on the disk. A machine fetching that revision
for the first time also downloads the archives (several hundred MB) first. A second, fully cached
`lake build` replays the identical report.

Approximate sizes, measured on a clean run:

| | |
|---|---|
| the clone (tracked files; `docs/blueprint/` is 5.4 MB of it) | ~6.9 MB |
| mathlib and the other dependencies, cached (in the working copy, `.lake/`) | ~7.6 GB |
| this project's own build (in `.lake/`) | ~68 MB |
| the blueprint package's own `blueprint/.lake/` (its own copy of the dependencies) | ~8.8 GB |
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
grep -c '^warning'                          audit.txt   # 0 on this branch and at v2.1-lint-clean (468 at v2-lean-formalization)
```

And the axiom profiles — which axioms, not just how many. Lean wraps long lines, so join them
first:

```bash
tr '\n' ' ' < audit.txt | grep -o 'depends on axioms: \[[^]]*\]' \
  | tr -d ' ' | sort | uniq -c | sort -rn
#  604 dependsonaxioms:[propext,Classical.choice,Quot.sound]
#   15 dependsonaxioms:[propext,Quot.sound]
#   15 dependsonaxioms:[propext]
```

These three are the standard axioms of Lean's own logic; nothing else appears anywhere in the
audit. At `v2.1-lint-clean` the profile is 603 / 16 / 15: one declaration, `Indexed.sum_bool_eq`,
gains `Classical.choice` here through mathlib's own `Bool.fintype` (the port notes, item 3). A build with nothing changed replays the same report from cache in about a second, so this
check can be repeated at any time without forcing a rebuild.

**Warnings.** The clone command above checks out `v2.2-lean4.34`, whose build prints **no
warnings**: `grep -c '^warning' audit.txt` is `0`, as at `v2.1-lint-clean` (Lean v4.29.1). `v2-lean-formalization`, the state of arXiv v2
(clone it with `--branch v2-lean-formalization`), prints 468 warnings from mathlib's own style linters
(unused `simp` arguments, line length, and similar) during the same build. They are not errors, and
the audit is identical at both tags; `grep -c '^error' audit.txt` is `0` at both.

See [`docs/what-is-proved.md`](docs/what-is-proved.md) for what the audit does and does not tell
you, and how Appendix A.3's declaration table binds it to the paper's claims.

### The dependency graph

`tools/extract.lean` walks the compiled proof terms and writes `nodes.tsv` and
`edges.tsv` — every declaration in the development, and every real dependency between them, with
compiler-generated auxiliaries passed through rather than counted. It sits outside
`InhomogeneousDeformations/` so an ordinary `lake build` does not elaborate it.

Run it deliberately, **from the repository root**, after building:

```bash
lake env lean --run tools/extract.lean
# nodes 1079  edges 11440   (1081 / 11450 at v2.1-lint-clean; the port notes, item 4)
```

`nodes.tsv` and `edges.tsv` are written into the directory you ran the command from, not next to
the script.
`docs/what-is-proved.md` §3 explains what the graph is used to check.

## The blueprint

[`blueprint/`](blueprint/) is a [verso-blueprint](https://github.com/leanprover/verso-blueprint) of
arXiv:2604.05252v2: the paper's statements, each linked to the declarations that the paper's
Appendix A.3 names, with the formalization status computed from them and a dependency graph. It is a
separate Lake package that requires this formalization by path; it changes nothing in
`InhomogeneousDeformations/`. The rendered site is committed in [`docs/blueprint/`](docs/blueprint/)
and served at <https://ritsumei-aoi.github.io/osp-triviality/blueprint/>. The statements are paraphrases; the paper is authoritative.

To rebuild it (after the build above; this fetches the blueprint's own copy of the dependencies):

```bash
cd blueprint
lake update && lake exe cache get
lake exe vbp build            # writes _out/site/html-multi/
```

Two checks, since `vbp build` exits 0 even when a declaration name does not resolve:

```bash
# 1. no warning from this package (the only warnings are inside VersoBlueprint itself)
lake exe vbp build 2>&1 | grep -E '^(warning|error)' | grep -v 'VersoBlueprint'   # nothing
# 2. the declaration lists equal the paper's Appendix A.3, node by node
cd .. && python3 blueprint/check_a3.py aoi2026_triviality_osp1_2n.tex .          # RESULT: ALL MATCH
```

The renderer records source locations as absolute paths of the machine that built it, so the
published copy is written by a script that makes them relative to the repository root and refuses to
leave any absolute local path. A fresh render then equals `docs/blueprint/` up to the build stamp
(the time, and the commit it was built from, shown on the front page) and the order of the
`<script>` blocks, which the renderer does not fix:

```bash
python3 blueprint/publish_site.py --check   # SAME (up to the build stamp and script order)
python3 blueprint/publish_site.py           # rewrites docs/blueprint/ from blueprint/_out/
```

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
