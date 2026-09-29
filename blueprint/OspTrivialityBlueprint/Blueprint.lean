import Verso
import VersoManual
import VersoBlueprint
import VersoBlueprint.Commands.Graph
import VersoBlueprint.Commands.Summary
import OspTrivialityBlueprint.Chapters.Sources
import OspTrivialityBlueprint.Chapters.Recovery
import OspTrivialityBlueprint.Chapters.Triviality
import OspTrivialityBlueprint.Chapters.Formalization

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "On the triviality of inhomogeneous deformations of osp(1|2n): a blueprint" =>

A blueprint of arXiv:2604.05252v2 against this repository's formalization. It is reference
material on the branch `lean-v4.34` (Lean v4.34.1): the paper cites the tag
`v2-lean-formalization`, and `main` is `v2.1-lint-clean` at Lean v4.29.1. The statements
are paraphrased from the paper, and each gives the paper's number; the headings (such as
"Proposition 2.1") are the blueprint's own numbering. The paper is authoritative. Each node
names the declarations that the paper's Appendix A assigns to it, and its status is computed
from them. The `uses` links are those of the paper's proofs.

{include 0 OspTrivialityBlueprint.Chapters.Sources}
{include 0 OspTrivialityBlueprint.Chapters.Recovery}
{include 0 OspTrivialityBlueprint.Chapters.Triviality}
{include 0 OspTrivialityBlueprint.Chapters.Formalization}

{blueprint_graph}
{blueprint_summary}
