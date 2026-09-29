import Verso
import VersoManual
import VersoBlueprint

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Sources and normalized generators" =>

Section 2 of the paper. The source algebra $`A_\beta` is generated over $`R = P \oplus \kappa P`
by even $`b_u` and one odd $`a`, with $`[b_u, b_v] = J_{uv}`, $`a^2 = \tfrac12` and
$`[b_u, a] = \beta_u \kappa`; $`A_B` is the same algebra with $`\beta = 0`.

:::lemma_ "lem_untwist"
*Lemma 2.2 (Untwisting the source).* The maps
$`\Phi : A_\beta \to A_B`, $`b_u \mapsto B_u + \beta_u \kappa a`, and
$`\Psi : A_B \to A_\beta`, $`B_u \mapsto b_u - \beta_u \kappa a`, fixing $`a` and $`R`, are
mutually inverse even $`R`-algebra maps.

Not formalized as an abstract statement: the development realizes the source inside one
ambient algebra rather than constructing the algebra presented by the relations (the paper's
Appendix A, "What is not formalized", items 1 and 2).
:::

:::lemma_ "lem_base"
*Lemma 2.3 (The undeformed algebra).* The map $`\iota_0 : \mathfrak g \to A_0`,
$`L_{uv} \mapsto L^0_{uv}`, $`F_u \mapsto F^0_u`, is injective, with image closed under the
supercommutator, and the induced bracket $`[\cdot,\cdot]_0` is given by the paper's
equations (2.4)–(2.6).

The coordinate presentation is formalized as item P1 of the paper's Appendix A (in the last
chapter below); this lemma itself, as a statement about $`A_0`, is not a formal node.
:::
