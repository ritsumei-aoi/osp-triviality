import Verso
import VersoManual
import VersoBlueprint
import InhomogeneousDeformations

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Recovery of the deformed bracket" =>

:::proposition "prop_recovery" (lean := "InhomogeneousDeformations.Source.iotaBetaR_injective, InhomogeneousDeformations.Source.prop_recovery, InhomogeneousDeformations.Source.coefficient_unique, InhomogeneousDeformations.Source.iotaBetaR_bracket_closed")
*Proposition 3.1 (Faithful recovery).* The map $`\iota_\beta` is injective with closed image.
The recovered bracket on coordinate elements $`X, Y \in \mathfrak g_P` is
$`[X,Y]_\beta = [X,Y]_0 + \kappa \Gamma_\beta(X,Y)`, where the odd $`P`-bilinear coefficient
$`\Gamma_\beta` is determined by the paper's equations (3.3)–(3.5).

This is item P4 of the paper's Appendix A: proved for the deformed source as realized in the
development. Item P4 also records that the coefficient is unique as an element of $`\mathfrak g_P`,
which the paper shows in the proof.
:::

:::proof "prop_recovery"
Transport through {uses "lem_untwist"}[] and compute in $`A_B` with {uses "lem_base"}[].
:::

:::theorem "item_P5" (lean := "InhomogeneousDeformations.Source.GammaBetaBasis_Fof_Lof_forced_FL_free")
*Item P5 (the fourth sector is forced).* The concrete algebra admits exactly one coefficient
on the $`(F, L)` sector, namely $`-\tfrac12(\beta_u L_{vw} + \beta_v L_{uw})`, the value that
(3.4) and super-skew symmetry give.
:::
