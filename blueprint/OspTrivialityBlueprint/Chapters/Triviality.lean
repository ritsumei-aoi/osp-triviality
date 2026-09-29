import Verso
import VersoManual
import VersoBlueprint
import InhomogeneousDeformations

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "An explicit odd coboundary" =>

:::theorem "thm_main" (lean := "InhomogeneousDeformations.Indexed.GammaBetaN_eq_deltaFN, InhomogeneousDeformations.Indexed.G4, InhomogeneousDeformations.Indexed.TBetaInv_TBeta, InhomogeneousDeformations.Indexed.TBeta_TBetaInv, InhomogeneousDeformations.Indexed.TBeta_isHomogR, InhomogeneousDeformations.Indexed.intertwining")
*Theorem 4.1 (Family triviality for every rank).* For every $`n \geq 1`, the coefficient of
{uses "prop_recovery"}[] satisfies $`\Gamma_\beta = \delta f_\beta`, with the explicit odd
primitive $`f_\beta` of the paper's (4.2). The maps
$`T_\beta = \mathrm{id} + \kappa (f_\beta)_R` and $`T_\beta^{-1} = \mathrm{id} - \kappa (f_\beta)_R`
are even $`R`-linear inverse maps, and $`[T_\beta X, T_\beta Y]_0 = T_\beta([X,Y]_\beta)` for
all $`X, Y \in \mathfrak g_R`.

Items P2 (the coboundary identity) and P3 (the change of generators) of the paper's
Appendix A together are this theorem.

The `uses` link to Proposition 3.1 is the paper's. In the formalization the two are joined by the
shared definition of $`\Gamma_\beta`, not by a proof dependency: the coboundary identity is proved
for the coordinate formulas directly, and item P4 recovers the same coefficient from the source
(Appendix A, "What is not formalized", item 2).
:::

:::corollary "cor_complex"
*Corollary 4.2 (Complex parameters).* For every $`\lambda \in \mathbb C^{2n}`, specialize
$`\beta_u \mapsto \lambda_u` and keep $`\kappa`. The specialized lifts are faithful, and the
paper's (4.2)–(4.4) define a trivialization over $`\Lambda_{\mathbb C}(\kappa)`.

Not formalized: the development is over $`\mathbb Q` (Appendix A, "What is not formalized",
item 1).
:::

:::proof "cor_complex"
Specialize {uses "thm_main"}[] and {uses "prop_recovery"}[]; specialization is a ring map.
:::
