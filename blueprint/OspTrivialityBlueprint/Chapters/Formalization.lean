import Verso
import VersoManual
import VersoBlueprint
import InhomogeneousDeformations

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "The formalized objects" =>

These nodes carry the paper's Appendix A items that are not a numbered statement of the body.

:::definition "item_P1" (lean := "InhomogeneousDeformations.Indexed.bracketN_isHomogN, InhomogeneousDeformations.Indexed.bracketN_super_skew_homog, InhomogeneousDeformations.Indexed.jacobiN_homog")
*Item P1 (the coordinate presentation is a Lie superalgebra of the stated shape).* The
bracket respects the $`\mathbb Z/2`-grading, is super-skew symmetric, and satisfies the
super-Jacobi identity, for arbitrary homogeneous module elements.
:::

:::definition "item_gR_checks" (lean := "InhomogeneousDeformations.Indexed.bracketR_iotaR_iotaR, InhomogeneousDeformations.Indexed.bracketR_super_skew_homog, InhomogeneousDeformations.Indexed.kappaMulR_sq, InhomogeneousDeformations.Indexed.kappaMulR_not_injective, InhomogeneousDeformations.Indexed.bracketRBeta_iotaR_iotaR")
*The presentation of $`\mathfrak g_R`.* $`\mathfrak g_R` is presented as
$`\mathfrak g_P \oplus \kappa \mathfrak g_P` on a doubled basis, not constructed. The
presentation is held accountable by four checks: it restricts to the undeformed bracket on
$`\mathfrak g_P`, it is super-skew, $`\kappa^2 = 0`, and multiplication by $`\kappa` is not
injective. The deformed bracket restricts likewise.
:::

:::theorem "item_N1" (lean := "InhomogeneousDeformations.Bridge.decodedBracket_eq_bracket")
*The $`n = 1` consistency check.* For one explicitly given datum at $`n = 1`, an external
description of the bracket table decodes to exactly the bracket defined internally. This is
a check on a single object, not a statement about a format or a program.
:::
