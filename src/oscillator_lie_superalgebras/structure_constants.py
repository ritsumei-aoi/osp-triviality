from __future__ import annotations

from typing import Dict, Mapping, Tuple


Number = complex


class LieSuperalgebra:
    def __init__(
        self,
        generators: Mapping[str, int],
        brackets: Mapping[Tuple[str, str], Mapping[str, Number]],
    ):
        """
        generators: dict[name] = parity (0 even, 1 odd)
        brackets: dict[(g1,g2)] = {g3: coeff, ...} meaning [g1,g2] = sum coeff*g3
        """
        self.generators = list(generators.keys())
        self.parity: Dict[str, int] = dict(generators)
        self.dim = len(self.generators)

        # Structure constants C[g1][g2][g3] = coeff in [g1,g2] = sum_g3 C*g3
        self.C: Dict[str, Dict[str, Dict[str, Number]]] = {
            g1: {g2: {g3: 0j for g3 in self.generators} for g2 in self.generators}
            for g1 in self.generators
        }

        for (g1, g2), result in brackets.items():
            p1 = self.parity[g1]
            p2 = self.parity[g2]
            for g3, coeff in result.items():
                c = complex(coeff)
                self.C[g1][g2][g3] = c
                if g1 != g2:
                    sign = -((-1) ** (p1 * p2))
                    self.C[g2][g1][g3] = complex(sign) * c
