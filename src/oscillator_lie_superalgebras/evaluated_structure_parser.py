"""
Parser for evaluated structure JSON files (Schema 3).

Extracts gamma coefficients (kappa-linear terms) and base brackets (kappa=0)
from a Schema 3 evaluated structure JSON file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import sympy as sp


class EvaluatedStructureParser:
    """Parse a Schema 3 evaluated structure JSON file.

    Provides extraction of:
    - gamma(X, Y): kappa-linear coefficients  → for cohomology/triviality check
    - base brackets [X, Y]: kappa=0 coefficients → for verification

    Args:
        json_path: Path to Schema 3 evaluated structure JSON.
    """

    def __init__(self, json_path: Union[str, Path]):
        self.path = Path(json_path)
        with self.path.open("r", encoding="utf-8") as f:
            self.data: Dict[str, Any] = json.load(f)

        self.schema_version: str = self.data["schema_version"]
        self.algebra_ref: Dict[str, Any] = self.data["algebra_reference"]
        self.gh_values: Dict[str, Any] = self.data["gh_values"]
        self.kappa_symbol: str = self.data["evaluated_brackets"]["kappa_symbol"]
        self._kappa = sp.Symbol(self.kappa_symbol)

        self._raw_brackets: Dict[str, Dict[str, str]] = (
            self.data["evaluated_brackets"]["brackets"]
        )

    # ------------------------------------------------------------------
    # algebra metadata
    # ------------------------------------------------------------------

    def get_family(self) -> str:
        return self.algebra_ref["family"]

    def get_m(self) -> int:
        return self.algebra_ref["m"]

    def get_n(self) -> int:
        return self.algebra_ref["n"]

    def get_gh_matrix(self) -> List[List]:
        return self.gh_values["matrix"]

    # ------------------------------------------------------------------
    # coefficient extraction helpers
    # ------------------------------------------------------------------

    def _parse_coeff(self, coeff_str: str) -> sp.Expr:
        """Parse a SymPy-compatible coefficient string."""
        return sp.sympify(coeff_str)


    def _kappa_coeff(self, expr: sp.Expr) -> complex:
        """Extract coefficient of kappa (kappa^1 term) from expr."""
        poly = sp.Poly(sp.expand(expr), self._kappa)
        c = poly.nth(1) if poly.degree() >= 1 else sp.Integer(0)
        return complex(c)

    def _constant_coeff(self, expr: sp.Expr) -> complex:
        """Extract kappa=0 (constant) term from expr."""
        return complex(expr.subs(self._kappa, 0))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def extract_gamma(
        self, tol: float = 1e-12
    ) -> Dict[Tuple[str, str], Dict[str, complex]]:
        """Extract kappa-linear coefficients (gamma) from evaluated brackets.

        gamma(X, Y) is defined by:
            [X, Y]_L = [X, Y] + kappa * gamma(X, Y)

        Returns:
            Dict mapping (gen1, gen2) -> {result_gen: complex_coeff}
            Zero entries (|coeff| < tol) are omitted.
        """
        result: Dict[Tuple[str, str], Dict[str, complex]] = {}

        for key, res_dict in self._raw_brackets.items():
            g1, g2 = key.split(",")
            gamma_entry: Dict[str, complex] = {}

            for out_gen, coeff_str in res_dict.items():
                expr = self._parse_coeff(coeff_str)
                c = self._kappa_coeff(expr)
                if abs(c) > tol:
                    gamma_entry[out_gen] = c

            if gamma_entry:
                result[(g1, g2)] = gamma_entry

        return result

    def extract_base_brackets(
        self, tol: float = 1e-12
    ) -> Dict[Tuple[str, str], Dict[str, complex]]:
        """Extract kappa=0 (base bracket) coefficients.

        Returns:
            Dict mapping (gen1, gen2) -> {result_gen: complex_coeff}
            Zero entries are omitted.
        """
        result: Dict[Tuple[str, str], Dict[str, complex]] = {}

        for key, res_dict in self._raw_brackets.items():
            g1, g2 = key.split(",")
            base_entry: Dict[str, complex] = {}

            for out_gen, coeff_str in res_dict.items():
                expr = self._parse_coeff(coeff_str)
                c = self._constant_coeff(expr)
                if abs(c) > tol:
                    base_entry[out_gen] = c

            if base_entry:
                result[(g1, g2)] = base_entry

        return result

    def summary(self) -> str:
        """Return a brief summary string."""
        gamma = self.extract_gamma()
        base = self.extract_base_brackets()
        alg = self.algebra_ref
        return (
            f"EvaluatedStructureParser: {alg['cartan_type']} "
            f"gh={self.gh_values['flat_repr']}  "
            f"non-zero gamma pairs={len(gamma)}  "
            f"non-zero base pairs={len(base)}"
        )
