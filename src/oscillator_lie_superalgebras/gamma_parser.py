"""
Parser for gamma structure JSON files.

This module loads the pre-computed gamma structures (2-cocycles) from JSON
and provides utilities to substitute concrete B matrix values.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
import sympy as sp


class GammaStructureParser:
    """Parser for gamma structure JSON files (schema v3.0 and v5.0).
    
    Gamma structures contain the κ-linear terms in extended brackets
    for inhomogeneous deformations of osp(1|2n).
    """
    
    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.schema_version = self.data["schema_version"]
        self.algebra_ref = self.data["algebra_reference"]
        
        # Parse parameter symbols from inhomogeneous matrix
        self.param_symbols = self._parse_parameters()
        
        # Parse gamma matrix (symbolic expressions)
        self.gamma = self._parse_gamma_matrix()
        
        self.metadata = self.data.get("computation_metadata", {})

        # v5: build gb_template for parameter mapping
        inhom = self.data["inhomogeneous_deformation"]["inhomogeneous_matrix"]
        self.gb_template = {}
        for i, row in enumerate(inhom["matrix"]):
            for j, param in enumerate(row):
                self.gb_template[param] = (i, j)

    def get_gb_shape(self):
        inhom = self.data["inhomogeneous_deformation"]["inhomogeneous_matrix"]
        return tuple(inhom["shape"])

    def substitute_gb(self, gb_dict: Dict[str, float]) -> Dict[Tuple[str, str], Dict[str, complex]]:
        """Substitute gb parameter values by name."""
        subs_dict = {self.param_symbols[k]: v for k, v in gb_dict.items()}
        gamma_numeric = {}
        for (g1, g2), result_dict in self.gamma.items():
            numeric_result = {}
            for res_gen, expr in result_dict.items():
                evaluated = expr.subs(subs_dict).evalf()
                numeric_val = complex(evaluated)
                if abs(numeric_val) > 1e-10:
                    numeric_result[res_gen] = numeric_val
            if numeric_result:
                gamma_numeric[(g1, g2)] = numeric_result
        return gamma_numeric

    
    def _parse_parameters(self) -> Dict[str, sp.Symbol]:
        """Parse deformation parameters from inhomogeneous matrix."""
        inhom_data = self.data["inhomogeneous_deformation"]
        matrix = inhom_data["inhomogeneous_matrix"]["matrix"]
        
        params = {}
        for row in matrix:
            for entry in row:
                if entry not in params:
                    params[entry] = sp.Symbol(entry, complex=True)
        return params
    
    def _parse_gamma_matrix(self) -> Dict[Tuple[str, str], Dict[str, sp.Expr]]:
        """Parse gamma matrix into symbolic expressions.
        
        Returns:
            Dictionary mapping (gen1, gen2) -> {result_gen: coefficient_expr}
        """
        gamma_dict = {}
        raw_gamma = self.data["gamma_matrix"]
        brackets = raw_gamma["brackets"] if "brackets" in raw_gamma else raw_gamma
        for pair_key, result_dict in brackets.items():
            g1, g2 = pair_key.split(',')
            parsed_result = {}
            for res_gen, coeff_dict in result_dict.items():
                expr = sp.Integer(0)
                for param_name, coeff_str in coeff_dict.items():
                    coeff_val = sp.sympify(coeff_str)
                    expr += coeff_val * self.param_symbols[param_name]
                parsed_result[res_gen] = expr
            gamma_dict[(g1, g2)] = parsed_result
        return gamma_dict

    def substitute_B(self, B_values: Union[List[float], Dict[str, float]]) -> Dict[Tuple[str, str], Dict[str, complex]]:
        """Substitute concrete values for B parameters.
        
        Args:
            B_values: Either a list of floats (ordered B_0, B_1, ...) or
                      a dict mapping parameter names to values.
        
        Returns:
            Dictionary mapping (gen1, gen2) -> {result_gen: numerical_coefficient}
        """
        if isinstance(B_values, (list, tuple)):
            param_names = sorted(self.param_symbols.keys(),
                                 key=lambda s: int(s.split('_')[1]))
            subs_dict = {self.param_symbols[name]: val
                         for name, val in zip(param_names, B_values)}
        else:
            subs_dict = {self.param_symbols[k]: v for k, v in B_values.items()}
        
        gamma_numeric = {}
        for (g1, g2), result_dict in self.gamma.items():
            numeric_result = {}
            for res_gen, expr in result_dict.items():
                evaluated = expr.subs(subs_dict).evalf()
                numeric_val = complex(evaluated)
                if abs(numeric_val) > 1e-10:
                    numeric_result[res_gen] = numeric_val
            if numeric_result:
                gamma_numeric[(g1, g2)] = numeric_result
        
        return gamma_numeric
    
    def get_shape(self) -> Tuple[int, int]:
        """Return the shape of the parameter matrix."""
        inhom_data = self.data["inhomogeneous_deformation"]
        return tuple(inhom_data["shape"])
    
    def get_algebra_info(self) -> Dict[str, Any]:
        """Return algebra reference information."""
        return self.algebra_ref