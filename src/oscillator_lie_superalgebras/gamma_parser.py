"""
Parser for gamma structure JSON files.

This module loads the pre-computed gamma structures (2-cocycles) from JSON
and provides utilities to substitute concrete gh matrix values.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
import sympy as sp


class GammaStructureParser:
    """Parser for gamma structure JSON files.
    
    Gamma structures contain the κ-linear terms in extended brackets
    for inhomogeneous deformations of osp(1|2n).
    """
    
    def __init__(self, json_path: str):
        """Load gamma structure from JSON file.
        
        Args:
            json_path: Path to gamma structure JSON file
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Extract metadata
        self.schema_version = self.data["schema_version"]
        self.algebra_ref = self.data["algebra_reference"]
        
        # Parse inhomogeneous matrix template
        self.gh_template = self._parse_gh_template()
        
        # Parse gamma matrix (symbolic expressions)
        self.gamma = self._parse_gamma_matrix()
        
        # Metadata
        self.metadata = self.data.get("computation_metadata", {})
    
    def _parse_gh_template(self) -> Dict[str, sp.Symbol]:
        """Parse the gh matrix template into symbolic parameters.
        
        Returns:
            Dictionary mapping parameter names (e.g., 'gh_a0_b1p') to SymPy symbols
        """
        inhom_data = self.data["inhomogeneous_deformation"]
        gh_matrix = inhom_data["inhomogeneous_matrix"]
        
        # Collect all unique parameter names
        params = {}
        for row in gh_matrix["matrix"]:
            for entry in row:
                if entry not in params:
                    params[entry] = sp.Symbol(entry, complex=True)
        
        return params
    
    def _parse_gamma_matrix(self) -> Dict[Tuple[str, str], Dict[str, sp.Expr]]:
        """Parse gamma matrix into symbolic expressions.
        
        Returns:
            Dictionary mapping (gen1, gen2) -> {result_gen: coefficient_expr}
            where coefficient_expr is a SymPy expression in terms of gh parameters
        """
        gamma_dict = {}
        raw_gamma = self.data["gamma_matrix"]
        
        for pair_key, result_dict in raw_gamma["brackets"].items():
            g1, g2 = pair_key.split(',')
            
            parsed_result = {}
            for res_gen, coeff_dict in result_dict.items():
                # coeff_dict maps gh parameter names to coefficient strings
                # e.g., {"gh_a0_b1p": "2*sqrt(2)", "gh_a0_b1m": "sqrt(2)"}
                expr = sp.Integer(0)
                for gh_param, coeff_str in coeff_dict.items():
                    coeff_val = sp.sympify(coeff_str)
                    expr += coeff_val * self.gh_template[gh_param]
                
                parsed_result[res_gen] = expr
            
            gamma_dict[(g1, g2)] = parsed_result
        
        return gamma_dict

    def substitute_gh(self, gh_values: Dict[str, Union[float, complex]]) -> Dict[Tuple[str, str], Dict[str, complex]]:
        """Substitute concrete values for gh matrix parameters.
        
        Args:
            gh_values: Dictionary mapping gh parameter names to numerical values
                       (e.g., {"gh_a0_b1p": 1.0, "gh_a0_b1m": 0.0})
        
        Returns:
            Dictionary mapping (gen1, gen2) -> {result_gen: numerical_coefficient}
            Coefficients are complex numbers (real values have imag=0)
        """
        subs_dict = {self.gh_template[k]: v for k, v in gh_values.items()}
        
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
    
    def get_gh_shape(self) -> Tuple[int, int]:
        """Return the shape of the gh matrix (fermion_count, boson_count)."""
        inhom_data = self.data["inhomogeneous_deformation"]
        return tuple(inhom_data["inhomogeneous_matrix"]["shape"])
    
    def get_algebra_info(self) -> Dict[str, Any]:
        """Return algebra reference information."""
        return self.algebra_ref
    
    def display_summary(self):
        """Print a summary of the loaded gamma structure."""
        alg = self.algebra_ref
        print(f"--- Gamma Structure for {alg['cartan_type']} ---")
        print(f"Schema version: {self.schema_version}")
        print(f"gh matrix shape: {self.get_gh_shape()}")
        print(f"gh parameters: {list(self.gh_template.keys())}")
        print(f"Non-zero gamma entries: {len(self.gamma)}")
        
        if self.gamma:
            sample_key = list(self.gamma.keys())[0]
            print(f"\nSample γ entry:")
            print(f"  γ({sample_key[0]}, {sample_key[1]}) = {self.gamma[sample_key]}")