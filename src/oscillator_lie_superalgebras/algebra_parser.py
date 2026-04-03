"""
Parser for OSp algebra structure JSON files.
Loads basis, parity, structure constants, and creates SymPy representations
of the generator realizations.

Version 3.0: Supports standard form (fermionic-first PBW ordering) with constant terms.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import sympy as sp


class OSpAlgebraParser:
    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Check schema version
        self.schema_version = self.data.get("schema_version", "unknown")
        if self.schema_version not in ["2.0", "3.0"]:
            print(f"Warning: Unexpected schema version {self.schema_version}")
            
        self.basis: List[str] = self.data["basis"]["even"] + self.data["basis"]["odd"]
        self.parity: Dict[str, int] = self.data["parity"]
        
        # Parse brackets into numeric/sympy forms
        self.brackets = self._parse_brackets()
        
        # Setup SymPy symbols for oscillators
        self.symbols = self._setup_sympy_symbols()
        
        # Parse generator realizations into SymPy expressions
        self.realizations = self._parse_realizations()
        
        # Oscillator relations
        self.oscillator_relations = self.data.get("oscillator_relations", {})
        self.symplectic_matrix = None
        if self.oscillator_relations:
            symplectic_data = self.oscillator_relations.get("symplectic_matrix", {})
            self.symplectic_matrix = symplectic_data.get("matrix")
        
        # PBW ordering (v3.0)
        self.ordering = None
        gen_real = self.data.get("generator_realization", {})
        if "ordering" in gen_real:
            self.ordering = gen_real["ordering"]

    def _parse_brackets(self) -> Dict[Tuple[str, str], Dict[str, sp.Expr]]:
        """Parse structure constants into a dictionary with SymPy expressions as coefficients."""
        parsed = {}
        raw_brackets = self.data["structure_constants"]["brackets"]
        
        for pair_str, result_dict in raw_brackets.items():
            g1, g2 = pair_str.split(',')
            parsed_res = {}
            for res_gen, coeff_str in result_dict.items():
                parsed_res[res_gen] = sp.sympify(coeff_str)
            parsed[(g1, g2)] = parsed_res
            
        return parsed

    def _setup_sympy_symbols(self) -> Dict[str, sp.Symbol]:
        """Create non-commutative SymPy symbols for bosonic and fermionic oscillators."""
        symbols = {}
        
        # Bosonic (b_i)
        bosons = []
        if "bosonic" in self.data["oscillator_generators"]:
            bosons = self.data["oscillator_generators"]["bosonic"]
        elif "bosons" in self.data["oscillator_generators"]:
            bosons = self.data["oscillator_generators"]["bosons"]["labels"]
        for b in bosons:
            symbols[b] = sp.Symbol(b, commutative=False)
        
        # Fermionic (a)
        fermions = []
        if "fermionic" in self.data["oscillator_generators"]:
            fermions = self.data["oscillator_generators"]["fermionic"]
        elif "fermions" in self.data["oscillator_generators"]:
            fermions = self.data["oscillator_generators"]["fermions"]["labels"]
        # Supplementary fermion (B(0,n) case)
        if "supplementary_fermion" in self.data["oscillator_generators"]:
            fermions = fermions + [self.data["oscillator_generators"]["supplementary_fermion"]["label"]]
        for a in fermions:
            symbols[a] = sp.Symbol(a, commutative=False)
        
        return symbols

    def _parse_realizations(self) -> Dict[str, sp.Expr]:
        """
        Convert JSON word representations into SymPy non-commutative expressions.
        Supports both v2.0 (terms) and v3.0 (standard_form) schemas.
        """
        realizations = {}
        raw_realizations = self.data["generator_realization"]["realizations"]
        
        for gen, data in raw_realizations.items():
            # Determine field name based on schema version
            if "standard_form" in data:
                terms_field = "standard_form"
            elif "terms" in data:
                terms_field = "terms"
            else:
                raise ValueError(f"No terms or standard_form field found for generator {gen}")
            
            expr = sp.Integer(0)
            for term in data[terms_field]:
                coeff = sp.sympify(term["coeff"])
                
                # Handle constant term (words: [])
                if len(term["words"]) == 0:
                    # Constant term: just add the coefficient
                    expr += coeff
                else:
                    # Create the product of non-commutative symbols
                    word_expr = sp.Integer(1)
                    for w in term["words"]:
                        word_expr = word_expr * self.symbols[w]
                    expr += coeff * word_expr
                
            realizations[gen] = expr
            
        return realizations

    def get_bracket(self, g1: str, g2: str) -> Dict[str, sp.Expr]:
        """Return the bracket result for [g1, g2]. Returns empty dict if bracket is 0."""
        return self.brackets.get((g1, g2), {})

    def display_summary(self):
        """Helper to print a summary of the loaded algebra."""
        alg = self.data["algebra"]
        print(f"--- Algebra: osp({alg['m']}|{alg['2n']}) [{alg['cartan_type']}] ---")
        print(f"Schema version: {self.schema_version}")
        print(f"Dimension: {len(self.basis)} (Even: {len(self.data['basis']['even'])}, Odd: {len(self.data['basis']['odd'])})")
        
        if self.ordering:
            print(f"PBW ordering: {self.ordering}")
        
        print("\nSample Realizations (in U(A)):")
        
        # Display first even and first odd
        first_even = self.data["basis"]["even"][0]
        first_odd = self.data["basis"]["odd"][0]
        print(f"  {first_even} = {self.realizations[first_even]}")
        print(f"  {first_odd} = {self.realizations[first_odd]}")
        
        print("\nSample Bracket:")
        sample_key = list(self.brackets.keys())[-1]
        print(f"  [{sample_key[0]}, {sample_key[1]}] = {self.brackets[sample_key]}")
