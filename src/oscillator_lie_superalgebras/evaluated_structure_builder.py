"""
Builder for evaluated structure JSON files (Schema 3).

Combines Schema 1 (algebra structure) and Schema 2 (gamma structure)
with a concrete gh matrix to produce fully evaluated brackets [X,Y]_L
= [X,Y] + kappa * gamma(X,Y)|_{gh=values}.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import sympy as sp

from oscillator_lie_superalgebras.gamma_parser import GammaStructureParser

GhMatrix = List[List[Union[int, float, str]]]


# ---------------------------------------------------------------------------
# gh encoding
# ---------------------------------------------------------------------------

class GhEncoder:
    """Encode gh matrix values to/from filename-safe flat strings."""

    # Special symbolic values
    _SPECIAL: Dict[str, str] = {
        "sqrt(2)": "r2",
        "-sqrt(2)": "mr2",
    }
    _SPECIAL_INV: Dict[str, str] = {v: k for k, v in _SPECIAL.items()}

    @classmethod
    def encode_value(cls, v: Union[int, float, str]) -> str:
        """Encode a single gh entry to filename-safe string.

        Rules:
          - Special sympy strings (sqrt(2), -sqrt(2)) -> r2, mr2
          - Negative integer  -> m<abs>
          - Positive integer  -> <int>
          - Negative float    -> m<int>p<frac>
          - Positive float    -> <int>p<frac>
          - Zero              -> 0
        """
        s = str(v).strip()

        if s in cls._SPECIAL:
            return cls._SPECIAL[s]

        try:
            sym = sp.sympify(s)
        except Exception:
            raise ValueError(f"Cannot encode gh value: {v!r}")

        if sym == 0:
            return "0"

        # Try rational / integer
        if sym.is_Integer:
            n = int(sym)
            return f"m{abs(n)}" if n < 0 else str(n)

        if sym.is_Rational:
            # represent as decimal string
            f = float(sym)
        elif sym.is_Float or isinstance(v, float):
            f = float(sym)
        else:
            raise ValueError(f"Cannot encode non-numeric gh value: {v!r}")

        sign = "m" if f < 0 else ""
        abs_f = abs(f)
        int_part = int(abs_f)
        frac = abs_f - int_part
        if frac == 0:
            return f"{sign}{int_part}"
        frac_str = f"{frac:.10f}".lstrip("0").replace(".", "").rstrip("0")
        return f"{sign}{int_part}p{frac_str}"

    @classmethod
    def encode_matrix(cls, gh: GhMatrix) -> str:
        """Row-major flatten and encode entire gh matrix."""
        flat = [cls.encode_value(v) for row in gh for v in row]
        return "_".join(flat)

    @classmethod
    def build_filename(cls, family: str, m: int, n: int, gh: GhMatrix) -> str:
        """Build the Schema 3 filename (without directory)."""
        flat = cls.encode_matrix(gh)
        return f"{family}_{m}_{n}_gh_{flat}.json"

    @classmethod
    def sympy_repr(cls, gh: GhMatrix) -> List[List[str]]:
        """Return gh as list-of-lists of SymPy-compatible strings."""
        return [[str(sp.sympify(str(v))) for v in row] for row in gh]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class EvaluatedStructureBuilder:
    """Build Schema 3 evaluated structure JSON.

    Args:
        structure_path: Path to Schema 1 algebra structure JSON.
        gamma_path: Path to Schema 2 gamma structure JSON.
    """

    SCHEMA_VERSION = "5.0"

    def __init__(self, structure_path: Union[str, Path], gamma_path: Union[str, Path]):
        self.structure_path = Path(structure_path)
        self.gamma_path = Path(gamma_path)

        with self.structure_path.open("r", encoding="utf-8") as f:
            self.structure_data = json.load(f)

        self.gamma_parser = GammaStructureParser(str(self.gamma_path))
        self.kappa = sp.Symbol("kappa")

        self._basis: List[str] = (
            self.structure_data["basis"]["even"]
            + self.structure_data["basis"]["odd"]
        )
        self._brackets0: Dict[Tuple[str, str], Dict[str, sp.Expr]] = (
            self._load_base_brackets()
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _load_base_brackets(self) -> Dict[Tuple[str, str], Dict[str, sp.Expr]]:
        raw = self.structure_data["structure_constants"]["brackets"]
        out: Dict[Tuple[str, str], Dict[str, sp.Expr]] = {}
        for key, res_dict in raw.items():
            g1, g2 = key.split(",")
            out[(g1, g2)] = {g: sp.sympify(c) for g, c in res_dict.items()}
        return out

    def _combine(
        self,
        gamma_num: Dict[Tuple[str, str], Dict[str, complex]],
    ) -> Dict[Tuple[str, str], Dict[str, sp.Expr]]:
        """Merge base brackets with kappa * gamma."""
        result: Dict[Tuple[str, str], Dict[str, sp.Expr]] = {}
        all_pairs = set(self._brackets0.keys()) | set(gamma_num.keys())

        for pair in all_pairs:
            res: Dict[str, sp.Expr] = {}

            if pair in self._brackets0:
                for g, c in self._brackets0[pair].items():
                    res[g] = res.get(g, sp.Integer(0)) + c

            if pair in gamma_num:
                for g, c_gamma in gamma_num[pair].items():
                    cg = sp.nsimplify(c_gamma)
                    res[g] = res.get(g, sp.Integer(0)) + self.kappa * cg

            res = {g: sp.simplify(c) for g, c in res.items()
                   if not sp.simplify(c).equals(sp.Integer(0))}
            if res:
                result[pair] = res

        return result

    def _coeff_to_str(self, expr: sp.Expr) -> str:
        """Convert SymPy expression to canonical string for JSON."""
        return str(sp.simplify(expr))



    def _check_gh_zero(self) -> str:
        """Verify that gh=0 reproduces Schema 1 structure constants."""
        param_names = list(self.gamma_parser.gh_template.keys())   # 旧: B_template
        zero_gh_values = {name: 0.0 for name in param_names}       # 旧: zero_values
        gamma_zero = self.gamma_parser.substitute_gh(zero_gh_values)  # 旧: substitute_B
        combined_zero = self._combine(gamma_zero)

        base = self._brackets0
        all_keys = set(base.keys()) | set(combined_zero.keys())
        for pair in all_keys:
            base_res = base.get(pair, {})
            eval_res = combined_zero.get(pair, {})
            all_gens = set(base_res.keys()) | set(eval_res.keys())
            for g in all_gens:
                c_base = sp.simplify(base_res.get(g, sp.Integer(0)))
                c_eval = sp.simplify(eval_res.get(g, sp.Integer(0)))
                if not c_base.equals(c_eval):
                    return "failed"
        return "passed"

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build(self, gh: GhMatrix, run_zero_check: bool = True,
          generated_by: str = "experiments/generate_evaluated_structure.py") -> Dict[str, Any]:
        param_names = list(self.gamma_parser.gh_template.keys())   # 旧: B_template
        flat_vals = [v for row in gh for v in row]
        if len(flat_vals) != len(param_names):
            raise ValueError(
                f"gh has {len(flat_vals)} entries, expected {len(param_names)}"
            )
        gh_dict = {name: float(sp.sympify(str(v))) for name, v in
                zip(param_names, flat_vals)}

        gamma_num = self.gamma_parser.substitute_gh(gh_dict)       # 旧: substitute_B

        combined = self._combine(gamma_num)

        # algebra reference
        alg = self.structure_data["algebra"]
        inhom = self.gamma_parser.data["inhomogeneous_deformation"]
        algebra_ref = {
            "family": alg["family"],
            "m": alg["m"],
            "n": alg["n"],
            "cartan_type": alg["cartan_type"],
            "structure_file": str(
                Path("../algebra_structures") / self.structure_path.name
            ),
            "gamma_file": str(
                Path("../gamma_structures") / self.gamma_path.name
            ),
            "structure_schema_version": self.structure_data.get("schema_version", "4.1"),
            "gamma_schema_version": self.gamma_parser.schema_version,
        }

        # gh_values section
        inhom_matrix = inhom["inhomogeneous_matrix"]
        gh_values: Dict[str, Any] = {
            "shape": inhom_matrix["shape"],
            "fermion_labels": inhom_matrix["fermion_basis_order"],
            "boson_labels": inhom_matrix["boson_basis_order"],
            "matrix": gh,
            "flat_repr": GhEncoder.encode_matrix(gh),
            "sympy_repr": GhEncoder.sympy_repr(gh),
            "description": inhom_matrix.get("description", ""),
        }


        # evaluated_brackets section — preserve Schema 1 key order, append new
        base_keys_ordered = list(self._brackets0.keys())
        extra_keys = [p for p in combined if p not in self._brackets0]
        ordered_pairs = base_keys_ordered + extra_keys

        brackets_out: Dict[str, Any] = {}
        for pair in ordered_pairs:
            if pair not in combined:
                continue
            key = f"{pair[0]},{pair[1]}"
            brackets_out[key] = {
                g: self._coeff_to_str(c)
                for g, c in combined[pair].items()
            }

        evaluated_brackets = {
            "description": (
                "Full brackets [X,Y]_L = [X,Y] + κ·γ(X,Y) with gh substituted. "
                "Coefficients are SymPy-compatible strings. Zero brackets are omitted."
            ),
            "kappa_symbol": "kappa",
            "brackets": brackets_out,
        }

        # metadata
        zero_check = self._check_gh_zero() if run_zero_check else "skipped"
        jst = timezone(timedelta(hours=9))
        metadata = {
            "generated_by": generated_by,
            "generation_timestamp": datetime.now(jst).isoformat(timespec="seconds"),
            "gh_zero_check": zero_check,
            "description": (
                "Generated by substituting gh values into gamma structure "
                "and combining with base structure constants."
            ),
        }

        return {
            "schema_version": self.SCHEMA_VERSION,
            "algebra_reference": algebra_ref,
            "gh_values": gh_values,
            "evaluated_brackets": evaluated_brackets,
            "metadata": metadata,
        }

    def save(self, gh: GhMatrix, output_dir: Union[str, Path], run_zero_check: bool = True,
             generated_by: str = "experiments/generate_evaluated_structure.py") -> Path:
        """Build and save Schema 3 JSON to output_dir.

        Returns:
            Path to the saved file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        alg = self.structure_data["algebra"]
        filename = GhEncoder.build_filename(
            alg["family"], alg["m"], alg["n"], gh
        )
        out_path = output_dir / filename

        data = self.build(gh, run_zero_check=run_zero_check, generated_by=generated_by)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return out_path
