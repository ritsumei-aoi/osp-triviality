"""
Compute gamma (2-cocycle) from gb matrix.

This module computes numerical gamma coefficients by substituting concrete
gb matrix values into pre-computed gamma structures (Schema 2 JSON files).
gb is the fermion×boson submatrix of the skew-supersymmetric Gram matrix G,
defined by: [b_j, a_i] = -gb[i][j] · κ  (v5.0 sign convention).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .gamma_parser import GammaStructureParser


GbMatrix = List[List[Union[int, float, complex]]]


def compute_gamma_from_gb(
    gamma_json_path: Union[str, Path],
    gb: GbMatrix,
) -> Dict[Tuple[str, str], Dict[str, complex]]:
    """Compute numerical gamma coefficients from a gb matrix.

    Loads a Schema 2 gamma structure JSON and substitutes the given gb matrix
    values into the symbolic gamma expressions.

    Sign convention (v5.0):
        [b_j, a_i] = G[b_j][a_i] · κ = -gb[i][j] · κ

    Args:
        gamma_json_path: Path to Schema 2 gamma structure JSON file
                         (e.g., "data/gamma_structures/B_0_1_gamma.json").
        gb: gb submatrix as list-of-lists (row=fermion in PBW order,
            col=boson in PBW order). Shape must match the JSON template.

    Returns:
        Dictionary mapping (gen1, gen2) -> {result_gen: complex_coefficient}.
        Only non-zero entries are included.

    Raises:
        FileNotFoundError: If gamma_json_path does not exist.
        ValueError: If gb shape does not match the JSON template.

    Example:
        >>> # B(0,1): gb shape (1, 2), e.g. gb = [[1, 0]]
        >>> gamma = compute_gamma_from_gb("data/gamma_structures/B_0_1_gamma.json", [[1, 0]])
    """
    parser = GammaStructureParser(str(gamma_json_path))
    gb_dict = _build_gb_dict(parser, gb)
    return parser.substitute_gb(gb_dict)


def _build_gb_dict(
    parser: GammaStructureParser,
    gb: GbMatrix,
) -> Dict[str, Union[float, complex]]:
    """Map flat gb matrix values to parser parameter names.

    Args:
        parser: Loaded GammaStructureParser instance.
        gb: gb matrix as list-of-lists.

    Returns:
        Dict mapping parameter names (e.g. 'gb_a0_b1p') to numerical values.

    Raises:
        ValueError: If gb shape does not match template.
    """
    param_names = list(parser.gb_template.keys())
    flat_vals = [v for row in gb for v in row]

    if len(flat_vals) != len(param_names):
        expected_shape = parser.get_gb_shape()
        raise ValueError(
            f"gb has {len(flat_vals)} entries, expected {len(param_names)} "
            f"(shape {expected_shape[0]}×{expected_shape[1]}). "
            f"Parameters: {param_names}"
        )

    return {
        name: complex(v) if isinstance(v, complex) else float(v)
        for name, v in zip(param_names, flat_vals)
    }