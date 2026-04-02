"""
Compute gamma (2-cocycle) from gh matrix.

This module computes numerical gamma coefficients by substituting concrete
gh matrix values into pre-computed gamma structures (Schema 2 JSON files).
gh is the fermion×boson submatrix of the skew-supersymmetric Gram matrix G,
defined by: [b_j, a_i] = -gh[i][j] · κ  (v5.0 sign convention).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .gamma_parser import GammaStructureParser


GhMatrix = List[List[Union[int, float, complex]]]


def compute_gamma_from_gh(
    gamma_json_path: Union[str, Path],
    gh: GhMatrix,
) -> Dict[Tuple[str, str], Dict[str, complex]]:
    """Compute numerical gamma coefficients from a gh matrix.

    Loads a Schema 2 gamma structure JSON and substitutes the given gh matrix
    values into the symbolic gamma expressions.

    Sign convention (v5.0):
        [b_j, a_i] = G[b_j][a_i] · κ = -gh[i][j] · κ

    Args:
        gamma_json_path: Path to Schema 2 gamma structure JSON file
                         (e.g., "data/gamma_structures/B_0_1_gamma.json").
        gh: gh submatrix as list-of-lists (row=fermion in PBW order,
            col=boson in PBW order). Shape must match the JSON template.

    Returns:
        Dictionary mapping (gen1, gen2) -> {result_gen: complex_coefficient}.
        Only non-zero entries are included.

    Raises:
        FileNotFoundError: If gamma_json_path does not exist.
        ValueError: If gh shape does not match the JSON template.

    Example:
        >>> # B(0,1): gh shape (1, 2), e.g. gh = [[1, 0]]
        >>> gamma = compute_gamma_from_gh("data/gamma_structures/B_0_1_gamma.json", [[1, 0]])
    """
    parser = GammaStructureParser(str(gamma_json_path))
    gh_dict = _build_gh_dict(parser, gh)
    return parser.substitute_gh(gh_dict)


def _build_gh_dict(
    parser: GammaStructureParser,
    gh: GhMatrix,
) -> Dict[str, Union[float, complex]]:
    """Map flat gh matrix values to parser parameter names.

    Args:
        parser: Loaded GammaStructureParser instance.
        gh: gh matrix as list-of-lists.

    Returns:
        Dict mapping parameter names (e.g. 'gh_a0_b1p') to numerical values.

    Raises:
        ValueError: If gh shape does not match template.
    """
    param_names = list(parser.gh_template.keys())
    flat_vals = [v for row in gh for v in row]

    if len(flat_vals) != len(param_names):
        expected_shape = parser.get_gh_shape()
        raise ValueError(
            f"gh has {len(flat_vals)} entries, expected {len(param_names)} "
            f"(shape {expected_shape[0]}×{expected_shape[1]}). "
            f"Parameters: {param_names}"
        )

    return {
        name: complex(v) if isinstance(v, complex) else float(v)
        for name, v in zip(param_names, flat_vals)
    }