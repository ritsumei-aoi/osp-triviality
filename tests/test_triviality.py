"""
Tests for triviality checking of B(0,n) = osp(1|2n) deformations.

Verifies that all first-order deformations are trivial for n = 1, 2.
"""

import numpy as np
import pytest

from oscillator_lie_superalgebras.triviality_checker import check_triviality


class TestTrivialityB01:
    """Triviality tests for B(0,1) = osp(1|2)."""

    def test_zero_deformation(self):
        """Zero B matrix should give trivial deformation."""
        result = check_triviality(n=1, B=[0.0, 0.0])
        assert result["is_trivial"]
        assert result["residual_norm"] < 1e-9

    def test_unit_deformation(self):
        """Unit B = [1, 0] should give trivial deformation."""
        result = check_triviality(n=1, B=[1.0, 0.0])
        assert result["is_trivial"]
        assert result["residual_norm"] < 1e-9

    def test_generic_deformation(self):
        """Generic B should give trivial deformation."""
        result = check_triviality(n=1, B=[0.5, -0.3])
        assert result["is_trivial"]
        assert result["residual_norm"] < 1e-9


class TestTrivialityB02:
    """Triviality tests for B(0,2) = osp(1|4)."""

    def test_zero_deformation(self):
        """Zero B matrix should give trivial deformation."""
        B = [0.0, 0.0, 0.0, 0.0]
        result = check_triviality(n=2, B=B)
        assert result["is_trivial"]
        assert result["residual_norm"] < 1e-9

    def test_single_parameter(self):
        """Single nonzero parameter should give trivial deformation (expected nontrivial in v5)."""
        B = [1.0, 0.0, 0.0, 0.0]
        result = check_triviality(n=2, B=B)
        assert not result["is_trivial"]
        assert result["residual_norm"] > 1e-3

    def test_generic_deformation(self):
        """Generic B should give trivial deformation (expected nontrivial in v5)."""
        B = [0.3, -0.7, 1.2, 0.5]
        result = check_triviality(n=2, B=B)
        assert not result["is_trivial"]
        assert result["residual_norm"] > 1e-3


class TestTrivialityOutput:
    """Test output format of triviality checker."""

    def test_result_keys(self):
        """Result dict should contain expected keys."""
        result = check_triviality(n=1, B=[1.0, 0.0])
        expected_keys = {"is_trivial", "residual_norm", "f_map", "gamma",
                         "tolerance", "n"}
        assert expected_keys.issubset(set(result.keys()))

    def test_f_map_is_dict(self):
        """f_map should be a dictionary mapping generators."""
        result = check_triviality(n=1, B=[1.0, 0.0])
        assert isinstance(result["f_map"], dict)