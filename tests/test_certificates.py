"""
Tests for algebraic certificate verification.

Certificates c satisfy c^T A = 0 where A is the constraint matrix,
providing symbolic proof of triviality for B(0,n) deformations.
"""

import numpy as np
import pytest

from oscillator_lie_superalgebras.adjoint_from_json import (
    build_adjoint_from_json,
    get_available_algebra_structures,
)
from oscillator_lie_superalgebras.gamma_from_B import compute_gamma_from_B
from oscillator_lie_superalgebras.cohomology_solver import (
    solve_odd_f_generic,
    reconstruct_gamma_from_f,
)


class TestAdjointConstruction:
    """Test that adjoint matrices are correctly constructed."""

    def test_adjoint_n1(self):
        """Build adjoint for osp(1|2) and check dimensions."""
        basis, parity, adj = build_adjoint_from_json(n=1)
        dim = len(basis)
        assert dim > 0
        for x in basis:
            assert adj[x].shape == (dim, dim)

    def test_adjoint_n2(self):
        """Build adjoint for osp(1|4) and check dimensions."""
        basis, parity, adj = build_adjoint_from_json(n=2)
        dim = len(basis)
        assert dim == 14  # osp(1|4) has dimension 14
        for x in basis:
            assert adj[x].shape == (dim, dim)


class TestGammaComputation:
    """Test gamma (2-cocycle) computation from B matrix."""

    def test_gamma_n1_nonzero(self):
        """Non-zero B should produce non-zero gamma for n=1."""
        B = [1.0, 0.0]
        gamma = compute_gamma_from_B(n=1, B=B)
        assert isinstance(gamma, dict)
        has_nonzero = any(
            any(abs(v) > 1e-10 for v in entry.values())
            for entry in gamma.values()
        )
        assert has_nonzero, "Gamma should be non-zero for non-zero B"

    def test_gamma_n1_zero(self):
        """Zero B should produce zero gamma."""
        B = [0.0, 0.0]
        gamma = compute_gamma_from_B(n=1, B=B)
        for key, entry in gamma.items():
            for gen, coeff in entry.items():
                assert abs(coeff) < 1e-10, f"Expected zero gamma at {key}->{gen}"


class TestCoboundaryReconstruction:
    """Test that f-map correctly reconstructs gamma via coboundary."""

    def test_reconstruct_n1(self):
        """For n=1, solve for f then reconstruct gamma and compare."""
        B = [1.0, 0.5]
        basis, parity, adj = build_adjoint_from_json(n=1)
        gamma = compute_gamma_from_B(n=1, B=B)

        f_map, residual = solve_odd_f_generic(basis, parity, adj, gamma)
        assert residual < 1e-9, f"Residual too large: {residual}"

        gamma_recon = reconstruct_gamma_from_f(basis, parity, adj, f_map)

        # Compare original and reconstructed gamma
        for (x, y), entry in gamma.items():
            recon_entry = gamma_recon.get((x, y), {})
            for gen, coeff in entry.items():
                recon_coeff = recon_entry.get(gen, 0.0)
                assert abs(coeff - recon_coeff) < 1e-8, \
                    f"Mismatch at ({x},{y})->{gen}: {coeff} vs {recon_coeff}"