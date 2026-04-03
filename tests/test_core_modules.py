"""
Basic functionality tests for core modules.

These tests ensure basic API compatibility and prevent regressions.
For detailed verification, see experiments/test_*.py and experiments/verify_*.py.
"""

import pytest
import numpy as np


#def test_gamma_from_B_basic():
#    """Test that compute_gamma_from_B runs without error."""
#    from oscillator_lie_superalgebras.gamma_from_B import compute_gamma_from_B
#    
#    # Basic smoke test
#    B = [1.0, 0.0]
#    gamma = compute_gamma_from_B(1, B)
#    
#    assert isinstance(gamma, dict)
#    assert len(gamma) > 0


def test_algebra_parser_loads():
    """Test that OSpAlgebraParser can load JSON."""
    from oscillator_lie_superalgebras.algebra_parser import OSpAlgebraParser
    from pathlib import Path
    
    json_path = Path(__file__).parent.parent / "data" / "algebra_structures" / "B_0_1_structure.json"
    parser = OSpAlgebraParser(str(json_path))
    
    assert len(parser.basis) > 0
    assert hasattr(parser, 'parity')


#def test_gamma_parser_loads():
#    """Test that GammaStructureParser can load JSON."""
#    from oscillator_lie_superalgebras.gamma_parser import GammaStructureParser
#    from pathlib import Path
#    
#    json_path = Path(__file__).parent.parent / "data" / "gamma_structures" / "osp_1_2_gamma.json"
#    parser = GammaStructureParser(str(json_path))
#    
#    assert len(parser.gamma) > 0
#    assert hasattr(parser, 'B_template')


def test_osp_generators_n1():
    """Test osp generators for n=1."""
    from oscillator_lie_superalgebras.osp_generators import build_osp12n_structure_constants
    
    basis, parity, brackets = build_osp12n_structure_constants(1)
    
    assert len(basis) == 5  # M_0_0, M_0_1, M_1_1, Q_0, Q_1
    assert len(parity) == 5
    assert len(brackets) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
