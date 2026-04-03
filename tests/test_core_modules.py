"""
Basic functionality tests for core modules.

These tests ensure basic API compatibility and prevent regressions.
For detailed verification, see experiments/test_*.py and experiments/verify_*.py.
"""

import pytest
import numpy as np


def test_algebra_parser_loads():
    """Test that OSpAlgebraParser can load JSON."""
    from oscillator_lie_superalgebras.algebra_parser import OSpAlgebraParser
    from pathlib import Path
    
    json_path = Path(__file__).parent.parent / "data" / "algebra_structures" / "B_0_1_structure.json"
    parser = OSpAlgebraParser(str(json_path))
    
    assert len(parser.basis) > 0
    assert hasattr(parser, 'parity')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
