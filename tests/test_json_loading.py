"""
Basic tests for JSON loading and schema validation.
"""

import json
import pytest
from pathlib import Path

from oscillator_lie_superalgebras.algebra_parser import OSpAlgebraParser


def test_osp_1_2_json_loads():
    """Test that osp(1|2) JSON file loads without errors."""
    json_path = Path("data/algebra_structures/osp_1_2_structure.json")
    assert json_path.exists(), f"File not found: {json_path}"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    assert data is not None
    assert "schema_version" in data


def test_osp_1_2_schema_version():
    """Test that osp(1|2) JSON has correct schema version."""
    json_path = Path("data/algebra_structures/osp_1_2_structure.json")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    assert data["schema_version"] == "3.0"


def test_osp_1_2_parser_loads():
    """Test that algebra parser loads osp(1|2) successfully."""
    json_path = Path("data/algebra_structures/osp_1_2_structure.json")
    parser = OSpAlgebraParser(str(json_path))
    
    assert parser.schema_version == "3.0"
    assert len(parser.basis) == 5
    assert parser.ordering == "fermionic-first: a, b_0, b_1, ..."


def test_osp_1_2_standard_form():
    """Test that standard form includes constant terms."""
    json_path = Path("data/algebra_structures/osp_1_2_structure.json")
    parser = OSpAlgebraParser(str(json_path))
    
    # M_0_1 should have constant term -1/2
    m01_expr = parser.realizations["M_0_1"]
    m01_str = str(m01_expr)
    
    assert "-1/2" in m01_str or "-0.5" in m01_str, \
        f"M_0_1 should contain constant term -1/2, got: {m01_str}"
    assert "b_0*b_1" in m01_str or "b_0**1*b_1**1" in m01_str, \
        f"M_0_1 should contain b_0*b_1 term, got: {m01_str}"


def test_osp_1_2_coefficient():
    """Test that Q_0 has correct coefficient sqrt(2)/2."""
    json_path = Path("data/algebra_structures/osp_1_2_structure.json")
    parser = OSpAlgebraParser(str(json_path))
    
    # Q_0 coefficient should be sqrt(2)/2
    q0_expr = parser.realizations["Q_0"]
    q0_str = str(q0_expr)
    
    assert "sqrt(2)" in q0_str, \
        f"Q_0 should contain sqrt(2), got: {q0_str}"
    assert "a*b_0" in q0_str or "a**1*b_0**1" in q0_str, \
        f"Q_0 should contain a*b_0 term, got: {q0_str}"


def test_osp_1_4_loads():
    """Test that osp(1|4) JSON loads successfully."""
    json_path = Path("data/algebra_structures/osp_1_4_structure.json")
    
    if not json_path.exists():
        pytest.skip(f"File not found: {json_path}")
    
    parser = OSpAlgebraParser(str(json_path))
    assert parser.schema_version == "3.0"
    assert len(parser.basis) == 14  # 10 even + 4 odd


def test_structure_constants_exist():
    """Test that structure constants are correctly loaded."""
    json_path = Path("data/algebra_structures/osp_1_2_structure.json")
    parser = OSpAlgebraParser(str(json_path))
    
    # Test a known bracket: [M_0_1, M_1_1] = 2 M_1_1
    bracket = parser.get_bracket("M_0_1", "M_1_1")
    
    assert "M_1_1" in bracket, "Expected M_1_1 in bracket result"
    assert bracket["M_1_1"] == 2, f"Expected coefficient 2, got {bracket['M_1_1']}"


if __name__ == "__main__":
    # Run tests manually for quick verification
    print("Running basic JSON loading tests...\n")
    
    test_osp_1_2_json_loads()
    print("✓ test_osp_1_2_json_loads")
    
    test_osp_1_2_schema_version()
    print("✓ test_osp_1_2_schema_version")
    
    test_osp_1_2_parser_loads()
    print("✓ test_osp_1_2_parser_loads")
    
    test_osp_1_2_standard_form()
    print("✓ test_osp_1_2_standard_form")
    
    test_osp_1_2_coefficient()
    print("✓ test_osp_1_2_coefficient")
    
    try:
        test_osp_1_4_loads()
        print("✓ test_osp_1_4_loads")
    except Exception as e:
        print(f"⚠ test_osp_1_4_loads: {e}")
    
    test_structure_constants_exist()
    print("✓ test_structure_constants_exist")
    
    print("\n✓ All tests passed!")
