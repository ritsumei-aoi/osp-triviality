"""
Basic tests for JSON loading and schema validation.
"""

import json
import pytest
from pathlib import Path

from oscillator_lie_superalgebras.algebra_parser import OSpAlgebraParser


def test_B01_json_loads():
    """Test that B(0,1) JSON file loads without errors."""
    json_path = Path("data/algebra_structures/B_0_1_structure.json")
    assert json_path.exists(), f"File not found: {json_path}"

    with open(json_path, 'r') as f:
        data = json.load(f)

    assert data is not None
    assert "schema_version" in data


def test_B01_schema_version():
    """Test that B(0,1) JSON has correct schema version (v5.0)."""
    json_path = Path("data/algebra_structures/B_0_1_structure.json")

    with open(json_path, 'r') as f:
        data = json.load(f)

    assert data["schema_version"] == "5.0"


def test_B01_parser_loads():
    """Test that algebra parser loads B(0,1) successfully."""
    json_path = Path("data/algebra_structures/B_0_1_structure.json")
    parser = OSpAlgebraParser(str(json_path))

    assert parser.schema_version == "5.0"
    assert len(parser.basis) == 5


def test_B02_loads():
    """Test that B(0,2) = osp(1|4) JSON loads successfully."""
    json_path = Path("data/algebra_structures/B_0_2_structure.json")

    if not json_path.exists():
        pytest.skip(f"File not found: {json_path}")

    parser = OSpAlgebraParser(str(json_path))
    assert parser.schema_version == "5.0"
    assert len(parser.basis) == 14  # 10 even + 4 odd
