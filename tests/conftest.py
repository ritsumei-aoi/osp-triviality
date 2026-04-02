"""
pytest configuration for oscillator-lie-superalgebras.

This test suite provides minimal regression tests for core functionality.
For detailed mathematical verification, see experiments/ directory.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
