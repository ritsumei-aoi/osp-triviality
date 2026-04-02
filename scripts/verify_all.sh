#!/bin/bash
# Run all verification scripts for osp-triviality.
# Usage: bash scripts/verify_all.sh

set -e

echo "=== osp-triviality: Full Verification Suite ==="
echo ""

echo "--- 1. Running test suite ---"
pytest tests/ -v
echo ""

echo "--- 2. Verifying dimension formula (dim C^1_mu = 6n - 2) ---"
python scripts/verify_dim_formula.py
echo ""

echo "--- 3. Verifying algebraic certificates ---"
python scripts/verify_certificates.py
echo ""

echo "--- 4. Checking triviality for B(0,2) ---"
python scripts/check_triviality.py --n 2
echo ""

echo "--- 5. Checking triviality for B(0,3) ---"
python scripts/check_triviality.py --n 3
echo ""

echo "=== All verifications passed ==="