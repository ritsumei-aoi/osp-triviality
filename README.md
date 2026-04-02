# osp-triviality

Computational verification code for the triviality of first-order
deformations of oscillator Lie superalgebras B(0,n) = osp(1|2n).

## Overview

This repository provides Python code to verify the main results of:

> *Triviality of first-order deformations of oscillator Lie superalgebras B(0,n)*

The code computationally verifies that every first-order deformation of
the oscillator Lie superalgebra B(0,n) = osp(1|2n) is trivial (equivalent
to the undeformed algebra) for small values of n, supporting the general
conjecture.

## Quick Start

```bash
pip install -e .
python scripts/check_triviality.py --n 2
python scripts/check_triviality.py --n 3
```

## Requirements

- Python >= 3.10
- NumPy >= 1.26
- SciPy >= 1.12
- SymPy >= 1.12

## Installation

```bash
git clone https://github.com/ritsumei-aoi/osp-triviality.git
cd osp-triviality
pip install -e ".[test]"
```

## Project Structure

```
osp-triviality/
├── src/oscillator_lie_superalgebras/   # Core library
│   ├── oscillator_algebra.py           # PBW rewriter engine
│   ├── B_generators.py                 # B(0,n) basis (Frappat notation)
│   ├── adjoint_from_json.py            # Adjoint representation matrices
│   ├── cohomology_solver.py            # Cohomology equation solver
│   ├── gamma_from_B.py                 # 2-cocycle computation
│   ├── triviality_checker.py           # High-level triviality API
│   └── ...                             # Parsers, builders, utilities
├── scripts/                            # Verification scripts
│   ├── check_triviality.py             # Main triviality verification
│   ├── verify_certificates.py          # Algebraic certificate verification
│   ├── verify_dim_formula.py           # dim C^1_μ = 6n-2 verification
│   ├── generate_structure.py           # Generate algebra structure JSON
│   └── generate_gamma.py              # Generate gamma structure JSON
├── data/                               # Pre-computed algebra data
│   ├── algebra_structures/             # Algebra structure constants (JSON)
│   └── gamma_structures/               # Gamma cocycle structures (JSON)
├── tests/                              # Test suite
└── docs/                               # Documentation
    ├── theory.md                       # Mathematical background
    └── schemas.md                      # JSON data schema specification
```

## Mathematical Background

For a Lie superalgebra **g** = B(0,n) = osp(1|2n), a first-order deformation
is determined by a 2-cocycle γ ∈ Z²(g, g). The deformation is *trivial* if
γ is a coboundary, i.e., there exists an odd linear map f: g → g such that:

```
γ(a,b) = (-1)^{p(a)} [a, f(b)] - (-1)^{(p(a)+1)p(b)} [b, f(a)] - f([a,b])
```

This code verifies triviality by:
1. Computing γ from the deformation parameters (B matrix)
2. Building adjoint representation matrices
3. Solving the cohomology equation for f via least-squares
4. Checking that the residual norm is below numerical tolerance

See [docs/theory.md](docs/theory.md) for details.

## Running Tests

```bash
pytest tests/ -v
```

## Verification Scripts

### Check triviality for B(0,n)

```bash
# Verify triviality for B(0,2) = osp(1|4)
python scripts/check_triviality.py --n 2

# Verify triviality for B(0,3) = osp(1|6)
python scripts/check_triviality.py --n 3
```

### Verify dimension formula

```bash
# Verify dim C^1_μ = 6n - 2 for n = 1, 2, 3
python scripts/verify_dim_formula.py
```

### Verify algebraic certificates

```bash
# Verify c^T A = 0 for all certificate families
python scripts/verify_certificates.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{osp-triviality,
  title={Computational verification code for triviality of osp(1|2n) deformations},
  author={ritsumei-aoi},
  year={2026},
  url={https://github.com/ritsumei-aoi/osp-triviality}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.