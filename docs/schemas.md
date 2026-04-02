# JSON Data Schema Specification

This document describes the JSON schemas used for storing algebra
structure constants and gamma (cocycle) structures.

## Overview

Two schema families are used:

| Schema | Version | Naming Convention | Purpose |
|--------|---------|-------------------|---------|
| Algebra Structure | v3.0 | `osp_1_{2n}_structure.json` | Basis, parity, brackets, oscillator realizations |
| Algebra Structure | v5.0 | `B_{m}_{n}_structure.json` | Extended format with metadata and relations |
| Gamma Structure | v3.0 | `osp_1_{2n}_gamma.json` | Symbolic cocycle expressions |
| Gamma Structure | v5.0 | `B_{m}_{n}_gamma.json` | Extended gamma with exchange relations |

## Schema v3.0: Algebra Structure

Used by the core library (`algebra_parser.py`, `adjoint_from_json.py`).

### Top-Level Fields

```json
{
  "schema_version": "3.0",
  "algebra": {
    "name": "osp(1|4)",
    "type": "B",
    "rank": [0, 2]
  },
  "basis": {
    "even": ["H_1", "H_2", "E_2d1_p", ...],
    "odd": ["E_d1_p", "E_d1_m", ...]
  },
  "parity": {
    "H_1": 0,
    "E_d1_p": 1,
    ...
  },
  "structure_constants": {
    "H_1": {
      "E_d1_p": {"E_d1_p": 1.0},
      ...
    },
    ...
  },
  "ordering": "fermionic-first: a, b_0, b_1, ...",
  "generator_realization": {
    "H_1": {"terms": [{"coeff": "1", "osc": "b_1_p*b_1_m"}]},
    ...
  }
}
```

### Structure Constants Format

The `structure_constants` section encodes the Lie superbracket [x, y] as:

```
structure_constants[x][y] = {basis_element: coefficient, ...}
```

where `[x, y] = Σ coefficient · basis_element`.

Conventions:
- Only non-zero brackets are stored
- Coefficients may be integers, floats, or strings like `"sqrt(2)/2"`
- The bracket is **not** stored symmetrically; [x,y] and [y,x] may both appear

## Schema v3.0: Gamma Structure

Used by `gamma_from_B.py` and `gamma_parser.py`.

### Top-Level Fields

```json
{
  "schema_version": "3.0",
  "algebra_reference": {
    "name": "osp(1|4)",
    "structure_file": "osp_1_4_structure.json"
  },
  "inhomogeneous_deformation": {
    "parameters": ["B_1", "B_2", "B_3", "B_4"],
    "gamma_matrix": {
      "Q_0": {
        "Q_0": {"terms": [{"coeff": "2*sqrt(2)", "params": ["B_1"]}]},
        ...
      }
    }
  }
}
```

### Gamma Matrix Format

The `gamma_matrix` encodes γ(x, y) symbolically in terms of the
deformation parameters B:

```
gamma_matrix[x][y] = {
  basis_element: {
    "terms": [
      {"coeff": "coefficient_expression", "params": ["B_i", ...]},
      ...
    ]
  }
}
```

The concrete numerical gamma is obtained by substituting B values.

## Schema v5.0: Algebra Structure

Extended format used by `check_triviality.py` and `evaluated_structure_parser.py`.

### Additional Sections (beyond v3.0)

```json
{
  "schema_version": "5.0",
  "algebra": {
    "family": "B",
    "m": 0,
    "n": 2,
    "name": "B(0,2) = osp(1|4)",
    "dimension": {"total": 14, "even": 10, "odd": 4}
  },
  "oscillator_generators": {
    "fermion_count": 1,
    "boson_count": 4,
    "fermion_labels": ["a_0"],
    "boson_labels": ["b_1_p", "b_1_m", "b_2_p", "b_2_m"]
  },
  "oscillator_relations": {
    "type": "CAR+CCR",
    "anticommutators": {...},
    "commutators": {...}
  },
  ...
}
```

### Key Differences from v3.0

- Explicit dimension metadata
- Oscillator generator labels and relations
- Family/m/n classification (supports B(m,n) for m ≥ 1)
- PBW ordering specification

## Schema v5.0: Gamma Structure

Extended format including exchange relations and gh parameters.

### Exchange Relations

```json
{
  "exchange_relations": {
    "convention": "[b_j, a_i] = -gh[i][j]·κ",
    "parameters": {
      "gh_a0_b1p": "gh[0][0]",
      "gh_a0_b1m": "gh[0][1]",
      ...
    }
  }
}
```

## Data Files Included

### algebra_structures/

| File | Algebra | Schema | Dimension |
|------|---------|--------|-----------|
| `osp_1_2_structure.json` | B(0,1) = osp(1\|2) | v3.0 | 4 |
| `osp_1_4_structure.json` | B(0,2) = osp(1\|4) | v3.0 | 14 |
| `osp_1_6_structure.json` | B(0,3) = osp(1\|6) | v3.0 | 28 |
| `B_0_2_structure.json` | B(0,2) = osp(1\|4) | v5.0 | 14 |
| `B_0_3_structure.json` | B(0,3) = osp(1\|6) | v5.0 | 28 |

### gamma_structures/

| File | Algebra | Schema | Parameters |
|------|---------|--------|------------|
| `osp_1_2_gamma.json` | B(0,1) | v3.0 | 2 (B₁, B₂) |
| `osp_1_4_gamma.json` | B(0,2) | v3.0 | 4 (B₁, ..., B₄) |
| `osp_1_6_gamma.json` | B(0,3) | v3.0 | 6 (B₁, ..., B₆) |
| `B_0_2_gamma.json` | B(0,2) | v5.0 | 4 (gh matrix) |
| `B_0_3_gamma.json` | B(0,3) | v5.0 | 6 (gh matrix) |

Structures for other values of n can be regenerated using the scripts:

```bash
python scripts/generate_structure.py
python scripts/generate_gamma.py
```