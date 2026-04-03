# JSON Data Schema Specification

This document describes the JSON schemas used for storing algebra
structure constants and gamma (cocycle) structures.

## Overview

Two schema families are used:

| Schema | Version | Naming Convention | Purpose |
|--------|---------|-------------------|---------|
| Algebra Structure | v5.0 | `B_{m}_{n}_structure.json` | Extended format with metadata and relations |
| Gamma Structure | v5.0 | `B_{m}_{n}_gamma.json` | Extended gamma with exchange relations |

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

Extended format including exchange relations and gb parameters.

### Exchange Relations

```json
{
  "exchange_relations": {
    "convention": "[b_j, a_i] = -gb[i][j]·κ",
    "parameters": {
      "gb_a0_b1p": "gb[0][0]",
      "gb_a0_b1m": "gb[0][1]",
      ...
    }
  }
}
```

## Data Files Included

### algebra_structures/

| File | Algebra | Schema | Dimension |
|------|---------|--------|-----------|
| `B_0_1_structure.json` | B(0,1) = osp(1\|2) | v5.0 | 4 |
| `B_0_2_structure.json` | B(0,2) = osp(1\|4) | v5.0 | 14 |
| `B_0_3_structure.json` | B(0,3) = osp(1\|6) | v5.0 | 28 |

### gamma_structures/

| File | Algebra | Schema | Parameters |
|------|---------|--------|------------|
| `B_0_1_gamma.json` | B(0,1) | v5.0 | 2 (gb matrix) |
| `B_0_2_gamma.json` | B(0,2) | v5.0 | 4 (gb matrix) |
| `B_0_3_gamma.json` | B(0,3) | v5.0 | 6 (gb matrix) |

Structures for other values of n can be regenerated using the scripts:

```bash
python scripts/generate_structure.py
python scripts/generate_gamma.py
```
