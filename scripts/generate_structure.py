"""
generate_B_structure_json.py

Generate JSON structure constants for B(m,n) = osp(2m+1|2n)
using the new oscillator algebra engine (oscillator_algebra.py).

Output directory: data/algebra_structures

Usage:
    python experiments/generate_B_structure_json.py
    python experiments/generate_B_structure_json.py --m 0 --n 1
    python experiments/generate_B_structure_json.py --m 1 --n 1
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict

# Ensure src and experiments are in path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))

from oscillator_lie_superalgebras.B_structure_builder import (
    build_algebra_section,
    build_oscillator_generators_section,
    build_oscillator_relations_section,
    build_basis_section,
    build_parity_section,
    build_generator_realization_section,
    build_metadata_section,
)


from oscillator_lie_superalgebras.oscillator_algebra import BmnAlgebra


# ---------------------------------------------------------------------------
# structure_constants section (new engine)
# ---------------------------------------------------------------------------

def build_structure_constants_section_v5(m: int, n: int) -> Dict[str, Any]:
    """
    Compute structure constants via BmnAlgebra (oscillator engine)
    and format them in v5.0 schema format.
    """
    alg = BmnAlgebra(m, n)
    raw_sc = alg.compute_structure_constants()

    brackets = {
        f"{g1},{g2}": res
        for (g1, g2), res in raw_sc.items()
    }

    note = (
        "Non-zero brackets [X, Y] = sum_Z c_Z Z. "
        "For odd-odd pairs: anticommutator {X,Y}. Otherwise: commutator [X,Y]. "
        "Coefficients follow implementation convention "
        "(E_{±δ_k} = a_0 b_k^±, E_{±ε_i} = ∓sqrt(2)*a_0*a_i^±); see metadata. "
        "Computed via oscillator algebra engine (oscillator_algebra.py)."
    )
    return {"description": note, "brackets": brackets}


# ---------------------------------------------------------------------------
# Top-level assembler
# ---------------------------------------------------------------------------

def build_json(m: int, n: int) -> Dict[str, Any]:
    print(f"\n[B({m},{n})] Building section builders...")
    algebra      = build_algebra_section(m, n)
    osc_gens     = build_oscillator_generators_section(m, n)
    osc_rels     = build_oscillator_relations_section(m, n)
    basis        = build_basis_section(m, n)
    parity       = build_parity_section(m, n)
    gen_real     = build_generator_realization_section(m, n)

    # metadata: reuse v5.0 but update generated_by
    metadata = build_metadata_section(m, n)
    metadata["generated_by"] = "experiments/generate_B_structure_json.py"
    metadata["computation_method"] = (
        "oscillator_algebra_engine (oscillator_algebra.py)"
    )

    print(f"[B({m},{n})] Computing structure constants via oscillator engine...")
    sc = build_structure_constants_section_v5(m, n)
    print(f"  non-zero bracket pairs: {len(sc['brackets'])}")

    return {
        "schema_version": "5.0",
        "algebra":                algebra,
        "oscillator_generators":  osc_gens,
        "oscillator_relations":   osc_rels,
        "basis":                  basis,
        "parity":                 parity,
        "generator_realization":  gen_real,
        "structure_constants":    sc,
        "metadata":               metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_CASES = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]


def main():
    parser = argparse.ArgumentParser(
        description="Generate B(m,n) structure constant JSON (v5.0 engine)"
    )
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    cases = [(args.m, args.n)] if (args.m is not None and args.n is not None) \
            else DEFAULT_CASES

    out_dir = root / "data" / "algebra_structures"

    out_dir.mkdir(parents=True, exist_ok=True)

    for m, n in cases:
        print(f"\n{'='*50}")
        print(f"Processing B({m},{n}) = osp({2*m+1}|{2*n})")
        print(f"{'='*50}")
        try:
            data = build_json(m, n)
            fname = out_dir / f"B_{m}_{n}_structure.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n  -> Saved: {fname}")
        except Exception as e:
            import traceback
            print(f"\n  ERROR in B({m},{n}): {e}")
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
