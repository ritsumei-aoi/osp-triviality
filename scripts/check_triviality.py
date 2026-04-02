#!/usr/bin/env python3
"""
Check triviality of inhomogeneous deformation from Schema 3 (v5.0) JSON.

Usage:
    python experiments/check_triviality_from_json.py --m 0 --n 1 --gh "[[-1, 0]]"
    python experiments/check_triviality_from_json.py --m 0 --n 1  # gh=0 check
    python experiments/check_triviality_from_json.py --m 0 --n 1 --gh "[[-1, 0]]" --tol 1e-6
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oscillator_lie_superalgebras.evaluated_structure_parser import EvaluatedStructureParser
from oscillator_lie_superalgebras.evaluated_structure_builder import EvaluatedStructureBuilder
from oscillator_lie_superalgebras.adjoint_from_json import build_adjoint_from_json_v41
from oscillator_lie_superalgebras.cohomology_solver import solve_odd_f_generic


TRIVIALITY_RESULTS_CSV = PROJECT_ROOT / "data" / "triviality_results.csv"


def _resolve_json_path(m: int, n: int, gh_flat: str, data_dir: Path) -> Path:
    """
    Resolve Schema 3 JSON path from (m, n, gh_flat).
    gh_flat: e.g. "m1_0" from --gh "[[-1, 0]]", or "0_0" for zero matrix.
    """
    fname = f"B_{m}_{n}_gh_{gh_flat}.json"
    return data_dir / fname


def _gh_to_flat(gh_list: list) -> str:
    """
    Convert gh matrix (nested list) to flat filename encoding.
    Rules: negative -> 'm' prefix, decimal point -> 'p', sqrt(2) -> 'r2'.
    Example: [[-1, 0]] -> "m1_0"
    """
    import math
    tokens = []
    for row in gh_list:
        for v in row:
            v = float(v)
            if abs(v - math.sqrt(2)) < 1e-10:
                tokens.append("r2")
            elif abs(v + math.sqrt(2)) < 1e-10:
                tokens.append("mr2")
            elif v < 0:
                abs_v = abs(v)
                s = f"{abs_v:g}"
                tokens.append("m" + s.replace(".", "p"))
            else:
                s = f"{v:g}"
                tokens.append(s.replace(".", "p"))
    return "_".join(tokens)


def _build_schema_paths(m: int, n: int, evaluated_data_dir: Path) -> tuple[Path, Path]:
    """
    Resolve Schema 1/2 JSON paths needed to generate Schema 3.
    If evaluated_data_dir ends with evaluated_structures, use its parent as data root.
    Otherwise fallback to PROJECT_ROOT/data.
    """
    data_root = (
        evaluated_data_dir.parent
        if evaluated_data_dir.name == "evaluated_structures"
        else PROJECT_ROOT / "data"
    )
    structure_path = data_root / "algebra_structures" / f"B_{m}_{n}_structure.json"
    gamma_path = data_root / "gamma_structures" / f"B_{m}_{n}_gamma.json"
    return structure_path, gamma_path


def _generate_missing_json(m: int, n: int, gh_list: list, evaluated_data_dir: Path) -> Path:
    """Generate missing Schema 3 JSON and return output path."""
    structure_path, gamma_path = _build_schema_paths(m, n, evaluated_data_dir)

    missing = [p for p in (structure_path, gamma_path) if not p.exists()]
    if missing:
        missing_str = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Schema 3 の自動生成に必要な入力ファイルが見つかりません:\n"
            f"{missing_str}"
        )

    builder = EvaluatedStructureBuilder(structure_path, gamma_path)
    return builder.save(
        gh=gh_list,
        output_dir=evaluated_data_dir,
        run_zero_check=True,
        generated_by="experiments/check_triviality_from_json.py",
    )


def _format_f_map(f_map: dict, tol: float = 1e-9) -> str:
    lines = []
    for src, dst_dict in f_map.items():
        if not dst_dict:
            continue
        terms = []
        for dst, coeff in dst_dict.items():
            if abs(coeff) > tol:
                if abs(coeff.imag) < tol:
                    terms.append(f"{coeff.real:.6g} {dst}")
                else:
                    terms.append(f"({coeff.real:.6g}+{coeff.imag:.6g}j) {dst}")
        if terms:
            lines.append(f"  f({src}) = {' + '.join(terms)}")
    return "\n".join(lines) if lines else "  f = 0 (zero map)"


def _serialize_f_map_for_csv(f_map: dict, tol: float = 1e-9) -> str:
    """Serialize odd map f into JSON string (only non-zero terms)."""
    serializable = {}
    for src, dst_dict in f_map.items():
        if not dst_dict:
            continue
        terms = {}
        for dst, coeff in dst_dict.items():
            if abs(coeff) <= tol:
                continue
            if abs(coeff.imag) < tol:
                terms[dst] = float(coeff.real)
            else:
                terms[dst] = {
                    "re": float(coeff.real),
                    "im": float(coeff.imag),
                }
        if terms:
            serializable[src] = terms
    return json.dumps(serializable, ensure_ascii=False, sort_keys=True)


def _append_triviality_result(
    csv_path: Path,
    json_path: Path,
    m: int,
    n: int,
    gh_list: list,
    gh_flat: str,
    tol: float,
    schema_version: str,
    basis: list,
    nonzero_gamma_pairs: int,
    residual_norm: float,
    is_trivial: bool,
    f_map: dict,
) -> bool:
    """
    Append result row to CSV iff the primary key (json filename) is new.
    Returns True when appended, False when skipped due to duplicate key.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "json_filename",
        "json_relative_path",
        "m",
        "n",
        "gh",
        "gh_flat",
        "schema_version",
        "basis_size",
        "nonzero_gamma_pairs",
        "tol",
        "residual_norm",
        "is_trivial",
        "odd_f_map_json",
        "logged_at",
    ]

    key = json_path.name
    existing_keys = set()
    needs_header = True

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                needs_header = False
                for row in reader:
                    row_key = row.get("json_filename")
                    if row_key:
                        existing_keys.add(row_key)

    if key in existing_keys:
        return False

    row = {
        "json_filename": key,
        "json_relative_path": str(json_path.relative_to(PROJECT_ROOT)),
        "m": m,
        "n": n,
        "gh": json.dumps(gh_list, ensure_ascii=False),
        "gh_flat": gh_flat,
        "schema_version": schema_version,
        "basis_size": len(basis),
        "nonzero_gamma_pairs": nonzero_gamma_pairs,
        "tol": f"{tol:.1e}",
        "residual_norm": f"{residual_norm:.6e}",
        "is_trivial": "true" if is_trivial else "false",
        "odd_f_map_json": _serialize_f_map_for_csv(f_map, tol=tol) if is_trivial else "",
        "logged_at": datetime.now().isoformat(timespec="seconds"),
    }

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check triviality of B(m,n) inhomogeneous deformation (Schema v5.0)."
    )
    parser.add_argument("--m", type=int, required=True, help="Fermionic pairs")
    parser.add_argument("--n", type=int, required=True, help="Bosonic pairs")
    parser.add_argument(
        "--gh",
        type=str,
        default=None,
        help='gh matrix as nested list, e.g. "[[-1, 0]]". Omit for zero matrix.',
    )
    parser.add_argument("--tol", type=float, default=1e-9, help="Numerical tolerance")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to evaluated_structures directory (default: data/evaluated_structures)",
    )
    args = parser.parse_args()

    m, n, tol = args.m, args.n, args.tol
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "evaluated_structures"

    # --- Parse gh ---
    if args.gh is not None:
        try:
            gh_list = json.loads(args.gh)
        except json.JSONDecodeError as e:
            print(f"[ERROR] --gh のパースに失敗しました: {e}", file=sys.stderr)
            sys.exit(1)
        gh_flat = _gh_to_flat(gh_list)
    else:
        # Zero matrix: shape [2m+1, 2n] of zeros
        rows = 2 * m + 1
        cols = 2 * n
        gh_list = [[0] * cols for _ in range(rows)]
        gh_flat = "_".join(["0"] * (rows * cols))

    # --- Resolve JSON path ---
    json_path = _resolve_json_path(m, n, gh_flat, data_dir)
    if not json_path.exists():
        print(f"[INFO] Schema 3 JSON が見つからないため自動生成します: {json_path}")
        try:
            json_path = _generate_missing_json(m, n, gh_list, data_dir)
            print(f"[INFO] 生成完了: {json_path.relative_to(PROJECT_ROOT)}")
        except Exception as e:
            print(f"[ERROR] Schema 3 JSON の自動生成に失敗しました: {e}", file=sys.stderr)
            sys.exit(1)

    # --- Step 1: Extract gamma ---
    print(f"Loading: {json_path.relative_to(PROJECT_ROOT)}")
    parser_obj = EvaluatedStructureParser(json_path)
    gamma = parser_obj.extract_gamma(tol=tol)

    # --- Step 2: Build adjoint matrices ---
    try:
        basis, parity, adjoint_matrices = build_adjoint_from_json_v41(m, n)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # --- Step 3: Solve for f ---
    f_map, residual_norm = solve_odd_f_generic(
        basis=basis,
        parity=parity,
        adjoint_matrices=adjoint_matrices,
        gamma=gamma,
        tol=tol,
    )

    # --- Step 4: Report ---
    is_trivial = residual_norm < tol

    print("=" * 60)
    print(f"Triviality Check: B({m},{n}),  gh = {gh_list}")
    print("=" * 60)
    print(f"  Schema version : {parser_obj.schema_version}")
    print(f"  Basis          : {basis}")
    print(f"  Non-zero γ pairs: {len(gamma)}")
    print(f"  Tolerance      : {tol:.1e}")
    print(f"  Residual norm  : {residual_norm:.4e}")
    print()
    print(f"  Result: {'✅ TRIVIAL' if is_trivial else '❌ NON-TRIVIAL'}")
    if is_trivial:
        print()
        print("  Odd linear map f:")
        print(_format_f_map(f_map, tol=tol))
    print("=" * 60)

    # --- Step 5: Persist result CSV ---
    appended = _append_triviality_result(
        csv_path=TRIVIALITY_RESULTS_CSV,
        json_path=json_path,
        m=m,
        n=n,
        gh_list=gh_list,
        gh_flat=gh_flat,
        tol=tol,
        schema_version=parser_obj.schema_version,
        basis=basis,
        nonzero_gamma_pairs=len(gamma),
        residual_norm=residual_norm,
        is_trivial=is_trivial,
        f_map=f_map,
    )
    if appended:
        print(f"[INFO] 判定結果を追記しました: {TRIVIALITY_RESULTS_CSV.relative_to(PROJECT_ROOT)}")
    else:
        print(f"[INFO] 既存レコードのため追記をスキップしました: {json_path.name}")


if __name__ == "__main__":
    main()