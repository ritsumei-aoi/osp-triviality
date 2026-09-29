#!/usr/bin/env python3
"""T5-T7 fixture extractor (stdlib only).

Reads ONLY the protected fixture `fixture/s_model_n1.json`, checks its
accepted identity (2720 bytes, the declared SHA256), and emits
`InhomogeneousDeformations/FixtureData.lean`: a literal `InhomogeneousDeformations.Decode.RawInput`
value transcribing every projected field, row, coefficient, exponent and
explicit zero of the raw structure, in the raw structure's own order.

The ONE declared transformation is the JSON string-valued
numerator/denominator -> typed Lean integer literal conversion; every
occurrence is reported below, field by field. No other transformation is
performed: no sorting, no merging, no normalization, no computed bracket
value, no assumed-success marker, no call to any other validator.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixture" / "s_model_n1.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "InhomogeneousDeformations" / "FixtureData.lean"

EXPECTED_BYTES = 2720
EXPECTED_SHA256 = "ec640c70f55a41a53529ecfe1a5facc934fb41974e568a1856d8267671317de8"


def lean_str(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def lean_str_list(xs) -> str:
    return "[" + ", ".join(lean_str(x) for x in xs) + "]"


def lean_nat_list(xs) -> str:
    return "[" + ", ".join(str(int(x)) for x in xs) + "]"


def lean_rat(term_coeff: dict, field_report: list, row_desc: str) -> str:
    """`{"numerator": "<str>", "denominator": "<str>"}` -> `Wire.Rat`
    literal. Declared string-to-typed-Int conversion; reported once per
    occurrence."""
    num_s = term_coeff["numerator"]
    den_s = term_coeff["denominator"]
    num_i = int(num_s)
    den_i = int(den_s)
    field_report.append(
        f"{row_desc}: numerator {num_s!r} -> {num_i}, denominator {den_s!r} -> {den_i}"
    )
    return f"{{ num := {num_i}, den := {den_i} : InhomogeneousDeformations.Wire.Rat }}"


def lean_poly_term(term: dict, field_report: list, row_desc: str) -> str:
    coeff = lean_rat(term["coefficient"], field_report, row_desc)
    exps = lean_nat_list(term["exponents"])
    return f"{{ coeff := {coeff}, exponents := {exps} : InhomogeneousDeformations.Wire.PolyTerm }}"


def lean_poly(terms: list, field_report: list, row_desc: str) -> str:
    return "[" + ", ".join(
        lean_poly_term(t, field_report, f"{row_desc} term#{i}") for i, t in enumerate(terms)
    ) + "]"


def lean_vector_entry(entry: dict, field_report: list, row_desc: str) -> str:
    basis = lean_str(entry["basis"])
    poly = lean_poly(entry["coefficient"], field_report, f"{row_desc} basis={entry['basis']}")
    return f"{{ basisId := {basis}, coeff := {poly} : InhomogeneousDeformations.Wire.VectorEntry }}"


def lean_vector(output: list, field_report: list, row_desc: str) -> str:
    return "[" + ", ".join(
        lean_vector_entry(e, field_report, row_desc) for e in output
    ) + "]"


def lean_row(entry: dict, idx: int, field_report: list) -> str:
    row_desc = f"row#{idx} {entry['inputs']}"
    inputs = lean_str_list(entry["inputs"])
    output = lean_vector(entry["output"], field_report, row_desc)
    return f"{{ inputs := {inputs}, output := {output} : InhomogeneousDeformations.Wire.Row }}"


LINE_LIMIT = 100
_CLOSERS = {"[": "]", "{": "}", "(": ")"}


def _parse_group(t: str, i: int):
    """Parse the bracket group opening at t[i]; return ((open, close, items), next index)."""
    op = t[i]
    cl = _CLOSERS[op]
    items, cur = [], []
    i += 1
    while True:
        ch = t[i]
        if ch == '"':
            j = i + 1
            while t[j] != '"':
                j += 1
            cur.append(t[i:j + 1])
            i = j + 1
            continue
        if ch in _CLOSERS:
            g, i = _parse_group(t, i)
            cur.append(g)
            continue
        if ch == cl:
            items.append(cur)
            if op == "{":
                if cur and isinstance(items[0][0], str) and items[0][0].startswith(" "):
                    items[0][0] = items[0][0][1:]
                if cur and isinstance(cur[-1], str) and cur[-1].endswith(" "):
                    cur[-1] = cur[-1][:-1]
            return (op, cl, items), i + 1
        if ch == "," and t[i + 1] == " ":
            items.append(cur)
            cur = []
            i += 2
            continue
        cur.append(ch)
        i += 1


def _sequence(t: str):
    out, i, buf = [], 0, ""
    while i < len(t):
        ch = t[i]
        if ch == '"':
            j = i + 1
            while t[j] != '"':
                j += 1
            buf += t[i:j + 1]
            i = j + 1
            continue
        if ch in _CLOSERS:
            if buf:
                out.append(buf)
                buf = ""
            g, i = _parse_group(t, i)
            out.append(g)
            continue
        buf += ch
        i += 1
    if buf:
        out.append(buf)
    return out


def _flat(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "".join(_flat(y) for y in x)
    op, cl, items = x
    inner = ", ".join("".join(_flat(y) for y in it) for it in items)
    return op + " " + inner + " " + cl if op == "{" else op + inner + cl


def _render_seq(nodes, col):
    lines, c = [""], col
    for x in nodes:
        if isinstance(x, str):
            lines[-1] += x
            c += len(x)
        else:
            f = _flat(x)
            if c + len(f) <= LINE_LIMIT:
                lines[-1] += f
                c += len(f)
            else:
                sub = _render_group(x, c)
                lines[-1] += sub[0]
                lines.extend(sub[1:])
                c = len(lines[-1]) if len(lines) > 1 else c
    return lines


def _render_group(g, col):
    op, cl, items = g
    ind = col + 2 if op == "{" else col + 1
    head = op + " " if op == "{" else op
    out = []
    for k, it in enumerate(items):
        sub = _render_seq(it, ind)
        sub[0] = (head if k == 0 else " " * ind) + sub[0]
        if k < len(items) - 1:
            sub[-1] += ","
        else:
            sub[-1] += " " + cl if op == "{" else cl
        out.extend(sub)
    return out


def wrap_long_lines(text: str) -> str:
    """Re-wrap every line over LINE_LIMIT along its bracket structure (Lean is whitespace-
    insensitive here), so that the generated file has 0 style-linter warnings; then break
    what is left at ` : ` (type ascriptions). Only whitespace changes."""
    res = []
    for line in text.split("\n"):
        if len(line) > LINE_LIMIT:
            ind = len(line) - len(line.lstrip())
            ls = _render_seq(_sequence(line[ind:]), ind)
            res.append(" " * ind + ls[0])
            res.extend(ls[1:])
        else:
            res.append(line)
    out = []
    for line in res:
        if len(line) <= LINE_LIMIT:
            out.append(line)
            continue
        ind = len(line) - len(line.lstrip())
        parts = line.split(" : ")
        cur = parts[0]
        for q in parts[1:]:
            piece = ": " + q
            if len(cur) + 1 + len(piece) > LINE_LIMIT:
                out.append(cur)
                cur = " " * (ind + 2) + piece
            else:
                cur += " " + piece
        out.append(cur)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--report", metavar="PATH", help="write the field-by-field report here (default: print it to stdout)")
    args = ap.parse_args()

    raw_bytes = FIXTURE_PATH.read_bytes()
    actual_len = len(raw_bytes)
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_len != EXPECTED_BYTES or actual_sha256 != EXPECTED_SHA256:
        print(
            f"REFUSED: fixture identity mismatch: bytes={actual_len} "
            f"(expected {EXPECTED_BYTES}), sha256={actual_sha256} "
            f"(expected {EXPECTED_SHA256})",
            file=sys.stderr,
        )
        return 1

    data = json.loads(raw_bytes.decode("utf-8"))

    field_report: list = []

    schema_version = data["schema_version"]

    coeff_domains = {d["id"]: d for d in data["coefficient_domains"]}
    q_id = "Q"
    p_dom = coeff_domains["P"]
    p_id = "P"
    p_base = p_dom["base"]
    p_variables = list(p_dom["variables"])

    module = data["modules"][0]
    module_id = module["id"]
    module_coeff_domain = module["coefficient_domain"]
    module_grading = module["grading"]
    basis_ids = [b["id"] for b in module["basis"]]
    basis_degrees = [b["degree"] for b in module["basis"]]

    op = data["operations"][0]
    op_id = op["id"]
    op_kind = op["kind"]
    op_module = op["module"]
    op_degree = op["degree"]
    op_scalar_behavior = op["scalar_behavior"]
    op_def = op["definition"]
    def_kind = op_def["kind"]
    coverage = op_def["coverage"]
    entries = op_def["entries"]

    sm = data["s_model"]
    sm_bracket = sm["bracket"]
    sm_module = sm["module"]
    sm_normalization = sm["normalization"]
    sm_rank = sm["rank"]
    sm_roles = list(sm["roles"].items())

    excluded_fields = ["claims", "maps", "provenance", "required_profiles", "model", "kind"]
    field_report.append(
        "excluded from RawInput (out of scope per T5's V-table non-goals): "
        + ", ".join(excluded_fields)
    )

    rows_lean = [lean_row(e, i, field_report) for i, e in enumerate(entries)]

    zero_rows = [e["inputs"] for e in entries if e["output"] == []]
    field_report.append(f"explicit zero rows preserved ({len(zero_rows)}): {zero_rows}")
    field_report.append(f"total rows preserved: {len(entries)}")

    roles_lean = ", ".join(f"({lean_str(k)}, {lean_str(v)})" for k, v in sm_roles)

    lean_source = f"""import InhomogeneousDeformations.Decode

/-!
# T5 — generated fixture datum

GENERATED by `tools/extract_fixture.py` from the protected fixture
`fixture/s_model_n1.json` (2720 bytes, SHA256
`ec640c70f55a41a53529ecfe1a5facc934fb41974e568a1856d8267671317de8`).
DO NOT EDIT BY HAND; regenerate via the script if the (never-touched)
fixture identity ever legitimately changes. Every projected field, row,
coefficient, exponent and explicit zero of the raw structure is preserved
in the raw structure's own order; the ONE transformation performed is the
declared string-numerator/denominator -> typed `ℤ` conversion (see the
extractor's own field-by-field report, printed by the script or written with `--report PATH`).
-/

namespace InhomogeneousDeformations

def fixtureRawInput : Decode.RawInput where
  schemaVersion := {lean_str(schema_version)}
  qId := {lean_str(q_id)}
  pId := {lean_str(p_id)}
  pBase := {lean_str(p_base)}
  pVariables := {lean_str_list(p_variables)}
  moduleId := {lean_str(module_id)}
  moduleCoeffDomain := {lean_str(module_coeff_domain)}
  moduleGrading := {lean_str(module_grading)}
  basisIds := {lean_str_list(basis_ids)}
  basisDegrees := {lean_nat_list(basis_degrees)}
  opId := {lean_str(op_id)}
  opKind := {lean_str(op_kind)}
  opModule := {lean_str(op_module)}
  opDegree := {op_degree}
  opScalarBehavior := {lean_str(op_scalar_behavior)}
  defKind := {lean_str(def_kind)}
  coverage := {lean_str(coverage)}
  rows := [{", ".join(rows_lean)}]
  smBracket := {lean_str(sm_bracket)}
  smModule := {lean_str(sm_module)}
  smNormalization := {lean_str(sm_normalization)}
  smRank := {sm_rank}
  smRoles := [{roles_lean}]

end InhomogeneousDeformations
"""

    OUT_PATH.write_text(wrap_long_lines(lean_source), encoding="utf-8")
    report = "\n".join(field_report) + "\n"
    print(f"OK: wrote {OUT_PATH} ({len(entries)} rows, {len(zero_rows)} zero rows)")
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"OK: field report at {args.report} ({len(field_report)} lines)")
    else:
        print("field report:")
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
