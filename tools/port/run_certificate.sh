#!/bin/bash
# The port certificate check. Usage, from anywhere, after `lake build` at the repository root: tools/port/run_certificate.sh
# 1. PortCertificate.lean must compile against the port, each cert_* with the standard axioms only.
# 2. Known-bad: the same file with hval1's constant 2 changed to 3 must FAIL to compile.
set -u
H=$(cd "$(dirname "$0")" && pwd); T=$(cd "$H/../.." && pwd); W=$(mktemp -d)
cp "$H/PortCertificate.lean" "$W/good.lean"
sed 's/then (2 : ℚ) else 0 :=/then (3 : ℚ) else 0 :=/' "$H/PortCertificate.lean" > "$W/bad.lean"
cmp -s "$W/good.lean" "$W/bad.lean" && { echo "known-bad was not generated"; exit 2; }
cd "$T"
lake env lean "$W/good.lean" > "$W/good.out" 2>&1; g=$?
lake env lean "$W/bad.lean"  > "$W/bad.out"  2>&1; b=$?
cat "$W/good.out"
echo "good: exit $g; errors $(grep -c 'error' "$W/good.out")"
tr '\n' ' ' < "$W/good.out" | sed 's/  */ /g' | grep -o "'[^']*' depends on axioms: \[[^]]*\]" > "$W/ax.txt"
echo "axiom entries: $(wc -l < "$W/ax.txt" | tr -d ' ') of 7; non-standard: $(grep -v -c -E 'axioms: \[(propext(, Classical\.choice)?(, Quot\.sound)?|propext, Quot\.sound)\]$' "$W/ax.txt")"
echo "known-bad: exit $b (must be nonzero); $(grep -m1 'error' "$W/bad.out")"
[ $g -eq 0 ] && [ $b -ne 0 ] && echo "CERTIFICATE PASS" || echo "CERTIFICATE FAIL"
