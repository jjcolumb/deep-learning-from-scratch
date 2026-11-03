#!/usr/bin/env bash
# Runs every sample in every language it has and compares the output with the
# numbers quoted in its post (samples/*/expected.txt).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
fail=0
for dir in samples/*/; do
  [ -f "$dir/expected.txt" ] || continue
  for lang in csharp python; do
    [ -d "$dir/$lang" ] || continue
    if [ "$lang" = csharp ]; then out=$(cd "$dir/csharp" && dotnet run -c Release -v q 2>&1)
    else out=$(cd "$dir/python" && "$PY" main.py 2>&1); fi
    if diff <(echo "$out") "$dir/expected.txt" > /dev/null; then echo "ok    $dir$lang"
    else echo "FAIL  $dir$lang"; diff <(echo "$out") "$dir/expected.txt" || true; fail=1; fi
  done
done
exit $fail
