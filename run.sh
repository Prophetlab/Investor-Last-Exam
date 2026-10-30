#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p local
if [[ ! -x local/.venv/bin/python ]]; then
  python_bin=${ILE_PYTHON:-python3}
  "$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required; set ILE_PYTHON to its executable"'
  "$python_bin" -m venv local/.venv
fi
fingerprint=$(sha256sum requirements.lock pyproject.toml | sha256sum | cut -d' ' -f1)
if [[ ! -f local/.installed ]] || [[ "$(cat local/.installed)" != "$fingerprint" ]]; then
  local/.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.lock
  local/.venv/bin/python -m pip install --no-deps -e .
  printf '%s\n' "$fingerprint" > local/.installed
fi
if [[ $# -eq 0 ]]; then set -- check --host --config; fi
exec local/.venv/bin/python run.py "$@"
