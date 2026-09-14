#!/usr/bin/env bash
# The whole check, in the order it runs everywhere.
#
# .github/workflows/ci.yml calls this script rather than repeating the steps, so
# a green run here is a green run there — there is no second list to drift.

set -uo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH. Install it, then re-run:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

PYVER="${UV_PYTHON:-$(cat .python-version 2>/dev/null || echo unknown)}"

pass=0
fail=0

step () {
  local label=$1
  shift
  printf '%-44s' "$label"
  local out
  if out=$("$@" 2>&1); then
    printf 'PASS\n'
    pass=$((pass + 1))
  else
    printf 'FAIL\n'
    printf '%s\n' "$out" | tail -30 | sed 's/^/    /'
    fail=$((fail + 1))
  fi
}

# The promise in the README is that `pip install openodke` needs no provider and no
# driver. That is only true if it is checked: this imports the package with the
# base dependencies alone, in a throwaway environment.
base_install_is_self_sufficient () {
  uv run --isolated --no-project --with . python -c "import openodke; openodke.Ontology(name='x').snippet('T')"
}

# The wheel is `openodke`; it installs both commands, the short `odke` and
# `openodke` (DECISIONS #22). Each must run from a clean environment and report
# the distribution name, and the import package must be `openodke` alone.
smoke_wheel () {
  local wheel out
  wheel=$(ls -t dist/openodke-*.whl 2>/dev/null | head -1) || return 1
  [ -n "$wheel" ] || return 1
  for cmd in odke openodke; do
    out=$(uv run --isolated --no-project --with "$wheel" "$cmd" --version) || return 1
    case "$out" in
      "openodke "*) ;;
      *) echo "$cmd --version printed: $out" >&2; return 1 ;;
    esac
  done
  uv run --isolated --no-project --with "$wheel" python -c "import openodke"
}

step "1. no live credentials present"        python3 scripts/assert_no_credentials.py
step "2. interpreter ${PYVER}"               uv python install
step "3. dependencies match the lock"        uv sync --locked
step "4. lint"                               uv run ruff check .
step "5. format"                             uv run ruff format --check .
step "6. types"                              uv run mypy
step "7. tests"                              uv run pytest
step "8. base install needs no extras"       base_install_is_self_sufficient
step "9. distribution builds"                uv build
step "10. wheel runs from a clean env"       smoke_wheel

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
