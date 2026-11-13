#!/usr/bin/env bash
# The check of this repo. CI runs it and skips the host-path step.
set -euo pipefail
cd "$(dirname "$0")/.."

# Report the coverage of `src/` by the lib tests. A box without the coverage
# tools skips it.
check_coverage() {
    if ! python3 scripts/coverage.py --check-tools; then
        echo "note: this box has no coverage tools, so the coverage did not run." >&2
        return 0
    fi
    python3 scripts/coverage.py .
}

if [ -z "${CI:-}" ]; then
    # This box gives the two roots, so this file names no path of a box.
    repo_root="/$(pwd | cut -d/ -f2)"
    home_root="/$(printf '%s' "$HOME" | cut -d/ -f2)"
    if git grep --untracked -n -I -E "(^|[^[:alnum:]_./-])($repo_root|$home_root)(/|\$)"; then
        echo "error: the lines above hold a host path. Write a path relative to the repo." >&2
        exit 1
    fi
fi

python3 scripts/coverage.py --selftest

cargo check --all-targets
cargo check --features examples-common
cargo test --lib -- --test-threads=1
check_coverage
cargo doc --no-deps
