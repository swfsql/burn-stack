#!/usr/bin/env bash
# The check of this repo. CI runs it and skips the two steps that read this
# box: the host paths and the prose.
#
# `--coverage-base BASE`: the coverage step compares the working tree with
# BASE, a git revision or a report of `scripts/coverage.py --save`. A path is
# relative to the repo. With no BASE, the step gives the coverage alone.
set -euo pipefail
cd "$(dirname "$0")/.."

coverage_base=""
while [ $# -gt 0 ]; do
    case "$1" in
        --coverage-base)
            coverage_base="${2:?--coverage-base takes a git revision or a report}"
            shift 2
            ;;
        *)
            echo "error: unknown argument \`$1\`." >&2
            exit 2
            ;;
    esac
done

# Fail when a file has more hard STE violations than it has on origin/main. A
# new file starts at zero. The linter is part of the `asd-ste100` skill.
check_prose() {
    local lint="$HOME/.claude/skills/asd-ste100/scripts/ste-lint.py"
    if [ ! -f "$lint" ]; then
        echo "note: this box has no ste-lint.py, so the prose check did not run." >&2
        return 0
    fi
    hard_violations() {
        python3 "$lint" --enable history --summary --baseline 1000000 "$1" | awk 'NR == 2 { print $1 }'
    }
    local base old_dir file before after failed=0
    base="$(git merge-base origin/main HEAD)"
    old_dir="$(mktemp -d)"
    while IFS= read -r file; do
        before=0
        if git cat-file -e "$base:$file" 2>/dev/null; then
            mkdir -p "$old_dir/$(dirname "$file")"
            git show "$base:$file" >"$old_dir/$file"
            before="$(hard_violations "$old_dir/$file")"
        fi
        after="$(hard_violations "$file")"
        if [ "$after" -gt "$before" ]; then
            echo "error: $file has $after hard STE violations. origin/main has $before." >&2
            python3 "$lint" --enable history "$file" >&2 || true
            failed=1
        fi
    done < <(
        {
            git diff --name-only --diff-filter=d "$base" -- '*.rs' '*.md'
            git ls-files --others --exclude-standard -- '*.rs' '*.md'
        } | grep -v '^\.scratch/' | sort -u
    )
    rm -rf "$old_dir"
    return "$failed"
}

# Report the coverage of `src/` by the lib tests, or compare it with
# `--coverage-base`. A box without the coverage tools skips it.
check_coverage() {
    if ! python3 scripts/coverage.py --check-tools; then
        echo "note: this box has no coverage tools, so the coverage did not run." >&2
        return 0
    fi
    if [ -n "$coverage_base" ]; then
        python3 scripts/coverage.py . --base "$coverage_base"
    else
        python3 scripts/coverage.py .
    fi
}

if [ -z "${CI:-}" ]; then
    # This box gives the two roots, so this file names no path of a box.
    repo_root="/$(pwd | cut -d/ -f2)"
    home_root="/$(printf '%s' "$HOME" | cut -d/ -f2)"
    if git grep --untracked -n -I -E "(^|[^[:alnum:]_./-])($repo_root|$home_root)(/|\$)"; then
        echo "error: the lines above hold a host path. Write a path relative to the repo." >&2
        exit 1
    fi
    check_prose
fi

python3 scripts/coverage.py --selftest

cargo check --all-targets
cargo check --features examples-common
cargo test --lib -- --test-threads=1
check_coverage
cargo doc --no-deps
