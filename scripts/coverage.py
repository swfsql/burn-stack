#!/usr/bin/env python3
"""The line, branch and condition coverage of the `src/` of a crate by its lib tests.

Usage:
  scripts/coverage.py [CRATE] [--files] [--save FILE] [--base BASE]
  scripts/coverage.py --check-tools
  scripts/coverage.py --selftest

CRATE is the directory of a crate, for example a family crate. The default is
the current directory. The script runs the lib tests of the crate under
`cargo llvm-cov` on nightly, with `-Zcoverage-options=condition`. Nightly rustc
has no MC/DC, and condition coverage is the nearest to it. The tests run on one
thread, as in `cargo test --lib -- --test-threads=1`.

The report counts the items of the `src/` of the crate. It does not count a
test file: a `tests.rs`, or a file in a `tests/` directory.

- Line: a line with code. A test covers it when the test runs it.
- Branch: one outcome (true or false) of one condition. The compiler instruments
  each condition of an `if` or of a `match` guard, and each operand of `&&` and
  `||`.
- Condition: one condition. A test suite covers it when it covers its two
  outcomes.

An item counts once for each source position. The instantiations of a generic
function add their counts. So the totals can differ from the summary of
`llvm-cov`. For example, `llvm-cov` does not count an outcome with no counter
in the binary. A constant condition has such an outcome, and so does each
condition of a function that no test calls. This script counts such an
outcome as not covered, as `llvm-cov` does for a line.

Compare: `--base` takes a git revision or a report that `--save` wrote. For a
revision, the script runs the lib tests of that revision in a git worktree at
`target/coverage-base/` of the crate. A revision with a relative path
dependency does not build there. Save a report of it instead.

The compare lists each item that the base covers and the working tree does not
cover. The `git diff` of the two `src/` trees matches each line of the base to
a line of the working tree. So an edit elsewhere in a file does not show as a
loss. An item on a line that the change edits or deletes shows as changed, not
as lost.

Each lost condition also shows as one or two lost branches. A lost branch can
show with no lost condition, when the base covers only that outcome of its
condition.

The script reports and does not judge. So it exits with 0 when it can compare
the reports. `--check-tools` exits with 1 when the box does not have
`cargo-llvm-cov` or the nightly toolchain. When nightly has no
`llvm-tools-preview`, `cargo llvm-cov` installs it through rustup, with no
prompt (`CARGO_LLVM_COV_SETUP=yes`). Set `CARGO_LLVM_COV_SETUP=no` to stop
that install. The script installs no other tool.
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

# The version of the file that `--save` writes.
FORMAT = 1
TOOLCHAIN = "nightly"
# `condition` includes `branch`. Do not give `cargo llvm-cov --branch`: its
# `-Zcoverage-options=branch` replaces this option.
COVERAGE_OPTION = "-Zcoverage-options=condition"
# The git worktree of a base revision, relative to the crate.
BASE_TREE = "target/coverage-base"
# The kinds of item. A count line names them in this order.
KINDS = ("lines", "branches", "conditions")


def is_test_file(path):
    """A test file is a `tests.rs`, or a file in a `tests/` directory."""
    parts = pathlib.PurePosixPath(path).parts
    return parts[-1] == "tests.rs" or "tests" in parts[:-1]


def counted_path(filename, root):
    """The path of a file relative to the crate root, or `None` when the report skips the file."""
    try:
        path = pathlib.PurePosixPath(filename).relative_to(root)
    except ValueError:
        return None
    if path.parts[0] != "src" or is_test_file(path):
        return None
    return path.as_posix()


def items(export, lcov, root):
    """The lines and the conditions of each counted file, from the two exports of llvm-cov.

    `export` is the JSON export and `lcov` is the lcov export. The result is
    `(lines, conditions)`. `lines` maps a file to `{line: count}`. `conditions`
    maps a file to `{(line0, column0, line1, column1): (true count, false count)}`.
    The instantiations of a generic function add their counts.
    """
    lines, path = {}, None
    for record in lcov.splitlines():
        if record.startswith("SF:"):
            path = counted_path(record[3:], root)
        elif record.startswith("DA:") and path is not None:
            line, count = map(int, record[3:].split(",")[:2])
            counts = lines.setdefault(path, {})
            counts[line] = counts.get(line, 0) + count
    conditions = {}
    for file in export["data"][0]["files"]:
        path = counted_path(file["filename"], root)
        if path is None:
            continue
        spans = conditions.setdefault(path, {})
        # llvm-cov calls a condition a branch region.
        for line0, column0, line1, column1, true, false, *_ in file["branches"]:
            span = (line0, column0, line1, column1)
            old_true, old_false = spans.get(span, (0, 0))
            spans[span] = (old_true + true, old_false + false)
    return lines, conditions


def totals(report):
    """`{kind: (covered, total)}` of the lines, the branches and the conditions."""
    lines = [count for counts in report["lines"].values() for count in counts.values()]
    pairs = [pair for spans in report["conditions"].values() for pair in spans.values()]
    outcomes = [count for pair in pairs for count in pair]
    return {
        "lines": (sum(count > 0 for count in lines), len(lines)),
        "branches": (sum(count > 0 for count in outcomes), len(outcomes)),
        "conditions": (sum(min(pair) > 0 for pair in pairs), len(pairs)),
    }


def git_diff(old_texts, new_texts):
    """The `git diff -U0` of two trees, from the text of each file."""
    with tempfile.TemporaryDirectory() as tmp:
        for side, texts in (("old", old_texts), ("new", new_texts)):
            pathlib.Path(tmp, side).mkdir()
            for path, text in texts.items():
                file = pathlib.Path(tmp, side, path)
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(text, encoding="utf-8")
        # The options override each git setting that changes the form of the output.
        done = subprocess.run(
            ["git", "-c", "core.quotePath=false", "diff", "--no-index", "--no-ext-diff", "--no-textconv"]
            + ["--no-color", "--no-prefix", "--text", "--find-renames", "--unified=0", "--inter-hunk-context=0"]
            + ["old", "new"],
            cwd=tmp,
            capture_output=True,
            encoding="utf-8",
        )
    # `git diff --no-index` exits with 1 when the trees differ.
    if done.returncode not in (0, 1):
        sys.exit(f"error: `git diff --no-index` failed: {done.stderr.strip()}")
    return done.stdout


def parse_diff(diff):
    """`{old path: (new path, hunks)}` of each old file that the diff changes.

    A hunk is `(old start, old count, new start, new count)`. A deleted file
    has the new path `None`.
    """
    entries, pending = [], 0
    for line in diff.splitlines():
        if pending:
            # The lines of a hunk. A line `\ No newline at end of file` is not one of them.
            pending -= line.startswith(("-", "+"))
            continue
        if line.startswith("diff --git "):
            entries.append({"old": None, "new": None, "hunks": []})
        for prefix, side in (("rename from ", "old"), ("--- ", "old"), ("rename to ", "new"), ("+++ ", "new")):
            if line.startswith(prefix):
                entries[-1][side] = line[len(prefix) :].removeprefix(f"{side}/")
        if line.startswith("@@ "):
            numbers = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line).groups()
            hunk = tuple(int(number) if number is not None else 1 for number in numbers)
            entries[-1]["hunks"].append(hunk)
            pending = hunk[1] + hunk[3]
    return {
        entry["old"]: (None if entry["new"] == "/dev/null" else entry["new"], entry["hunks"])
        for entry in entries
        if entry["old"] not in (None, "/dev/null")
    }


def mapped_line(hunks, line):
    """The line in the new file of a line of the old file, or `None` when the change edits or deletes it."""
    shift = 0
    for old_start, old_count, _, new_count in hunks:
        if old_count and old_start <= line < old_start + old_count:
            return None
        # A hunk with no old line adds its lines after the line `old_start`.
        last = old_start + old_count - 1 if old_count else old_start
        if last < line:
            shift += new_count - old_count
    return line + shift


def mapped_span(hunks, span):
    """The span in the new file of a span of the old file, or `None` when the change touches it."""
    line0, column0, line1, column1 = span
    lines = [mapped_line(hunks, line) for line in range(line0, line1 + 1)]
    if None in lines or lines[-1] - lines[0] != line1 - line0:
        return None
    return (lines[0], column0, lines[-1], column1)


def compare_items(base, new):
    """The covered items of the base that are lost, and those on changed lines.

    The result is `(lost, changed)`. Each maps a file to `{"lines": [...],
    "branches": [(span, outcome), ...], "conditions": [span, ...]}`. A lost
    item has the path and the line of the working tree. A changed item has
    those of the base. The hunks of a deleted file hold all its lines, so each
    item of the file is changed.
    """
    maps = parse_diff(git_diff(base["texts"], new["texts"]))
    lost, changed = {}, {}

    def add(kind, side, path, item):
        side.setdefault(path, {name: [] for name in KINDS})[kind].append(item)

    for path, counts in base["lines"].items():
        new_path, hunks = maps.get(path, (path, []))
        for line, count in counts.items():
            if count == 0:
                continue
            target = mapped_line(hunks, line)
            if target is None:
                add("lines", changed, path, line)
            elif new["lines"].get(new_path, {}).get(target, 0) == 0:
                add("lines", lost, new_path, target)
    for path, spans in base["conditions"].items():
        new_path, hunks = maps.get(path, (path, []))
        for span, pair in spans.items():
            target = mapped_span(hunks, span)
            new_pair = new["conditions"].get(new_path, {}).get(target, (0, 0))
            for index, outcome in enumerate(("true", "false")):
                if pair[index] == 0:
                    continue
                if target is None:
                    add("branches", changed, path, (span, outcome))
                elif new_pair[index] == 0:
                    add("branches", lost, new_path, (target, outcome))
            if min(pair) == 0:
                continue
            if target is None:
                add("conditions", changed, path, span)
            elif min(new_pair) == 0:
                add("conditions", lost, new_path, target)
    return lost, changed


def ranges(numbers):
    """Sorted numbers as ranges, for example `3–5, 9`."""
    runs = []
    for number in sorted(numbers):
        if runs and number == runs[-1][1] + 1:
            runs[-1][1] = number
        else:
            runs.append([number, number])
    return ", ".join(f"{first}–{last}" if last > first else f"{first}" for first, last in runs)


def listing(files):
    """The items of each file, one line for each kind."""
    if not files:
        return ["None."]
    out = []
    for path in sorted(files):
        out.append(path)
        found = files[path]
        lines, branches, conditions = found["lines"], found["branches"], found["conditions"]
        if lines:
            out.append(f"  lines       {ranges(lines)}")
        if branches:
            in_order = sorted(branches, key=lambda item: (item[0], item[1] == "false"))
            named = ", ".join(f"{span[0]}:{span[1]} {outcome}" for span, outcome in in_order)
            out.append(f"  branches    {named}")
        if conditions:
            named = ", ".join(f"{span[0]}:{span[1]}" for span in sorted(conditions))
            out.append(f"  conditions  {named}")
    return out


def count_line(heading, files):
    counts = (f"{sum(len(found[kind]) for found in files.values())} {kind}" for kind in KINDS)
    return f"## {heading}: {', '.join(counts)}"


def fraction(covered, total):
    return f"{covered} of {total} ({covered / total:.1%})" if total else "0 of 0"


def compare(base, new):
    """The text of the compare of a base report and the report of the working tree."""
    out = []
    if base["crate"] != new["crate"]:
        out += [f"WARNING: the base measures `{base['crate']}`, and the working tree measures `{new['crate']}`.", ""]
    old_totals, new_totals = totals(base), totals(new)
    out += ["| measure | base | working tree |", "|---|---|---|"]
    for kind in old_totals:
        out.append(f"| {kind} | {fraction(*old_totals[kind])} | {fraction(*new_totals[kind])} |")
    lost, changed = compare_items(base, new)
    out += ["", count_line("Lost items", lost), ""]
    out += ["The base covers each item below, and the working tree does not. A line is a line of the working tree.", ""]
    out += listing(lost)
    out += ["", count_line("Covered items on changed lines", changed), ""]
    out += ["The base covers each item below. The change edits or deletes its line. A line is a line of the base.", ""]
    out += listing(changed)
    return "\n".join(out) + "\n"


def summary(report, files):
    """The totals of a report. With `files`, also the totals of each file."""
    out = ["| measure | covered |", "|---|---|"]
    for kind, pair in totals(report).items():
        out.append(f"| {kind} | {fraction(*pair)} |")
    if files:
        out += ["", "| file | lines | branches | conditions |", "|---|---|---|---|"]
        for path in sorted(report["lines"].keys() | report["conditions"].keys()):
            one = totals({key: {path: report[key].get(path, {})} for key in ("lines", "conditions")})
            out.append(f"| {path} | " + " | ".join(fraction(*pair) for pair in one.values()) + " |")
    return "\n".join(out) + "\n"


# The keys of a report that JSON holds as they are.
PLAIN_KEYS = ("crate", "commit", "dirty", "texts")


def to_json(report):
    out = {key: report[key] for key in PLAIN_KEYS}
    out["lines"] = {path: sorted(counts.items()) for path, counts in report["lines"].items()}
    out["conditions"] = {
        path: sorted([*span, *pair] for span, pair in spans.items()) for path, spans in report["conditions"].items()
    }
    return {"format": FORMAT, **out}


def from_json(data):
    if data.get("format") != FORMAT:
        sys.exit(f"error: the report has the format {data.get('format')}. This script reads the format {FORMAT}.")
    out = {key: data[key] for key in PLAIN_KEYS}
    out["lines"] = {path: dict(pairs) for path, pairs in data["lines"].items()}
    out["conditions"] = {
        path: {tuple(entry[:4]): tuple(entry[4:]) for entry in entries} for path, entries in data["conditions"].items()
    }
    return out


def output(command):
    """The stdout of a command, or `None` when the command fails or does not exist."""
    try:
        done = subprocess.run(command, capture_output=True, encoding="utf-8")
    except FileNotFoundError:
        return None
    return done.stdout if done.returncode == 0 else None


def missing_tools():
    """The names of the coverage tools that this box does not have, and that a
    run cannot install. `cargo llvm-cov` installs the `llvm-tools-preview`
    (see `measure`)."""
    missing = []
    if output(["cargo", "llvm-cov", "--version"]) is None:
        missing.append("cargo-llvm-cov")
    if output(["rustup", "component", "list", "--toolchain", TOOLCHAIN, "--installed"]) is None:
        missing.append(f"the {TOOLCHAIN} toolchain")
    return missing


def git(cwd, *args):
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, encoding="utf-8")
    if done.returncode != 0:
        sys.exit(f"error: `git {' '.join(args)}` failed: {done.stderr.strip()}")
    return done.stdout.strip()


def measure(root, target=None):
    """The report of the crate in the directory `root`, from a run of its lib tests.

    `target` is the cargo target directory. The default is the one of cargo.
    """
    env = dict(os.environ)
    env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} {COVERAGE_OPTION}".strip()
    # When nightly has no `llvm-tools-preview`, `cargo llvm-cov` installs it
    # through rustup. `yes` skips its prompt. An explicit `no` stops the install.
    env.setdefault("CARGO_LLVM_COV_SETUP", "yes")
    # `cargo llvm-cov` has no `--target-dir`. It reads the directory of `cargo metadata`.
    if target is not None:
        env["CARGO_TARGET_DIR"] = str(target)
    cargo = ["cargo", f"+{TOOLCHAIN}", "llvm-cov"]

    def run(args, stdout):
        done = subprocess.run(cargo + args, cwd=root, env=env, stdout=stdout, encoding="utf-8")
        if done.returncode != 0:
            sys.exit(f"error: `{' '.join(cargo + args)}` failed in {root}.")
        return done.stdout

    # With `--no-report`, `cargo llvm-cov` keeps the counts of the earlier runs
    # and adds them to the report. So delete them first.
    run(["clean", "--profraw-only"], sys.stderr)
    # The output of the tests goes to stderr, so stdout holds only the report.
    run(["--lib", "--no-report", "--", "--test-threads=1"], sys.stderr)
    export = json.loads(run(["report", "--json", "--skip-functions"], subprocess.PIPE))
    lcov = run(["report", "--lcov"], subprocess.PIPE)
    lines, conditions = items(export, lcov, root.as_posix())
    texts = {}
    for file in sorted((root / "src").rglob("*.rs")):
        path = file.relative_to(root).as_posix()
        if not is_test_file(path):
            texts[path] = file.read_text(encoding="utf-8")
    with open(root / "Cargo.toml", "rb") as manifest:
        crate = tomllib.load(manifest)["package"]["name"]
    return {
        "crate": crate,
        "commit": git(root, "rev-parse", "HEAD"),
        "dirty": bool(git(root, "status", "--porcelain", "--", ".")),
        "texts": texts,
        "lines": lines,
        "conditions": conditions,
    }


def measure_revision(root, revision):
    """The report of a git revision of the crate in `root`, from a run in the worktree `BASE_TREE`."""
    commit = git(root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    inner = root.relative_to(git(root, "rev-parse", "--show-toplevel"))
    tree = root / BASE_TREE
    if (tree / ".git").exists():
        git(tree, "checkout", "--quiet", "--force", "--detach", commit)
    else:
        # `prune` forgets a worktree whose directory a `cargo clean` deleted.
        git(root, "worktree", "prune")
        git(root, "worktree", "add", "--quiet", "--detach", str(tree), commit)
    base_root = tree / inner
    # When the revision has no `Cargo.lock`, the base builds with the dependencies of the working tree.
    lock = (inner / "Cargo.lock").as_posix()
    tracked = output(["git", "-C", str(root), "cat-file", "-e", f"{commit}:{lock}"]) is not None
    if not tracked and (root / "Cargo.lock").exists():
        shutil.copyfile(root / "Cargo.lock", base_root / "Cargo.lock")
    report = measure(base_root, target=base_root / "target")
    report["dirty"] = False
    return report


def label(report):
    changes = ", with uncommitted changes" if report["dirty"] else ""
    return f"{report['commit'][:12]}{changes}"


def main():
    parser = argparse.ArgumentParser(
        description="The line, branch and condition coverage of the src/ of a crate by its lib tests."
    )
    parser.add_argument(
        "crate",
        nargs="?",
        default=".",
        type=pathlib.Path,
        help="the directory of the crate. The default is the current directory",
    )
    parser.add_argument(
        "--base",
        metavar="BASE",
        help="a git revision, or a report that --save wrote. The script compares the working tree with it",
    )
    parser.add_argument("--save", metavar="FILE", type=pathlib.Path, help="write the report of the working tree to FILE")
    parser.add_argument("--files", action="store_true", help="without --base, also give the coverage of each file")
    parser.add_argument(
        "--check-tools", action="store_true", help="exit with 1 when the box does not have the coverage tools"
    )
    parser.add_argument("--selftest", action="store_true", help="run the tests of this script")
    args = parser.parse_args()

    if args.selftest:
        run_selftests(SELFTESTS)
        return
    missing = missing_tools()
    if args.check_tools:
        if missing:
            sys.exit(f"missing: {', '.join(missing)}")
        return
    if missing:
        sys.exit(f"error: this box does not have {', '.join(missing)}.")

    root = args.crate.resolve()
    new = measure(root)
    if args.save is not None:
        args.save.write_text(json.dumps(to_json(new)), encoding="utf-8")
    if args.base is None:
        print(f"# Coverage of the `src/` of {new['crate']} at {label(new)}\n")
        print(summary(new, args.files), end="")
        return
    if pathlib.Path(args.base).is_file():
        base = from_json(json.loads(pathlib.Path(args.base).read_text(encoding="utf-8")))
    else:
        base = measure_revision(root, args.base)
    print(f"# Coverage of the `src/` of {new['crate']}: {args.base} ({label(base)}) -> working tree ({label(new)})\n")
    print(compare(base, new), end="")


def assert_lines(text, wanted):
    """Each wanted line is in the text. Runs of blanks count as one blank."""
    lines = {" ".join(line.split()) for line in text.splitlines()}
    missing = [line for line in wanted if line not in lines]
    assert not missing, f"missing {missing} in:\n{text}"


# Self-test: the inputs are inline llvm-cov exports and source texts.

EXPORT = {
    "data": [
        {
            "files": [
                {
                    "filename": "/r/src/a.rs",
                    "branches": [
                        [3, 8, 3, 9, 5, 0, 0, 0, 4],
                        # A second instantiation of the same condition.
                        [3, 8, 3, 9, 0, 2, 0, 0, 4],
                        [3, 13, 3, 14, 0, 0, 0, 0, 4],
                        [7, 4, 7, 9, 1, 1, 0, 0, 4],
                    ],
                },
                {"filename": "/r/src/a/tests.rs", "branches": [[1, 1, 1, 2, 1, 0, 0, 0, 4]]},
                {"filename": "/r/src/tests/util.rs", "branches": [[1, 1, 1, 2, 1, 0, 0, 0, 4]]},
                {"filename": "/r/tests/it.rs", "branches": [[1, 1, 1, 2, 1, 0, 0, 0, 4]]},
                {"filename": "/other/src/b.rs", "branches": [[1, 1, 1, 2, 1, 0, 0, 0, 4]]},
            ]
        }
    ]
}

LCOV = """\
SF:/r/src/a.rs
DA:1,3
DA:2,0
DA:3,5
DA:7,0
end_of_record
SF:/r/src/a/tests.rs
DA:1,1
end_of_record
SF:/r/src/tests/util.rs
DA:1,1
end_of_record
SF:/r/tests/it.rs
DA:1,1
end_of_record
SF:/other/src/b.rs
DA:1,1
end_of_record
"""


def a_report_counts_the_items_of_src_without_the_tests():
    lines, conditions = items(EXPORT, LCOV, "/r")
    assert set(lines) == {"src/a.rs"}, lines
    assert set(conditions) == {"src/a.rs"}, conditions
    # The two instantiations at 3:8 cover both outcomes. 3:13 covers none.
    assert totals({"lines": lines, "conditions": conditions}) == {
        "lines": (2, 4),
        "branches": (4, 6),
        "conditions": (2, 3),
    }


BASE_A = """\
fn f(x: bool) -> u8 {
    if x {
        1
    } else {
        2
    }
}
fn g() {}
"""

# A new first line moves each line down by one. The change also edits `g`.
NEW_A = "// A comment.\n" + BASE_A.replace("fn g() {}", "fn g() { }")


def make_report(texts, lines, conditions=None):
    return {"crate": "c", "commit": "0" * 40, "dirty": False, "texts": texts, "lines": lines, "conditions": conditions or {}}


def section(text, heading):
    """The part of the text from the heading that starts with `heading` to the next heading."""
    part = text[text.index(f"\n## {heading}") + 1 :]
    end = part.find("\n## ")
    return part if end < 0 else part[:end]


def the_compare_follows_each_line_through_the_diff():
    base = make_report(
        {"src/a.rs": BASE_A},
        {"src/a.rs": {1: 4, 2: 4, 3: 3, 5: 1, 7: 4, 8: 2}},
        {"src/a.rs": {(2, 8, 2, 9): (3, 1)}},
    )
    # A base can be a saved report.
    base = from_json(json.loads(json.dumps(to_json(base))))
    new = make_report(
        {"src/a.rs": NEW_A},
        {"src/a.rs": {2: 4, 3: 4, 4: 3, 6: 0, 8: 4, 9: 0}},
        {"src/a.rs": {(3, 8, 3, 9): (3, 0)}},
    )
    text = compare(base, new)
    lost = section(text, "Lost items")
    # The line 6 of the working tree is the line 5 of the base. The lines above it moved, and
    # the tests still cover them.
    assert_lines(
        lost,
        ["## Lost items: 1 lines, 1 branches, 1 conditions", "src/a.rs", "lines 6", "branches 3:8 false", "conditions 3:8"],
    )
    # The change edits the line 8 of the base, so its item is changed, not lost.
    assert_lines(section(text, "Covered items on changed lines"), ["src/a.rs", "lines 8"])


def the_compare_follows_a_renamed_file():
    renamed = "fn r() {\n    1;\n}\n"
    # The diff of the deleted line `-- x` is `--- x`, as the header of a file.
    dashes = "-- x\nfn d() {\n    2;\n}\n"
    base = make_report(
        {"src/r.rs": renamed, "src/gone.rs": "fn gone() {}\n", "src/d.rs": dashes},
        {"src/r.rs": {1: 1, 2: 1, 3: 1}, "src/gone.rs": {1: 1}, "src/d.rs": {2: 1, 3: 1, 4: 1}},
    )
    new = make_report(
        {"src/s.rs": renamed, "src/d.rs": dashes.removeprefix("-- x\n")},
        {"src/s.rs": {1: 1, 2: 0, 3: 1}, "src/d.rs": {1: 1, 2: 0, 3: 1}},
    )
    text = compare(base, new)
    assert_lines(
        section(text, "Lost items"),
        ["## Lost items: 2 lines, 0 branches, 0 conditions", "src/s.rs", "src/d.rs", "lines 2"],
    )
    assert_lines(
        section(text, "Covered items on changed lines"),
        ["## Covered items on changed lines: 1 lines, 0 branches, 0 conditions", "src/gone.rs", "lines 1"],
    )


def a_covered_condition_that_loses_one_outcome_is_lost():
    base = make_report({"src/a.rs": BASE_A}, {"src/a.rs": {2: 4}}, {"src/a.rs": {(2, 8, 2, 9): (3, 1)}})
    new = make_report({"src/a.rs": BASE_A}, {"src/a.rs": {2: 4}}, {"src/a.rs": {(2, 8, 2, 9): (3, 0)}})
    assert_lines(
        section(compare(base, new), "Lost items"),
        ["## Lost items: 0 lines, 1 branches, 1 conditions", "src/a.rs", "branches 2:8 false", "conditions 2:8"],
    )


def a_lost_branch_of_a_condition_that_the_base_does_not_cover_loses_no_condition():
    # The base covers only the true outcome of the condition.
    base = make_report({"src/a.rs": BASE_A}, {"src/a.rs": {2: 4}}, {"src/a.rs": {(2, 8, 2, 9): (3, 0)}})
    new = make_report({"src/a.rs": BASE_A}, {"src/a.rs": {2: 4}}, {"src/a.rs": {(2, 8, 2, 9): (0, 0)}})
    assert_lines(
        section(compare(base, new), "Lost items"),
        ["## Lost items: 0 lines, 1 branches, 0 conditions", "src/a.rs", "branches 2:8 true"],
    )


def a_covered_condition_on_a_changed_line_is_changed():
    edited = BASE_A.replace("if x {", "if !x {")
    base = make_report({"src/a.rs": BASE_A}, {"src/a.rs": {2: 4}}, {"src/a.rs": {(2, 8, 2, 9): (3, 1)}})
    new = make_report({"src/a.rs": edited}, {"src/a.rs": {2: 4}}, {"src/a.rs": {(2, 8, 2, 10): (1, 3)}})
    text = compare(base, new)
    assert_lines(section(text, "Lost items"), ["## Lost items: 0 lines, 0 branches, 0 conditions"])
    assert_lines(
        section(text, "Covered items on changed lines"),
        [
            "## Covered items on changed lines: 1 lines, 2 branches, 1 conditions",
            "src/a.rs",
            "lines 2",
            "branches 2:8 true, 2:8 false",
            "conditions 2:8",
        ],
    )


def run_selftests(tests):
    """Run each test. A failed test sets the exit status."""
    failed = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except (Exception, SystemExit) as error:
            failed += 1
            print(f"FAIL {test.__name__}: {type(error).__name__}: {error}")
    if failed:
        sys.exit(f"{failed} of {len(tests)} tests failed")


SELFTESTS = [
    a_report_counts_the_items_of_src_without_the_tests,
    the_compare_follows_each_line_through_the_diff,
    the_compare_follows_a_renamed_file,
    a_covered_condition_that_loses_one_outcome_is_lost,
    a_lost_branch_of_a_condition_that_the_base_does_not_cover_loses_no_condition,
    a_covered_condition_on_a_changed_line_is_changed,
]


if __name__ == "__main__":
    main()
