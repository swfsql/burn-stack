#!/usr/bin/env bash
#
# setup-box.sh: create the directories that the docs of our repos name with a
# `..` path.
#
# The box is the tree of directories around the repos, for example `../burn/`
# and `../papers/hnet/`. Each repo has a list, `docs/box.tsv`, of the directories
# that its docs name. The script gets only what a list names.
#
# Usage
# -----
#   scripts/setup-box.sh [--root DIR] [LIST ...]
#
#   --root DIR  The directory that holds the box. The default is the parent of
#               this repo.
#   LIST        A list to read. The default is the list of this repo, then the
#               list of each repo in the root. A repo that the first list clones
#               is also in the root.
#
# Rules
# -----
# - The script does not change a directory or a file that exists. It prints a
#   note when the HEAD of an existing git checkout is not at its pin.
# - A source can disappear and come back, so a list has no mark for a dead
#   source. A failure gives a warning, and the script continues with the next
#   entry. The exit status stays 0. Only a usage error or a missing tool gives
#   the status 2.
# - The script does not run downloaded files. Each download goes to a new
#   temporary directory in the root, and moves into place when it is complete.
#
# List format
# -----------
# One line for each entry, with columns that a tab separates. A line that starts
# with `#` is a comment. `-` marks an empty column.
#
#   dir      The path from the root. It cannot be absolute or hold `..`.
#   kind     `git`, `arxiv` or `file`.
#   origin   git: the clone URL. arxiv: the paper ID. file: the URL.
#   version  git: a commit or a tag, or `-` for the default branch.
#            arxiv: `vN`. file: `-`, because the URL holds the commit.
#   aliases  The citation aliases of `docs/source-terms.md` that need the entry.
#   note     A line that the script prints when it creates the entry.
#
# A version is exact because `docs/source-terms.md` cites `file:line`. A new
# version of a source can move each line that a citation names.

set -uo pipefail

# A repo that is gone must give a failure. It must not ask for a password.
export GIT_TERMINAL_PROMPT=0

usage() {
    echo "Usage: setup-box.sh [--root DIR] [LIST ...]" >&2
    exit 2
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(dirname -- "$script_dir")
root=$(dirname -- "$repo")
lists=()

while (($#)); do
    case $1 in
    --root)
        (($# >= 2)) || usage
        root=$2
        shift 2
        ;;
    -*) usage ;;
    *)
        lists+=("$1")
        shift
        ;;
    esac
done

for tool in git curl tar; do
    if ! command -v "$tool" >/dev/null; then
        echo "error: $tool is not available." >&2
        exit 2
    fi
done

if ! mkdir -p -- "$root" || ! root=$(cd -- "$root" && pwd); then
    echo "error: the root directory is not available." >&2
    exit 2
fi

warnings=()
created=0
present=0
work=
tmp_count=0

cleanup() {
    if [[ -n $work ]]; then
        rm -rf -- "$work"
    fi
}
trap cleanup EXIT

warn() {
    warnings+=("$1")
    echo "warning: $1" >&2
}

# Set `tmp` to a new empty directory. The first call makes the work directory, so
# a run with nothing to get writes nothing. The work directory is in the root, so
# the final `mv` stays on one file system.
new_tmp() {
    if [[ -z $work ]]; then
        if ! work=$(mktemp -d "$root/.setup-box.XXXXXX"); then
            work=
            return 1
        fi
    fi
    tmp_count=$((tmp_count + 1))
    tmp="$work/$tmp_count"
    mkdir -p -- "$tmp"
}

# The clause of a warning that names the citations without a source.
lost() {
    if [[ $1 != - ]]; then
        echo " Citations with $1 cannot resolve."
    fi
}

# Print a note if the HEAD of the existing git checkout `$1` is not at the pin `$2`.
check_pin() {
    local dest=$1 version=$2 head pin
    if [[ ! -e $dest/.git || $version == - ]]; then
        return
    fi
    head=$(git -C "$dest" rev-parse --verify -q HEAD 2>/dev/null)
    pin=$(git -C "$dest" rev-parse --verify -q "$version^{commit}" 2>/dev/null)
    if [[ $head != "$pin" ]]; then
        echo "note: ${dest#"$root"/}: HEAD is ${head:0:12}, and the pin is $version. The script does not change it."
    fi
}

# Write the URL `$1` to the file `$2`. Set `err` to the message of a failure.
download() {
    err=$(curl --fail --silent --show-error --location --connect-timeout 20 \
        --max-time 300 --retry 2 --output "$2" -- "$1" 2>&1)
}

# Each `get_*` function makes `$tmp/entry` from the entry in the variables of
# `read_entry`. A function returns non-zero after it gives its warning.

get_git() {
    local err
    if [[ $version == - ]]; then
        if ! err=$(git clone --quiet --filter=blob:none -- "$origin" "$tmp/entry" 2>&1); then
            warn "$dir: the clone failed: ${err##*$'\n'}.$(lost "$aliases")"
            return 1
        fi
        return 0
    fi
    if ! err=$(git clone --quiet --filter=blob:none --no-checkout -- "$origin" "$tmp/entry" 2>&1); then
        warn "$dir: the clone failed: ${err##*$'\n'}.$(lost "$aliases")"
        return 1
    fi
    if git -C "$tmp/entry" checkout --quiet --detach "$version" 2>/dev/null; then
        return 0
    fi
    if ! err=$(git -C "$tmp/entry" checkout --quiet 2>&1); then
        warn "$dir: the checkout failed: ${err##*$'\n'}.$(lost "$aliases")"
        return 1
    fi
    if [[ $aliases == - ]]; then
        warn "$dir: the pin $version is not available. The clone stays on the default branch."
    else
        warn "$dir: the pin $version is not available. The clone stays on the default branch. Line citations with $aliases can be wrong."
    fi
    return 0
}

get_arxiv() {
    local err
    if ! download "https://arxiv.org/e-print/$origin$version" "$tmp/source.tgz"; then
        warn "$dir: the download of arXiv $origin$version failed: $err.$(lost "$aliases")"
        return 1
    fi
    mkdir -p -- "$tmp/entry"
    if ! err=$(tar -xzf "$tmp/source.tgz" -C "$tmp/entry" --no-same-owner --no-same-permissions 2>&1); then
        warn "$dir: the extraction of arXiv $origin$version failed: ${err##*$'\n'}.$(lost "$aliases")"
        return 1
    fi
}

get_file() {
    local err
    if ! download "$origin" "$tmp/entry"; then
        warn "$dir: the download failed: $err.$(lost "$aliases")"
        return 1
    fi
}

read_entry() {
    local dest="$root/$dir"
    case /$dir/ in
    //* | */../* | */./*)
        warn "$dir: the path is not a relative path in the root. The script skips the entry."
        return
        ;;
    esac
    if [[ -e $dest || -L $dest ]]; then
        present=$((present + 1))
        if [[ $kind == git ]]; then
            check_pin "$dest" "$version"
        fi
        return
    fi
    case $kind in
    git | arxiv | file) ;;
    *)
        warn "$dir: the kind \"$kind\" is not known. The script skips the entry."
        return
        ;;
    esac
    echo "get: $dir"
    if ! new_tmp; then
        warn "$dir: the script cannot write to the root."
        return
    fi
    if ! "get_$kind"; then
        return
    fi
    if ! mkdir -p -- "$(dirname -- "$dest")" || ! mv -- "$tmp/entry" "$dest"; then
        warn "$dir: the script cannot move the entry into place."
        return
    fi
    created=$((created + 1))
    if [[ $note != - ]]; then
        echo "note: $dir: $note"
    fi
}

# Descriptor 3 holds the list, so a command in the loop cannot read a line of it.
read_list() {
    local dir kind origin version aliases note
    if [[ ! -f $1 ]]; then
        warn "$1: the list does not exist."
        return
    fi
    while IFS=$'\t' read -r dir kind origin version aliases note <&3 || [[ -n ${dir:-} ]]; do
        if [[ -z $dir || $dir == \#* ]]; then
            continue
        fi
        aliases=${aliases:--}
        note=${note:--}
        read_entry
    done 3<"$1"
}

if ((${#lists[@]})); then
    for list in "${lists[@]}"; do
        read_list "$list"
    done
else
    own="$repo/docs/box.tsv"
    read_list "$own"
    # This glob expands after the first list, so it finds the repos of that list.
    for list in "$root"/*/docs/box.tsv; do
        if [[ -f $list && ! $list -ef $own ]]; then
            read_list "$list"
        fi
    done
fi

echo "done: created $created, present $present, warnings ${#warnings[@]}."
for warning in ${warnings[@]+"${warnings[@]}"}; do
    echo "warning: $warning"
done
exit 0
