#!/usr/bin/env python3
"""Give twin commits on the release branches the same subject, split symlink commits, push.

    python tools/align_branch_messages.py --dry-run
    python tools/align_branch_messages.py

Release branches (six-digit names on origin) are compared with the newest one. Commits that
exist on only one branch since the merge base are paired with their twin by order, subject
similarity and shared file names. Each pair gets one Conventional Commits subject (the better
of the two, or one inferred from the first verb and the touched directory); bodies are kept.
A commit that adds symlinks together with other changes is split in two, symlinks first. History
is rewritten with git-filter-repo in a fresh clone of origin, force-pushed with a lease, and the
local branches are then reset to the new origin branches. Needs git-filter-repo.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from git_filter_repo import Commit, FilteringOptions, RepoFilter

CONVENTIONAL = re.compile(r"^(\w+)(\([^)]+\))?!?: ")
TYPES = {
    "add": "feat",
    "extend": "feat",
    "read": "feat",
    "fix": "fix",
    "correct": "fix",
    "set": "fix",
    "join": "fix",
    "move": "refactor",
    "rename": "refactor",
    "replace": "refactor",
    "rework": "refactor",
    "refactor": "refactor",
    "reorganize": "refactor",
    "import": "refactor",
    "use": "refactor",
    "write": "refactor",
    "label": "refactor",
    "build": "refactor",
}
Info = tuple[str, set[str]]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True, capture_output=True
    ).stdout


def unique_commits(repo: Path, own: str, other: str) -> dict[str, Info]:
    """Map hash to (subject, files) for commits on `own` that are not on `other`."""
    out = git(
        repo,
        "log",
        "--reverse",
        "--format=%x00%H%x09%s",
        "--name-only",
        f"origin/{other}..origin/{own}",
    )
    commits = {}
    for block in out.split("\0")[1:]:
        head, *files = block.strip().split("\n")
        sha, subject = head.split("\t", 1)
        commits[sha] = (subject, {f for f in files if f})
    return commits


def stem(subject: str) -> str:
    return CONVENTIONAL.sub("", subject).lower().replace("_", " ")


def score(a: Info, b: Info) -> float:
    jaccard = len(a[1] & b[1]) / (len(a[1] | b[1]) or 1)
    return 0.6 * SequenceMatcher(None, stem(a[0]), stem(b[0])).ratio() + 0.4 * jaccard


def pair(
    left: dict[str, Info], right: dict[str, Info], threshold: float
) -> list[tuple[str, str]]:
    """Order-preserving alignment of the two histories that maximizes the summed score."""
    xs, ys = list(left), list(right)
    gain = [[score(left[x], right[y]) for y in ys] for x in xs]
    best = [[0.0] * (len(ys) + 1) for _ in range(len(xs) + 1)]
    for i in range(1, len(xs) + 1):
        for j in range(1, len(ys) + 1):
            match = gain[i - 1][j - 1]
            best[i][j] = max(
                best[i - 1][j],
                best[i][j - 1],
                best[i - 1][j - 1] + (match if match >= threshold else 0.0),
            )
    pairs, i, j = [], len(xs), len(ys)
    while i and j:
        if best[i][j] == best[i - 1][j]:
            i -= 1
        elif best[i][j] == best[i][j - 1]:
            j -= 1
        else:
            pairs.append((xs[i - 1], ys[j - 1]))
            i, j = i - 1, j - 1
    return pairs[::-1]


def conventional(subject: str, files: set[str]) -> str:
    if CONVENTIONAL.match(subject):
        return subject
    verb = subject.split()[0].lower()
    dirs = Counter(f.split("/")[0] for f in files if "/" in f)
    scope = f"({dirs.most_common(1)[0][0]})" if len(dirs) == 1 else ""
    kind = TYPES.get(verb, "chore")
    text = (
        subject.split(maxsplit=1)[1]
        if verb == kind
        else f"{subject[0].lower()}{subject[1:]}"
    )
    return f"{kind}{scope}: {text}"


def canonical(infos: list[Info]) -> str:
    return max(
        (conventional(*i) for i in infos),
        key=lambda s: (bool(CONVENTIONAL.match(s)), len(s)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Local clone whose origin is rewritten",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55, help="Minimum match score"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the new subjects and stop"
    )
    args = parser.parse_args()

    url = git(args.repo, "remote", "get-url", "origin").strip()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "clone"
        subprocess.run(["git", "clone", "-q", url, str(work)], check=True)
        branches = sorted(
            b.strip().removeprefix("origin/")
            for b in git(work, "branch", "-r").splitlines()
            if re.fullmatch(r"origin/\d{6}", b.strip())
        )
        newest = branches[-1]
        subjects: dict[bytes, bytes] = {}
        for branch in branches[:-1]:
            left, right = (
                unique_commits(work, branch, newest),
                unique_commits(work, newest, branch),
            )
            for x, y in pair(left, right, args.threshold):
                new = canonical([left[x], right[y]])
                for sha, old in ((x, left[x][0]), (y, right[y][0])):
                    subjects[sha.encode()] = new.encode()
                    if old != new:
                        print(f"{branch} {sha[:7]}  {old}\n     ->  {new}")
            print(
                f"{branch}: {len(left)} / {len(right)} commits, {len(subjects)} retitled so far"
            )
        if args.dry_run:
            return

        old = {b: git(work, "rev-parse", f"origin/{b}").strip() for b in branches}
        for branch in branches:
            git(work, "update-ref", f"refs/heads/{branch}", f"origin/{branch}")

        def retitle(commit: Commit, _metadata: dict) -> None:
            links = [
                c for c in commit.file_changes if c.type == b"M" and c.mode == b"120000"
            ]
            renamed = any(c.type == b"D" for c in commit.file_changes)
            if (
                commit.original_id
                and links
                and not renamed
                and len(links) < len(commit.file_changes)
            ):
                names = " and ".join(c.filename.decode() for c in links)
                split = Commit(
                    commit.branch,
                    commit.author_name,
                    commit.author_email,
                    commit.author_date,
                    commit.committer_name,
                    commit.committer_email,
                    commit.committer_date,
                    f"feat: add {names} symlinks to parent directory\n".encode(),
                    links,
                    commit.parents,
                )
                repo_filter.insert(split, direct_insertion=True)
                commit.parents = [split.id]
                commit.file_changes = [c for c in commit.file_changes if c not in links]
            subject, _, body = commit.message.partition(b"\n")
            commit.message = subjects.get(commit.original_id, subject) + b"\n" + body

        options = FilteringOptions.parse_args(
            ["--refs", *[f"refs/heads/{b}" for b in branches], "--force"]
        )
        repo_filter = RepoFilter(options, commit_callback=retitle)
        os.chdir(work)
        repo_filter.run()
        os.chdir(args.repo)
        for b in branches:
            git(
                work,
                "push",
                f"--force-with-lease=refs/heads/{b}:{old[b]}",
                url,
                f"refs/heads/{b}",
            )
    git(args.repo, "fetch", "origin")
    current = git(args.repo, "branch", "--show-current").strip()
    for b in branches:
        if b != current:
            git(args.repo, "branch", "-f", "--track", b, f"origin/{b}")
        elif not git(args.repo, "status", "--porcelain"):
            git(args.repo, "reset", "--hard", f"origin/{b}")
            git(args.repo, "branch", f"--set-upstream-to=origin/{b}", b)


if __name__ == "__main__":
    main()
