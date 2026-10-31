"""Keep pyUFS/bin and pyUFS/fix as symlinks to directories in the parent of pyUFS."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def link_parent_dir(name: str, create: bool) -> Path:
    """Return REPO.parent/name, linked from REPO/name.

    With `create` the parent directory is made when missing. Otherwise a missing
    parent directory, or a REPO/name that is not a link to it, is an error.
    """
    target = REPO.parent / name
    link = REPO / name
    if create:
        target.mkdir(exist_ok=True)
    if not target.is_dir():
        raise SystemExit(
            f"ERROR: {target} not found; create it or link it with: ln -s /path/to/{name} {target}"
        )
    if link.is_symlink() and link.resolve() == target.resolve():
        return target
    if link.is_symlink() and create:
        link.unlink()
    elif link.exists() or link.is_symlink():
        raise SystemExit(
            f"ERROR: {link} is not a link to {target}; replace it with: ln -sfn ../{name} {link}"
        )
    elif not create:
        raise SystemExit(f"ERROR: {link} missing; link it with: ln -s ../{name} {link}")
    link.symlink_to(Path("..") / name)
    return target
