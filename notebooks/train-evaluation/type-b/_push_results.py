"""
notebooks/train-evaluation/type-b/_push_results.py

Push training results (figures, metrics) to GitHub.
Run via:  %run -i _push_results.py   (the -i flag shares the notebook namespace)

Requires in notebook namespace:
    REPO_DIR   : str  — path to the cloned repo (e.g. '/content/repo')
    COMMIT_MSG : str  — commit message

Design notes
------------
- results dirs may be symlinks → Google Drive.  Git cannot add files through
  a symlink that points outside the working tree (exit 128).
- We resolve each symlink to its real Drive path, copy files into the actual
  repo tree, then add only the files that exist (never staging deletions).
"""

import shutil
import subprocess
from pathlib import Path

# ── helpers ────────────────────────────────────────────────────────────────────

def _git(args: list) -> None:
    subprocess.run(['git', '-C', REPO_DIR] + args, check=True)  # noqa: F821


def _copy_drive_results_to_repo(repo: Path) -> list[str]:
    """
    For each results sub-directory that is a Drive symlink, copy its files
    into the real repo tree.  Returns a list of repo-relative paths added.
    """
    added: list[str] = []
    for sub in ['figures/type-b', 'metrics/type-b']:
        link = repo / 'src' / 'pipelines' / 'results' / sub
        real = link.resolve()          # Drive path if symlink, same path otherwise

        if real == link:
            # Already a real directory — files are already in the repo tree
            for f in link.glob('*'):
                if f.is_file():
                    added.append(str(f.relative_to(repo)))
            continue

        # link → Drive: copy each file from Drive into the repo tree
        if link.is_symlink():
            link.unlink()              # remove the symlink
        link.mkdir(parents=True, exist_ok=True)

        if not real.exists():
            print(f'[warn] Drive path does not exist: {real} — skipping {sub}/')
            continue

        for src_file in real.glob('*'):
            if not src_file.is_file():
                continue
            dst_file = link / src_file.name
            shutil.copy2(str(src_file), str(dst_file))
            rel = str(dst_file.relative_to(repo))
            added.append(rel)
            print(f'  [copied] {rel}')

    return added


# ── main ───────────────────────────────────────────────────────────────────────

_git(['config', 'user.email', 'yeonkim112599@gmail.com'])
_git(['config', 'user.name', 'Seoyeon Kim'])

_repo  = Path(REPO_DIR)   # noqa: F821
_files = _copy_drive_results_to_repo(_repo)

if not _files:
    print('No result files found — nothing to push.')
else:
    # Add only files that exist; never stage deletions of other tracked files
    _git(['add', '--'] + _files)

    _status = subprocess.run(
        ['git', '-C', REPO_DIR, 'status', '--porcelain'],  # noqa: F821
        capture_output=True, text=True,
    )
    if not _status.stdout.strip():
        print('Nothing to commit — results already up to date.')
    else:
        _git(['commit', '-m', COMMIT_MSG])   # noqa: F821
        _git(['push'])
        print(f'Pushed {len(_files)} file(s): {COMMIT_MSG}')  # noqa: F821
