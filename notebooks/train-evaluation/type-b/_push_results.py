"""
notebooks/train-evaluation/type-b/_push_results.py

Push training results (figures, metrics) to GitHub.
Run this via:  %run -i _push_results.py

Requires REPO_DIR and COMMIT_MSG to be defined in the notebook before calling.
    COMMIT_MSG = 'Add results: cnn_1layer x tfidf_lsa'

Why copy instead of git-add through symlink:
  results dirs are symlinked to Google Drive (/content/drive/...).
  Git treats the Drive path as outside the working tree and returns exit 128.
  We copy the files back into the repo, add them, commit, then push.
"""

import shutil
import subprocess
from pathlib import Path


def _git(args):
    subprocess.run(['git', '-C', REPO_DIR] + args, check=True)


_git(['config', 'user.email', 'yeonkim112599@gmail.com'])
_git(['config', 'user.name', 'Seoyeon Kim'])

# ── Copy Drive-backed results into the actual repo tree ────────────────────────
_repo = Path(REPO_DIR)
for _sub in ['figures', 'metrics']:
    _src = _repo / 'src' / 'pipelines' / 'results' / _sub   # may be a symlink → Drive
    _dst = _repo / 'src' / 'pipelines' / 'results' / _sub

    # Resolve the real source path (Drive) if it's a symlink
    _real_src = _src.resolve()

    if _real_src != _dst:
        # Remove the symlink and replace with a real directory containing copies
        if _dst.is_symlink():
            _dst.unlink()
        _dst.mkdir(parents=True, exist_ok=True)
        for _f in _real_src.iterdir():
            if _f.is_file():
                shutil.copy2(str(_f), str(_dst / _f.name))
        print(f'[copied] Drive/{_sub}/ → repo/src/pipelines/results/{_sub}/')
    else:
        print(f'[skip copy] {_sub}/ is already a real directory')

# ── Stage and push ─────────────────────────────────────────────────────────────
_git(['add',
      'src/pipelines/results/figures/',
      'src/pipelines/results/metrics/'])

_status = subprocess.run(
    ['git', '-C', REPO_DIR, 'status', '--porcelain'],
    capture_output=True, text=True
)
if not _status.stdout.strip():
    print('Nothing to commit — results already up to date.')
else:
    _git(['commit', '-m', COMMIT_MSG])
    _git(['push'])
    print(f'Pushed: {COMMIT_MSG}')
