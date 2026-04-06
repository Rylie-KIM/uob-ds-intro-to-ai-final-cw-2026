"""
notebooks/train-evaluation/type-b/_push_results.py

Push training results (figures, metrics) to GitHub.
Run this via:  %run _push_results.py

Requires REPO_DIR and COMMIT_MSG to be defined before calling.
    COMMIT_MSG = 'Add results: cnn_1layer x tfidf_lsa'
"""

import subprocess

def _git(args):
    subprocess.run(['git', '-C', REPO_DIR] + args, check=True)

_git(['config', 'user.email', 'yeonkim112599@gmail.com'])
_git(['config', 'user.name', 'Seoyeon Kim'])

_git(['add',
      'src/pipelines/results/figures/',
      'src/pipelines/results/metrics/'])

result = subprocess.run(
    ['git', '-C', REPO_DIR, 'status', '--porcelain'],
    capture_output=True, text=True
)
if not result.stdout.strip():
    print('Nothing to commit — results already up to date.')
else:
    _git(['commit', '-m', COMMIT_MSG])
    _git(['push'])
    print(f'Pushed: {COMMIT_MSG}')
