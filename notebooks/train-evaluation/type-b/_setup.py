"""
notebooks/train-evaluation/type-b/_setup.py

Common setup for all Type-B experiment notebooks.
Run this via:  %run _setup.py

Requires these variables to be defined before calling:
    MODE            : 'local' or 'colab'
    LOCAL_REPO_DIR  : absolute path to repo (local mode)
    GITHUB_REPO_URL : GitHub repo URL (colab mode)
    DRIVE_BASE      : Google Drive base path (colab mode)
"""

import os, sys, subprocess, shutil, time
from pathlib import Path

if MODE == 'local':
    REPO_DIR = LOCAL_REPO_DIR
    print(f'Local mode. Repo: {REPO_DIR}')

elif MODE == 'colab':
    from google.colab import drive, userdata
    drive.mount('/content/drive')

    REPO_DIR      = '/content/repo'
    DRIVE_DATA    = f'{DRIVE_BASE}/data'
    DRIVE_RESULTS = f'{DRIVE_BASE}/results'

    # ── Clone or pull repo ─────────────────────────────────────────────────────
    try:
        token = userdata.get('GITHUB_TOKEN')
        repo_url_auth = GITHUB_REPO_URL.replace('https://', f'https://{token}@')
    except Exception:
        print('WARNING: GITHUB_TOKEN not found in Colab Secrets.')
        repo_url_auth = GITHUB_REPO_URL

    if not os.path.exists(REPO_DIR):
        print('Cloning repo...')
        subprocess.run(['git', 'clone', repo_url_auth, REPO_DIR], check=True)
    else:
        print('Repo exists. Pulling latest...')
        subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)

    # ── Copy images to Colab local disk (file-by-file, resume-safe) ───────────
    LOCAL_IMG_BASE = Path('/content/images')
    for dtype in ['type-b']:
        local_img = LOCAL_IMG_BASE / dtype
        drive_img = Path(DRIVE_DATA) / 'images' / dtype
        repo_img  = Path(REPO_DIR) / 'src' / 'data' / 'images' / dtype

        local_img.mkdir(parents=True, exist_ok=True)
        src_files   = sorted(drive_img.glob('*.png'))
        total       = len(src_files)
        local_files = set(f.name for f in local_img.glob('*.png'))
        missing     = [f for f in src_files if f.name not in local_files]

        if not missing:
            print(f'[skip] images/{dtype} — all {total} files on local disk')
        else:
            print(f'Copying {len(missing)}/{total} images ({dtype})  ({total - len(missing)} already exist)')
            t0 = time.time()
            for i, src in enumerate(missing, 1):
                shutil.copy2(str(src), str(local_img / src.name))
                if i % 500 == 0 or i == len(missing):
                    elapsed = time.time() - t0
                    eta     = elapsed / i * (len(missing) - i)
                    print(f'  {i}/{len(missing)} ({i/len(missing)*100:.0f}%)  elapsed={elapsed:.0f}s  eta={eta:.0f}s', end='\r')
            print(f'\n[done] {len(missing)} images copied in {time.time()-t0:.1f}s')

        if repo_img.is_symlink():
            os.unlink(str(repo_img))
        elif repo_img.exists():
            shutil.rmtree(str(repo_img))
        os.symlink(str(local_img), str(repo_img))
        print(f'[symlinked] images/{dtype} → local disk')

    # ── Symlink results → Drive ────────────────────────────────────────────────
    for sub in ['checkpoints', 'metrics', 'figures']:
        drive_sub = Path(DRIVE_RESULTS) / sub
        drive_sub.mkdir(parents=True, exist_ok=True)
        repo_sub = Path(REPO_DIR) / 'src' / 'pipelines' / 'results' / sub
        repo_sub.parent.mkdir(parents=True, exist_ok=True)
        if not repo_sub.exists():
            os.symlink(str(drive_sub), str(repo_sub))
            print(f'[symlinked] results/{sub} → Drive')

    # ── Symlink embedding .pt → Drive ─────────────────────────────────────────
    drive_emb = Path(DRIVE_RESULTS) / 'embeddings'
    drive_emb.mkdir(parents=True, exist_ok=True)
    repo_emb = Path(REPO_DIR) / 'src' / 'embeddings' / 'computed-embeddings' / 'type-b' / 'results'
    repo_emb.mkdir(parents=True, exist_ok=True)
    for pt in drive_emb.glob('*.pt'):
        dest = repo_emb / pt.name
        if not dest.exists():
            os.symlink(str(pt), str(dest))
            print(f'[symlinked] {pt.name}')

    print(f'\nColab mode ready. Repo: {REPO_DIR}')

# ── sys.path ───────────────────────────────────────────────────────────────────
for p in [
    REPO_DIR,
    str(Path(REPO_DIR) / 'src'),
    str(Path(REPO_DIR) / 'src' / 'embeddings' / 'non-pretrained'),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Device ────────────────────────────────────────────────────────────────────
import torch
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')
if device == 'cuda':
    print(f'GPU   : {torch.cuda.get_device_name(0)}')
