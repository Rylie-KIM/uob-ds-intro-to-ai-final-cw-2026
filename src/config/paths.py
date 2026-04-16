"""
src/config/paths.py
Centralised path constants for the project.
Import from here instead of constructing paths in individual scripts.
"""
from pathlib import Path

# Project root  (src/config/paths.py -> src/config -> src -> root)
ROOT = Path(__file__).resolve().parent.parent.parent

# ── Type-B ────────────────────────────────────────────────────────────────────
TYPE_B_SENTENCES = ROOT / 'src' / 'data' / 'type-b' / 'sentences_b.csv'
TYPE_B_IMAGE_MAP = ROOT / 'src' / 'data' / 'type-b' / 'image_map_b.csv'
TYPE_B_IMAGES    = ROOT / 'src' / 'data' / 'images' / 'type-b'
TYPE_B_SPLITS    = ROOT / 'src' / 'data' / 'type-b' / 'splits'

# ── Type-A ────────────────────────────────────────────────────────────────────
TYPE_A_SENTENCES = ROOT / 'src' / 'data' / 'type-a' / 'sentences_a.csv'
TYPE_A_IMAGE_MAP = ROOT / 'src' / 'data' / 'type-a' / 'image_map_a.csv'
TYPE_A_IMAGES    = ROOT / 'src' / 'data' / 'images' / 'type-a'

# ── Type-C ────────────────────────────────────────────────────────────────────
TYPE_C_SENTENCES = ROOT / 'src' / 'data' / 'type-c' / 'sentences_c.csv'
TYPE_C_IMAGE_MAP = ROOT / 'src' / 'data' / 'type-c' / 'image_map_c.csv'
TYPE_C_IMAGES    = ROOT / 'src' / 'data' / 'images' / 'type-c'

# ── Pre-computed embeddings ───────────────────────────────────────────────────
EMBED_RESULTS_B = ROOT / 'src' / 'embeddings' / 'computed-embeddings' / 'type-b' / 'results'
EMBED_RESULTS_A = ROOT / 'src' / 'embeddings' / 'computed-embeddings' / 'type-a' / 'results'
EMBED_RESULTS_C = ROOT / 'src' / 'embeddings' / 'computed-embeddings' / 'type-c' / 'results'

# ── Training outputs ──────────────────────────────────────────────────────────
CHECKPOINTS_B            = ROOT / 'src' / 'pipelines' / 'results' / 'checkpoints' / 'type-b'
CHECKPOINTS_B_NORMED     = CHECKPOINTS_B / 'normalised'
CHECKPOINTS_B_S2_NON_NORMED = CHECKPOINTS_B / 's2' / 'non-normalised'
CHECKPOINTS_B_S2_NORMED     = CHECKPOINTS_B / 's2' / 'normalised'

METRICS_B            = ROOT / 'src' / 'pipelines' / 'results' / 'metrics' / 'type-b'
METRICS_B_NON_NORMED = METRICS_B / 'non-normalised'    # training logs for base runs
METRICS_B_NORMED     = METRICS_B / 'normalised'        # training logs for normalised runs
METRICS_B_S2_NON_NORMED = METRICS_B / 's2-non-normalised'
METRICS_B_S2_NORMED     = METRICS_B / 's2-normalised'
PREDICTIONS_B               = METRICS_B / 'prediction'
PREDICTIONS_B_NORMED        = METRICS_B / 'prediction-normalised'
PREDICTIONS_B_S2            = METRICS_B / 'prediction-s2'
PREDICTIONS_B_S2_NORMED     = METRICS_B / 'prediction-s2-normalised'
PREDICTIONS_B_COMBINED      = METRICS_B / 'prediction-combined'
PREDICTIONS_B_COMMERCIAL_AI = METRICS_B / 'prediction-commercial-ai'
METRICS_FINAL        = ROOT / 'src' / 'pipelines' / 'results' / 'metrics' / 'type-b' / 'final'

# ── Fine-tuned SBERT checkpoints ─────────────────────────────────────────────
SBERT_CKPT_B = ROOT / 'results' / 'checkpoints' / 'sbert_finetuned_typeb'
SBERT_CKPT_A = ROOT / 'results' / 'checkpoints' / 'sbert_finetuned_typea'
SBERT_CKPT_C = ROOT / 'results' / 'checkpoints' / 'sbert_finetuned_typec'

# ── Results / figures ─────────────────────────────────────────────────────────
RESULTS_DIR          = ROOT / 'results'
FIGURES_DIR          = ROOT / 'src' / 'pipelines' / 'results' / 'figures' / 'type-b'
FIGURES_EVAL_B              = FIGURES_DIR / 'evaluation'
FIGURES_EVAL_NORM_B         = FIGURES_DIR / 'evaluation' / 'normalised'
FIGURES_EVAL_CMP_B          = FIGURES_DIR / 'evaluation' / 'comparison'
FIGURES_EVAL_B_S2           = FIGURES_DIR / 'evaluation' / 's2'
FIGURES_EVAL_B_S2_NON_NORMED = FIGURES_DIR / 'evaluation' / 's2' / 'non-normalised'
FIGURES_EVAL_B_S2_NORMED    = FIGURES_DIR / 'evaluation' / 's2' / 'normalised'
FIGURES_EVAL_B_COMBINED     = FIGURES_DIR / 'evaluation' / 'combined'

# ── Config directory ──────────────────────────────────────────────────────────
CONFIG_DIR = ROOT / 'src' / 'config'


if __name__ == '__main__':
    pairs = [
        ('ROOT',            ROOT),
        ('TYPE_B_SENTENCES', TYPE_B_SENTENCES),
        ('TYPE_B_IMAGE_MAP', TYPE_B_IMAGE_MAP),
        ('TYPE_B_IMAGES',    TYPE_B_IMAGES),
        ('EMBED_RESULTS_B',  EMBED_RESULTS_B),
        ('CHECKPOINTS_B',    CHECKPOINTS_B),
        ('METRICS_B',        METRICS_B),
    ]
    for name, p in pairs:
        print(f'{name:<22} {p}  [{"OK" if p.exists() else "missing"}]')
