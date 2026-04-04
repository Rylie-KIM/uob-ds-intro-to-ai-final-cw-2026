from pathlib import Path
import sys
# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent.parent.parent.parent.parent
# _HERE     = Path(__file__).resolve().parent
# _DATA_DIR = _ROOT / 'src' / 'data' / 'type-b'
# _OUT_DIR  = _HERE / 'results'

# Add repo root and non-pretrained dir (hyphenated name can't be imported normally)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src' / 'embeddings' / 'non-pretrained'))

from one_hot_encoding import OneHot

onehot = OneHot()

## Add OneHot Embeddings

onehot.process()

sys.path.insert(0, str(_ROOT / 'src' / 'embeddings' / 'pretrained'))

from tinybert_pooler_embeddings import TinyBertPoolerEmbedder

tinyBert_embedder = TinyBertPoolerEmbedder()
tinyBert_embedder.process()

from sbert_embeddings import SBERTEmbedder

sbert_embedder = SBERTEmbedder()

## Configure sbert class so it can update the master csv....
# Check out my tinybert and och embedding objects for how i did it - if you have a better way then just implement it

from pretrained_word2vec_embeddings import PretrainedWord2VecEmbedder

pt_word2vec = PretrainedWord2VecEmbedder()









