from pathlib import Path
import sys
# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent.parent.parent.parent.parent

# Add repo root and non-pretrained dir (hyphenated name can't be imported normally)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src' / 'embeddings' / 'pretrained'))

from only_type_a_TB_pooler import TinyBertPoolerEmbedder

tinyBert_embedder = TinyBertPoolerEmbedder()
tinyBert_embedder.process()

from only_type_a_TB_mean import TinyBertMeanEmbedder

tinyBert_mean = TinyBertMeanEmbedder()
tinyBert_mean.process()

from only_type_a_B_pooler import BertPoolerEmbedder

bert_pooler = BertPoolerEmbedder()
bert_pooler.process()

from only_type_a_B_mean import BertMeanEmbedder
bert_mean = BertMeanEmbedder()
bert_mean.process()

from only_type_a_sbert_embeddings import SBERTEmbedder

sbert_embedder = SBERTEmbedder()
sbert_embedder.process()

from only_type_a_p_w2v import PretrainedWord2VecEmbedder

p_w2vec = PretrainedWord2VecEmbedder()
p_w2vec.process()





