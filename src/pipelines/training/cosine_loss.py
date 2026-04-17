import torch.nn as nn
import torch.nn.functional as F

class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_emb, truth_emb):
        return (1.0 - F.cosine_similarity(pred_emb, truth_emb, dim=1)).mean()