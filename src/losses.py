
import math
import torch
import torch.nn.functional as F

def info_nce(emb_a, emb_b, temperature=0.07):
    emb_a = F.normalize(emb_a, dim=-1)
    emb_b = F.normalize(emb_b, dim=-1)
    logits = emb_a @ emb_b.t() / temperature
    labels = torch.arange(emb_a.size(0), device=emb_a.device)
    return F.cross_entropy(logits, labels)

def topic_loss(logits, targets):
    return F.cross_entropy(logits, targets)

def hashing_loss(hash_logits, emb):
    # Push logits to saturate and align with a random projection sign pattern
    l_bin = (1 - torch.tanh(hash_logits).abs()).mean()
    with torch.no_grad():
        rp = torch.randn(emb.size(-1), hash_logits.size(-1), device=emb.device) / math.sqrt(emb.size(-1))
        target = (emb @ rp >= 0).float() * 2 - 1
    l_align = F.mse_loss(torch.tanh(hash_logits), target)
    return 0.5 * l_bin + 0.5 * l_align

def spectral_loss(spec_a, spec_b):
    # Encourage similar spectral signatures for positives
    return 1 - F.cosine_similarity(spec_a, spec_b, dim=-1).mean()
