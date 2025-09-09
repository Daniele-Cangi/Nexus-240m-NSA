
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Norms & Activations ----------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# ---------- Transformer Block (Colab-friendly) ----------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden, dropout=0.05, rope_theta=10000.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = SwiGLU(d_model, ffn_hidden)
        self.rope_theta = rope_theta  # reserved for a fuller RoPE implementation
    def forward(self, x, attn_mask=None):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x

# ---------- Topic-Aware Pooling ----------
class TopicAwarePooling(nn.Module):
    def __init__(self, d_model, n_topics=16):
        super().__init__()
        self.cls_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.topic_head = nn.Linear(d_model, n_topics)
    def forward(self, x):
        # x: [B,T,C]; use token 0 as CLS surrogate
        cls = x[:, 0]
        mean = x.mean(dim=1)
        w = torch.sigmoid(self.cls_gate(cls))  # [B,1]
        pooled = w * cls + (1 - w) * mean
        logits = self.topic_head(pooled)  # [B,topics]
        return pooled, logits

# ---------- Residual Hashing Bridge ----------
class ResidualHashingBridge(nn.Module):
    def __init__(self, in_dim, bits=64):
        super().__init__()
        self.proj = nn.Linear(in_dim, bits)
    def forward(self, x):
        z = self.proj(x)
        b = (z >= 0).float() * 2 - 1  # STE-like hard sign (in forward)
        return z, b

# ---------- Full Model with Dual-Head + NSA ----------
class NexusEmb240MNSA(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg["d_model"], cfg["n_heads"], cfg["ffn_hidden"], cfg["dropout"], cfg["rope_theta"])
            for _ in range(cfg["n_layers"])
        ])
        self.norm = RMSNorm(cfg["d_model"])
        self.pool = TopicAwarePooling(cfg["d_model"], cfg["topic_head"]["n_topics"])

        # Dual-head projections
        self.head_sem = nn.Linear(cfg["d_model"], cfg["dual_head"]["semantic_dim"])
        self.head_ent = nn.Linear(cfg["d_model"], cfg["dual_head"]["entity_dim"])
        self.gate = nn.Linear(cfg["d_model"], 2)

        # Neural Spectral Anchoring (NSA)
        self.spec_proj = nn.Linear(cfg["d_model"], 256)
        self.freq_matrix = nn.Parameter(torch.randn(256, 64) * 0.02)  # learnable spectral basis
        # Final reducer to 768D
        self.spec_reduce = nn.Linear(cfg["dual_head"]["semantic_dim"] + cfg["dual_head"]["entity_dim"] + 64,
                                     cfg["embedding_dim"])

        # Residual Hashing Bridge
        self.rhb = ResidualHashingBridge(cfg["embedding_dim"], cfg["residual_hashing"]["bits"])

    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids)  # [B,T,C]
        for blk in self.blocks:
            x = blk(x, attn_mask=None)
        x = self.norm(x)
        pooled, topic_logits = self.pool(x)  # [B,C], [B,topics]

        # Dual-head (semantic + entity)
        sem = F.normalize(self.head_sem(pooled), dim=-1)  # [B,512]
        ent = F.normalize(self.head_ent(pooled), dim=-1)  # [B,256]
        g = torch.softmax(self.gate(pooled), dim=-1)      # [B,2]
        sem_scaled = sem * g[:, 0:1]
        ent_scaled = ent * g[:, 1:2]

        # NSA
        spec_in = self.spec_proj(pooled)                 # [B,256]
        spec_freq = torch.matmul(spec_in, self.freq_matrix)  # [B,64]
        spec_out = torch.tanh(spec_freq)                 # [-1,1]

        # Concat + reduce to final emb
        concat = torch.cat([sem_scaled, ent_scaled, spec_out], dim=-1)  # [B, 512+256+64=832]
        emb = self.spec_reduce(concat)                                   # [B,768]
        emb = F.normalize(emb, dim=-1)

        # Hashing (detach emb to keep it stable early on)
        rhb_logits, rhb_bits = self.rhb(emb.detach())
        return emb, topic_logits, rhb_logits, rhb_bits, spec_out
