
import argparse, json, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model import NexusEmb240MNSA
from src.losses import info_nce, topic_loss, hashing_loss, spectral_loss
from src.data_loader import PairDataset
from src.utils import set_seed

def encode_batch(tokenizer, texts, max_len=128, device="cuda"):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc["input_ids"].to(device)

def collate_fn(batch, tokenizer, max_len, device):
    A = [a for a,b,t in batch]; B = [b for a,b,t in batch]
    T = torch.tensor([t for a,b,t in batch], dtype=torch.long, device=device)
    a_ids = encode_batch(tokenizer, A, max_len=max_len, device=device)
    b_ids = encode_batch(tokenizer, B, max_len=max_len, device=device)
    return a_ids, b_ids, T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--tokenizer_model", type=str, required=True, help="sentencepiece model path")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="ckpts")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    cfg = json.load(open(args.config))
    model = NexusEmb240MNSA(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Load a simple HF tokenizer wrapper for the local SentencePiece
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=False, trust_remote_code=True)

    ds = PairDataset(args.pairs)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len, device), drop_last=True)

    step = 0
    model.train()
    while step < args.steps:
        for a_ids, b_ids, topics in dl:
            emb_a, logit_a, hash_a, _, spec_a = model(a_ids)
            emb_b, logit_b, hash_b, _, spec_b = model(b_ids)

            L_main = info_nce(emb_a, emb_b, temperature=0.08)
            L_topic = 0.05*(topic_loss(logit_a, topics) + topic_loss(logit_b, topics))
            L_hash  = 0.02*(hashing_loss(hash_a, emb_a) + hashing_loss(hash_b, emb_b))
            L_spec  = 0.05*spectral_loss(spec_a, spec_b)

            loss = L_main + L_topic + L_hash + L_spec
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)

            step += 1
            if step % 50 == 0:
                print(f"step {step}: loss={loss.item():.4f} (main={L_main.item():.4f}, topic={L_topic.item():.4f}, hash={L_hash.item():.4f}, spec={L_spec.item():.4f})")
            if step % 500 == 0:
                ckpt = os.path.join(args.save_dir, f"step_{step}.pt")
                torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt)
            if step >= args.steps:
                break
    print("Training done.")

if __name__ == "__main__":
    main()
