
import argparse, json, numpy as np, torch
from transformers import AutoTokenizer
from src.model import NexusEmb240MNSA

def embed_texts(model, tokenizer, texts, max_len=128, device="cuda"):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        emb, *_ = model(enc["input_ids"].to(device))
        return emb.cpu().numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tokenizer_model", type=str, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(args.config))
    model = NexusEmb240MNSA(cfg).to(device)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=False, trust_remote_code=True)

    # Tiny toy eval: check that similar pairs are closer than random
    A = ["BTC funding rates rise with open interest", "ENTSO-E publishes grid load metrics"]*32
    B_pos = ["Open interest increases often drive funding rate changes", "European grid load data via ENTSO-E"]*32
    B_neg = ["Banana smoothie recipe and tips", "How to paint a wall quickly"]*32

    Ea = embed_texts(model, tok, A)
    Eb_pos = embed_texts(model, tok, B_pos)
    Eb_neg = embed_texts(model, tok, B_neg)

    def cos(a,b): return (a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+1e-9)
    s_pos = cos(Ea, Eb_pos).mean()
    s_neg = cos(Ea, Eb_neg).mean()
    print(json.dumps({"cos_pos_mean": float(s_pos), "cos_neg_mean": float(s_neg), "margin": float(s_pos - s_neg)}, indent=2))

if __name__ == "__main__":
    main()
