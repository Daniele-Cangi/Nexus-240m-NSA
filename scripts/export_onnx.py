
import argparse, json, torch, os
from src.model import NexusEmb240MNSA

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None, help="optional .pt state dict")
    p.add_argument("--out", type=str, default="nexus_emb_240m_nsa.onnx")
    p.add_argument("--seq_len", type=int, default=128)
    args = p.parse_args()

    cfg = json.load(open(args.config))
    model = NexusEmb240MNSA(cfg)
    if args.checkpoint and os.path.exists(args.checkpoint):
        sd = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(sd.get("model", sd))

    model.eval()
    dummy = torch.randint(0, cfg["vocab_size"], (1, args.seq_len), dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy,),
            args.out,
            input_names=["input_ids"],
            output_names=["embedding","topic_logits","hash_logits","hash_bits","spectral_out"],
            dynamic_axes={"input_ids":{0:"batch",1:"seq"}},
            opset_version=17
        )
    print(f"Exported ONNX: {args.out}")

if __name__ == "__main__":
    main()
