
import argparse, pathlib
import sentencepiece as spm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, required=True, help="Path to raw text file (one or more, comma-separated)")
    p.add_argument("--vocab", type=int, default=48000)
    p.add_argument("--out_prefix", type=str, default="tokenizer_spm_48k")
    args = p.parse_args()

    inputs = args.corpus.split(",")
    input_str = ",".join(inputs)

    spm.SentencePieceTrainer.Train(
        input=input_str,
        model_prefix=args.out_prefix,
        vocab_size=args.vocab,
        character_coverage=0.9995,
        model_type="unigram",
        user_defined_symbols=[
            "BTC","ETH","SOL","ADA","Layer2","rollup","mempool","slippage","basis","funding rate",
            "ENTSO-E","grid load","kWh","capacity factor","dispatch","curtailment",
            "JVL","EtherCAT","ROS2","UR","AMR","SDK","API","RAG","vector DB"
        ]
    )
    print(f"Tokenizer written: {args.out_prefix}.model / .vocab")

if __name__ == "__main__":
    main()
