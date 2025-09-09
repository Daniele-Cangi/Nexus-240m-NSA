# Nexus-240m-NSA
A compact, dual-head embedding model (semantic + entity) with Neural Spectral Anchoring (NSA) and Residual Hashing for efficient vector search on edge devices.
[README.md](https://github.com/user-attachments/files/22231916/README.md)
# NEXUS-EMB-240M-NSA (Starter Kit)

A compact, dual-head embedding model optimized for inference on low-power devices (`edge-first`). This starter kit is designed to give you a significant head start, providing the necessary tools to train, evaluate, and export a state-of-the-art custom embedding model for your data.

### Key Features That Make a Difference

The NEXUS-EMB-240M-NSA architecture stands out by integrating non-conventional techniques that enhance both its precision and efficiency.

* **Dual-Head Architecture (Semantic & Entity)**: Unlike traditional models that produce a single vector, this model generates two separate vectors: one focused on understanding general meaning (`semantic`) and another on identifying specific entities and terms (`entity`). Combining them creates a final 768-dimensional embedding that is exceptionally rich and detailed, improving accuracy in complex search tasks.
* **Neural Spectral Anchoring (NSA)**: This advanced feature maps embeddings into a spectral space, optimizing the relationships between data in a way that makes vector search not only more accurate but also more efficient. It's an innovative technique that goes beyond standard supervised training to give the model a deeper understanding of your data's structure.
* **Residual Hashing Bridge**: For applications that demand extreme search speed, the integrated 64-bit hashing bridge supports rapid candidate pre-filtering. This dramatically reduces the search space and accelerates queries without sacrificing final accuracy.
* **Matryoshka Embeddings**: Flexibility is key. The model supports resizing its final embeddings to 768, 512, or 256 dimensions. This allows you to tailor the model to your application and device requirements, optimally balancing performance and memory needs.

---

### Quickstart Guide to Your First Training Run

This repository is optimized for immediate use in Google Colab. Follow these steps to train, evaluate, and export the model.

1.  **Environment Setup**
    ```bash
    pip install torch==2.4.0 transformers sentencepiece einops faiss-cpu
    ```

2.  **Train the Tokenizer**
    Train a custom SentencePiece tokenizer on your text corpus to ensure the model understands your specific domain.
    ```bash
    python scripts/build_tokenizer.py --corpus path/to/your/corpus.txt --vocab 48000 --out_prefix tokenizer_spm_48k
    ```

3.  **Train the Model**
    Run the core model training using your own data pairs.
    ```bash
    python scripts/train.py \
      --config configs/nexus_emb_240m.json \
      --pairs data/your_pairs.jsonl \
      --tokenizer_model tokenizer_spm_48k.model \
      --batch 64 --max_len 128 --steps 1000
    ```

4.  **Basic Evaluation**
    Perform a quick evaluation to verify the model is learning correctly, ensuring positive pairs are closer to each other than negative ones.
    ```bash
    python scripts/eval_mteb_lite.py \
      --config configs/nexus_emb_240m.json \
      --tokenizer_model tokenizer_spm_48k.model
    ```

5.  **Export to ONNX**
    Export the model to a standardized format optimized for inference across a wide range of hardware.
    ```bash
    python scripts/export_onnx.py \
      --config configs/nexus_emb_240m.json \
      --out artifacts/nexus_emb_240m_nsa.onnx --seq_len 128
    ```

---

### Notes for Advanced Training

* **For a full and robust training run**, we strongly recommend implementing techniques like **`hard-negative mining`** and **`Knowledge Distillation (KD)`** from larger, more powerful teacher models.
* To ensure stability on smaller hardware, features like **`RoPE`** and **`FlashAttention`** are disabled by default. You can enable them later to further optimize performance.
