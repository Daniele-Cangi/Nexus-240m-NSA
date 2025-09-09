
import json
from torch.utils.data import Dataset

class PairDataset(Dataset):
    """Expects JSONL lines with: {"text_a": str, "text_b": str, "topic": int}"""
    def __init__(self, path):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                self.items.append((j["text_a"], j["text_b"], j.get("topic", 0)))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]
