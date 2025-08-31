from dataclasses import dataclass
from typing import List, Dict
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from tqdm.auto import tqdm

class MultiViewContrastive:
    def __init__(self, os_name, sr_name, as_name, dep_name):
        self.os = SentenceTransformer(os_name)
        self.sr = SentenceTransformer(sr_name)
        self.asv= SentenceTransformer(as_name)
        self.dep= SentenceTransformer(dep_name)

    def encode_views(self, x:str, tuples=None):
        return {
            "os": self.os.encode([x], normalize_embeddings=True)[0],
            "sr": self.sr.encode([" ".join(t.pos_ for t in self.sr[0].tokenize(x))], normalize_embeddings=True)[0] if hasattr(self.sr, "tokenize") else self.sr.encode([x], normalize_embeddings=True)[0],
            "as": self.asv.encode([x], normalize_embeddings=True)[0],
            "dep": self.dep.encode([x], normalize_embeddings=True)[0],
        }

class InfoNCEDataset(Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]["text"]

def train_retriever(train_rows: List[Dict], out_dir: str,
                    os_name, sr_name, as_name, dep_name,
                    batch_size=64, lr=2e-5, epochs=3, temperature=0.07):
    # For simplicity, train 4 independent SBERT models with the same supervised triplets
    def to_examples(rows):
        ex=[]
        for r in rows:
            x = r["text"]
            # use positive as slight augmentation (self-positive)
            ex.append(InputExample(texts=[x, x]))
        return ex

    for name in [os_name, sr_name, as_name, dep_name]:
        model = SentenceTransformer(name)
        train_dl = DataLoader(to_examples(train_rows), batch_size=batch_size, shuffle=True)
        loss = losses.MultipleNegativesRankingLoss(model)
        warmup = max(100, int(len(train_dl)*epochs*0.1))
        model.fit(train_objectives=[(train_dl, loss)],
                  epochs=epochs, warmup_steps=warmup, output_path=f"{out_dir}/{name.split('/')[-1]}")
