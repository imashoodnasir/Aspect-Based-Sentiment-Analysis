from dataclasses import dataclass
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy

@dataclass
class ViewEncoders:
    os_model: SentenceTransformer
    sr_model: SentenceTransformer
    as_model: SentenceTransformer
    dep_model: SentenceTransformer
    nlp: any

    @classmethod
    def build(cls, os_name, sr_name, as_name, dep_name):
        return cls(
            os_model=SentenceTransformer(os_name),
            sr_model=SentenceTransformer(sr_name),
            as_model=SentenceTransformer(as_name),
            dep_model=SentenceTransformer(dep_name),
            nlp=spacy.load("en_core_web_sm")
        )

    def h_os(self, x: str) -> np.ndarray:
        return self.os_model.encode([x], normalize_embeddings=True)[0]

    def h_sr(self, x: str) -> np.ndarray:
        doc = self.nlp(x)
        pos = " ".join([t.pos_ for t in doc])
        return self.sr_model.encode([pos], normalize_embeddings=True)[0]

    def h_as(self, x: str, tuples_gold=None) -> np.ndarray:
        # If gold tuples known (training), build AS-Text; else, fallback to sentence itself.
        if tuples_gold:
            templ = "; ".join([f"The aspect {a} is {s}" for a,s in tuples_gold])
        else:
            templ = x
        return self.as_model.encode([templ], normalize_embeddings=True)[0]

    def h_dep(self, x: str) -> np.ndarray:
        doc = self.nlp(x)
        motifs = []
        for tok in doc:
            if tok.dep_ in ("amod","acomp","dobj","nsubj","cop","advmod"):
                motifs.append(f"{tok.head.pos_}->{tok.dep_}->{tok.pos_}")
        sk = " ".join(motifs) if motifs else "none"
        return self.dep_model.encode([sk], normalize_embeddings=True)[0]

def cos_sim(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8))
