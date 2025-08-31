from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from .views import ViewEncoders, cos_sim
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@dataclass
class RetrievalWeights:
    alpha_os: float
    alpha_sr: float
    alpha_as: float
    alpha_dep: float

def mmr_select(cands: List[int],
               sim_matrix: np.ndarray,
               k: int,
               beta: float) -> List[int]:
    selected = []
    remaining = set(cands)
    while remaining and len(selected) < k:
        if not selected:
            i = int(np.argmax(sim_matrix.mean(axis=1)))
            selected.append(i); remaining.discard(i); continue
        best_i, best_score = None, -1e9
        for i in list(remaining):
            rel = sim_matrix[i].mean()
            div = max([sim_matrix[i, j] for j in selected] + [0.0])
            score = beta * rel - (1 - beta) * div
            if score > best_score:
                best_i, best_score = i, score
        selected.append(best_i); remaining.discard(best_i)
    return selected

class RaSVRetriever:
    def __init__(self, enc: ViewEncoders, weights: RetrievalWeights,
                 lambda_uncert: float = 0.5, device: str = "cuda",
                 llm_name: str = None):
        self.enc = enc
        self.w = weights
        self.lambda_uncert = lambda_uncert
        self.device = device
        if llm_name:
            self.tok = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name, device_map="auto")
        else:
            self.tok = self.llm = None

    def _joint_score(self, hx: Dict[str,np.ndarray], hj: Dict[str,np.ndarray]) -> float:
        s = 0.0
        s += self.w.alpha_os  * cos_sim(hx["os"],  hj["os"])
        s += self.w.alpha_sr  * cos_sim(hx["sr"],  hj["sr"])
        s += self.w.alpha_as  * cos_sim(hx["as"],  hj["as"])
        s += self.w.alpha_dep * cos_sim(hx["dep"], hj["dep"])
        return s

    def _entropy_probe(self, prompt: str) -> float:
        if self.llm is None: return 0.0
        toks = self.tok(prompt, return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            out = self.llm(**toks)
        logits = out.logits[0, -1]  # next-token
        p = torch.softmax(logits, dim=-1)
        h = -(p * torch.log(p + 1e-12)).sum().item()
        return h

    def encode_views(self, x: str) -> Dict[str,np.ndarray]:
        return {
            "os": self.enc.h_os(x),
            "sr": self.enc.h_sr(x),
            "as": self.enc.h_as(x),
            "dep": self.enc.h_dep(x),
        }

    def rank(self, x_star: str, pool_texts: List[str], probe_template: str = None) -> List[Tuple[int,float]]:
        hx = self.encode_views(x_star)
        H_pool = [self.encode_views(t) for t in pool_texts]
        base = np.array([self._joint_score(hx, hj) for hj in H_pool], dtype=float)

        if self.llm and probe_template:
            h_star = self._entropy_probe(probe_template.format(x=x_star))
            discounts = []
            for t in pool_texts:
                h_with = self._entropy_probe(probe_template.format(x=x_star+"\nDemo:\n"+t))
                discounts.append(h_with - h_star)
            base = base - self.lambda_uncert * np.array(discounts)

        order = np.argsort(-base)
        return [(int(i), float(base[i])) for i in order]

    def select(self, x_star: str, pool_texts: List[str], k: int, beta: float, probe_template: str=None) -> List[int]:
        scores = self.rank(x_star, pool_texts, probe_template)
        idx = [i for i,_ in scores]
        # build simple sim matrix for MMR using joint scores pairwise
        H = [self.encode_views(t) for t in pool_texts]
        simM = np.zeros((len(H), len(H)))
        for i in range(len(H)):
            for j in range(len(H)):
                simM[i,j] = self._joint_score(H[i], H[j])
        chosen_local = mmr_select(idx, simM, k, beta)
        return chosen_local[:k]
