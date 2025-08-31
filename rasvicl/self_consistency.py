from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .prompting import build_prompt

@dataclass
class SCDecoder:
    model: AutoModelForCausalLM
    tok: AutoTokenizer
    device: str = "cuda"

    @classmethod
    def build(cls, llm_name: str):
        tok = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(llm_name, device_map="auto")
        return cls(model=model, tok=tok)

    def decode_once(self, prompt: str, max_new_tokens=128, temperature=0.7, top_p=0.95) -> str:
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **ids, do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0], skip_special_tokens=True)
        return text.split("Answer:")[-1].strip()

def parse_tuples(s: str) -> List[Tuple[str,str]]:
    # Expect a bracketed list of (aspect, polarity). Robust fallback parsing.
    import re
    pairs = re.findall(r"\(([^,]+),\s*([^)]+)\)", s)
    return [(a.strip(), b.strip()) for a,b in pairs]

def majority_vote(list_of_tuple_sets: List[List[Tuple[str,str]]], pi=0.5) -> List[Tuple[str,str]]:
    flat=[]
    for ts in list_of_tuple_sets:
        flat += ts
    uniq = set(flat)
    out=[]
    for z in uniq:
        cnt = sum([1 for ts in list_of_tuple_sets if z in ts])
        if cnt/len(list_of_tuple_sets) >= pi:
            out.append(z)
    return out
