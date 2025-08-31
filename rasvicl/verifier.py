from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import spacy
from .text_utils import find_all_spans, iou_char, normalize_tuples

@dataclass
class Verifier:
    nlp: any
    tau: float = 0.5
    w: np.ndarray = np.array([0.5, 0.3, 0.2])  # presence, polarity, dependency
    b: float = 0.0
    dep_len_cap: int = 4

    @classmethod
    def build(cls):
        return cls(nlp=spacy.load("en_core_web_sm"))

    def _presence(self, z, x) -> float:
        aspect = z[0]
        spans = find_all_spans(x)
        max_iou = 0.0
        for s in spans:
            max_iou = max(max_iou, iou_char(s, self._first_match_span(aspect, x)))
        return max_iou

    def _first_match_span(self, phrase: str, text: str):
        import re
        m = re.search(re.escape(phrase), text, re.IGNORECASE)
        if not m: return (0,0)
        return (m.start(), m.end())

    def _polarity(self, z, x) -> float:
        aspect, sent = z
        doc = self.nlp(x)
        # take neighbors within 3 deps
        anchors = {"positive": ["good","great","excellent","nice","love"],
                   "neutral":  ["ok","average","fine","neutral"],
                   "negative": ["bad","poor","terrible","hate","awful"]}
        lex = anchors.get(sent.lower(), anchors["neutral"])
        score=0.0; count=0
        for t in doc:
            if t.text.lower() in lex:
                score += 1.0; count+=1
        return min(1.0, score / max(1,count or 1))

    def _dep_ok(self, z, x) -> float:
        aspect = z[0]; doc = self.nlp(x)
        a_tokens = [t for t in doc if t.text.lower() in aspect.lower().split()]
        if not a_tokens: return 0.0
        for t in a_tokens:
            cur=t; steps=0
            while cur.head!=cur and steps<self.dep_len_cap:
                steps+=1; cur=cur.head
            if steps<=self.dep_len_cap: return 1.0
        return 0.0

    def conf(self, z, x) -> float:
        psi = np.array([self._presence(z,x), self._polarity(z,x), self._dep_ok(z,x)])
        logit = float(np.dot(self.w, psi) + self.b)
        return 1/(1+np.exp(-logit))

    def filter(self, candidates: List[Tuple[str,str]], x: str, tau: float=None) -> List[Tuple[str,str]]:
        tau = tau or self.tau
        out=[]
        for z in normalize_tuples(candidates):
            if self.conf(z, x) >= tau:
                out.append(z)
        return out
