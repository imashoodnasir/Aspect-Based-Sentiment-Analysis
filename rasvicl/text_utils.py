from typing import List, Tuple
import re

def find_all_spans(text: str) -> List[Tuple[int, int]]:
    tokens = text.split()
    spans = []
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            s = " ".join(tokens[i:j+1])
            # Map back to char indices (approx via regex)
            for m in re.finditer(re.escape(s), text):
                spans.append((m.start(), m.end()))
    return spans

def iou_char(span_a: Tuple[int,int], span_b: Tuple[int,int]) -> float:
    a1,a2 = span_a; b1,b2 = span_b
    inter = max(0, min(a2,b2) - max(a1,b1))
    union = max(a2,b2) - min(a1,b1)
    return 0.0 if union == 0 else inter/union

def normalize_tuples(tuples: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    out=[]
    for a,s in tuples:
        out.append((" ".join(a.split()).strip().lower(), s.strip().lower()))
    return out
