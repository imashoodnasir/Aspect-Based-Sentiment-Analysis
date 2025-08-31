from typing import Dict, List, Tuple
from collections import Counter

def agree(pred: List[Tuple[str,str]], gold: List[Tuple[str,str]]) -> int:
    return len(set(pred) & set(gold))

def grid_optimize(pool: List[Dict],
                  candidate_sets: List[Dict[str,str]],
                  infer_fn) -> Dict[str,str]:
    # candidate_sets: list of label dictionaries {"pos":"savory", "neu":"bland", "neg":"stale"}
    best, best_score = None, -1
    for V in candidate_sets:
        score = 0
        for ex in pool:
            pred = infer_fn(ex["text"], V)  # returns list of tuples
            score += agree(pred, [tuple(x) for x in ex.get("tuples",[])])
        if score > best_score:
            best, best_score = V, score
    return best or {"pos":"positive", "neu":"neutral", "neg":"negative"}

def anchors_from_corpus(pool: List[Dict]) -> Dict[str,float]:
    cnt = Counter()
    for ex in pool:
        for _,s in ex.get("tuples",[]):
            cnt[s.lower()] += 1
    tot = sum(cnt.values()) or 1
    return {k:v/tot for k,v in cnt.items()}
