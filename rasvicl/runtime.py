from typing import Dict, List, Tuple
from .views import ViewEncoders
from .retriever import RaSVRetriever, RetrievalWeights
from .prompting import build_prompt
from .self_consistency import SCDecoder, parse_tuples, majority_vote
from .verifier import Verifier

def run_rasvicl(example: Dict,
                pool: List[Dict],
                encoders: ViewEncoders,
                llm_name: str,
                k=8, beta=0.5,
                sc_n=20, sc_temps=(0.3,0.5,0.7,0.9),
                alphas=(0.25,0.25,0.25,0.25),
                lambda_uncert=0.5,
                verbalizers=None):
    retr = RaSVRetriever(encoders, RetrievalWeights(*alphas), lambda_uncert, llm_name=llm_name)
    pool_texts = [p["text"] for p in pool]
    sel_idx = retr.select(example["text"], pool_texts, k=k, beta=beta, probe_template="Input: {x}\nOutput: []")
    demos = [(pool[i]["text"], pool[i].get("tuples", [])) for i in sel_idx]

    prompt = build_prompt(example["text"], demos, verbalizers or {"pos":"positive","neu":"neutral","neg":"negative"})
    sc = SCDecoder.build(llm_name)
    outs=[]
    temps=list(sc_temps)
    for t in range(sc_n):
        outs.append(parse_tuples(sc.decode_once(prompt, temperature=temps[t % len(temps)])))
    voted = majority_vote(outs, pi=0.5)

    ver = Verifier.build()
    final = ver.filter(voted, example["text"], tau=0.5)
    return final, {"selected_indices": sel_idx, "prompt": prompt}
