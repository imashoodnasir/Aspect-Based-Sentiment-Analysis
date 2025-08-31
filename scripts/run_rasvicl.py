import argparse, json
from rasvicl.config import ModelConfig, RetrievalConfig, SelfConsistencyConfig
from rasvicl.data import read_jsonl, write_jsonl
from rasvicl.views import ViewEncoders
from rasvicl.runtime import run_rasvicl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--pool_path", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    mc = ModelConfig()
    enc = ViewEncoders.build(mc.os_encoder, mc.sr_encoder, mc.as_encoder, mc.dep_encoder)

    data = read_jsonl(args.data_path)
    pool = read_jsonl(args.pool_path)

    outputs=[]
    for ex in data:
        tuples, meta = run_rasvicl(ex, pool, encoders=enc, llm_name=mc.llm_name)
        outputs.append({"text": ex["text"], "pred": tuples, "meta": {"selected": meta["selected_indices"]}})
        print(tuples)

    write_jsonl(args.out_path, outputs)

if __name__=="__main__":
    main()
