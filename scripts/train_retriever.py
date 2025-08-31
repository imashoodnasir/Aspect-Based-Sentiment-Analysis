import argparse
from rasvicl.data import read_jsonl
from rasvicl.retriever_train import train_retriever
from rasvicl.config import ModelConfig, RetrieverTrainConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.train_json)
    mc = ModelConfig()
    train_retriever(rows, args.out_dir,
                    mc.os_encoder, mc.sr_encoder, mc.as_encoder, mc.dep_encoder)

if __name__=="__main__":
    main()
