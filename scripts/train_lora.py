import argparse
from rasvicl.data import read_jsonl
from rasvicl.lora_adapt import train_lora
from rasvicl.prompting import build_prompt

def answer_from_gold(r):
    # Use gold tuples as answer text (teacher forcing)
    return " " + str(r.get("tuples", []))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fewshot_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--llm_name", default="mistralai/Mistral-7B-Instruct-v0.2")
    args = ap.parse_args()

    rows = read_jsonl(args.fewshot_json)
    def prompt_fn(r): 
        return f"Task: extract (aspect, polarity) tuples.\nReview: {r['text']}\nAnswer:"
    train_lora(args.llm_name, rows, args.out_dir, prompt_fn=prompt_fn, answer_fn=answer_from_gold)

if __name__=="__main__":
    main()
