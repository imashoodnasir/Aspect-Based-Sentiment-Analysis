# RaSV-ICL: Retrieval-Augmented Self-Verified In-Context Learning for Aspect-Based Sentiment Analysis

This repository contains the reference implementation of **RaSV-ICL**, a framework for zero- and few-shot **Aspect-based Sentiment Analysis (ABSA)** featuring:

- Multi-view demo retrieval (OS/SR/AS/DEP) with MMR diversity and uncertainty-aware reweighting
- Schema-constrained prompting with domain-optimized verbalizers
- Self-consistency decoding plus a lightweight tuple verifier
- Optional few-shot LoRA adapters (≤1% params) with KL-regularized training
- Contrastive (InfoNCE) retriever training across views

> Benchmarks: Lap14, Rest14, Books, Clothing. Zero-shot RaSV-ICL attains F1 = 75.9, 79.3, 61.7, 57.3. With 20 labels/domain via LoRA: 78.6, 82.0, 65.7, 61.3. Robust under 30% noisy demos (≤6 F1 drop).

---

## Installation

```bash
git clone https://github.com/imashoodnasir/Aspect-Based-Sentiment-Analysis.git
cd Aspect-Based-Sentiment-Analysis
python -m venv .venv && source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Requirements (key): torch, transformers, peft, sentence-transformers, faiss-cpu, spacy, scikit-learn, datasets, tqdm.

---

## Project Layout

```
Aspect-Based-Sentiment-Analysis/
├─ rasvicl/
│  ├─ config.py                 # Model & training configs
│  ├─ data.py                   # JSONL IO helpers
│  ├─ views.py                  # OS/SR/AS/DEP encoders + embeddings
│  ├─ retriever.py              # Joint scoring, uncertainty, MMR selection
│  ├─ prompting.py              # Schema-constrained prompts + verbalizers
│  ├─ self_consistency.py       # Diverse decoding + majority vote
│  ├─ verifier.py               # Span, polarity, dependency checks
│  ├─ lora_adapt.py             # Few-shot LoRA (CE + γ·KL)
│  ├─ retriever_train.py        # InfoNCE/MNRL training for views
│  └─ text_utils.py             # Utilities (IoU, normalization)
├─ scripts/
│  ├─ run_rasvicl.py            # Zero-shot pipeline
│  ├─ train_retriever.py        # Train multi-view retriever
│  └─ train_lora.py             # Train LoRA adapters (10–20 labels)
├─ requirements.txt
└─ README.md
```

---

## Data Format

Use JSONL with fields:
```json
{"text": "The keyboard is great but the screen is dim.",
 "tuples": [["keyboard","positive"], ["screen","negative"]]}
```

- `data/` holds test/eval sets per domain (Lap14, Rest14, Books, Clothing).
- `pool/` holds candidate demonstrations (labeled or pseudo-labeled) used for ICL.

---

## Zero-shot Inference (RaSV-ICL)

```bash
python scripts/run_rasvicl.py   --data_path data/rest14_test.jsonl   --pool_path pool/rest14_pool.jsonl   --out_path outputs/rasvicl_rest14_pred.jsonl
```

Edit `rasvicl/config.py` to set your LLM (e.g., `mistralai/Mistral-7B-Instruct-v0.2` or any compatible instruction-tuned model).

Outputs contain predicted tuples for each review plus selected demo indices.

---

## Train the Multi-view Retriever (InfoNCE)

```bash
python scripts/train_retriever.py   --train_json data/rest14_train.jsonl   --out_dir ckpts/retriever
```

This trains 4 light sentence encoders (OS/SR/AS/DEP) using MultipleNegativesRankingLoss as an InfoNCE-style objective.

---

## Few-shot LoRA (10–20 labels per domain)

```bash
python scripts/train_lora.py   --fewshot_json data/rest14_10shot.jsonl   --out_dir ckpts/lora_rest14   --llm_name mistralai/Mistral-7B-Instruct-v0.2
```

- Loss: L = L_CE + γ·L_KL, where L_KL regularizes toward the zero-shot posterior.
- Target modules: last-layer k/v projections (≤1% params).

To run inference with the adapted model, load the LoRA checkpoint using PEFT or merge it with the base model.

---

## Reproducing Table Numbers

1. Run zero-shot on all four datasets → collect P/R/F1.
2. Train LoRA with 10 and 20 labels → rerun inference.
3. Robustness: randomly corrupt 10–30% of pool demos and re-run.
4. Self-consistency: vary n in {1,5,10,20}.
5. Verbalizers: compare generic, domain-tuned, optimized; see `rasvicl/verbalizer.py`.

We provide example scripts in `scripts/` to automate these sweeps.

---

## Figures

- Fig.1–2: Pipeline + Multi-view retrieval
- Fig.3: Schema + verbalizers (objective)
- Fig.4: Self-consistency & verifier (confidence gating)
- Fig.5: Few-shot LoRA adaptation (CE + γ·KL)
- Fig.6: Retriever training (per-view InfoNCE)
- Fig.7–10: Runtime, Ablations, Robustness, Cross-domain

All vector assets (SVG/PNG) can be generated via the figure scripts (optional).

---

## Tips & Troubleshooting

- CUDA OOM: reduce max_new_tokens and n_samples, or switch to a smaller LLM.
- spaCy model: ensure `python -m spacy download en_core_web_sm`.
- Tokenizer padding: set `pad_token_id` to EOS for some LLMs if needed.
- FAISS: for large pools, swap the in-memory cosine for FAISS IVF/Flat.

---

## License

MIT License (see `LICENSE` if added). For the LLM and encoders, follow their respective licenses.
