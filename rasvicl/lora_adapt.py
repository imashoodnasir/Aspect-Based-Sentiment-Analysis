from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

class FewShotSet(Dataset):
    def __init__(self, rows, prompt_fn: Callable[[Dict], str], answer_fn: Callable[[Dict], str]):
        self.rows = rows; self.build_prompt = prompt_fn; self.build_answer = answer_fn
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def sequence_kl(p_logits, q_logits):
    p = torch.log_softmax(p_logits, dim=-1)
    q = torch.log_softmax(q_logits, dim=-1)
    return torch.sum(torch.exp(p) * (p - q), dim=-1).mean()

def train_lora(zs_model_name: str, rows: List[Dict], out_dir: str, r=8, lr=1e-4, epochs=3, gamma_kl=0.5,
               prompt_fn=None, answer_fn=None):
    tok = AutoTokenizer.from_pretrained(zs_model_name, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(zs_model_name, device_map="auto")
    frozen = AutoModelForCausalLM.from_pretrained(zs_model_name, device_map="auto")  # zero-shot posterior
    frozen.eval(); 
    peft_cfg = LoraConfig(r=r, lora_alpha=16, target_modules=["k_proj","v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(base, peft_cfg)

    ds = FewShotSet(rows, prompt_fn, answer_fn)
    def collate(batch):
        prompts = [prompt_fn(r) for r in batch]
        answers = [answer_fn(r) for r in batch]
        texts = [p + " " + a for p,a in zip(prompts, answers)]
        return tok(texts, return_tensors="pt", padding=True).to(model.device), prompts
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    num_steps = epochs*len(dl)
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(10, num_steps//10), num_training_steps=num_steps)

    model.train()
    for _ in range(epochs):
        for batch, prompts in tqdm(dl, leave=False):
            labels = batch["input_ids"].clone()
            out = model(**batch, labels=labels)
            loss_ce = out.loss
            with torch.no_grad():
                out_zs = frozen(**batch, labels=labels)
            # KL on final token positions (approximate)
            kl = sequence_kl(out.logits[:,-1,:], out_zs.logits[:,-1,:])
            loss = loss_ce + gamma_kl*kl
            loss.backward(); opt.step(); sch.step(); opt.zero_grad()
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
