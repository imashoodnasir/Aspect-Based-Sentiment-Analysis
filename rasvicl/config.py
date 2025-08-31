from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    # LLM for generation (frozen for zero-shot)
    llm_name: str = "mistralai/Mistral-7B-Instruct-v0.2"  # change to what you have access to
    max_new_tokens: int = 128
    temperature: float = 0.3
    top_p: float = 0.95

    # Sentence encoders (can be shared backbone)
    os_encoder: str = "sentence-transformers/all-mpnet-base-v2"
    sr_encoder: str = "sentence-transformers/all-mpnet-base-v2"
    as_encoder: str = "sentence-transformers/all-mpnet-base-v2"
    dep_encoder: str = "sentence-transformers/all-mpnet-base-v2"

    device: str = "cuda"

@dataclass
class RetrievalConfig:
    k: int = 8
    alpha_os: float = 0.25
    alpha_sr: float = 0.25
    alpha_as: float = 0.25
    alpha_dep: float = 0.25
    beta_mmr: float = 0.5
    lambda_uncert: float = 0.5

@dataclass
class SelfConsistencyConfig:
    n_samples: int = 20
    temperatures: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])
    pi_majority: float = 0.5
    tau_conf: float = 0.5

@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["k_proj", "v_proj"])
    lr: float = 1e-4
    epochs: int = 3
    gamma_kl: float = 0.5

@dataclass
class RetrieverTrainConfig:
    batch_size: int = 64
    lr: float = 2e-5
    epochs: int = 3
    temperature: float = 0.07
    embed_dim: int = 768
