from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .dataset import PromptRecord


def select_balanced_records(records: Sequence[PromptRecord], max_prompts: int | None) -> list[PromptRecord]:
    if max_prompts is None or max_prompts >= len(records):
        return list(records)
    if max_prompts <= 0:
        raise ValueError("max_prompts must be positive")

    harmful = [record for record in records if record.label == "harmful"]
    benign = [record for record in records if record.label == "benign"]
    selected: list[PromptRecord] = []
    while len(selected) < max_prompts and (harmful or benign):
        if harmful and len(selected) < max_prompts:
            selected.append(harmful.pop(0))
        if benign and len(selected) < max_prompts:
            selected.append(benign.pop(0))
    return selected


def resolve_device(device: str):
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "On Colab, set Runtime -> Change runtime type -> T4 GPU, then restart and run all cells."
        )
    return device


def resolve_dtype(dtype: str, device: str):
    import torch

    if dtype == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User: {prompt}\nAssistant:"


def pool_last_token(hidden_state, attention_mask):
    import torch

    lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[batch_indices, lengths]


def extract_hidden_states(
    records: Sequence[PromptRecord],
    model_name: str,
    output_path: str | Path,
    max_length: int = 512,
    device: str = "auto",
    dtype: str = "auto",
) -> tuple[int, int, int]:
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "output_hidden_states": True,
        "low_cpu_mem_usage": True,
    }
    if resolved_device == "cuda":
        model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if resolved_device != "cuda":
        model.to(resolved_device)
    model.config.use_cache = False
    model.eval()

    vectors = []
    with torch.no_grad():
        for record in tqdm(records, desc="extracting hidden states"):
            text = format_prompt(tokenizer, record.prompt)
            batch = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            batch = {key: value.to(resolved_device) for key, value in batch.items()}
            outputs = model(**batch, output_hidden_states=True, use_cache=False)
            per_layer = []
            for hidden_state in outputs.hidden_states:
                pooled = pool_last_token(hidden_state, batch["attention_mask"])
                per_layer.append(pooled[0].detach().float().cpu().numpy())
            vectors.append(np.stack(per_layer, axis=0))
            del outputs, batch
            if resolved_device == "cuda":
                torch.cuda.empty_cache()

    X = np.stack(vectors, axis=0).astype("float32")
    y = np.array([record.target for record in records], dtype="int64")
    ids = np.array([record.id for record in records])
    labels = np.array([record.label for record in records])
    categories = np.array([record.category for record in records])
    prompts = np.array([record.prompt for record in records])
    layers = np.arange(X.shape[1], dtype="int64")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        ids=ids,
        labels=labels,
        categories=categories,
        prompts=prompts,
        layers=layers,
        model_name=np.array(model_name),
    )
    return X.shape


def create_synthetic_hidden_states(
    records: Sequence[PromptRecord],
    output_path: str | Path,
    n_layers: int = 25,
    hidden_dim: int = 96,
    signal_start_layer: int = 10,
    seed: int = 42,
) -> tuple[int, int, int]:
    import numpy as np

    rng = np.random.default_rng(seed)
    y = np.array([record.target for record in records], dtype="int64")
    X = rng.normal(0, 1, size=(len(records), n_layers, hidden_dim)).astype("float32")
    signal = rng.normal(0, 1, size=(hidden_dim,)).astype("float32")
    signal = signal / np.linalg.norm(signal)
    for layer in range(signal_start_layer, n_layers):
        strength = 0.25 + 0.08 * (layer - signal_start_layer)
        X[:, layer, :] += (2 * y[:, None] - 1) * signal[None, :] * strength

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        ids=np.array([record.id for record in records]),
        labels=np.array([record.label for record in records]),
        categories=np.array([record.category for record in records]),
        prompts=np.array([record.prompt for record in records]),
        layers=np.arange(n_layers, dtype="int64"),
        model_name=np.array("synthetic-control"),
    )
    return X.shape
