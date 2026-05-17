# Pre-Refusal Signatures

> **Can we detect harmful intent inside a language model before it generates a single token?**

Current safety systems often inspect model outputs after text has already been produced. This project studies a more proactive signal: whether harmful-intent prompts are linearly decodable from intermediate hidden states in a small instruction-tuned LLM before decoding begins.

The pipeline extracts layer-wise hidden states from `Qwen/Qwen2.5-1.5B-Instruct`, trains one linear probe per layer, visualizes where the signal emerges, and demonstrates a simple pre-generation guard based on the best layer.

This is a mechanistic-interpretability research prototype, not a production moderation system.

## Core Idea

```text
User prompt
  -> chat template
  -> model forward pass only
  -> hidden states from every layer
  -> last-token pooled layer vectors
  -> layer-wise logistic regression probes
  -> best-layer signal + early guard demo
```

If harmful-intent information is present before generation, a monitor can in principle act earlier than output filters. The goal is not to claim a finished safety system, but to test whether the internal representation contains a useful pre-refusal signal.

## Why This Matters

| Standard output filtering | Pre-refusal hidden-state probing |
| --- | --- |
| Reacts after tokens are generated | Measures the prompt representation before decoding |
| Operates on text | Operates on internal activations |
| Can miss subtle jailbreak setup | Can expose linearly decodable intent signals |
| Hard to inspect mechanistically | Produces layer-wise curves and probe directions |

## Current Status

The repository currently includes:

- A complete end-to-end probing pipeline.
- A curated 80-prompt dataset with balanced harmful-intent and benign examples.
- A Colab notebook for running Qwen on a T4 GPU.
- Synthetic smoke-test artifacts proving the pipeline works before the real Qwen run.

The next research step is to run the Colab notebook and replace the synthetic smoke-test figures with real Qwen results.

## Repository Layout

```text
pre-refusal-signatures/
|-- configs/
|   |-- default.yaml
|-- data/
|   |-- prompts.jsonl
|   |-- README.md
|-- docs/
|   |-- limitations.md
|   |-- methodology.md
|   |-- plans/
|-- figures/
|   |-- layer_accuracy_curve.png
|   |-- best_layer_pca.png
|   |-- confusion_matrix.png
|-- notebooks/
|   |-- run_qwen_colab.ipynb
|-- reports/
|   |-- layer_probe_metrics.csv
|   |-- error_analysis.md
|-- scripts/
|   |-- 00_validate_dataset.py
|   |-- 01_extract_hidden_states.py
|   |-- 02_train_layer_probes.py
|   |-- 03_make_figures.py
|   |-- 04_run_guard_demo.py
|-- src/
|   |-- pre_refusal_signatures/
|-- tests/
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/00_validate_dataset.py --data data/prompts.jsonl --min-per-label 40
pytest -q
```

## Run On Colab / T4

Use the notebook:

```text
notebooks/run_qwen_colab.ipynb
```

Recommended Colab workflow:

1. Open the notebook in Google Colab.
2. Set `Runtime -> Change runtime type -> T4 GPU`.
3. Set:

```python
REPO_URL = "https://github.com/DaviBonetto/algoverse.git"
```

4. Run all cells.
5. Download `pre_refusal_qwen_results.zip`.
6. Copy the extracted `figures/`, `reports/`, and selected `outputs/` files back into this repo.

The notebook runs the real model pipeline:

```bash
python scripts/01_extract_hidden_states.py --config configs/default.yaml --device cuda
python scripts/02_train_layer_probes.py --states outputs/hidden_states.npz
python scripts/03_make_figures.py --states outputs/hidden_states.npz --metrics reports/layer_probe_metrics.csv
python scripts/04_run_guard_demo.py --prompt "Explain photosynthesis simply." --device cuda
pytest -q
```

If a T4 hits memory limits:

```python
MAX_PROMPTS = 40
MAX_LENGTH = 256
```

The notebook now uses these T4-safe values by default. After a successful run, set `MAX_PROMPTS = None` and `MAX_LENGTH = 512` for the full 80-prompt extraction.

## Local Smoke Test

This mode does not download Qwen. It injects a controlled synthetic signal into later layers so the full analysis pipeline can be tested quickly.

```bash
python scripts/01_extract_hidden_states.py --config configs/default.yaml --synthetic
python scripts/02_train_layer_probes.py --states outputs/hidden_states.npz
python scripts/03_make_figures.py --states outputs/hidden_states.npz --metrics reports/layer_probe_metrics.csv
python scripts/04_run_guard_demo.py --prompt "Explain photosynthesis simply."
pytest -q
```

Synthetic smoke-test result:

| Best layer | Accuracy | F1 | ROC-AUC |
| ---: | ---: | ---: | ---: |
| 24 | 0.900 | 0.897 | 0.938 |

These numbers are not the scientific claim. They are a pipeline sanity check. The real result should be generated with Qwen using the Colab notebook.

## Figures

Layer-wise probe performance:

![Layer-wise probe performance](figures/layer_accuracy_curve.png)

Best-layer PCA projection:

![Best-layer PCA projection](figures/best_layer_pca.png)

Best-layer confusion matrix:

![Confusion matrix](figures/confusion_matrix.png)

## Dataset

`data/prompts.jsonl` contains 80 curated prompts:

- 40 sanitized harmful-intent prompts.
- 40 benign prompts across math, coding, science, history, writing, health, productivity, career, creative writing, and everyday assistance.

The harmful-intent examples are intentionally written at the intent level. They do not include operational instructions, target details, quantities, exploit code, or procedural steps.

## Method Details

For each prompt:

1. Apply the model's chat template.
2. Run a forward pass with `output_hidden_states=True`.
3. Pool the final non-padding token from every layer.
4. Store `X` with shape `(num_prompts, num_layers, hidden_dim)`.
5. Train an independent logistic regression probe per layer.
6. Select the best layer by F1, using accuracy as a tie-breaker.

The probe is deliberately simple. In interpretability work, linear probes are useful because they ask whether information is accessible in the representation without adding a large nonlinear classifier.

## Early Guard Demo

The guard combines:

- Best-layer harmful-intent probability from the linear probe.
- Distance from the benign centroid at the same layer.

It returns:

- `ALLOW`: low estimated risk.
- `REVIEW`: borderline probability or unusually far from benign examples.
- `FLAG`: above the high-recall harmful threshold.

This is a proof-of-concept monitor. It is not calibrated enough for deployment.

## Limitations

- The dataset is small and manually curated.
- The prompts are static, not adaptive jailbreaks.
- Linear separability is not causal evidence.
- PCA plots can hide structure present in higher dimensions.
- Results from a 1.5B model may not transfer to frontier models.
- The early guard is a research sketch, not a complete safety layer.

## Future Work

- Run the same protocol on larger models and multiple instruction-tuned families.
- Compare last-token pooling with mean pooling and instruction-token pooling.
- Add confidence intervals over random dataset splits.
- Test held-out jailbreak families and adaptive paraphrases.
- Use activation patching to test whether the probe direction is causally involved.
- Integrate a streaming inference hook that checks hidden states before decoding.

## Citation

```bibtex
@software{pre_refusal_signatures_2026,
  title = {Pre-Refusal Signatures: Early Detection of Harmful Intent via Layer-Wise Hidden-State Probing},
  author = {Bonetto, David},
  year = {2026},
  url = {https://github.com/DaviBonetto/algoverse}
}
```
