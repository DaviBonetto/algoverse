# Pre-Refusal Signatures

*"Current safety systems filter outputs. We ask: can we detect harmful intent in the model's internal representations before a single token is generated?"*

This repository is a compact mechanistic-interpretability project for AI safety. It extracts layer-wise hidden states from a small instruction-tuned language model, trains linear probes on each layer, and tests whether harmful-intent prompts are detectable before decoding begins.

The project is intentionally scoped as a research prototype: it studies whether a harmful-intent signal is linearly decodable from intermediate representations. It is not a production safety classifier and should not be used as the only guardrail for deployed systems.

## Method

```text
prompt -> tokenizer/chat template -> model forward pass only
       -> hidden states from every layer -> last-token pooling
       -> layer-wise logistic regression probes
       -> best-layer analysis + early guard demo
```

The core question is whether intermediate layers contain a useful pre-generation signal. For each prompt, the pipeline stores one vector per layer. A linear probe is trained independently at each layer using stratified cross-validation. The resulting accuracy/F1 curve shows where the signal becomes easiest to decode.

## Repository Layout

```text
pre-refusal-signatures/
|-- configs/default.yaml
|-- data/prompts.jsonl
|-- scripts/
|   |-- 00_validate_dataset.py
|   |-- 01_extract_hidden_states.py
|   |-- 02_train_layer_probes.py
|   |-- 03_make_figures.py
|   |-- 04_run_guard_demo.py
|-- src/pre_refusal_signatures/
|-- tests/
|-- docs/
|-- figures/
|-- reports/
|-- outputs/
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/00_validate_dataset.py --data data/prompts.jsonl --min-per-label 40
```

## Colab / T4 Run

The easiest way to produce the real Qwen results is to run the Colab notebook:

```text
notebooks/run_qwen_colab.ipynb
```

Recommended Colab setup:

1. Runtime -> Change runtime type -> T4 GPU.
2. Set `REPO_URL` in the notebook to your GitHub repo URL.
3. Run all cells.
4. Download `pre_refusal_qwen_results.zip`.
5. Copy the zip contents back into this repository to update `figures/`, `reports/`, and `outputs/`.

The notebook runs:

```bash
python scripts/01_extract_hidden_states.py --config configs/default.yaml --device cuda
python scripts/02_train_layer_probes.py --states outputs/hidden_states.npz
python scripts/03_make_figures.py --states outputs/hidden_states.npz --metrics reports/layer_probe_metrics.csv
pytest -q
```

If a T4 hits CUDA memory limits, set these notebook values:

```python
MAX_PROMPTS = 40
MAX_LENGTH = 256
```

For a fast end-to-end smoke test without downloading a model:

```bash
python scripts/01_extract_hidden_states.py --config configs/default.yaml --synthetic
python scripts/02_train_layer_probes.py --states outputs/hidden_states.npz
python scripts/03_make_figures.py --states outputs/hidden_states.npz --metrics reports/layer_probe_metrics.csv
python scripts/04_run_guard_demo.py --prompt "Explain photosynthesis simply."
pytest -q
```

For the intended model run:

```bash
python scripts/01_extract_hidden_states.py --config configs/default.yaml --device cuda
python scripts/02_train_layer_probes.py --states outputs/hidden_states.npz
python scripts/03_make_figures.py --states outputs/hidden_states.npz --metrics reports/layer_probe_metrics.csv
python scripts/04_run_guard_demo.py --prompt "Explain photosynthesis simply." --device cuda
```

Use `--device cpu` if no GPU is available. CPU extraction is slower but still conceptually identical.

## Dataset

`data/prompts.jsonl` contains 80 curated prompts:

- 40 harmful-intent prompts, sanitized to avoid operational harmful instructions.
- 40 benign prompts spanning math, coding, science, history, writing, health, productivity, and everyday assistance.

The dataset is small by design. It is meant to support a rapid, inspectable probing study rather than a robust benchmark.

## Expected Outputs

After running the pipeline:

- `outputs/hidden_states.npz`: layer-wise hidden-state tensor and metadata.
- `reports/layer_probe_metrics.csv`: accuracy, precision, recall, F1, and ROC-AUC per layer.
- `outputs/probe_predictions.csv`: cross-validated probabilities for the best layer.
- `figures/layer_accuracy_curve.png`: where the signal emerges across depth.
- `figures/best_layer_pca.png`: 2D PCA projection of the best layer.
- `figures/confusion_matrix.png`: best-layer classification errors.
- `reports/error_analysis.md`: qualitative false positive/false negative analysis.

## Early Guard Demo

The guard is deliberately simple:

1. Fit the best-layer linear probe.
2. Estimate a threshold that targets high recall on harmful-intent examples.
3. Compare the prompt vector against the benign centroid.
4. Return `ALLOW`, `REVIEW`, or `FLAG`.

This is a systems sketch, not a deployment recommendation. Its purpose is to show how interpretability signals could become pre-generation monitoring signals.

## Results

Run the scripts to generate local results. The synthetic smoke-test mode should show a clear layer-wise emergence curve because it injects a known signal into later layers. The real model run is the research result and should be reported with the exact model, device, dataset size, and random seed used.

When reporting results, use careful language:

> "We find evidence that harmful-intent labels are linearly decodable from intermediate hidden states in this small curated setting."

Avoid stronger claims unless the dataset, model coverage, and adversarial evaluation are expanded.

## Limitations

- The dataset is small and manually curated.
- Harmful-intent prompts are static, not adaptive adversarial attacks.
- Linear probes can detect correlations without explaining causality.
- PCA visualizations can hide separability that exists in higher dimensions.
- Small instruction-tuned models may not reflect the behavior of frontier systems.
- This is not a replacement for output filtering, policy classifiers, or human review.

## Future Work

- Scale to larger models and more diverse prompt distributions.
- Compare pooling strategies: last token, mean pooling, instruction-token pooling.
- Run activation patching to test whether the probe direction is causally involved.
- Add calibration curves and confidence intervals.
- Evaluate adaptive jailbreak prompts and distribution shift.
- Integrate a streaming inference hook that checks hidden states before decoding.

## Citation

If this project is useful, cite the repository:

```bibtex
@software{pre_refusal_signatures_2026,
  title = {Pre-Refusal Signatures: Early Detection of Harmful Intent via Layer-Wise Hidden-State Probing},
  author = {Bonetto, David},
  year = {2026},
  url = {https://github.com/your-user/pre-refusal-signatures}
}
```
