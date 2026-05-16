# Pre-Refusal Signatures Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible mechanistic-interpretability safety repo that detects harmful-intent prompts from layer-wise hidden states before generation.

**Architecture:** The project is a small Python research pipeline: curate prompts, extract per-layer hidden-state vectors from an instruction-tuned small LLM, train linear probes per layer, visualize where the signal emerges, and package a simple early-abort guard. The repo must be credible even with a small dataset by emphasizing reproducibility, cross-validation, qualitative error analysis, and honest limitations.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, scikit-learn, numpy, pandas, matplotlib, seaborn, optional UMAP.

---

## Target Repository Tree

```text
pre-refusal-signatures/
|-- README.md
|-- LICENSE
|-- .gitignore
|-- requirements.txt
|-- pyproject.toml
|-- configs/
|   |-- default.yaml
|-- data/
|   |-- prompts.jsonl
|   |-- README.md
|-- scripts/
|   |-- 00_validate_dataset.py
|   |-- 01_extract_hidden_states.py
|   |-- 02_train_layer_probes.py
|   |-- 03_make_figures.py
|   |-- 04_run_guard_demo.py
|-- src/
|   |-- pre_refusal_signatures/
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- dataset.py
|   |   |-- extraction.py
|   |   |-- probing.py
|   |   |-- guard.py
|   |   |-- plotting.py
|   |   |-- reporting.py
|-- outputs/
|   |-- .gitkeep
|-- figures/
|   |-- .gitkeep
|-- reports/
|   |-- .gitkeep
|-- tests/
|   |-- test_dataset.py
|   |-- test_probing.py
|   |-- test_guard.py
|-- docs/
|   |-- methodology.md
|   |-- limitations.md
|   |-- plans/
|   |   |-- 2026-05-16-pre-refusal-signatures.md
```

## Two-Hour Execution Strategy

The mandatory artifact is an end-to-end repo that can run on a small curated dataset and produce figures. The stretch artifact is a full Qwen2.5-1.5B-Instruct extraction if compute is available.

### Block 1: Repo Skeleton and Research Framing

**Files:**
- Create: `README.md`
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `configs/default.yaml`
- Create: `docs/methodology.md`
- Create: `docs/limitations.md`

**Steps:**
1. Create the directory tree exactly as listed above.
2. Add dependencies: `torch`, `transformers`, `accelerate`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `pyyaml`, `tqdm`, `pytest`.
3. Set default model to `Qwen/Qwen2.5-1.5B-Instruct`, pooling to `last_token`, probe model to `LogisticRegression`, CV folds to `5`.
4. Write README opening with: "Current safety systems filter outputs. We ask: can we detect harmful intent in the model's internal representations before a single token is generated?"

**Verify:**
- `python -m pytest` can discover tests, even before tests are populated.
- `python -m pip install -r requirements.txt` succeeds in the intended environment.

### Block 2: Dataset With Safety Hygiene

**Files:**
- Create: `data/prompts.jsonl`
- Create: `data/README.md`
- Create: `src/pre_refusal_signatures/dataset.py`
- Create: `scripts/00_validate_dataset.py`
- Create: `tests/test_dataset.py`

**Dataset schema:**
```json
{"id":"harmful_001","label":"harmful","category":"cyber_abuse","prompt":"<sanitized harmful-intent prompt>"}
```

**Steps:**
1. Create 80 total prompts: 40 harmful-intent, 40 benign.
2. Avoid operationally detailed harmful instructions; keep harmful prompts as intent-level red-team prompts suitable for safety evaluation.
3. Balance benign prompts across math, coding, explanation, summarization, history, science, and casual QA.
4. Implement JSONL loading with schema validation, duplicate-id checks, label checks, and minimum class-count checks.
5. Add unit tests for valid rows, missing keys, invalid labels, and duplicate IDs.

**Verify:**
- Run: `python scripts/00_validate_dataset.py --data data/prompts.jsonl`
- Expected: prints total count, label counts, category counts, and exits with code 0.
- Run: `pytest tests/test_dataset.py -q`
- Expected: all tests pass.

### Block 3: Hidden-State Extraction

**Files:**
- Create: `src/pre_refusal_signatures/config.py`
- Create: `src/pre_refusal_signatures/extraction.py`
- Create: `scripts/01_extract_hidden_states.py`

**Steps:**
1. Load tokenizer and model with `output_hidden_states=True`.
2. Use chat template when available; otherwise format prompt plainly.
3. Run the prompt forward pass only; do not generate output tokens.
4. For each layer, pool hidden states at the final non-padding token.
5. Save compressed arrays to `outputs/hidden_states.npz` with keys: `X`, `y`, `ids`, `layers`, `model_name`.
6. Support `--max-prompts` for fast smoke tests and `--device auto/cpu/cuda`.

**Verify:**
- Run: `python scripts/01_extract_hidden_states.py --config configs/default.yaml --max-prompts 6 --device cpu`
- Expected: creates `outputs/hidden_states.npz` and prints shape like `(6, num_layers, hidden_dim)`.

### Block 4: Layer-Wise Linear Probes

**Files:**
- Create: `src/pre_refusal_signatures/probing.py`
- Create: `src/pre_refusal_signatures/reporting.py`
- Create: `scripts/02_train_layer_probes.py`
- Create: `tests/test_probing.py`

**Steps:**
1. For each layer, fit a logistic regression probe using stratified 5-fold CV.
2. Store accuracy, precision, recall, F1, ROC-AUC if both classes are present.
3. Use a fixed random seed for reproducibility.
4. Save `reports/layer_probe_metrics.csv`.
5. Save `outputs/probe_predictions.csv` with per-example predicted probabilities for error analysis.

**Verify:**
- Run: `python scripts/02_train_layer_probes.py --states outputs/hidden_states.npz`
- Expected: creates `reports/layer_probe_metrics.csv` with one row per layer.
- Run: `pytest tests/test_probing.py -q`
- Expected: all tests pass on synthetic arrays.

### Block 5: Figures and Error Analysis

**Files:**
- Create: `src/pre_refusal_signatures/plotting.py`
- Create: `scripts/03_make_figures.py`

**Figures:**
1. `figures/layer_accuracy_curve.png`: probe accuracy/F1 by layer.
2. `figures/best_layer_pca.png`: PCA scatter at best probe layer.
3. `figures/confusion_matrix.png`: confusion matrix at best layer.
4. `reports/error_analysis.md`: false positives, false negatives, and interpretation.

**Steps:**
1. Select best layer by mean CV F1, not just accuracy.
2. Plot a horizontal baseline at majority-class accuracy.
3. Use PCA instead of UMAP as the default to avoid extra dependency risk.
4. Generate a short markdown report with best layer, top metrics, and caveats.

**Verify:**
- Run: `python scripts/03_make_figures.py --states outputs/hidden_states.npz --metrics reports/layer_probe_metrics.csv`
- Expected: three PNGs and one markdown report are created.

### Block 6: Early Guard Demo

**Files:**
- Create: `src/pre_refusal_signatures/guard.py`
- Create: `scripts/04_run_guard_demo.py`
- Create: `tests/test_guard.py`

**Guard design:**
- Primary guard: probability threshold from the best-layer linear probe.
- Secondary guard: distance from benign centroid as an interpretable baseline.

**Steps:**
1. Fit probe on all available data at the selected best layer.
2. Compute a threshold that targets high recall on harmful-intent prompts.
3. Print decision: `ALLOW`, `FLAG`, or `REVIEW`.
4. Include a demo prompt path but avoid generating model completions.

**Verify:**
- Run: `python scripts/04_run_guard_demo.py --prompt "Explain photosynthesis simply."`
- Expected: prints layer used, harmful probability, centroid distance, and decision.
- Run: `pytest tests/test_guard.py -q`
- Expected: all tests pass.

### Block 7: README Polish and GitHub Presentation

**Files:**
- Modify: `README.md`
- Modify: `docs/methodology.md`
- Modify: `docs/limitations.md`

**README sections:**
1. Motivation
2. Method
3. Quickstart
4. Reproduce results
5. Results
6. Early guard demo
7. Limitations
8. Future work
9. Citation / attribution

**Key claims to make carefully:**
- Say "evidence of linearly decodable harmful-intent signal" instead of "proof of refusal signature."
- Say "small curated dataset" and "not robust to adaptive adversaries."
- Emphasize this is a probing study, not a production safety filter.

**Verify:**
- Fresh clone path commands in README run in order.
- Figures referenced in README exist.
- No hidden-state artifacts larger than GitHub-friendly limits are committed.

## Final Validation Checklist

- [ ] `python scripts/00_validate_dataset.py --data data/prompts.jsonl`
- [ ] `python scripts/01_extract_hidden_states.py --config configs/default.yaml --max-prompts 6 --device cpu`
- [ ] `python scripts/02_train_layer_probes.py --states outputs/hidden_states.npz`
- [ ] `python scripts/03_make_figures.py --states outputs/hidden_states.npz --metrics reports/layer_probe_metrics.csv`
- [ ] `python scripts/04_run_guard_demo.py --prompt "Explain photosynthesis simply."`
- [ ] `pytest -q`
- [ ] README contains generated metrics and all figure links resolve.

## Commit Plan

1. `git commit -m "chore: scaffold research repository"`
2. `git commit -m "feat: add dataset validation and hidden-state extraction"`
3. `git commit -m "feat: add layer-wise probes and visualizations"`
4. `git commit -m "feat: add early guard demo and documentation"`

## Risk Controls

- If Qwen download or memory fails, use a smaller fallback model configured in `configs/default.yaml`, then document the fallback.
- If full 80-prompt extraction is slow, run `--max-prompts 20` first and keep the pipeline demonstrable.
- If metrics are unstable, foreground the methodology and report confidence/limitations instead of overstating results.
- If PCA separation is weak, keep the figure and explain that the linear probe may exploit directions not visible in 2D PCA.
