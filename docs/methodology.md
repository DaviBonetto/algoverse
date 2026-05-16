# Methodology

## Research Question

Can harmful-intent prompts be detected from a model's intermediate hidden states before the model generates any output token?

## Model

The default target model is `Qwen/Qwen2.5-1.5B-Instruct`. The pipeline performs a forward pass over the formatted user prompt and extracts `outputs.hidden_states` from every layer. No assistant response is generated.

## Representation

For each layer, the project uses last-token pooling: the hidden state at the final non-padding prompt token. This gives one vector per prompt per layer.

```text
X shape = (num_prompts, num_layers, hidden_dim)
y shape = (num_prompts,)
```

## Probe

Each layer gets an independent logistic regression probe with standardized features and stratified cross-validation. The point is not to train a new neural model. The point is to test whether the information is linearly accessible from the representation.

Metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

The best layer is selected by F1, then accuracy as a tie-breaker.

## Early Guard

The guard demo combines two simple signals:

- Best-layer linear probe probability.
- Distance from the benign centroid at the same layer.

The guard returns:

- `ALLOW`: low probability and close to benign centroid.
- `REVIEW`: borderline probability or unusual benign-centroid distance.
- `FLAG`: probability above the high-recall harmful threshold.

## Safety Hygiene

The harmful-intent prompts are sanitized. They indicate misuse intent but avoid procedural details that would enable harm.

