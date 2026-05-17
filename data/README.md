# Prompt Dataset

This directory contains prompt datasets for probing studies.

## `prompts.jsonl`

The original pilot dataset contains 80 short prompts:

- 40 `harmful` prompts: sanitized harmful-intent or policy-evasion requests. These are phrased at the intent level and intentionally omit operational details.
- 40 `benign` prompts: normal requests across math, coding, science, history, writing, and everyday assistance.

The goal is not to provide a red-team benchmark for deployment. The goal is to test whether harmful-intent labels are linearly decodable from intermediate hidden states in a small instruction-tuned language model.

Schema:

```json
{"id":"harmful_001","label":"harmful","category":"cyber_abuse","prompt":"..."}
```

Fields:

- `id`: unique prompt identifier.
- `label`: either `harmful` or `benign`.
- `category`: coarse category used for error analysis.
- `prompt`: user-facing prompt text.

Safety note: prompts are intentionally sanitized. They indicate harmful intent without providing procedural steps, quantities, code, target details, or evasion instructions.

## `prompts_v2.jsonl`

The harder paper-style dataset contains 56 prompts:

- 20 harmful-intent / benign counterfactual pairs.
- 20 hard negatives that reuse safety-relevant vocabulary with benign intent.
- 8 easy benign prompts.
- 8 easy harmful-intent prompts.

Extra metadata fields support stronger evaluation:

- `family`: used for family-heldout generalization.
- `difficulty`: distinguishes counterfactual pairs, hard negatives, and easy controls.
- `pair_id`: links harmful/benign minimal pairs.
- `intent_type`: a more specific semantic tag for analysis.

The purpose of v2 is to reduce lexical shortcuts and test whether hidden-state probes detect intent rather than obvious keywords.
