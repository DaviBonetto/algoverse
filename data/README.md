# Prompt Dataset

This dataset contains 80 short prompts for a probing study:

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

