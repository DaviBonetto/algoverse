# Limitations

This project is a small probing study. It should be read as an interpretability prototype, not a production safety system.

## Dataset Limits

The dataset contains 80 curated prompts. That is enough for a clear end-to-end research artifact, but not enough for robust deployment conclusions. Manual curation can also create artifacts: the probe may learn wording differences rather than a general harmful-intent representation.

## Model Limits

The default model is a small instruction-tuned LLM. Results may not transfer to larger models, different training recipes, or models with different refusal behavior.

## Probe Limits

A linear probe shows that information is decodable. It does not prove that the model uses that direction causally when deciding whether to refuse or comply.

## Adversarial Limits

The prompts are static. An adaptive attacker could search for phrasing that avoids the probe. A stronger study should include held-out jailbreak families and adversarially optimized prompts.

## Systems Limits

The early guard demo is intentionally simple. A production design would need calibration, latency testing, monitoring, privacy review, and defense-in-depth with other classifiers and policy checks.

