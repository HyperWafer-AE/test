# Limitations

- TerminalBench observations are truncated to 5,000 chars, so dependencies are lower-bound estimates.
- Dataset-server partial loading may produce fewer rows if the network fails; metadata records the actual count.
- Confidence intervals are trace-bootstrap estimates, not a full hierarchical model.
- The severity model is a problem-opportunity proxy, not a runtime speedup experiment.
- KV stability is inferred from repeated prefixes/tokens and is not direct KV-cache telemetry.
- Object future-window prediction uses all tool steps up to 20,000 evaluation points for runtime control on large samples.
