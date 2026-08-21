# SCAPE_TOKEN_CONTRACT

Contract:
- rollout token ids, teacher-rescore token ids, training token ids, and tool-call span token ids must be carried as ids, not decoded and re-encoded for the loss path.
- `tool_span.py` parses tool-call text only for span/audit regression tests; production loss records must keep token id fields in `ComponentTransitionRecord`.
- GPT-OSS/Harmony should use official Harmony build/render contracts; Harness-1/Qwen should use the model chat template.

Implemented tests cover legal curate/end_search spans and invalid-tool rejection. Full tokenizer-in-tokenizer-out parity against live SCAPE rollout remains a paper-grade blocker.
