# SCAPE_EASYOPD_ARCHITECTURE

`scape_component_opd` is implemented as an EasyOPD method-local extension under `easyopd/methods/scape_component_opd/`.

Boundaries:
- EasyOPD/verl own distributed training, rollout workers, FSDP, vLLM, checkpointing, and optimizer execution.
- SCAPE-specific logic lives in `ComponentSpec`, component registry, projection utilities, state snapshot/delta, tool span parser, hooks, live agent loop, real closed-loop evaluator, and CLI.
- SCAPE/Harness-1 runtime is source-of-truth for live multi-turn Search execution.

Completed training/evaluation path:
- `scripts/run_easyopd.py --method scape_component_opd ... --dry-run` validates EasyOPD registry/config.
- `scripts/scape_component_opd.py train ...` builds a verl `main_ppo` command.
- A one-step verl smoke completed with Qwen3-1.7B, vLLM rollout, FSDP actor, checkpoint save, and SCAPE method config attached.
- `scripts/scape_component_opd_actual_lora_smoke.py` validates projected-action CE style LoRA update and adapter reload.
- `SCAPEAgentLoop.run_live_search` executes live Harness-1 `search_corpus -> read_document -> final` multi-turn actions.
- `SCAPERealClosedLoopEvaluator` runs the same live loop under a single evaluator contract.
