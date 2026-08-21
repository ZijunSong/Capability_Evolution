# VERL_PATCH_AUDIT

## Modified verl files

### `verl/workers/actor/dp_actor.py`

Reason: the approved `/opt/scape-easyopd-smoke7` runtime can run Qwen3/verl with `use_remove_padding=false`, but does not include `flash_attn`. Upstream imported `flash_attn.bert_padding` at module import time whenever CUDA was visible, blocking even the non-remove-padding smoke path.

Patch: a minimal `# ============ [EasyOPD:SCAPE_COMPONENT_OPD] ============` guarded fallback imports `einops.rearrange` and defers the hard failure to `index_first_axis/pad_input/unpad_input` if a code path actually requires flash-attn.

Safe default: when `flash_attn` is installed, upstream behavior is unchanged. If `use_remove_padding=true` without flash-attn, the fallback raises an explicit `ModuleNotFoundError` instead of silently changing behavior.

### `verl/trainer/main_ppo.py`

Existing local EasyOPD compatibility patch restores `all_special_tokens_extended` for vLLM 0.8.5 + Transformers 5.x. This patch predates the final SCAPE smoke and remains required by this local upstream bundle.
