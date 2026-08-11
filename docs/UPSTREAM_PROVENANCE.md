# UPSTREAM_PROVENANCE

## Harness-1

| Field | Value |
|---|---|
| Upstream repo | https://github.com/pat-jj/harness-1 |
| Pin strategy | git submodule at `external/harness-1` |
| Pinned commit (bootstrap) | `8ac4012167858f6478fb2a8fd840e4550e2af161` |
| Model checkpoint | https://huggingface.co/pat-jj/harness-1 |
| Train data | https://huggingface.co/datasets/pat-jj/harness-1-train-data |
| Paper | https://arxiv.org/abs/2606.02373 |

## Integration policy

- Prefer wrappers / adapters / env overrides in `scape/`.
- Do **not** pile SCAPE method changes into upstream sources.
- If an upstream hook is missing, keep a minimal patch under `patches/` and document it here.

## Local mirror used at bootstrap

If GitHub clone is unavailable, the pin may be initialized from the local mirror:

`/data/ppnm/BiSHOP/harness-1-upstream` @ `8ac4012`.
