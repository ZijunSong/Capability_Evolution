# CLAUDE.md

## Environment Rules

- Do not run torch, vLLM, or other GPU-heavy Python workloads from `/mnt` environments. `/mnt` is a JuiceFS mount and can hang or fail during torch/vLLM startup and model loading.
- Use Python/conda/venv environments under `/opt` for SCAPE GPU experiments. If the required `/opt` environment does not exist, create one under `/opt` before launching experiments.
- It is fine to keep repositories, datasets, models, and output artifacts under `/mnt`; only the executable environment must live outside JuiceFS.
- For SCAPE H100/H20 runs, prefer a named `/opt/scape-*` environment and record its interpreter path in `RUN_MANIFEST.json`.
