# UPSTREAM_LOCK

- EasyOPD upstream SHA: `277b76fb675a11b0236a9c86573207251ac41727`
- Runtime: `/opt/scape-easyopd-smoke7`
- Python: 3.12.13
- torch: 2.6.0+cu124
- transformers: 5.15.0
- peft: 0.20.0
- ray: 2.47.1
- vllm: 0.8.5.post1
- GPUs: 8 × NVIDIA H100 80GB HBM3
- BF16 matmul: pass on all 8
- SCAPE/Harness live deps: installed in `/opt/scape-easyopd-smoke7`
- verl core modified: yes, minimal SCAPE_COMPONENT_OPD fallback in `verl/workers/actor/dp_actor.py`
- Status: `SCAPE_EASYOPD_READY`
