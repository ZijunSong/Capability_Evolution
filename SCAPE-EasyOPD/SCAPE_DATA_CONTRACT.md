# SCAPE_DATA_CONTRACT

`ComponentTransitionRecord` in `types.py` standardizes SCAPE OPD data fields: query/trajectory/turn ids, component identity, state/view hashes, prompt/response token ids, masks, logprobs, component event/effect, projected action, visibility validity, rewards, visible docs, curated pre/post state, split, and seed.

Collectors write JSONL/CSV under `outputs/scape_easyopd/components/<component>/...`. Tiny verl smoke data is written only under `outputs/scape_easyopd/framework/tiny_data/` and is marked non-paper-grade.
