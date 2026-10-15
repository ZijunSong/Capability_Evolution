# Propose a harness component

You may read failure labels, trajectory ids, prior component source, and rejected reasons from the packet. You may not read gold document ids, answer strings, or confirmation-set labels.

Write a general mechanism. Do not paste a query id, a gold document id, or a question-specific search string into the component, manifest, guidance, or recipe.

Each proposal changes one component unless the packet asks for prune, merge, or recombine. Include:

- the component diff under `components/`, `guidance/`, and `recipes/`
- `hypothesis.json` with failure type, trajectory id, mechanism, expected fix, and rollback
- `adaptation.json` naming the source component, the student tool schema, where the information comes from, and the skip reason when no native action exists
- one positive trigger, one negative trigger, one missing-evidence case, and one invalid-argument case

Do not edit the trainer, the scorer, the split manifest, or another candidate's directory.
