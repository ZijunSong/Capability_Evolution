# Adapt a component into a student action

The teacher may use the component's private view. The projected target must be one action the student can execute from the same visible state.

Prefer, in order:

1. an explicit native action that reproduces the effect, such as `curate` on a document the student has already seen
2. skip internal events until a later native anchor that does not depend on those hidden results
3. the first step of a native recipe, then wait for the student's real observation

If the action needs a document id, a tool, or a fact the student cannot access, emit SKIP and record the reason. Do not emit a second action in the same target. Do not copy the student's prompt and call it the teacher condition.
