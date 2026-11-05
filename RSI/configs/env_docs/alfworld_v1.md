# ALFWorld runtime interface (v1, probed from the live environment)

You control an agent in a TextWorld household. Each turn you send exactly one
`command` tool call whose `text` is a short imperative for the world engine.

## Command grammar (verified against the engine's admissible commands)

Every location and object has a number suffix (`bed 1`, `drawer 2`,
`desklamp 1`). The engine accepts ONLY well-formed commands from its own
vocabulary; malformed ones return "Nothing happens" and waste your action
budget. Verb templates:

- `go to <receptacle>` — walk to a location seen in the room description.
- `take <object> from <receptacle>` — pick up (e.g. `take book 1 from bed 1`).
- `put <object> in/on <receptacle>` — place a held object
  (e.g. `put candle 1 on toilet 1`).
- `open <receptacle>` / `close <receptacle>` — fridges, drawers, cabinets,
  microwaves, safes.
- `examine <object>` — inspect something (in the room, on a surface, or held).
- `use <object>` — operate a device. **`use desklamp 1` is how you turn on a
  lamp** (not "turn on desklamp"); using the lamp while holding an object
  completes "examine with desklamp" tasks.
- `heat <object> with microwave 1` / `cool <object> with fridge 1` /
  `clean <object> with sink 1` / `clean <object> with sinkbasin 1` — appliance
  verbs for heat/cool/clean task types.
- `inventory`, `look`, `help` — status queries.

## Reading the feedback

After each command the engine replies with the outcome. "Nothing happens"
means the command was not valid in the current state — do NOT repeat it;
change verb, object numbering, or location. Surfaces list their contents when
you arrive (`You arrive at bed 1. On the bed 1, you see ...`). Objects you can
act on next are listed in the `admissible_commands` line appended to each
observation — treat it as the authoritative menu of legal actions and prefer
commands from it verbatim.

## Task types

`look_at_obj_in_light` (examine an object with a lamp: take the object,
`go to` the lamp, `use <lamp>`), `pick_and_place_simple` /
`pick_clean_then_place_in_recep` / `pick_heat_then_place_in_recep` /
`pick_cool_then_place_in_recep` / `pick_two_obj_and_place` (fetch, optionally
process, then `put` in the goal receptacle). The task sentence after "Your
task is to:" defines the goal; finish only when it is satisfied.
