# AppWorld runtime interface (v3, contract per refactor guide PR-01)

> v3 documents only vendor-verified calls.  (v1 taught two non-existent
> interfaces; it is kept at `appworld_v1.md` for historical reproduction.
> Every NEW run must use this file.)

Each turn you submit exactly ONE `python_exec` action. Its `code` runs in a
persistent Python session — variables you define stay available in later
turns.

## Objects injected into the session

- `apis` — the app API collection. Already imported: do NOT run
  `import apis` or `from appworld import apis`. The task's allowed apps are
  attributes, e.g. `apis.phone`, `apis.gmail`, `apis.spotify`,
  `apis.todoist`, `apis.simple_note`, `apis.venmo`, `apis.splitwise`,
  `apis.amazon`, `apis.file_system`, `apis.admin`, `apis.supervisor`,
  `apis.api_docs` (a given task exposes a subset).
- `requester` — low-level HTTP requester (rarely needed; prefer `apis`).

## Output rule (important)

Expression return values are NOT guaranteed to appear in your observation.
Whenever you need to READ a result, wrap it in `print(...)`:

```python
print(apis.api_docs.show_app_descriptions())
```

A bare `apis.api_docs.show_app_descriptions()` may execute successfully and
show you nothing.

## How to discover the API surface (all three verified)

```python
# 1. List the app names and descriptions:
print(apis.api_docs.show_app_descriptions())

# 2. List one app's API names and descriptions:
print(apis.api_docs.show_api_descriptions(app_name="spotify"))

# 3. Show one API's full doc (path, method, parameters, types, required):
print(apis.api_docs.show_api_doc(app_name="spotify", api_name="login"))
```

Every API call returns a
dict-like response (or a list of them); errors are raised as Python
exceptions with a `message` field — read them.

## Authentication

Authentication is app-specific — there is no universal auth call.
Before authenticating, look up the app's own auth API:

```python
print(apis.api_docs.show_api_doc(app_name="<app>", api_name="login"))
```

then follow THAT doc — the parameters, account fields and response structure
are app-specific. Publicly accessible supervisor account passwords can be
listed via `apis.supervisor` APIs (discover them the same way); act on the
account the task names. Do not invent credentials.

## Conventions

- Do not invent API names, parameters, or observation contents; query
  `api_docs` when unsure.
- Pagination: list APIs typically take `page_index` — iterate until an empty
  page before concluding an entity does not exist.
- The supervisor is the user you are helping; act on the account named in
  the task, verify entity identity before mutating, and confirm effects
  after mutating (re-read the record / check the affected row count).
- **Finishing**: call `Finish` with your final answer when the task asks a
  question (a number, yes/no, or a specific entity) or `null` if it does
  not. The runtime submits the answer to the supervisor for you — do not
  call `complete_task` yourself.
