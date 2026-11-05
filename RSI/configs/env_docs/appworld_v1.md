# AppWorld runtime interface (v1, probed from the live environment)

Each turn you submit exactly ONE `python_exec` action. Its `code` runs in a
persistent IPython session — variables you define stay available in later turns.

## Objects injected into the session

- `apis` — the app API collection. The task's allowed apps are attributes,
  e.g. `apis.phone`, `apis.gmail`, `apis.spotify`, `apis.todoist`,
  `apis.simple_note`, `apis.venmo`, `apis.splitwise`, `apis.amazon`,
  `apis.file_system`, `apis.admin`, `apis.supervisor`, `apis.api_docs`
  (a given task exposes a subset).
- `requester` — low-level HTTP requester (rarely needed; prefer `apis`).

## How to discover the API surface (verified commands)

- List the app names and descriptions:
  `apis.api_docs.show_app_descriptions()`
- List one app's API names:
  `apis.api_docs.show_app_apis(app_name='phone')`
- Show one API's full doc (path, method, parameters, types, required):
  `apis.api_docs.show_api_doc(app_name='simple_note', api_name='create_note')`
- Every API call returns a dict-like response (or a list of them); errors are
  raised as Python exceptions with a `message` field — read them.

## Authentication

Most apps require an access token before use:
`account = apis.<app>.authenticate(**accounts.<app>)` — the supervisor's own
credentials live in `apis.supervisor` (e.g.
`apis.supervisor.show_account_passwords()`), and task instructions tell you
whose account to act on. Do not invent credentials.

## Conventions

- Do not invent API names, parameters, or observation contents; query
  `api_docs` when unsure.
- Pagination: list APIs typically take `page_index` — iterate until an empty
  page before concluding an entity does not exist.
- The supervisor is the user you are helping; act on the account named in the
  task, verify entity identity before mutating, and confirm effects after
  mutating.
- **Answering the supervisor**: if the task asks a question (e.g. "How many…?",
  "What is the total…?"), compute the answer and submit it with
  `apis.supervisor.complete_task(answer=<answer>)` BEFORE finishing — the
  answer must be concise (a number, yes/no, or a specific entity). If the task
  is not a question (e.g. "Start my music player"), finish without an answer.
  After mutating, verify the change took effect (re-read the record / check
  the affected row count) before concluding it succeeded.
