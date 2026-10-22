import json
import urllib.request

fn = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": "Search corpus",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
payload = {
    "model": "GLM-4-32B-0414",
    "messages": [{"role": "user", "content": "Search the corpus for smoke test query."}],
    "tools": [fn],
    "tool_choice": "auto",
    "max_tokens": 256,
    "temperature": 0.0,
}
req = urllib.request.Request(
    "http://127.0.0.1:8090/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read().decode())
except Exception as e:
    print("REQ_FAIL", e)
    raise SystemExit(1)
print("RAW_RESPONSE", json.dumps(body, ensure_ascii=False)[:1000])
msg = (body.get("choices") or [{}])[0].get("message", {})
print("CONTENT", repr(msg.get("content")))
print("TOOL_CALLS", json.dumps(msg.get("tool_calls"), ensure_ascii=False))
