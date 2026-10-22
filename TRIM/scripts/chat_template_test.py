import json
from transformers import AutoTokenizer

c = json.load(open("/mnt/songzijun/models/GLM-4-32B-0414/config.json"))
assert c.get("architectures") == ["Glm4ForCausalLM"], c.get("architectures")
assert "quantization_config" not in c
tok = AutoTokenizer.from_pretrained("/mnt/songzijun/models/GLM-4-32B-0414", trust_remote_code=True)
msg = [{"role": "user", "content": "Search the corpus for smoke test query."}]
fn = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": "x",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
tools = [fn]
ids = tok.apply_chat_template(msg, tools=tools, add_generation_prompt=True, tokenize=False)
print("GLM_CHAT_TEMPLATE_OK", {"chars": len(ids)})
print("RENDERED_PROMPT", repr(ids)[:400])
