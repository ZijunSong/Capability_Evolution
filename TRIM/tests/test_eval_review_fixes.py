"""R01–R07 eval-review fixes: verify HTTP, token budget, messages, metrics, out dir."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from trim.eval.harness1_api_eval import (
    _cfg_number,
    assert_fresh_eval_dir,
    count_tool_calls_from_turns,
    normalize_query_metrics,
    summarize_api_traces,
)
from trim.local_backend.corpus_store import LocalCorpusStore
from trim.local_backend.tools import LocalReadDocumentTool
from trim.upstream_harness1.api_adapter import ChatCompletionsClient
from trim.upstream_harness1.env_bridge import (
    API_FORMAT_RETRY_PROMPT,
    clip_observation_text,
    flatten_message_text,
    openai_messages_from_env,
)
from trim.upstream_harness1.token_count import prefix_to_token_budget, whitespace_token_counter


@dataclass
class FakeSchema:
    name: str
    description: str = ""
    parameters: dict | None = None

    def model_dump(self):
        return {"name": self.name, "description": self.description, "parameters": self.parameters or {}}


class FakeObservation:
    def __init__(self, observations, sources, tool_metadata=None):
        self.observations = list(observations)
        self.sources = list(sources)
        self.tool_metadata = list(tool_metadata) if tool_metadata is not None else [None] * len(self.observations)


class FakeAction:
    def __init__(self, name: str = "search_corpus", source: str = "call_1"):
        self.tools = [SimpleNamespace(tool_schema=SimpleNamespace(name=name))]
        self.params = [{}]
        self.sources = [source]
        self.reasoning = None

    def as_iter(self):
        return zip(self.tools, self.params, self.sources)


class FakeTrajectory:
    def __init__(self, actions_and_observations, id=None):
        self.actions_and_observations = actions_and_observations

    def to_openai_format(self):
        messages = []
        pending_calls = []
        for item in self.actions_and_observations:
            if isinstance(item, FakeAction):
                calls = []
                for tool, _params, source in item.as_iter():
                    calls.append(
                        {
                            "id": str(source),
                            "type": "function",
                            "function": {"name": tool.tool_schema.name, "arguments": "{}"},
                        }
                    )
                pending_calls = [c["id"] for c in calls]
                messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            else:
                for text, source in zip(item.observations, item.sources):
                    if source == "user":
                        messages.append({"role": "user", "content": text})
                    else:
                        call_id = str(source) if source else (pending_calls[0] if pending_calls else "tool")
                        messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        return messages


class FakeWM:
    def __init__(self, pool: int):
        self._pool = pool

    def get_pool_size(self):
        return self._pool


class FakeEnv:
    def __init__(self, *, obs_text: str, turns_since_curate: int = 0, pool: int = 0):
        self.system_prompt = "sys"
        self._turns_since_curate = turns_since_curate
        self.wm = FakeWM(pool)
        self.text_token_counter = whitespace_token_counter
        self._action = FakeAction()
        self._obs = FakeObservation([obs_text], [self._action.sources[0]])

    def selected_context_window(self):
        return {
            "wm_text": None,
            "recent_actions": [self._action],
            "recent_observations": [self._obs],
            "result_summaries": [""],
        }


FAKE_MODS = {"Trajectory": FakeTrajectory, "Observation": FakeObservation}


def test_r01_verify_payload_omits_tool_fields():
    client = ChatCompletionsClient(base_url="http://127.0.0.1:9/v1", model="v", max_tokens=2048)
    payload = client.build_payload([{"role": "user", "content": "CLAIM"}], tools=None, max_tokens=80, temperature=0.0)
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "parallel_tool_calls" not in payload
    assert payload["max_tokens"] == 80
    assert payload["temperature"] == 0.0
    with_tools = client.build_payload([{"role": "user", "content": "q"}], tools=[{"type": "function"}])
    assert "tools" in with_tools
    assert with_tools["tool_choice"] == "auto"


def test_r01_local_verify_issues_one_http_request():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from trim.local_backend.auxiliary import LocalVerifierClient, OpenAIChatShim

    hits: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            hits.append(body)
            payload = {
                "choices": [{"message": {"content": "yes. matches the document."}, "finish_reason": "stop"}]
            }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        shim = OpenAIChatShim(LocalVerifierClient(base_url=url, model="fixed-local-verifier", require_loopback=True))
        resp = shim.create(
            model="fixed-local-verifier",
            messages=[{"role": "user", "content": "CLAIM: x\n\nDOCUMENT:\ny"}],
            temperature=0.0,
            max_tokens=80,
            timeout=20,
        )
        assert resp.choices[0].message.content.lower().startswith("yes")
        assert len(hits) == 1
        assert "tools" not in hits[0]
        assert hits[0]["max_tokens"] == 512
    finally:
        server.shutdown()


def test_r02_read_truncates_oversized_single_chunk():
    store = LocalCorpusStore.from_memory([{"id": "long", "text": "word " * 5000}])
    before = store.checksum_of("long")
    read = LocalReadDocumentTool(
        store=store,
        schema=FakeSchema("read_document"),
        token_counter=whitespace_token_counter,
        max_tokens=4096,
    )
    text, _ = read({"doc_id": "long"})
    body = text.split("\n", 1)[-1]
    assert whitespace_token_counter(body) <= 4096
    assert whitespace_token_counter(body) >= 4000
    assert "word" in body
    assert store.checksum_of("long") == before
    assert prefix_to_token_budget("a b c d e", 2, whitespace_token_counter).split() == ["a", "b"]


def test_r03_retry_nudge_and_obs_clip():
    tail = "_TAIL_UNIQUE_EVIDENCE"
    long_obs = "HEAD_UNIQUE_" + ("x" * (40011 - 12 - len(tail))) + tail
    assert len(long_obs) == 40011
    env = FakeEnv(obs_text=long_obs, turns_since_curate=1, pool=2)
    msgs = openai_messages_from_env(env, FAKE_MODS, retry=False, max_obs_chars=15000)
    blob = json.dumps(msgs)
    assert "truncated, 40011 chars total" in blob
    assert tail not in blob
    assert "HEAD_UNIQUE_" in blob
    assert "call curate NOW" in blob
    retry_msgs = openai_messages_from_env(env, FAKE_MODS, retry=True, max_obs_chars=15000)
    retry_blob = json.dumps(retry_msgs)
    assert API_FORMAT_RETRY_PROMPT in retry_blob
    assert "call curate NOW" not in retry_blob
    assistant_ids = []
    tool_ids = []
    for msg in retry_msgs:
        if msg.get("role") == "assistant":
            assistant_ids.extend(str(c.get("id")) for c in (msg.get("tool_calls") or []))
        if msg.get("role") == "tool":
            tool_ids.append(str(msg.get("tool_call_id") or ""))
    assert tool_ids
    assert set(tool_ids) <= set(assistant_ids)


def test_r03_identical_messages_without_retry_flag():
    env = FakeEnv(obs_text="short")
    a = openai_messages_from_env(env, FAKE_MODS, retry=False)
    b = openai_messages_from_env(env, FAKE_MODS, retry=False)
    assert a == b
    c = openai_messages_from_env(env, FAKE_MODS, retry=True)
    assert a != c


def test_r04_summary_f1_from_precision_recall_not_missing_field():
    metrics = normalize_query_metrics({"precision": 1.0, "recall": 1.0, "trajectory_recall": 1.0})
    assert metrics["f1"] == 1.0
    assert metrics["f1_missing"] is False
    summary = summarize_api_traces([metrics])
    assert summary["f1"] == 1.0
    missing = normalize_query_metrics({"reward": 0.0})
    assert missing["f1"] is None
    assert missing["f1_missing"] is True
    miss_sum = summarize_api_traces([missing])
    assert miss_sum["f1"] == 0.0
    assert miss_sum["f1_missing"] == 1
    assert miss_sum["cohort_denominator"] == 1


def test_r04_tool_call_counts_are_not_turn_counts():
    turns = [
        {
            "parse_error": None,
            "api_response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "search_corpus"}},
                                {"function": {"name": "fan_out_search"}},
                            ]
                        }
                    }
                ]
            },
        }
    ]
    counts = count_tool_calls_from_turns(turns)
    assert counts["n_tool_calls"] == 2
    assert counts["n_fan_out_calls"] == 1
    summary = summarize_api_traces([normalize_query_metrics({"precision": 0.0, "recall": 0.0}, turns=turns)])
    assert summary["mean_tool_calls_per_query"] == 2
    assert summary["mean_turns"] == 1


def test_r05_format_error_step_metrics_reach_summary():
    metrics = normalize_query_metrics(
        {"reward": 0.0},
        step_metrics={"format_error": 1.0, "no_error": 0.0, "reward": -0.2},
        done=True,
    )
    assert metrics["format_error"] == 1.0
    assert metrics["ended"] is True
    assert metrics["reward"] == -0.2
    summary = summarize_api_traces([metrics])
    assert summary["format_error_rate"] == 1.0


def test_r06_temperature_zero_is_not_replaced():
    assert _cfg_number({"temperature": 0.0}, "temperature", 1.0) == 0.0
    assert _cfg_number({}, "temperature", 1.0) == 1.0
    assert _cfg_number({"temperature": None}, "temperature", 1.0) == 1.0


def test_r07_reuse_out_dir_is_rejected(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "PER_QUERY.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already has"):
        assert_fresh_eval_dir(out)
    fresh = tmp_path / "fresh"
    assert_fresh_eval_dir(fresh) is None


def test_clip_observation_helper_matches_upstream_suffix():
    text = "a" * 40011
    clipped = clip_observation_text(text, 15000)
    assert clipped.startswith("a" * 15000)
    assert "40011 chars total" in clipped
    assert clipped.endswith("chars total)")


def test_budget_single_turn_large_obs_is_clipped():
    def char_counter(text: str) -> int:
        return len(str(text))

    long_obs = "BLOCK " * 8000
    env = FakeEnv(obs_text=long_obs)
    env.text_token_counter = char_counter
    msgs = openai_messages_from_env(
        env,
        FAKE_MODS,
        token_counter=char_counter,
        max_obs_chars=15000,
        prompt_token_budget=4000,
    )
    assert char_counter(flatten_message_text(msgs)) <= 4000


def test_cohort_final_answer_recall_mixed_perfect_and_empty_curated():
    perfect = normalize_query_metrics(
        {"precision": 1.0, "recall": 1.0, "final_answer_recall": 1.0, "n_curated": 3.0}
    )
    empty = normalize_query_metrics(
        {
            "precision": 0.0,
            "recall": 0.0,
            "trajectory_recall": 0.25,
            "n_curated": 0.0,
            "no_error": 1.0,
        }
    )
    assert empty["final_answer_recall"] == 0.0
    summary = summarize_api_traces([perfect, empty])
    assert summary["final_answer_recall"] == 0.5
    assert summary["final_answer_recall_missing"] == 0
    assert summary["cohort_denominator"] == 2


def test_cohort_format_error_counts_as_zero_quality():
    ok = normalize_query_metrics({"precision": 1.0, "recall": 1.0, "final_answer_recall": 1.0})
    bad = normalize_query_metrics(
        {"reward": -0.2},
        step_metrics={"format_error": 1.0, "no_error": 0.0, "reward": -0.2},
        done=True,
    )
    assert bad["recall"] == 0.0
    assert bad["final_answer_recall"] == 0.0
    summary = summarize_api_traces([ok, bad])
    assert summary["recall"] == 0.5
    assert summary["f1"] == 0.5
    assert summary["format_error_rate"] == 0.5
