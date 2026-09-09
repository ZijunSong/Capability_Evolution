"""Local verifier / reranker clients. Missing config is an error, not a stub."""

from __future__ import annotations

from typing import Any, Callable, Sequence
from urllib.parse import urlparse

from trim.upstream_harness1.api_adapter import ChatCompletionsClient


def _assert_local_url(url: str, *, name: str) -> None:
    parsed = urlparse(str(url))
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            f"{name} must be a loopback URL for local_bm25 offline runs; got {url!r}"
        )


class LocalVerifierClient:
    """OpenAI-compatible chat client used only by original exec_verify_claim."""

    def __init__(self, *, base_url: str, model: str, api_key: str | None = None, require_loopback: bool = True):
        if require_loopback:
            _assert_local_url(base_url, name="verify_base_url")
        self.base_url = base_url
        self.model = model
        self._http = ChatCompletionsClient(base_url=base_url, model=model, api_key=api_key, temperature=0.0)

    def chat(self, **kwargs: Any) -> Any:
        """Shape expected by original exec_verify_claim (OpenAI SDK-like)."""
        messages = kwargs.get("messages") or []
        tools = kwargs.get("tools")
        max_tokens = kwargs.get("max_tokens")
        timeout = kwargs.get("timeout")
        temperature = kwargs.get("temperature")
        response = self._http.complete(
            list(messages),
            list(tools) if tools else None,
            max_tokens=None if max_tokens is None else int(max_tokens),
            timeout_s=None if timeout is None else float(timeout),
            temperature=None if temperature is None else float(temperature),
        )
        return _OpenAICompat(response, model=self.model)


class _Msg:
    def __init__(self, payload: dict[str, Any]):
        self.content = payload.get("content")
        self.role = payload.get("role")
        self.tool_calls = payload.get("tool_calls")


class _Choice:
    def __init__(self, payload: dict[str, Any]):
        self.message = _Msg((payload.get("message") or {}))
        self.finish_reason = payload.get("finish_reason")


class _OpenAICompat:
    def __init__(self, response: dict[str, Any], *, model: str):
        self.choices = [_Choice(c) for c in (response.get("choices") or [])]
        self.model = model
        self.usage = response.get("usage")

    @property
    def chat(self) -> "_OpenAICompat":
        return self

    def completions(self) -> Any:
        return self

    def create(self, **kwargs: Any) -> "_OpenAICompat":
        del kwargs
        return self


class OpenAIChatShim:
    """`client.chat.completions.create` wrapper around LocalVerifierClient."""

    def __init__(self, inner: LocalVerifierClient):
        self._inner = inner
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> _OpenAICompat:
        return self._inner.chat(**kwargs)


class LocalHttpReranker:
    identity = "local_http_reranker"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        token_counter: Callable[[str], int] | None = None,
        max_tokens: int | None = None,
        require_loopback: bool = True,
        api_key: str | None = None,
    ) -> None:
        if require_loopback:
            _assert_local_url(base_url, name="reranker_base_url")
        from harness.rerank import RerankResult  # type: ignore[import-not-found]

        self._result_cls = RerankResult
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.token_counter = token_counter
        self.max_tokens = max_tokens
        self._http = ChatCompletionsClient(base_url=base_url, model=model, api_key=api_key, temperature=0.0)

    def __call__(self, query: str, documents: Sequence[str], max_tokens: int | None = None) -> list[Any]:
        import json
        import urllib.request

        # Prefer a dedicated /rerank if the local server exposes it; otherwise fail loudly.
        payload = json.dumps({"model": self.model, "query": query, "documents": list(documents)}).encode("utf-8")
        url = self.base_url
        if url.endswith("/v1"):
            url = url[: -len("/v1")]
        req = urllib.request.Request(
            url + "/rerank",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"local reranker at {url}/rerank failed: {exc}. "
                "Do not silently skip rerank. Pass --reranker none if this run has no reranker."
            ) from exc
        results = body.get("results") or body.get("data") or []
        ranked: list[Any] = []
        for item in results:
            idx = int(item.get("index") if item.get("index") is not None else item.get("original_index"))
            score = float(item.get("relevance_score") or item.get("score") or 0.0)
            doc = documents[idx] if 0 <= idx < len(documents) else str(item.get("document") or "")
            ranked.append(self._result_cls(document=doc, score=score, original_index=idx))
        budget = max_tokens if max_tokens is not None else self.max_tokens
        if self.token_counter is not None and budget is not None:
            kept: list[Any] = []
            used = 0
            for row in ranked:
                n = self.token_counter(row.document)
                row.tokens = n
                if used + n > budget:
                    break
                kept.append(row)
                used += n
            return kept
        return ranked
