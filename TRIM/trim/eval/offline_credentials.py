"""Local eval/train never calls OpenAI; keep dummy keys so import-time clients start.

Pyserini's ``lucene`` import pulls ``pyserini.encode._openai``, which constructs
``openai.OpenAI()`` at module load. An unset *or empty* ``OPENAI_API_KEY``
(common after loading a blank ``.env``) raises before BM25 can run. TRIM retrieval
is local Lucene / JSONL overlap, so a placeholder key is enough and no content
moderation / embedding audit is performed.
"""

from __future__ import annotations

import os

LOCAL_OPENAI_API_KEY = "sk-local-offline"

_PLACEHOLDER_ENV = {
    "OPENAI_API_KEY": LOCAL_OPENAI_API_KEY,
    "CHROMA_API_KEY": "local-offline",
    "CHROMA_DATABASE": "local-offline",
}


def _missing(value: str | None) -> bool:
    return value is None or not str(value).strip()


def ensure_local_offline_credentials(env: dict[str, str] | None = None) -> dict[str, str]:
    """Fill blank OpenAI/Chroma credentials so local BM25 eval can start."""
    target = env if env is not None else os.environ
    for key, placeholder in _PLACEHOLDER_ENV.items():
        if _missing(target.get(key)):
            target[key] = placeholder
    return target
