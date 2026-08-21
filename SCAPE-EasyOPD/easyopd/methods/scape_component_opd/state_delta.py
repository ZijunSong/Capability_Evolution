from __future__ import annotations

from typing import Any


def curated_delta(pre: list[str], post: list[str]) -> dict[str, Any]:
    pre_s = [str(x) for x in pre]
    post_s = [str(x) for x in post]
    return {
        "add_ids": [x for x in post_s if x not in pre_s],
        "remove_ids": [x for x in pre_s if x not in post_s],
        "curated_ids_pre": pre_s,
        "curated_ids_post": post_s,
    }
