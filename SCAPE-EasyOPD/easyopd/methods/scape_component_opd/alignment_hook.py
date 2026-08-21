from __future__ import annotations


def assert_token_ids_equal(a: list[int], b: list[int], *, label: str = "token_ids") -> None:
    if list(a) != list(b):
        raise AssertionError(f"{label} mismatch")
