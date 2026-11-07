"""PR-08 tests (guide §12 test_environment_target_validation + §12.1 the
three manual-review counterexamples)."""

from __future__ import annotations

from rsi.contracts.action import Finish, ToolCall
from rsi.contracts.observation import ToolSpec
from rsi.opd.validators import (
    VERDICT_INVALID,
    VERDICT_UNCERTAIN,
    VERDICT_VALID,
    validate_alfworld_target,
    validate_appworld_target,
    validate_webshop_target,
)

ALF_CATALOG = (
    ToolSpec(
        name="command",
        description="text command",
        parameters_schema={"type": "object", "properties": {"text": {"type": "string"}},
                           "required": ["text"]},
    ),
)
WS_CATALOG = (
    ToolSpec(
        name="click",
        description="click",
        parameters_schema={"type": "object", "properties": {"target": {"type": "string"}},
                           "required": ["target"]},
    ),
    ToolSpec(
        name="search",
        description="search",
        parameters_schema={"type": "object", "properties": {"query": {"type": "string"}},
                           "required": ["query"]},
    ),
)
AW_CATALOG = (
    ToolSpec(
        name="python_exec",
        description="code",
        parameters_schema={"type": "object", "properties": {"code": {"type": "string"}},
                           "required": ["code"]},
    ),
)


# --- ALFWorld ----------------------------------------------------------------

def test_alfworld_unknown_command_rejected():
    v = validate_alfworld_target(
        action=ToolCall(name="command", arguments={"text": "take book 2 from shelf 9"}),
        catalog=ALF_CATALOG,
        visible_state={"admissible_commands": ["go to bed 1", "take book 1 from bed 1"]},
    )
    assert v.verdict == VERDICT_INVALID
    assert "command_not_in_current_admissible_set" in v.reason_codes


def test_alfworld_visible_command_accepted():
    v = validate_alfworld_target(
        action=ToolCall(name="command", arguments={"text": "take book 1 from bed 1"}),
        catalog=ALF_CATALOG,
        visible_state={"admissible_commands": ["go to bed 1", "take book 1 from bed 1"]},
    )
    assert v.verdict == VERDICT_VALID


def test_alfworld_no_admissible_is_uncertain():
    v = validate_alfworld_target(
        action=ToolCall(name="command", arguments={"text": "look"}),
        catalog=ALF_CATALOG,
        visible_state={},
    )
    assert v.verdict == VERDICT_UNCERTAIN


# --- WebShop: counterexample one (legality != state alignment) ---------------

def test_webshop_buy_from_wrong_page_rejected():
    """§12.1 #1: student is on the SEARCH page, teacher clicked through to the
    item page.  Buy Now is a legal tool+argument but NOT legal at this
    state — must be rejected, not waved through by schema validity."""
    v = validate_webshop_target(
        action=ToolCall(name="click", arguments={"target": "Buy Now"}),
        catalog=WS_CATALOG,
        visible_state={"page_type": "search", "clickables": ["search_button"]},
    )
    assert v.verdict == VERDICT_INVALID


def test_webshop_buy_ok_on_item_page_with_options():
    v = validate_webshop_target(
        action=ToolCall(name="click", arguments={"target": "Buy Now"}),
        catalog=WS_CATALOG,
        visible_state={"page_type": "item", "clickables": ["Buy Now"],
                       "required_options_satisfied": True},
    )
    assert v.verdict == VERDICT_VALID


def test_webshop_buy_blocked_without_options():
    v = validate_webshop_target(
        action=ToolCall(name="click", arguments={"target": "Buy Now"}),
        catalog=WS_CATALOG,
        visible_state={"page_type": "item", "clickables": ["Buy Now"],
                       "required_options_satisfied": False},
    )
    assert v.verdict == VERDICT_INVALID
    assert any("options" in r for r in v.reason_codes)


def test_webshop_title_not_clickable():
    v = validate_webshop_target(
        action=ToolCall(name="click", arguments={"target": "SHOE XYZ (red, size 9)"}),
        catalog=WS_CATALOG,
        visible_state={"page_type": "results", "clickables": ["SHOE XYZ (red, size 9)", "page 2"]},
    )
    assert v.verdict == VERDICT_VALID  # it IS clickable here


# --- AppWorld ----------------------------------------------------------------

def test_appworld_syntax_error_rejected():
    v = validate_appworld_target(
        action=ToolCall(name="python_exec", arguments={"code": "def x(:\n  pass"}),
        catalog=AW_CATALOG,
        visible_state={"session_variables": [], "visible_api_names": []},
    )
    assert v.verdict == VERDICT_INVALID
    assert any(r.startswith("python_syntax_error") for r in v.reason_codes)


def test_appworld_fork_only_variable_rejected():
    """§12.1 #2 (AppWorld form): the teacher privately created ``token`` in
    its fork; the student's session never defined it.  Even though the DB is
    unchanged and the code is syntactically valid, the target is NOT
    realizable at the student state."""
    v = validate_appworld_target(
        action=ToolCall(
            name="python_exec",
            arguments={"code": "print(apis.spotify.search(token))"},
        ),
        catalog=AW_CATALOG,
        visible_state={"session_variables": [], "visible_api_names": ["spotify"]},
    )
    assert v.verdict == VERDICT_INVALID
    assert any("session_variables_missing" in r for r in v.reason_codes)


def test_appworld_session_variable_ok():
    v = validate_appworld_target(
        action=ToolCall(
            name="python_exec",
            arguments={"code": "print(apis.spotify.search(token))"},
        ),
        catalog=AW_CATALOG,
        visible_state={"session_variables": ["token"], "visible_api_names": ["spotify"]},
    )
    assert v.verdict == VERDICT_VALID


def test_appworld_dynamic_construct_uncertain():
    v = validate_appworld_target(
        action=ToolCall(
            name="python_exec",
            arguments={"code": "obj = globals()['x']\nprint(obj)"},
        ),
        catalog=AW_CATALOG,
        visible_state={"session_variables": [], "visible_api_names": []},
    )
    assert v.verdict == VERDICT_UNCERTAIN


def test_appworld_wrong_type_rejected():
    v = validate_appworld_target(
        action=ToolCall(name="python_exec", arguments={"code": 123}),
        catalog=AW_CATALOG,
        visible_state={},
    )
    assert v.verdict == VERDICT_INVALID


# --- three-value semantics ---------------------------------------------------

def test_finish_always_valid():
    for fn in (validate_alfworld_target, validate_webshop_target, validate_appworld_target):
        cat = {"alfworld": ALF_CATALOG, "webshop": WS_CATALOG, "appworld": AW_CATALOG}
        v = fn(action=Finish(answer=None), catalog=cat, visible_state={})
        assert v.verdict == VERDICT_VALID
