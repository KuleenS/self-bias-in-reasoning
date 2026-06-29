from selfbias.evaluate import parse_judgment
from selfbias.prompts import evaluation_messages, generation_messages
from selfbias.data.base import ReasoningExample


def test_parse_judgment_json():
    assert parse_judgment('{"valid": true}') == (True, False)
    assert parse_judgment('blah {"valid": false} trailing') == (False, False)


def test_parse_judgment_regex_fallback():
    assert parse_judgment('the answer is "valid": true') == (True, False)


def test_parse_judgment_error():
    assert parse_judgment("no verdict here") == (None, True)
    assert parse_judgment("") == (None, True)


def test_generation_messages_include_instruction():
    ex = ReasoningExample("0", "gsm8k", "2+2?", "4", "numeric")
    msgs = generation_messages(ex)
    assert msgs[0]["role"] == "user"
    assert "Answer:" in msgs[0]["content"]
    assert "2+2?" in msgs[0]["content"]


def test_evaluation_messages_truncates():
    long_chain = "x" * 50000
    msgs = evaluation_messages("q", long_chain, max_reasoning_chars=100)
    assert "[truncated]" in msgs[0]["content"]
