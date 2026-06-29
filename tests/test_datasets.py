import os

import pytest

from selfbias.data.base import list_datasets, load_examples

EXPECTED = {
    "gsm8k", "math500", "aime", "gpqa_diamond", "mmlu_pro",
    "arc_challenge", "bbh", "folio", "logiqa", "commonsense_qa",
}


def test_all_loaders_registered():
    assert EXPECTED.issubset(set(list_datasets()))


def test_folio_local_loader():
    exs = load_examples("folio", limit=5)
    assert len(exs) == 5
    ex = exs[0]
    assert ex.dataset == "folio"
    assert ex.answer_type == "mcq"
    assert ex.gold_answer in {"A", "B", "C"}
    assert "Premises:" in ex.question


@pytest.mark.skipif(not os.getenv("SELFBIAS_NET_TESTS"), reason="set SELFBIAS_NET_TESTS=1 to hit HF")
def test_gsm8k_network():
    exs = load_examples("gsm8k", limit=3)
    assert len(exs) == 3
    assert exs[0].answer_type == "numeric"
    assert exs[0].gold_answer.lstrip("-").isdigit()
