from selfbias.answers import extract_answer, grade, is_correct


def test_numeric_extraction_and_match():
    assert grade("...so the total is 42.\nAnswer: 42", "42", "numeric") == ("42", True)
    assert grade("Answer: 1,000", "1000", "numeric")[1] is True
    assert grade("The result is \\boxed{7}", "7", "numeric")[1] is True
    assert grade("Answer: 41", "42", "numeric")[1] is False


def test_mcq_extraction_and_match():
    assert grade("Reasoning...\nAnswer: B", "B", "mcq") == ("B", True)
    assert grade("therefore (C) is right", "C", "mcq")[1] is True
    assert grade("answer: a", "A", "mcq")[1] is True
    assert grade("Answer: D", "A", "mcq")[1] is False


def test_freeform_match():
    assert grade("Answer: Paris", "Paris", "freeform")[1] is True
    assert grade("Answer: paris .", "Paris", "freeform")[1] is True
    # numeric fallback inside freeform
    assert grade("Answer: 3.0", "3", "freeform")[1] is True
    assert grade("Answer: London", "Paris", "freeform")[1] is False


def test_missing_answer_is_incorrect():
    assert extract_answer("", "numeric") is None
    assert is_correct(None, "5", "numeric") is False
