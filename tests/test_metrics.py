from agentlab.evals.metrics import contains, exact_match, extract_number, f1, numbers_match


def test_em_ignores_articles_and_case():
    assert exact_match("The Eiffel Tower", ["eiffel tower"]) == 1.0


def test_f1_partial():
    assert 0 < f1("Eiffel Tower Paris", ["Eiffel Tower"]) < 1


def test_contains():
    assert contains("It was Guido van Rossum.", ["Guido van Rossum"]) == 1.0


def test_extract_number():
    assert extract_number("The answer is $1,234.50.") == 1234.5
    assert numbers_match("42.", "42")