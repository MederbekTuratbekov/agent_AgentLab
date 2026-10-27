import pytest

from agentlab.agent.calc import safe_eval


def test_basic():
    assert safe_eval("200 * 0.15") == 30
    assert safe_eval("((6 - 2) * 3) + 25") == 37


def test_blocks_import():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hi')")


def test_blocks_attribute_access():
    with pytest.raises(ValueError):
        safe_eval("().__class__")


def test_blocks_huge_power():
    with pytest.raises(ValueError):
        safe_eval("9 ** 9999")