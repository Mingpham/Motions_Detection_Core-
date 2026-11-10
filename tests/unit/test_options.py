import pytest

from movecore.core.config import ConfigError
from movecore.core.options import OptionChecks, is_number

CHECKS = OptionChecks("modules.demo")


def test_is_number_rejects_bools():
    assert is_number(3) and is_number(0.5)
    assert not is_number(True) and not is_number("3")


def test_messages_name_the_section_and_option():
    with pytest.raises(ConfigError, match=r"'modules.demo.size' must be an integer from 1, got 0"):
        CHECKS.integer("size", 0, 1)


def test_unknown_options_are_listed():
    with pytest.raises(ConfigError, match="Unknown 'modules.demo' options: colour, sise"):
        CHECKS.reject_unknown({"sise": 1, "colour": 2, "size": 3}, {"size"})


@pytest.mark.parametrize(
    ("check", "value"),
    [
        (lambda v: CHECKS.integer("n", v, 0, 10), 11),
        (lambda v: CHECKS.odd("n", v), 4),
        (lambda v: CHECKS.number("n", v), -1),
        (lambda v: CHECKS.number("n", v, at_most=1), 1.5),
        (lambda v: CHECKS.choice("n", v, ["a", "b"]), "c"),
        (lambda v: CHECKS.boolean("n", v), "yes"),
    ],
)
def test_bad_values_raise(check, value):
    with pytest.raises(ConfigError):
        check(value)


def test_good_values_pass():
    CHECKS.integer("n", 5, 0, 10)
    CHECKS.odd("n", 5)
    CHECKS.number("n", 1, at_most=1)
    CHECKS.choice("n", "a", ["a", "b"])
    CHECKS.boolean("n", False)
