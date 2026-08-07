"""The template-tournament verdict as a soft purity flag.

A decisive non-Ia winner should flag needs_classification; an Ia winner or an
indecisive (small-margin) non-Ia should not.
"""
from run_tonight import _template_flags_nonia, TEMPLATE_NONIA_MARGIN

M = TEMPLATE_NONIA_MARGIN


def test_decisive_non_ia_flags():
    assert _template_flags_nonia("IIn", M + 0.1) is True
    assert _template_flags_nonia("IIP", M + 1.0) is True


def test_ia_winner_never_flags():
    assert _template_flags_nonia("Ia", 10.0) is False


def test_indecisive_non_ia_does_not_flag():
    assert _template_flags_nonia("IIn", M - 0.1) is False


def test_missing_or_bad_inputs_do_not_flag():
    assert _template_flags_nonia(None, 5.0) is False
    assert _template_flags_nonia("IIn", float("nan")) is False
    assert _template_flags_nonia("IIn", None) is False
