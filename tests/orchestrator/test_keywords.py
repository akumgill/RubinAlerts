"""Tests for the structured, clamped controlled-keyword vocabulary and the
ingestion-side override ("non-negotiable") flag.

All in-memory: Target objects built directly and a temp CSV for ingestion; no
network/DB.
"""

import logging
import math

from orchestrator.models import Target
from orchestrator.normalize import load_targets_csv, parse_keywords
from orchestrator.prioritizer import (
    compute_composite_score,
    _keyword_adjustment,
    KEYWORD_WEIGHTS,
    KEYWORD_ADJ_CLAMP,
)


def test_free_text_notes_no_longer_scored():
    """The old substring false-fire is GONE: a target whose free-text notes
    contain 'too' (in "too faint") but has no structured keywords gets a zero
    keyword adjustment."""
    t = Target(name='T', ra_deg=150.0, dec_deg=2.0, priority=2,
               notes='too faint to bother', keywords=[])
    _, bd = compute_composite_score(t)
    assert bd['keyword_adj'] == 0.0
    assert bd['keyword_term'] == 0.0


def test_structured_single_tag():
    """A single recognized structured tag contributes its exact weight."""
    t = Target(name='T', ra_deg=150.0, dec_deg=2.0, priority=2,
               keywords=['urgent'])
    _, bd = compute_composite_score(t)
    assert math.isclose(bd['keyword_adj'], 0.2)


def test_structured_tags_sum_then_clamp():
    """Stacked tags sum then CLAMP to KEYWORD_ADJ_CLAMP (not the raw sum)."""
    # 0.3 + 0.2 + 0.25 = 0.75 raw -> clamps to 0.5.
    t = Target(name='T', ra_deg=150.0, dec_deg=2.0, priority=2,
               keywords=['high_priority', 'urgent', 'too'])
    _, bd = compute_composite_score(t)
    assert math.isclose(bd['keyword_adj'], KEYWORD_ADJ_CLAMP)
    assert math.isclose(bd['keyword_adj'], 0.5)
    # Sanity: the unclamped sum really would have exceeded the clamp.
    raw = KEYWORD_WEIGHTS['high_priority'] + KEYWORD_WEIGHTS['urgent'] \
        + KEYWORD_WEIGHTS['too']
    assert raw > KEYWORD_ADJ_CLAMP


def test_negative_tags_clamp_low():
    """Net demotions clamp at -KEYWORD_ADJ_CLAMP too."""
    # -0.3 + -0.3 + -0.2 = -0.8 raw -> clamps to -0.5.
    adj = _keyword_adjustment(['backup', 'low_priority', 'filler'])
    assert math.isclose(adj, -KEYWORD_ADJ_CLAMP)


def test_unknown_tags_ignored_in_scoring():
    """Unknown tags in the keyword list do not affect the adjustment."""
    adj = _keyword_adjustment(['urgent', 'totally_made_up', 'nonsense'])
    assert math.isclose(adj, 0.2)


def test_override_tag_contributes_zero_to_adjustment():
    """An override token is not an additive weight (it sets mandatory at
    ingestion); if it ever appears in the keyword list it scores 0."""
    adj = _keyword_adjustment(['override'])
    assert adj == 0.0


def test_parse_keywords_normalizes_and_flags_override():
    """parse_keywords normalizes case/spaces, drops unknowns (with warning),
    and reports the override flag separately."""
    kws, mandatory = parse_keywords('High Priority; classification needed, '
                                    'override, garbage_tag', 'T')
    assert mandatory is True
    assert 'high_priority' in kws
    assert 'classification_needed' in kws
    assert 'override' not in kws  # override is a flag, not a scored tag
    assert 'garbage_tag' not in kws


def test_unknown_tag_warns(caplog):
    """parse_keywords logs a warning for each dropped unknown token."""
    with caplog.at_level(logging.WARNING):
        kws, _ = parse_keywords('urgent, not_a_real_keyword', 'OBJ123')
    assert kws == ['urgent']
    assert any('not_a_real_keyword' in r.message for r in caplog.records)


def test_ingestion_parses_keywords_column(tmp_path):
    """load_targets_csv parses a comma-separated keywords column, normalizes
    case/spaces, validates against the vocabulary, and sets mandatory."""
    csv = tmp_path / 'targets.csv'
    csv.write_text(
        "name,ra,dec,program,keywords\n"
        "SN_A,150.0,2.0,P,\"High Priority, urgent\"\n"
        "SN_B,151.0,2.5,P,\"mandatory, near peak\"\n"
        "SN_C,152.0,3.0,P,\"bogus_tag\"\n"
    )
    targets = load_targets_csv(str(csv))
    by_name = {t.name: t for t in targets}

    a = by_name['SN_A']
    assert set(a.keywords) == {'high_priority', 'urgent'}
    assert a.mandatory is False

    b = by_name['SN_B']
    assert b.mandatory is True
    assert 'near_peak' in b.keywords
    assert 'mandatory' not in b.keywords  # override is a flag, not a tag

    c = by_name['SN_C']
    assert c.keywords == []  # unknown token dropped
    assert c.mandatory is False


def test_ingestion_override_keyword_sets_mandatory(tmp_path):
    """A bare 'override' keyword sets Target.mandatory and contributes nothing
    to the additive keyword adjustment."""
    csv = tmp_path / 'targets.csv'
    csv.write_text(
        "name,ra,dec,program,keywords\n"
        "SN_OVR,150.0,2.0,P,override\n"
    )
    t = load_targets_csv(str(csv))[0]
    assert t.mandatory is True
    assert t.keywords == []
    _, bd = compute_composite_score(t)
    assert bd['keyword_adj'] == 0.0
