"""WP3 Bayesian recommender tests."""

from contracts import EVENT_LABELS
from recommender.bayesian_network import BayesianRecommender

rec = BayesianRecommender()


def _ranking(evidence):
    return rec.recommend(evidence)


def test_returns_all_events_as_normalised_distribution():
    ranked = _ranking({})
    assert [r["event"] for r in ranked] and len(ranked) == len(EVENT_LABELS)
    assert set(r["event"] for r in ranked) == set(EVENT_LABELS)
    assert abs(sum(r["prob"] for r in ranked) - 1.0) < 1e-6
    # Sorted descending.
    probs = [r["prob"] for r in ranked]
    assert probs == sorted(probs, reverse=True)


def test_arts_solo_relaxed_favours_museum():
    ranked = _ranking({"Interest": "Arts", "GroupSize": "Solo", "ActivityLevel": "Relaxed"})
    top2 = [r["event"] for r in ranked[:2]]
    assert "Museum" in top2


def test_food_outdoor_favours_food():
    ranked = _ranking({"Interest": "Food", "Setting": "Outdoor", "Budget": "Low"})
    assert ranked[0]["event"] == "Food"


def test_active_night_large_sporty_is_high_energy():
    ranked = _ranking({"Interest": "Sports", "ActivityLevel": "Active",
                       "GroupSize": "Large", "TimeOfDay": "Night"})
    # High-energy entertainment events should dominate.
    top3 = [r["event"] for r in ranked[:3]]
    assert "Sports" in top3 or "Nightlife" in top3


def test_partial_evidence_supported():
    # A single observed slot still yields a valid ranking.
    ranked = _ranking({"Interest": "Music"})
    assert abs(sum(r["prob"] for r in ranked) - 1.0) < 1e-6


def test_explain_mentions_event():
    ev = {"Interest": "Arts", "GroupSize": "Solo"}
    top = _ranking(ev)[0]["event"]
    text = rec.explain(top, ev)
    assert top in text and len(text) > 10


def test_invalid_values_ignored():
    # Illegal slot values must not crash or change the legal-evidence behaviour.
    ranked = _ranking({"Interest": "Food", "Budget": "Bananas", "Bogus": "x"})
    assert ranked[0]["event"] in EVENT_LABELS
