"""Bayesian event recommendation engine (WP3).

A small, explainable three-layer Bayesian network built with pyAgrum:

    Layer 1 (evidence, 6 nodes)  Budget GroupSize ActivityLevel Setting TimeOfDay Interest
                                    |        |          |          |        |        |
    Layer 2 (latent, 3 nodes)        SocialContext   EnergyProfile      VenueType
                                          \\              |               /
    Layer 3 (output)                            EventRec  (8 events)

Parent sets (each evidence pair drives one interpretable latent factor):
    VenueType      <- Interest, Setting
    SocialContext  <- GroupSize, Budget
    EnergyProfile  <- ActivityLevel, TimeOfDay
    EventRec       <- VenueType, SocialContext, EnergyProfile

The CPTs are generated from soft rules grounded in the proposal's Event Catalogue
(§5) rather than hand-written tables, which keeps them consistent and explainable.
Variable elimination over EventRec returns the ranked recommendations; partial
evidence is supported (unobserved nodes fall back to their priors).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import pyAgrum as gum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts import EVENT_LABELS, EVIDENCE_LABELS, Evidence, Result, validate_evidence  # noqa: E402

# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------

VENUE = ("Cultural", "Entertainment", "Nature", "Culinary")
SOCIAL = ("Intimate", "Social", "Mass")
ENERGY = ("Low", "Medium", "High")
EVENTS = tuple(EVENT_LABELS)

# ---------------------------------------------------------------------------
# Soft rules for the latent CPTs (base distribution * modifier, then normalised)
# ---------------------------------------------------------------------------

# VenueType <- Interest (base), Setting (modifier)
_INTEREST_VENUE = {
    "Arts":   {"Cultural": 0.70, "Entertainment": 0.15, "Nature": 0.05, "Culinary": 0.10},
    "Music":  {"Cultural": 0.15, "Entertainment": 0.70, "Nature": 0.05, "Culinary": 0.10},
    "Food":   {"Cultural": 0.10, "Entertainment": 0.10, "Nature": 0.10, "Culinary": 0.70},
    "Sports": {"Cultural": 0.05, "Entertainment": 0.55, "Nature": 0.35, "Culinary": 0.05},
}
_SETTING_VENUE = {
    "Indoor":  {"Cultural": 1.2, "Entertainment": 1.1, "Nature": 0.4, "Culinary": 1.1},
    "Outdoor": {"Cultural": 0.7, "Entertainment": 0.9, "Nature": 2.5, "Culinary": 0.9},
    "Either":  {"Cultural": 1.0, "Entertainment": 1.0, "Nature": 1.0, "Culinary": 1.0},
}

# SocialContext <- GroupSize (base), Budget (modifier)
_GROUP_SOCIAL = {
    "Solo":  {"Intimate": 0.75, "Social": 0.20, "Mass": 0.05},
    "Small": {"Intimate": 0.40, "Social": 0.50, "Mass": 0.10},
    "Large": {"Intimate": 0.05, "Social": 0.45, "Mass": 0.50},
}
_BUDGET_SOCIAL = {
    "Low":  {"Intimate": 1.1, "Social": 1.0, "Mass": 0.9},
    "Med":  {"Intimate": 1.0, "Social": 1.0, "Mass": 1.0},
    "High": {"Intimate": 1.0, "Social": 1.05, "Mass": 1.1},
}

# EnergyProfile <- ActivityLevel (base), TimeOfDay (modifier)
_ACTIVITY_ENERGY = {
    "Relaxed":  {"Low": 0.70, "Medium": 0.25, "High": 0.05},
    "Moderate": {"Low": 0.20, "Medium": 0.60, "High": 0.20},
    "Active":   {"Low": 0.05, "Medium": 0.30, "High": 0.65},
}
_TIME_ENERGY = {
    "Day":     {"Low": 1.1, "Medium": 1.0, "High": 0.9},
    "Evening": {"Low": 1.0, "Medium": 1.1, "High": 1.0},
    "Night":   {"Low": 0.8, "Medium": 1.0, "High": 1.3},
}

# Event profiles (which latent values each event prefers) — from the Catalogue.
_EVENT_PROFILE = {
    "Museum":     {"venue": ("Cultural",),               "social": ("Intimate", "Social"), "energy": ("Low",)},
    "Concert":    {"venue": ("Entertainment",),          "social": ("Social", "Mass"),     "energy": ("Medium",)},
    "Sports":     {"venue": ("Entertainment",),          "social": ("Mass",),              "energy": ("High",)},
    "Food":       {"venue": ("Culinary",),               "social": ("Social", "Mass"),     "energy": ("Low", "Medium")},
    "Outdoor":    {"venue": ("Nature",),                 "social": ("Intimate",),          "energy": ("High",)},
    "Nightlife":  {"venue": ("Entertainment",),          "social": ("Social",),            "energy": ("High",)},
    "Workshop":   {"venue": ("Cultural",),               "social": ("Intimate",),          "energy": ("Medium",)},
    "Networking": {"venue": ("Cultural", "Entertainment"), "social": ("Social",),          "energy": ("Medium",)},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_var(name: str, labels) -> gum.LabelizedVariable:
    var = gum.LabelizedVariable(name, name, len(labels))
    for i, label in enumerate(labels):
        var.changeLabel(i, label)
    return var


def _normalised(scores: Dict[str, float], order) -> List[float]:
    total = float(sum(scores[k] for k in order)) or 1.0
    return [scores[k] / total for k in order]


def _combine(base: Dict[str, float], modifier: Dict[str, float], order) -> List[float]:
    return _normalised({k: base[k] * modifier[k] for k in order}, order)


def _event_distribution(v: str, s: str, e: str) -> List[float]:
    """Distribution over the 8 events for a latent combination (v, s, e)."""
    scores = {}
    for event in EVENTS:
        prof = _EVENT_PROFILE[event]
        vm = 1.0 if v in prof["venue"] else 0.20
        sm = 1.0 if s in prof["social"] else 0.25
        em = 1.0 if e in prof["energy"] else 0.25
        scores[event] = vm * sm * em + 1e-3  # small floor avoids hard zeros
    return _normalised(scores, EVENTS)


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def build_network() -> gum.BayesNet:
    bn = gum.BayesNet("EventRecommendation")

    # Layer 1 — evidence nodes (uniform priors so missing evidence is neutral).
    for slot, labels in EVIDENCE_LABELS.items():
        bn.add(_make_var(slot, labels))
        bn.cpt(slot).fillWith([1.0 / len(labels)] * len(labels))

    # Layer 2 — latent factors.
    bn.add(_make_var("VenueType", VENUE))
    bn.add(_make_var("SocialContext", SOCIAL))
    bn.add(_make_var("EnergyProfile", ENERGY))

    # Layer 3 — output.
    bn.add(_make_var("EventRec", EVENTS))

    # Arcs.
    bn.addArc("Interest", "VenueType");      bn.addArc("Setting", "VenueType")
    bn.addArc("GroupSize", "SocialContext"); bn.addArc("Budget", "SocialContext")
    bn.addArc("ActivityLevel", "EnergyProfile"); bn.addArc("TimeOfDay", "EnergyProfile")
    for latent in ("VenueType", "SocialContext", "EnergyProfile"):
        bn.addArc(latent, "EventRec")

    # Latent CPTs.
    for interest in EVIDENCE_LABELS["Interest"]:
        for setting in EVIDENCE_LABELS["Setting"]:
            bn.cpt("VenueType")[{"Interest": interest, "Setting": setting}] = \
                _combine(_INTEREST_VENUE[interest], _SETTING_VENUE[setting], VENUE)

    for group in EVIDENCE_LABELS["GroupSize"]:
        for budget in EVIDENCE_LABELS["Budget"]:
            bn.cpt("SocialContext")[{"GroupSize": group, "Budget": budget}] = \
                _combine(_GROUP_SOCIAL[group], _BUDGET_SOCIAL[budget], SOCIAL)

    for activity in EVIDENCE_LABELS["ActivityLevel"]:
        for tod in EVIDENCE_LABELS["TimeOfDay"]:
            bn.cpt("EnergyProfile")[{"ActivityLevel": activity, "TimeOfDay": tod}] = \
                _combine(_ACTIVITY_ENERGY[activity], _TIME_ENERGY[tod], ENERGY)

    # EventRec CPT (over the latent combinations).
    for v in VENUE:
        for s in SOCIAL:
            for e in ENERGY:
                bn.cpt("EventRec")[{"VenueType": v, "SocialContext": s, "EnergyProfile": e}] = \
                    _event_distribution(v, s, e)

    return bn


# ---------------------------------------------------------------------------
# Recommender (implements the contracts.Recommender protocol)
# ---------------------------------------------------------------------------

_VENUE_DESC = {"Cultural": "a cultural", "Entertainment": "an entertainment",
               "Nature": "an outdoor", "Culinary": "a culinary"}
_SOCIAL_DESC = {"Intimate": "intimate", "Social": "social", "Mass": "lively crowd"}
_ENERGY_DESC = {"Low": "low-key", "Medium": "moderate", "High": "high-energy"}


class BayesianRecommender:
    def __init__(self) -> None:
        self.bn = build_network()

    def _inference(self, evidence: Evidence):
        ie = gum.LazyPropagation(self.bn)
        clean = validate_evidence(evidence)
        if clean:
            ie.setEvidence(dict(clean))
        ie.makeInference()
        return ie

    def recommend(self, evidence: Evidence) -> Result:
        ie = self._inference(evidence)
        post = ie.posterior("EventRec")
        results = [{"event": event, "prob": float(post[i])}
                   for i, event in enumerate(EVENTS)]
        results.sort(key=lambda r: r["prob"], reverse=True)
        return results

    def explain(self, event: str, evidence: Evidence) -> str:
        ie = self._inference(evidence)

        def _map(node, order):
            post = ie.posterior(node)
            return order[max(range(len(order)), key=lambda i: post[i])]

        venue = _map("VenueType", VENUE)
        social = _map("SocialContext", SOCIAL)
        energy = _map("EnergyProfile", ENERGY)
        return ("I suggest %s because your preferences point to %s venue with a %s "
                "vibe and a %s feel." % (event, _VENUE_DESC[venue],
                                         _SOCIAL_DESC[social], _ENERGY_DESC[energy]))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rec = BayesianRecommender()
    samples = [
        {"Budget": "Low", "GroupSize": "Small", "ActivityLevel": "Relaxed",
         "Setting": "Outdoor", "TimeOfDay": "Day", "Interest": "Food"},
        {"Interest": "Arts", "GroupSize": "Solo", "ActivityLevel": "Relaxed"},
        {"Interest": "Music", "GroupSize": "Large", "TimeOfDay": "Night",
         "ActivityLevel": "Active"},
        {},  # no evidence -> prior ranking
    ]
    for ev in samples:
        print("\nEvidence:", ev or "(none)")
        ranked = rec.recommend(ev)
        for r in ranked[:3]:
            print("   %-11s %5.1f%%" % (r["event"], r["prob"] * 100))
        top = ranked[0]["event"]
        print("   ->", rec.explain(top, ev))
