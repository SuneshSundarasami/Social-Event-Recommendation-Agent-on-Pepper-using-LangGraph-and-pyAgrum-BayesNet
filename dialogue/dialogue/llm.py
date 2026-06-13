"""Evidence parsers — the LLM interface layer (WP2).

Two interchangeable implementations of ``parse(user_text, focus_slot) -> dict``:

  * GroqEvidenceParser  — Groq (free open models, e.g. Llama 3.3) via the
    OpenAI-compatible API. Maps a free-form reply to discrete evidence values
    (and may fill several slots from one answer).
  * ScriptedParser      — deterministic keyword matcher; no network. Used for
    offline tests and as a graceful fallback if the LLM call fails.

Both ALWAYS pass their output through ``schema.whitelist`` so only legal label
values can ever reach the Bayesian network.
"""

from __future__ import annotations

import json
import logging
import os
import re

from dialogue.schema import EVIDENCE_LABELS, whitelist
from dialogue.slots import SLOT_HINTS

log = logging.getLogger("wp2.llm")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _schema_description() -> str:
    return "\n".join(
        f'- {slot}: one of {list(opts)}' for slot, opts in EVIDENCE_LABELS.items()
    )


class GroqEvidenceParser:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str = GROQ_BASE_URL) -> None:
        from openai import OpenAI

        # Accept the standard GROQ_API_KEY or the GROK_KEY name used in .env.
        key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROK_KEY")
        if not key:
            raise RuntimeError("Groq API key not set (GROQ_API_KEY / GROK_KEY in .env).")
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=key, base_url=base_url)

    def parse(self, user_text: str, focus_slot: str | None = None,
              question: str | None = None) -> dict:
        system = (
            "You convert a user's casual reply about social-event preferences into "
            "discrete values. Extract every preference clearly implied by the reply. "
            "Return a JSON object whose keys are a subset of the slots below and whose "
            "values are EXACTLY one of the allowed options; omit a slot if it is "
            "unclear. Do not invent or guess values.\n"
            "Rules:\n"
            "- Do NOT fill a slot from a vague, generic, or merely agreeable reply "
            "('great', 'cool', 'ok', 'sure', 'yeah', 'sounds good', 'nice'). If the "
            "reply does not clearly indicate the value, OMIT that slot.\n"
            "- Use a catch-all option (Setting='Either', ActivityLevel='Moderate') "
            "ONLY when the user explicitly says they have no preference — never as a "
            "fallback for an unclear answer.\n"
            "- Greetings/pleasantries ('good day', 'good morning', 'good evening', "
            "'good night', 'hello', 'hi', 'sir', 'maam') are NOT preferences — never "
            "set TimeOfDay from a greeting.\n"
            "- Respect negation: 'not hungry', 'no sports', 'not into music' mean the "
            "user does NOT want that — do not select it.\n"
            "- Time: tonight / late = Night; this evening / after work = Evening; "
            "today / this afternoon / morning = Day.\n"
            "- Group: girlfriend / partner / a friend / a few people = Small; "
            "alone / by myself = Solo; big crowd / lots of people = Large.\n\n"
            f"Slots and allowed values:\n{_schema_description()}"
        )
        ctx = ""
        if question:
            ctx += f'The user was just asked: "{question}". '
        if focus_slot:
            ctx += f'That question was mainly about "{focus_slot}". '
        user = ctx + f'User reply: "{user_text}". Return only the JSON object.'

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("Groq parse failed (%s); returning no evidence.", exc)
            return {}
        return whitelist(data if isinstance(data, dict) else {})


class GroqQuestionFramer:
    """Generates the next question conversationally with Groq.

    Given what's already known and what's still missing, it asks ONE short, warm
    question (fresh wording each time) about one or two unknown fields — never
    reading the options out like a form. Falls back (returns None) on any error
    so the graph uses the slot's canned question instead.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str = GROQ_BASE_URL, temperature: float = 0.8) -> None:
        from openai import OpenAI

        key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROK_KEY")
        if not key:
            raise RuntimeError("Groq API key not set (GROQ_API_KEY / GROK_KEY in .env).")
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self.client = OpenAI(api_key=key, base_url=base_url)

    def frame(self, filled: dict, missing: list[str], target: str,
              attempt: int = 0, feedback: str | None = None) -> str | None:
        known = ", ".join("%s=%s" % (k, v) for k, v in filled.items()) or "nothing yet"
        target_hint = SLOT_HINTS.get(target, target)
        others = [m for m in missing if m != target]
        system = (
            "You are Pepper, a warm, friendly social robot helping someone choose a "
            "social event. Ask ONE short, open, natural question (max ~20 words) to "
            "learn the TARGET below. Make it experiential and conversational — weave "
            "in what you already know as context — so the person answers freely and "
            "you can INFER the value from what they say. Do NOT read out the allowed "
            "options or ask a rigid 'this or that' question, and avoid yes/no "
            "questions. Vary your phrasing; never ask about things already known. "
            "Return only the question.\n"
            "Example of the right style: 'What sounds most fun to you on a low-key "
            "evening out?'"
        )
        user = "Already known: %s.\nTARGET to learn now: %s — %s.\n" % (
            known, target, target_hint)
        if others:
            user += "Other still-unknown (optional): %s.\n" % ", ".join(others)
        if feedback:
            # Reframe / conflict note from the Evaluator or ConflictResolver. The
            # note itself says how to ask (open rephrase, or either/or for a clash).
            user += "%s\n" % feedback
        user += "Ask your next question."
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=self.temperature,
                max_tokens=60,
            )
            return (resp.choices[0].message.content or "").strip() or None
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("Groq question framing failed (%s); using canned question.", exc)
            return None


class ScriptedParser:
    """Keyword-based parser — deterministic, offline. Also a sane LLM fallback."""

    KEYWORDS: dict[str, dict[str, list[str]]] = {
        "Budget": {
            "Low": ["cheap", "low", "budget", "inexpensive", "affordable", "free"],
            "Med": ["medium", "moderate", "mid", "average", "reasonable"],
            "High": ["expensive", "high", "premium", "luxury", "fancy", "splurge"],
        },
        "GroupSize": {
            "Solo": ["solo", "alone", "myself", "just me", "by myself"],
            "Small": ["small", "few friends", "couple of", "a few", "small group"],
            "Large": ["large", "big group", "lots of", "many", "crowd", "big"],
        },
        "ActivityLevel": {
            "Relaxed": ["relaxed", "chill", "calm", "easy", "laid back", "quiet"],
            "Moderate": ["moderate", "medium", "balanced", "normal"],
            "Active": ["active", "energetic", "sporty", "intense", "lively"],
        },
        "Setting": {
            "Indoor": ["indoor", "inside", "indoors"],
            "Outdoor": ["outdoor", "outside", "outdoors", "open air", "nature"],
            "Either": ["either", "no preference", "doesn't matter", "any", "whatever"],
        },
        "TimeOfDay": {
            "Day": ["day", "daytime", "morning", "afternoon", "noon"],
            "Evening": ["evening", "dinner", "sunset"],
            "Night": ["night", "late", "nightlife", "midnight"],
        },
        "Interest": {
            "Arts": ["arts", "art", "museum", "gallery", "culture", "painting"],
            "Music": ["music", "concert", "gig", "band", "live music"],
            "Food": ["food", "eat", "cuisine", "restaurant", "foodie", "tasting"],
            "Sports": ["sports", "sport", "game", "match", "fitness", "athletic"],
        },
    }

    def parse(self, user_text: str, focus_slot: str | None = None,
              question: str | None = None) -> dict:
        text = " " + user_text.lower() + " "
        found: dict[str, str] = {}
        for slot, values in self.KEYWORDS.items():
            for value, kws in values.items():
                if any(re.search(r"\b" + re.escape(kw) + r"\b", text) for kw in kws):
                    found[slot] = value
                    break
        return whitelist(found)
