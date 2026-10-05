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
import random
import re

from dialogue.schema import EVIDENCE_LABELS, whitelist
from dialogue.slots import SLOT_HINTS

log = logging.getLogger("wp2.llm")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.1-8b-instant"


def _schema_description() -> str:
    return "\n".join(
        f'- {slot}: one of {list(opts)}' for slot, opts in EVIDENCE_LABELS.items()
    )


def _create_completion(client, model: str, reasoning_state: list, base_kwargs: dict):
    """``chat.completions.create``, transparently trying Groq's
    ``reasoning_format="hidden"`` extension first.

    Reasoning models (e.g. Qwen3) otherwise return their raw ``<think>...``
    chain-of-thought as the message content, which — combined with a small
    ``max_tokens`` meant for a short final answer — gets truncated mid-thought
    before ever producing one. ``reasoning_format="hidden"`` strips the
    thinking from the response, but the model still spends real budget on it
    internally, so ``max_tokens`` is floored well above what a plain model
    would need. Non-reasoning models reject the parameter outright (400), so
    the result is cached per instance (via the ``reasoning_state`` 1-item
    list: ``[None]`` unknown, ``[True]``/``[False]`` known) to avoid retrying
    on every call once it's known either way.
    """
    from openai import BadRequestError

    if reasoning_state[0] is not False:
        kwargs = dict(base_kwargs, extra_body={"reasoning_format": "hidden"})
        if "max_tokens" in kwargs:
            # The reasoning chain itself is stochastic in length and can run to
            # several thousand tokens before the model reaches a final answer —
            # 800 was observed to still truncate mid-thought on this framer's
            # fuller prompt. 4096 leaves real headroom; still much cheaper than
            # letting a reframe/default cycle repeat because every attempt
            # silently comes back empty.
            kwargs["max_tokens"] = max(kwargs["max_tokens"], 4096)
        try:
            resp = client.chat.completions.create(model=model, **kwargs)
            reasoning_state[0] = True
            return resp
        except BadRequestError as exc:
            if "reasoning_format" not in str(exc):
                raise
            reasoning_state[0] = False
    return client.chat.completions.create(model=model, **base_kwargs)


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
        self._reasoning_state = [None]

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
            resp = _create_completion(self.client, self.model, self._reasoning_state, {
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            })
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("Groq parse failed (%s); returning no evidence.", exc)
            return {}
        return whitelist(data if isinstance(data, dict) else {})


# Randomly nudged into the prompt each call so repeated turns (or repeated
# runs, since Router now also randomizes which slot is asked) don't converge
# on the same template — temperature alone tends to still favor one phrasing
# for a near-identical prompt.
_STYLE_HINTS = (
    "Ask with calm, genuine curiosity.",
    "Ask in a courteous, attentive tone, like a good host.",
    "Ask using a brief, natural everyday comparison.",
    "Ask plainly and efficiently, with quiet warmth.",
    "Ask with understated, dry wit — one subtle touch, nothing more.",
    "Ask in a considerate, professional tone.",
)


class GroqQuestionFramer:
    """Generates the next question conversationally with Groq.

    Given what's already known and what's still missing, it asks ONE short,
    composed question (fresh wording AND a fresh angle each time, nudged by a
    randomly chosen style hint) about one or two unknown fields — never
    reading the options out like a form. Falls back (returns None) on any
    error so the graph uses the slot's canned question instead.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str = GROQ_BASE_URL, temperature: float = 0.7) -> None:
        from openai import OpenAI

        key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROK_KEY")
        if not key:
            raise RuntimeError("Groq API key not set (GROQ_API_KEY / GROK_KEY in .env).")
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self.client = OpenAI(api_key=key, base_url=base_url)
        self._reasoning_state = [None]

    def frame(self, filled: dict, missing: list[str], target: str,
              attempt: int = 0, feedback: str | None = None) -> str | None:
        known = ", ".join("%s=%s" % (k, v) for k, v in filled.items()) or "nothing yet"
        target_hint = SLOT_HINTS.get(target, target)
        others = [m for m in missing if m != target]
        style = random.choice(_STYLE_HINTS)
        system = (
            "You are JARVIS, a composed, courteous social-event assistant running on "
            "a Pepper robot. Ask ONE short, open, natural question (max ~20 words) to "
            "learn the TARGET below. Make it experiential and conversational — weave "
            "in what you already know as context — so the person answers freely and "
            "you can INFER the value from what they say. Do NOT read out the allowed "
            "options or ask a rigid 'this or that' question, and avoid yes/no "
            "questions. Keep the tone professional and understated — at most a subtle, "
            "dry touch of wit, used sparingly, never slang or forced jokes. Never reuse "
            "the same wording, structure, or example twice; genuinely vary your "
            "phrasing and angle every time. Return only the question.\n"
            "Examples of the right style (for different targets — don't copy these "
            "verbatim, just match the tone):\n"
            "- 'Would you prefer something more refined, or are you happy to keep it "
            "simple?'\n"
            "- 'Will it just be you this evening, or is anyone joining you?'\n"
            "- 'What kind of pace suits you tonight — relaxed, or something livelier?'\n"
            "- 'Any particular interests I should factor in?'"
        )
        user = "Already known: %s.\nTARGET to learn now: %s — %s.\n" % (
            known, target, target_hint)
        if others:
            user += "Other still-unknown (optional): %s.\n" % ", ".join(others)
        if feedback:
            # Reframe / conflict note from the Evaluator or ConflictResolver. The
            # note itself says how to ask (open rephrase, or either/or for a clash).
            user += "%s\n" % feedback
        user += "Style for this question: %s\n" % style
        user += "Ask your next question."
        try:
            resp = _create_completion(self.client, self.model, self._reasoning_state, {
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": self.temperature,
                "max_tokens": 60,
            })
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
