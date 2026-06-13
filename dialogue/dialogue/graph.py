"""The dialogue StateGraph (WP2, PLAN.md §5).

Named agent roles:

    Router           picks the next unknown slot (or finishes).
    QuestionFramer   asks about that slot (LLM-framed, reframed on retry) and
                     captures the reply.
    AnswerParser     maps the reply to candidate evidence (with question context;
                     one answer may fill several slots) and drops catch-all
                     guesses for vague replies.
    ConflictResolver detects when the new answer contradicts an already-decided
                     slot; if so it re-opens that slot and asks the QuestionFramer
                     to pose a clear either/or question to settle it.
    Evaluator        judges whether the reply produced a usable value for the
                     target slot: commit (-> Router), reframe (-> QuestionFramer),
                     or apply a default after a few attempts.

Flow:
    __start__ -> Router -> QuestionFramer -> AnswerParser -> ConflictResolver
    ConflictResolver --(conflict)----> QuestionFramer      (disambiguate)
    ConflictResolver --(no conflict)-> Evaluator
    Evaluator        --(usable/default)-> Router
    Evaluator        --(not usable)----> QuestionFramer    (reframe)
    Router           --(nothing missing)-> Finish -> __end__
"""

from __future__ import annotations

from typing import Optional

# On Python < 3.12 pydantic (used by the LangGraph API/Studio to derive schemas)
# requires typing_extensions.TypedDict, not typing.TypedDict.
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from dialogue.schema import whitelist
from dialogue.slots import DEFAULT_SLOTS, Slot

# Catch-all values an LLM tends to reach for when guessing; trust them for the
# asked slot only if the user explicitly signalled no preference.
_WEAK_VALUES = {"Setting": "Either", "ActivityLevel": "Moderate"}
_NO_PREFERENCE_CUES = (
    "either", "any", "anything", "no pref", "dont care", "don't care",
    "doesnt matter", "doesn't matter", "does not matter", "whatever",
    "up to you", "both", "flexible", "not fussed",
)


def _has_no_preference_cue(text: str) -> bool:
    low = text.lower()
    return any(cue in low for cue in _NO_PREFERENCE_CUES)


class DialogueState(TypedDict):
    evidence: dict          # the slots filled so far (the contract type)
    target: Optional[str]   # the slot currently being resolved
    last_question: str      # the question just asked (context for the parser)
    last_user_text: str
    candidate: dict         # cleaned parser output awaiting commit
    attempts: int           # reframe attempts on the current target
    feedback: Optional[str] # note for the QuestionFramer to reframe / disambiguate
    conflict: bool          # set by ConflictResolver to route a disambiguation
    turn_log: list          # transcript (explainability + eval evidence)
    done: bool


def build_graph(parser, input_provider, framer=None,
                slots: list[Slot] | None = None, max_attempts: int = 2):
    slots = slots or DEFAULT_SLOTS
    by_name = {s.name: s for s in slots}
    order = [s.name for s in slots]

    def missing_of(evidence: dict) -> list[str]:
        return [name for name in order if name not in evidence]

    # -- Router: choose the next unknown slot (resets the reframe state) --------
    def Router(state: DialogueState) -> dict:
        missing = missing_of(state["evidence"])
        if not missing:
            return {"done": True, "target": None}
        return {"done": False, "target": missing[0], "attempts": 0, "feedback": None}

    # -- QuestionFramer: ask about the target, reframing if asked to -----------
    def QuestionFramer(state: DialogueState) -> dict:
        target = state["target"]
        missing = missing_of(state["evidence"])
        question = None
        if framer is not None:
            try:
                question = framer.frame(state["evidence"], missing, target,
                                        attempt=state["attempts"],
                                        feedback=state["feedback"])
            except Exception:
                question = None
        if not question:
            question = by_name[target].question  # offline / fallback wording
        reply = input_provider.ask(question)
        log = state["turn_log"] + [{
            "target": target, "q": question, "a": reply,
            "feedback": state["feedback"],
        }]
        return {"last_question": question, "last_user_text": reply, "turn_log": log}

    # -- AnswerParser: reply -> cleaned candidate evidence ---------------------
    def AnswerParser(state: DialogueState) -> dict:
        candidate = whitelist(parser.parse(state["last_user_text"],
                                           focus_slot=state["target"],
                                           question=state["last_question"]))
        # Distrust a catch-all value for the asked slot unless the user explicitly
        # said "no preference" — otherwise a vague "great" becomes a fake answer.
        target = state["target"]
        if candidate.get(target) == _WEAK_VALUES.get(target) \
                and not _has_no_preference_cue(state["last_user_text"]):
            candidate.pop(target, None)
        return {"candidate": candidate}

    # -- ConflictResolver: spot contradictions with already-decided slots ------
    def ConflictResolver(state: DialogueState) -> dict:
        candidate = state["candidate"]
        evidence = state["evidence"]
        conflicts = [k for k, v in candidate.items()
                     if k in evidence and evidence[k] != v]
        if not conflicts:
            return {"conflict": False}

        slot = conflicts[0]
        old, new = evidence[slot], candidate[slot]
        # Commit any non-conflicting new info from this same reply, then re-open
        # the contradicted slot so the QuestionFramer can settle it.
        other_new = {k: v for k, v in candidate.items()
                     if k not in evidence and k not in conflicts}
        reopened = {**evidence, **other_new}
        reopened.pop(slot, None)
        feedback = (
            'The user earlier indicated %s=%s but now seems to say %s ("%s"). '
            'These conflict — ask a short, clear either/or question to confirm '
            'whether they mean %s or %s.' % (slot, old, new, state["last_user_text"],
                                             old, new)
        )
        return {"evidence": reopened, "candidate": {}, "target": slot,
                "feedback": feedback, "attempts": 0, "conflict": True}

    # -- Evaluator: usable? commit | reframe | default -------------------------
    def Evaluator(state: DialogueState) -> dict:
        target = state["target"]
        candidate = state["candidate"]
        # Commit only NEW slots — never silently overwrite (conflicts handled above).
        new = {k: v for k, v in candidate.items() if k not in state["evidence"]}
        evidence = {**state["evidence"], **new}
        update: dict = {"evidence": evidence, "candidate": {}}

        if target in evidence:                     # usable (directly or multi-fill)
            update["feedback"] = None
            update["attempts"] = 0
            return update

        attempts = state["attempts"] + 1
        if attempts > max_attempts:                # give up: sensible default
            evidence[target] = by_name[target].default
            update["evidence"] = evidence
            update["attempts"] = 0
            update["feedback"] = None
            input_provider.notify("No worries, I'll go with %s for now." % evidence[target])
            return update

        update["attempts"] = attempts
        update["feedback"] = (
            'The previous answer "%s" did not make their %s clear. Ask about %s again '
            'a different, open and natural way — do NOT list the options; you may give '
            'one concrete everyday example to nudge them.'
            % (state["last_user_text"], target, target)
        )
        return update

    def Finish(state: DialogueState) -> dict:
        return {"done": True}

    def route_from_router(state: DialogueState) -> str:
        return "Finish" if state["done"] else "QuestionFramer"

    def route_from_conflict(state: DialogueState) -> str:
        return "QuestionFramer" if state["conflict"] else "Evaluator"

    def route_from_evaluator(state: DialogueState) -> str:
        return "Router" if state["target"] in state["evidence"] else "QuestionFramer"

    g = StateGraph(DialogueState)
    g.add_node("Router", Router)
    g.add_node("QuestionFramer", QuestionFramer)
    g.add_node("AnswerParser", AnswerParser)
    g.add_node("ConflictResolver", ConflictResolver)
    g.add_node("Evaluator", Evaluator)
    g.add_node("Finish", Finish)

    g.add_edge(START, "Router")
    g.add_conditional_edges("Router", route_from_router,
                            {"QuestionFramer": "QuestionFramer", "Finish": "Finish"})
    g.add_edge("QuestionFramer", "AnswerParser")
    g.add_edge("AnswerParser", "ConflictResolver")
    g.add_conditional_edges("ConflictResolver", route_from_conflict,
                            {"QuestionFramer": "QuestionFramer", "Evaluator": "Evaluator"})
    g.add_conditional_edges("Evaluator", route_from_evaluator,
                            {"Router": "Router", "QuestionFramer": "QuestionFramer"})
    g.add_edge("Finish", END)
    return g.compile()


def initial_state() -> DialogueState:
    return {
        "evidence": {},
        "target": None,
        "last_question": "",
        "last_user_text": "",
        "candidate": {},
        "attempts": 0,
        "feedback": None,
        "conflict": False,
        "turn_log": [],
        "done": False,
    }
