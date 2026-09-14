"""Bounded LLM worker: answering form questions the profile cannot map.

Scope limits, by design:
  * Last resort only. Every deterministic rule in `answers.py` is tried
    first; this runs only when none matched and nobody is available to ask.
  * Auto-submit only. Under `--review` a person is watching, so an unmapped
    question escalates to them instead — a human answer beats a generated one.
  * Structured output only. The reply is parsed, type-checked against the
    question kind, and length-capped before it can reach a form.
  * Fails closed. Any error, malformed reply or out-of-range answer falls
    back to the deterministic guess rather than inventing something worse.

The reference implementation this is modelled on returned free text with no
schema and fell back to the candidate's years of experience on failure,
which puts a number into a free-text box. Both are avoided here.

Author: Jashan Sadioura
"""

from __future__ import annotations

import json
import logging
import re

from jobflow.core.models import JobPosting, Profile

log = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 3000   # bounds token spend per question
MAX_ANSWER_CHARS = 300         # forms silently truncate beyond this

SYSTEM_PROMPT = """You answer questions on a job application form, as the \
candidate, in their voice.

Return ONLY a JSON object, no prose, with these keys:
  "answer": the answer text, as the candidate would write it
  "confident": true only if the candidate's details actually support this

Rules for "answer":
  * A question asking for a number (years, count, duration, rating) gets
    digits only: "5", not "5 years".
  * A yes/no question gets exactly "Yes" or "No".
  * A question offering options gets one option, copied exactly.
  * Anything else gets at most two sentences, under 300 characters.
  * Never repeat the question. Never explain yourself. Never invent
    qualifications, employers, or dates the candidate has not stated.

Set "confident" to false when the candidate's details do not support an
honest answer. A false there is safer than a plausible invention: the answer
goes to a real employer under the candidate's name."""


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a model response.

    Tolerates fenced code blocks and leading prose, which models emit even
    when told not to.
    """
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class QuestionAnswerer:
    """Answers unmapped form questions via an OpenAI-compatible endpoint.

    `client` is injected so this is testable without network access and
    works against any compatible provider (OpenAI, Ollama, LM Studio).
    """

    def __init__(
        self,
        profile: Profile,
        client=None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.profile = profile
        self.client = client
        self.model = model

    @property
    def enabled(self) -> bool:
        return self.client is not None

    # -- prompt ----------------------------------------------------------

    def _candidate_block(self) -> str:
        p = self.profile
        pro, comp = p.professional, p.compensation
        return f"""CANDIDATE
Name: {p.identity.full_name}
Headline: {pro.headline}
Years of experience: {pro.years_of_experience}
Holds a masters degree: {"yes" if pro.has_masters else "no"}
Most recent employer: {pro.recent_employer}
Location: {p.location.city}, {p.location.country}
Notice period: {comp.notice_period_days} days
Work authorisation: {p.eligibility.work_authorization}
Summary:
{pro.summary.strip()}"""

    def _build_prompt(
        self, question: str, options: list[str] | None, posting: JobPosting | None
    ) -> str:
        parts = [self._candidate_block()]

        if posting is not None and posting.description.strip():
            desc = posting.description[:MAX_DESCRIPTION_CHARS]
            if len(posting.description) > MAX_DESCRIPTION_CHARS:
                desc += "\n[description truncated]"
            parts.append(f"JOB POSTING\nTitle: {posting.title}\n"
                         f"Company: {posting.company}\nDescription:\n{desc}")

        if options:
            listed = "\n".join(f"  - {o}" for o in options[:25])
            parts.append(f"QUESTION\n{question}\n\n"
                         f"Choose exactly one of these options:\n{listed}")
        else:
            parts.append(f"QUESTION\n{question}")

        return "\n\n".join(parts)

    # -- answering -------------------------------------------------------

    def answer(
        self,
        question: str,
        options: list[str] | None = None,
        posting: JobPosting | None = None,
    ) -> str | None:
        """Answer one question, or None when it cannot be answered honestly.

        None is the safe outcome: the caller falls back to its deterministic
        guess, which is at least predictable.
        """
        if not self.enabled or not question.strip():
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": self._build_prompt(question, options, posting)},
                ],
                temperature=0,          # deterministic for auditability
                max_tokens=300,
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as e:
            log.warning("AI answer unavailable for %r: %s", question[:60], e)
            return None

        payload = _extract_json(raw)
        if payload is None:
            log.warning("unparseable AI answer for %r: %.120s", question[:60], raw)
            return None

        if payload.get("confident") is not True:
            # The model itself says the profile does not support an answer.
            log.info("AI declined to answer %r (not supported by profile).",
                     question[:60])
            return None

        answer = str(payload.get("answer", "")).strip()
        if not answer:
            return None
        if len(answer) > MAX_ANSWER_CHARS:
            log.warning("AI answer for %r exceeded %d chars; discarding.",
                        question[:60], MAX_ANSWER_CHARS)
            return None

        # A choice question must return one of the offered options verbatim,
        # otherwise selecting it would silently fail against the dropdown.
        if options:
            for opt in options:
                if answer.strip().lower() == opt.strip().lower():
                    return opt
            log.warning("AI answer %r is not one of the offered options; "
                        "discarding.", answer[:60])
            return None

        return answer


# ---------------------------------------------------------------------------
#
# Thanks for using JobFlow.
#
# Built because an application carries your name, so the tool should stop and
# ask rather than guess. If it helped your search, pass it on to someone else
# who is looking.
#
# Jashan Sadioura  ·  https://www.linkedin.com/in/sadioura-jashan/
#
# ---------------------------------------------------------------------------
