"""Bounded LLM worker: job-description fit scoring.

Scope limits, by design:
  * Advisory only. A score never submits or blocks an application on its own;
    it ranks and it can recommend a skip below a threshold the user sets.
  * Judgement only. Anything with a testable correct answer (years required,
    blocked phrases, dedupe) belongs in the deterministic screener.
  * Structured output only. Responses are parsed into FitAssessment and
    rejected if malformed — no free text reaches the decision path.
  * Fails open. An API error yields None, and the run continues on
    deterministic screening alone.

Author: Jashan Sadioura
"""

from __future__ import annotations

import json
import logging
import re

from jobflow.core.models import FitAssessment, JobPosting, Profile

log = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 6000  # bounds token spend per posting

SYSTEM_PROMPT = """You assess how well a candidate matches a job posting.

Return ONLY a JSON object, no prose, with these keys:
  "score": integer 0-100, how well the candidate fits
  "rationale": one or two sentences explaining the score
  "matched_strengths": array of specific candidate strengths the posting asks for
  "gaps": array of specific requirements the candidate does not evidence

Scoring guide:
  80-100  strong match; core requirements are clearly evidenced
  60-79   good match; most requirements met, some gaps
  40-59   partial match; meaningful gaps in core requirements
  0-39    weak match; the role targets a different profile

Judge only what the posting and candidate summary state. Do not infer
seniority from title alone, and do not reward keyword overlap that is not
backed by described experience."""


def _build_user_prompt(profile: Profile, posting: JobPosting) -> str:
    p = profile.professional
    desc = posting.description[:MAX_DESCRIPTION_CHARS]
    if len(posting.description) > MAX_DESCRIPTION_CHARS:
        desc += "\n[description truncated]"

    return f"""CANDIDATE
Headline: {p.headline}
Years of experience: {p.years_of_experience}
Holds a masters degree: {"yes" if p.has_masters else "no"}
Summary:
{p.summary.strip()}

JOB POSTING
Title: {posting.title}
Company: {posting.company}
Location: {posting.location}
Description:
{desc}"""


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a model response.

    Tolerates fenced code blocks and leading prose, which models emit even
    when told not to.
    """
    text = text.strip()
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


class FitScorer:
    """Scores postings via an OpenAI-compatible chat endpoint.

    `client` is injected so the scorer is testable without network access
    and works against any compatible provider (OpenAI, Ollama, LM Studio).
    """

    def __init__(
        self,
        profile: Profile,
        client=None,
        model: str = "gpt-4o-mini",
        min_score: int = 50,
    ) -> None:
        self.profile = profile
        self.client = client
        self.model = model
        self.min_score = min_score

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def score(self, posting: JobPosting) -> FitAssessment | None:
        """Assess one posting. Returns None when unavailable or malformed."""
        if not self.enabled:
            return None
        if not posting.description.strip():
            log.debug("no description for %s; skipping fit scoring", posting.job_id)
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(self.profile, posting)},
                ],
                temperature=0,          # deterministic for auditability
                max_tokens=500,
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as e:
            # Fail open: the deterministic layer still governs the decision.
            log.warning("fit scoring unavailable for %s: %s", posting.job_id, e)
            return None

        payload = _extract_json(raw)
        if payload is None:
            log.warning("unparseable fit response for %s: %.120s", posting.job_id, raw)
            return None

        try:
            score = int(payload.get("score", -1))
        except (TypeError, ValueError):
            log.warning("non-numeric score for %s", posting.job_id)
            return None
        if not 0 <= score <= 100:
            log.warning("score %s out of range for %s", score, posting.job_id)
            return None

        def _str_list(key: str) -> list[str]:
            v = payload.get(key, [])
            return [str(x) for x in v][:10] if isinstance(v, list) else []

        rationale = str(payload.get("rationale", "")).strip()
        if not rationale:
            log.warning("missing rationale for %s", posting.job_id)
            return None

        try:
            return FitAssessment(
                score=score,
                rationale=rationale,
                matched_strengths=_str_list("matched_strengths"),
                gaps=_str_list("gaps"),
            )
        except Exception as e:
            log.warning("invalid fit assessment for %s: %s", posting.job_id, e)
            return None

    def recommends_skip(self, fit: FitAssessment | None) -> bool:
        """True when a score is present and below threshold.

        A missing score never recommends a skip — absence of evidence is not
        evidence of poor fit.
        """
        return fit is not None and fit.score < self.min_score


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
