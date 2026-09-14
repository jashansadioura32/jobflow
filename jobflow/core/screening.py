"""Deterministic screening.

Every check here has a single testable correct answer, so none of it is
delegated to a model. Cheap, certain rejections run before any LLM call:
a posting blocked on work authorization should never cost a token.

Author: Jashan Sadioura
"""

from __future__ import annotations

import re

from jobflow.core.models import (
    Evaluation,
    Finding,
    JobPosting,
    Profile,
    SearchConfig,
    SkipReason,
    Decision,
)

# "5+ years", "5-7 years", "minimum of 5 years", "at least 5 yrs"
_YEARS_PATTERNS = [
    re.compile(r"(\d{1,2})\s*\+\s*(?:years?|yrs?)", re.I),
    re.compile(r"(\d{1,2})\s*(?:-|to|–)\s*\d{1,2}\s*(?:years?|yrs?)", re.I),
    re.compile(r"(?:minimum|min\.?|at least|least)\s*(?:of\s*)?(\d{1,2})\s*(?:years?|yrs?)", re.I),
    re.compile(r"(\d{1,2})\s*(?:years?|yrs?)\s*(?:of\s*)?(?:relevant\s*|professional\s*|industry\s*)?experience", re.I),
]

# Spelled-out numbers appear often enough in senior postings to be worth handling.
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15,
}
_WORD_YEARS = re.compile(
    r"\b(" + "|".join(_WORD_NUMBERS) + r")\s*(?:\+\s*)?(?:years?|yrs?)", re.I
)


def extract_years_required(description: str) -> int | None:
    """Return the lowest credible 'years required' in a description."""
    found = extract_years_evidence(description)
    return found[0][0] if found else None


def extract_years_evidence(description: str) -> list[tuple[int, str]]:
    """Every credible years-figure found, lowest first, with its phrase.

    The lowest match wins deliberately: a posting saying "3+ years
    required, 5+ preferred" is a 3-year role. Taking the max would skip
    roles the candidate is qualified for.

    The matched phrase is returned alongside the number so a skip can be
    audited later. A bare figure in the log ("requires 12y") is impossible
    to distinguish from a misparse of "12 years of excellence" without it.
    """
    if not description:
        return []

    found: list[tuple[int, str]] = []
    for pattern in _YEARS_PATTERNS:
        for m in pattern.finditer(description):
            try:
                found.append((int(m.group(1)), " ".join(m.group(0).split())))
            except (ValueError, IndexError):
                continue

    for m in _WORD_YEARS.finditer(description):
        word = m.group(1).lower()
        if word in _WORD_NUMBERS:
            found.append((_WORD_NUMBERS[word], " ".join(m.group(0).split())))

    # Values above 25 are almost always company age or boilerplate.
    credible = [(y, phrase) for y, phrase in found if 0 < y <= 25]
    return sorted(credible, key=lambda pair: pair[0])


def _contains_phrase(haystack_lower: str, phrase: str) -> bool:
    """Word-boundary-aware phrase match.

    Prevents substring false positives — the reason "secret" is never
    matched bare (it hits "Company Secretary" and "trade secrets policy").
    """
    escaped = re.escape(phrase.lower().strip())
    return re.search(rf"(?<!\w){escaped}(?!\w)", haystack_lower) is not None


class DeterministicScreener:
    """Applies certain rules in ascending order of cost."""

    def __init__(self, profile: Profile, config: SearchConfig) -> None:
        self.profile = profile
        self.config = config

    def screen(
        self, posting: JobPosting, applied_ids: set[str] | None = None
    ) -> Evaluation:
        """Screen one posting. APPLY here means 'no deterministic blocker'."""
        findings: list[Finding] = []
        applied_ids = applied_ids or set()

        # 1. Dedupe — cheapest possible check.
        if posting.job_id in applied_ids:
            return self._skip(
                posting, SkipReason.ALREADY_APPLIED, findings,
                Finding(
                    dimension="deduplication",
                    passed=False,
                    detail=f"job_id {posting.job_id} is already in the application log",
                    severity="blocker",
                ),
            )
        findings.append(
            Finding(dimension="deduplication", passed=True, detail="not previously applied")
        )

        # 2. Company exclusions — exact, case-insensitive.
        company_lower = posting.company.lower().strip()
        for excluded in self.config.exclusions.companies:
            if company_lower == excluded.lower().strip():
                return self._skip(
                    posting, SkipReason.COMPANY_EXCLUDED, findings,
                    Finding(
                        dimension="company_exclusion",
                        passed=False,
                        detail=f"{posting.company!r} is on the exclusion list",
                        severity="blocker",
                    ),
                )
        findings.append(
            Finding(dimension="company_exclusion", passed=True, detail="company not excluded")
        )

        desc_lower = posting.description_lower

        # 3. Work-authorization blockers.
        for phrase in self.config.exclusions.description_blockers:
            if _contains_phrase(desc_lower, phrase):
                return self._skip(
                    posting, SkipReason.BLOCKED_PHRASE, findings,
                    Finding(
                        dimension="work_authorization",
                        passed=False,
                        detail=f"description contains blocking phrase {phrase!r}",
                        severity="blocker",
                    ),
                )
        findings.append(
            Finding(
                dimension="work_authorization",
                passed=True,
                detail="no work-authorization blockers found",
            )
        )

        # 4. Clearance requirements.
        if not self.profile.eligibility.has_security_clearance:
            for phrase in self.config.exclusions.clearance_blockers:
                if _contains_phrase(desc_lower, phrase):
                    return self._skip(
                        posting, SkipReason.CLEARANCE_REQUIRED, findings,
                        Finding(
                            dimension="security_clearance",
                            passed=False,
                            detail=f"description requires clearance ({phrase!r})",
                            severity="blocker",
                        ),
                    )
        findings.append(
            Finding(
                dimension="security_clearance",
                passed=True,
                detail="no clearance requirement detected",
            )
        )

        # 5. Experience gate.
        gate = self.config.experience_gate
        if gate.enabled:
            years = posting.experience_required
            evidence = ""
            if years is None:
                matches = extract_years_evidence(posting.description)
                if matches:
                    years, phrase = matches[0]
                    # Record what was matched, and what else was on the page,
                    # so a wrong skip is diagnosable from the log alone.
                    others = ", ".join(p for _, p in matches[1:4])
                    evidence = f' (matched "{phrase}"'
                    evidence += f'; also saw {others})' if others else ")"

            if years is None:
                findings.append(
                    Finding(
                        dimension="experience",
                        passed=True,
                        detail="no explicit experience requirement found; not treated as a blocker",
                        severity="info",
                    )
                )
            else:
                ceiling = self.profile.experience_ceiling
                if years > ceiling:
                    return self._skip(
                        posting, SkipReason.EXPERIENCE_TOO_HIGH, findings,
                        Finding(
                            dimension="experience",
                            passed=False,
                            detail=(
                                f"posting requires {years}y; ceiling is {ceiling}y "
                                f"({self.profile.professional.years_of_experience}y experience"
                                f"{' + 2y masters buffer' if self.profile.professional.has_masters else ''})"
                                f"{evidence}"
                            ),
                            severity="blocker",
                        ),
                    )
                findings.append(
                    Finding(
                        dimension="experience",
                        passed=True,
                        detail=f"posting requires {years}y, within ceiling of {ceiling}y",
                    )
                )

        return Evaluation(posting=posting, decision=Decision.APPLY, findings=findings)

    @staticmethod
    def _skip(
        posting: JobPosting,
        reason: SkipReason,
        findings: list[Finding],
        blocker: Finding,
    ) -> Evaluation:
        return Evaluation(
            posting=posting,
            decision=Decision.SKIP,
            skip_reason=reason,
            findings=[*findings, blocker],
        )


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
