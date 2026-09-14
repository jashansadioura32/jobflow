"""Form question → answer resolution.

Deterministic and ordered. Specific patterns are tested before general ones,
because "current CTC in lakhs" must not be caught by a bare "salary" rule.

A question with no confident match returns None rather than a guess: an
unanswered field escalates to a human, while a wrong answer reaches an
employer. Those failure modes are not symmetric.

Author: Jashan Sadioura
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from jobflow.core.models import Profile


@dataclass(frozen=True)
class AnswerRule:
    name: str
    matches: Callable[[str], bool]
    answer: Callable[[Profile], str]


def _has(*needles: str) -> Callable[[str], bool]:
    return lambda q: all(n in q for n in needles)


def _any_of(*needles: str) -> Callable[[str], bool]:
    return lambda q: any(n in q for n in needles)


def _all_and_any(
    required: tuple[str, ...], any_of: tuple[str, ...]
) -> Callable[[str], bool]:
    return lambda q: all(r in q for r in required) and any(a in q for a in any_of)


_SALARY_WORDS = ("salary", "compensation", "ctc", "remuneration", "pay expectation")
_CURRENT_WORDS = ("current", "present", "existing", "latest")


def build_rules() -> list[AnswerRule]:
    """Rules in priority order — first match wins."""
    return [
        # ---- identity ----
        AnswerRule("first_name", _all_and_any(("first",), ("name",)),
                   lambda p: p.identity.first_name),
        AnswerRule("last_name", _all_and_any(("last",), ("name",)),
                   lambda p: p.identity.last_name),
        AnswerRule("full_name",
                   lambda q: _has("name")(q) and _any_of("full", "legal")(q),
                   lambda p: p.identity.full_name),
        AnswerRule("email", _any_of("email", "e-mail"),
                   lambda p: str(p.identity.email)),
        AnswerRule("phone_country_code", _has("country", "code"),
                   lambda p: p.identity.phone_country_code),
        AnswerRule("phone", _any_of("phone", "mobile", "contact number"),
                   lambda p: p.identity.phone),

        # ---- location ----
        AnswerRule("city", _any_of("city", "current location", "which city"),
                   lambda p: p.location.city),
        AnswerRule("state", _any_of("state", "province"),
                   lambda p: p.location.state),
        AnswerRule("zipcode", _any_of("zip", "postal", "pin code"),
                   lambda p: p.location.zipcode),
        AnswerRule("country", lambda q: "country" in q and "code" not in q,
                   lambda p: p.location.country),
        AnswerRule("street", _any_of("street", "address line", "address"),
                   lambda p: p.location.street),

        # ---- notice period (before salary: "notice period in months") ----
        AnswerRule("notice_months", _all_and_any(("notice",), ("month",)),
                   lambda p: p.compensation.notice_period_months),
        AnswerRule("notice_weeks", _all_and_any(("notice",), ("week",)),
                   lambda p: p.compensation.notice_period_weeks),
        AnswerRule("notice_days", _has("notice"),
                   lambda p: str(p.compensation.notice_period_days)),

        # ---- compensation: unit-specific first ----
        AnswerRule(
            "current_salary_lakhs",
            lambda q: any(w in q for w in _SALARY_WORDS)
            and any(w in q for w in _CURRENT_WORDS) and "lakh" in q,
            lambda p: p.compensation.current_lakhs,
        ),
        AnswerRule(
            "current_salary_monthly",
            lambda q: any(w in q for w in _SALARY_WORDS)
            and any(w in q for w in _CURRENT_WORDS)
            and any(w in q for w in ("month", "per month", "monthly")),
            lambda p: p.compensation.current_monthly,
        ),
        AnswerRule(
            "current_salary",
            lambda q: any(w in q for w in _SALARY_WORDS)
            and any(w in q for w in _CURRENT_WORDS),
            lambda p: str(p.compensation.current_ctc),
        ),
        AnswerRule(
            "desired_salary_lakhs",
            lambda q: any(w in q for w in _SALARY_WORDS) and "lakh" in q,
            lambda p: p.compensation.desired_lakhs,
        ),
        AnswerRule(
            "desired_salary_monthly",
            lambda q: any(w in q for w in _SALARY_WORDS)
            and any(w in q for w in ("month", "per month", "monthly")),
            lambda p: p.compensation.desired_monthly,
        ),
        AnswerRule(
            "desired_salary",
            lambda q: any(w in q for w in _SALARY_WORDS),
            lambda p: str(p.compensation.desired_salary),
        ),

        # ---- professional ----
        AnswerRule(
            "years_experience",
            lambda q: _any_of("years of experience", "years experience",
                              "how many years", "total experience")(q),
            lambda p: str(p.professional.years_of_experience),
        ),
        AnswerRule("headline", _has("headline"),
                   lambda p: p.professional.headline),
        AnswerRule("summary", _any_of("summary", "about you", "tell us about"),
                   lambda p: p.professional.summary.strip()),
        AnswerRule("employer", _any_of("current employer", "recent employer",
                                       "current company"),
                   lambda p: p.professional.recent_employer),
        AnswerRule("linkedin", _has("linkedin"),
                   lambda p: p.identity.linkedin_url),
        AnswerRule("website", _any_of("website", "portfolio", "blog", "github"),
                   lambda p: p.identity.website),
        # A bare "link" is the catch-all phrasing forms use ("Personal link",
        # "Any relevant links?"). It sits *after* the linkedin rule above so
        # "LinkedIn profile link" still resolves to the LinkedIn URL, and
        # after website so the more specific words win when both appear.
        # "Link to resume" is a file-upload prompt, not a portfolio question;
        # answering it with a website URL is confidently wrong, so it is
        # excluded and left to escalate.
        AnswerRule("link",
                   lambda q: _any_of("link", "url")(q) and "resume" not in q
                   and "cv" not in q,
                   lambda p: p.identity.website),

        # ---- eligibility ----
        # Order matters and is load-bearing. LinkedIn's common phrasing is
        # "Are you legally authorized to work in X without sponsorship?" --
        # a single question containing both "authorized" and "sponsorship".
        # A bare sponsorship rule matched it first and answered "No", the
        # exact opposite of the truth. Authorization is therefore tested
        # before sponsorship, and the combined phrasing before both.
        AnswerRule(
            "authorized_without_sponsorship",
            lambda q: _any_of("authorized", "authorised", "eligible")(q)
            and _any_of("sponsor", "sponsorship")(q),
            lambda p: "No" if p.eligibility.requires_visa_sponsorship else "Yes",
        ),
        AnswerRule(
            "authorization",
            _any_of("authorized to work", "authorised to work", "work authorization",
                    "legally authorized", "legally authorised", "right to work",
                    "employment eligibility"),
            lambda p: "Yes",
        ),
        AnswerRule(
            "visa",
            _any_of("visa", "sponsorship", "require sponsorship"),
            lambda p: "Yes" if p.eligibility.requires_visa_sponsorship else "No",
        ),
        AnswerRule("citizenship", _any_of("citizenship", "citizen status"),
                   lambda p: p.eligibility.work_authorization),
        AnswerRule("clearance", _has("clearance"),
                   lambda p: "Yes" if p.eligibility.has_security_clearance else "No"),

        # ---- standard screening questions ----
        # Answered from explicit config, never guessed: a wrong answer to a
        # criminal-record or non-compete question reaches an employer under
        # your name. Ordered most-specific first, as everywhere else here.
        AnswerRule(
            "criminal_record",
            _any_of("criminal", "convicted", "felony", "misdemeanor",
                    "criminal record", "criminal history"),
            lambda p: p.screening.criminal_record,
        ),
        AnswerRule(
            "background_check",
            lambda q: _any_of("background check", "background screening",
                              "background verification")(q),
            lambda p: p.screening.background_check_consent,
        ),
        AnswerRule("drug_test", _any_of("drug test", "drug screen"),
                   lambda p: p.screening.drug_test_consent),

        AnswerRule("night_shift", _any_of("night shift", "night shifts",
                                          "graveyard shift", "rotational shift"),
                   lambda p: p.screening.willing_night_shift),
        AnswerRule("weekends", _any_of("weekend", "weekends"),
                   lambda p: p.screening.willing_weekends),
        AnswerRule("relocate", _any_of("relocate", "relocation"),
                   lambda p: p.screening.willing_relocate),
        # "travel" alone is too broad: "Travel allowance expected" wants a
        # number, and answering "Yes" to it puts a word in a money field.
        # Willingness phrasings alone are not enough either -- "What travel
        # allowance do you expect?" contains "do you" -- so anything asking
        # for an amount is excluded outright.
        AnswerRule(
            "travel",
            lambda q: _any_of("travel", "travelling", "traveling")(q)
            and not _any_of("allowance", "reimburse", "amount", "expense",
                            "how much", "budget")(q)
            and _any_of("willing", "able", "comfortable", "open to",
                        "prepared", "can you", "do you")(q),
            lambda p: p.screening.willing_travel,
        ),
        AnswerRule("overtime", _has("overtime"),
                   lambda p: p.screening.willing_overtime),

        # Before the generic "employed" rule: "applied ... before" is a
        # different question from "are you employed".
        AnswerRule(
            "applied_before",
            lambda q: _any_of("applied", "application")(q)
            and _any_of("before", "previously", "past", "last 6 months",
                        "six months", "in the last")(q),
            lambda p: p.screening.applied_before_recently,
        ),
        AnswerRule(
            "previously_employed",
            lambda q: _any_of("previously employed", "worked for us",
                              "worked here", "former employee",
                              "ever been employed")(q),
            lambda p: p.screening.previously_employed_here,
        ),
        AnswerRule(
            "related_to_employee",
            _any_of("relative", "related to", "family member", "friend who works"),
            lambda p: p.screening.related_to_employee,
        ),
        AnswerRule("non_compete", _any_of("non-compete", "non compete",
                                          "noncompete", "restrictive covenant"),
                   lambda p: p.screening.non_compete_agreement),

        AnswerRule("currently_employed", _any_of("currently employed",
                                                 "presently employed"),
                   lambda p: p.screening.currently_employed),
        AnswerRule("driving_licence", _any_of("driving licence",
                                              "driving license", "driver's licence",
                                              "driver's license", "valid licence",
                                              "valid license"),
                   lambda p: p.screening.has_driving_licence),
        AnswerRule("own_equipment", _any_of("own laptop", "own computer",
                                            "own device", "own equipment",
                                            "reliable internet"),
                   lambda p: p.screening.has_own_equipment),
        AnswerRule("start_immediately", _any_of("start immediately",
                                                "join immediately",
                                                "immediate joiner"),
                   lambda p: p.screening.can_start_immediately),

        # ---- EEO ----
        AnswerRule("gender", _any_of("gender", "sex"),
                   lambda p: p.demographics.gender),
        AnswerRule("ethnicity", _any_of("ethnicity", "race", "hispanic"),
                   lambda p: p.demographics.ethnicity),
        AnswerRule("disability", _has("disab"),
                   lambda p: p.demographics.disability_status),
        AnswerRule("veteran", _any_of("veteran", "military"),
                   lambda p: p.demographics.veteran_status),

        # ---- misc phrasings the reference handles ----
        AnswerRule("proficiency", _has("proficiency"), lambda p: "Professional"),
        AnswerRule(
            "how_heard",
            lambda q: _any_of("hear about", "come across", "how did you find")(q),
            lambda p: "LinkedIn",
        ),
        AnswerRule("cover_letter", _has("cover letter"),
                   lambda p: p.professional.summary.strip()),
        AnswerRule("start_date", _any_of("when can you start", "how soon can you join",
                                         "available to start", "earliest start"),
                   lambda p: "Immediately"
                   if p.compensation.notice_period_days == 0
                   else f"{p.compensation.notice_period_days} days"),

        # ---- self-rating ----
        AnswerRule(
            "confidence",
            lambda q: bool(re.search(r"scale of 1\s*[-–to]+\s*10", q)),
            lambda p: str(p.defaults.confidence_level),
        ),
    ]


# Options a yes/no-shaped guess should prefer, in order. Affirmative first:
# the overwhelmingly common unmapped required question on a job application
# is a willingness or agreement check.
_GUESS_PREFERENCES = ("yes", "agree", "i do", "i have", "i am", "professional")


class AnswerResolver:
    """Resolves form questions to profile-backed answers."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self._rules = build_rules()

    def resolve(self, question: str) -> tuple[str, str] | None:
        """Return (answer, rule_name), or None if no rule matches confidently."""
        q = question.lower().strip()
        if not q:
            return None
        for rule in self._rules:
            try:
                if rule.matches(q):
                    value = rule.answer(self.profile)
                    # An empty profile value means "not provided" — escalate
                    # rather than submit a blank.
                    if value and value.strip():
                        return value.strip(), rule.name
                    return None
            except Exception:
                continue
        return None

    def resolve_choice(
        self, question: str, options: list[str]
    ) -> tuple[str, str] | None:
        """Resolve a select/radio question to one of `options`."""
        resolved = self.resolve(question)
        if not resolved:
            return None
        answer, rule = resolved
        answer_low = answer.lower()

        for opt in options:
            if opt.lower().strip() == answer_low:
                return opt, rule
        for opt in options:
            o = opt.lower()
            if answer_low in o or o in answer_low:
                return opt, rule

        # Yes/No questions where the option text is phrased differently.
        if answer_low in {"yes", "no"}:
            for opt in options:
                if opt.lower().startswith(answer_low):
                    return opt, rule
        return None

    # -- guessing -------------------------------------------------------
    #
    # Everything above refuses to answer without a confident match. These
    # two deliberately do not: with no human to escalate to, an unanswered
    # required question blocks the whole application. A guess is recorded
    # as a guess so it can be audited afterwards.

    def guess_choice(self, options: list[str]) -> tuple[str, str] | None:
        """Pick the least-bad option for an unmapped choice question."""
        real = [o for o in options
                if o.strip() and o.strip().lower() not in {
                    "select an option", "choose", "-", "--"}]
        if not real:
            return None
        for want in _GUESS_PREFERENCES:
            for opt in real:
                if opt.strip().lower().startswith(want):
                    return opt, "guess"
        return real[0], "guess"

    def guess_text(self, question: str) -> tuple[str, str]:
        """Produce a plausible free-text answer for an unmapped question.

        A numeric-sounding question gets years of experience, which is what
        the reference falls back to; anything else gets a neutral string.
        """
        q = question.lower()
        if _any_of("how many", "number of", "years", "how long")(q):
            return str(self.profile.professional.years_of_experience), "guess"
        return "Yes", "guess"


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
