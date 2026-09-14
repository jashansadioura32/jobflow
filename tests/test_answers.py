"""Tests for answer resolution.

Focused on rule-ordering collisions — "current CTC in lakhs" must not fall
through to the generic salary rule — and on the guarantee that `resolve`
returns None for an unknown question rather than inventing an answer.

Guessing lives in separate `guess_*` methods that callers opt into, so the
refusal above stays intact for anyone who has a human to escalate to.

Author: Jashan Sadioura
"""

from __future__ import annotations

import pytest

from jobflow.core.answers import AnswerResolver
from jobflow.core.models import (
    Compensation, Defaults, Demographics, Eligibility, Identity,
    Location, Professional, Profile,
)


@pytest.fixture
def resolver(tmp_path):
    resume = tmp_path / "r.pdf"
    resume.write_bytes(b"%PDF-1.4")
    return AnswerResolver(Profile(
        identity=Identity(
            first_name="Alex", last_name="Doe",
            email="test@example.com", phone="9999999999",
            phone_country_code="+91",
            linkedin_url="https://www.linkedin.com/in/example-user/",
            website="https://example.com/",
        ),
        location=Location(city="Springfield", street="Springfield", state="Example State",
                          zipcode="500001", country="India"),
        professional=Professional(
            headline="AI & Data Product Manager", years_of_experience=6,
            has_masters=True, recent_employer="Acme Corp", resume_path=resume,
            summary="Product manager specializing in AI and data products.",
        ),
        compensation=Compensation(current_ctc=1200000, desired_salary=1500000,
                                  notice_period_days=0),
        eligibility=Eligibility(requires_visa_sponsorship=False),
        demographics=Demographics(gender="Male", ethnicity="Asian",
                                  disability_status="No", veteran_status="No"),
        defaults=Defaults(confidence_level=8),
    ))


@pytest.mark.parametrize("question,expected", [
    ("First name", "Alex"),
    ("Last name", "Doe"),
    ("Email address", "test@example.com"),
    ("Mobile phone number", "9999999999"),
    ("Phone country code", "+91"),
    ("What is your current city?", "Springfield"),
    ("State", "Example State"),
    ("Postal code", "500001"),
    ("How many years of experience do you have?", "6"),
    ("LinkedIn profile URL", "https://www.linkedin.com/in/example-user/"),
    ("Portfolio website", "https://example.com/"),
    ("Do you require visa sponsorship?", "No"),
    ("Gender", "Male"),
    ("On a scale of 1-10, how strong is your SQL?", "8"),
])
def test_basic_resolution(resolver, question, expected):
    got = resolver.resolve(question)
    assert got is not None, f"no rule matched {question!r}"
    assert got[0] == expected


# ---- the ordering collisions ----

def test_current_ctc_variants(resolver):
    assert resolver.resolve("What is your current CTC?")[0] == "1200000"
    assert resolver.resolve("Current CTC in lakhs")[0] == "12.00"
    assert resolver.resolve("Current salary per month")[0] == "100000"


def test_desired_salary_variants(resolver):
    assert resolver.resolve("Expected CTC")[0] == "1500000"
    assert resolver.resolve("Expected CTC in lakhs")[0] == "15.00"
    assert resolver.resolve("Desired salary per month")[0] == "125000"


def test_current_beats_generic_salary(resolver):
    """'current' must win over the generic desired-salary rule."""
    assert resolver.resolve("Your present salary")[0] == "1200000"


def test_notice_period_units(resolver):
    assert resolver.resolve("Notice period in days")[0] == "0"
    assert resolver.resolve("Notice period in months")[0] == "0"


def test_monthly_answers_are_integers(resolver):
    """Forms often reject decimals in numeric fields."""
    for q in ("Current salary per month", "Desired salary per month"):
        assert "." not in resolver.resolve(q)[0]


# ---- escalation instead of guessing ----

def test_unknown_question_returns_none(resolver):
    assert resolver.resolve("What is your favourite colour?") is None
    assert resolver.resolve("Describe a conflict with a coworker") is None
    assert resolver.resolve("") is None


def test_empty_profile_value_escalates(tmp_path):
    """A blank profile field must escalate, not submit an empty string."""
    resume = tmp_path / "r.pdf"
    resume.write_bytes(b"%PDF")
    r = AnswerResolver(Profile(
        identity=Identity(first_name="Alex", last_name="Doe",
                          email="a@b.com", phone="9999999999"),
        location=Location(city="Springfield", country="India"),
        professional=Professional(years_of_experience=6, resume_path=resume),
        compensation=Compensation(current_ctc=1, desired_salary=1, notice_period_days=0),
        eligibility=Eligibility(),
        demographics=Demographics(),  # all blank
    ))
    assert r.resolve("Gender") is None
    assert r.resolve("Veteran status") is None


# ---- select/radio ----

def test_resolve_choice_exact_and_fuzzy(resolver):
    assert resolver.resolve_choice("Gender", ["Male", "Female", "Decline"])[0] == "Male"
    got = resolver.resolve_choice(
        "Do you require visa sponsorship?",
        ["Yes, I require sponsorship", "No, I do not require sponsorship"],
    )
    assert got[0].startswith("No")


def test_resolve_choice_no_match_returns_none(resolver):
    assert resolver.resolve_choice("Gender", ["Option A", "Option B"]) is None


# ---------------- eligibility ordering ----------------

def test_authorized_without_sponsorship_is_yes(resolver):
    """Regression: the combined phrasing was answered backwards.

    "Legally authorized ... without sponsorship?" contains both "authorized"
    and "sponsorship". A bare sponsorship rule matched first and answered
    "No" -- the opposite of the truth for someone who needs no sponsorship.
    """
    for q in (
        "Are you legally authorized to work in India without sponsorship?",
        "Are you legally authorised to work without requiring sponsorship?",
        "Are you eligible to work here without visa sponsorship?",
    ):
        answer, rule = resolver.resolve(q)
        assert answer == "Yes", q
        assert rule == "authorized_without_sponsorship"


def test_plain_authorization_and_plain_sponsorship_stay_distinct(resolver):
    assert resolver.resolve("Are you legally authorized to work in India?")[0] == "Yes"
    assert resolver.resolve(
        "Will you now or in the future require sponsorship?")[0] == "No"


def test_employment_eligibility_resolves(resolver):
    answer, _ = resolver.resolve("Please confirm your employment eligibility")
    assert answer == "Yes"


def test_newly_mapped_questions(resolver):
    assert resolver.resolve("How did you hear about this job?")[0] == "LinkedIn"
    assert resolver.resolve("What is your proficiency in English?")[0] == "Professional"
    assert resolver.resolve("How soon can you join us?")[0] == "Immediately"


# ---------------- guessing ----------------

def test_resolve_still_refuses_to_guess(resolver):
    """The default path is unchanged: no confident match means None."""
    assert resolver.resolve("Describe a conflict with a coworker") is None


def test_guess_choice_prefers_an_affirmative_option(resolver):
    answer, rule = resolver.guess_choice(["Select an option", "No", "Yes"])
    assert answer == "Yes"
    assert rule == "guess"


def test_guess_choice_skips_placeholders(resolver):
    answer, _ = resolver.guess_choice(["Select an option", "Maybe"])
    assert answer == "Maybe"


def test_guess_choice_returns_none_without_real_options(resolver):
    assert resolver.guess_choice(["Select an option", "", "-"]) is None


def test_guess_text_uses_experience_for_numeric_questions(resolver):
    answer, rule = resolver.guess_text("How many years of Kubernetes do you have?")
    assert answer == "6"
    assert rule == "guess"


# ---------------- link phrasings ----------------

def test_bare_link_questions_resolve_to_the_website(resolver):
    """Forms often ask for "a link" with no other noun.

    The reference answers these; JobFlow matched only website/portfolio/
    blog/github, so these fell through unanswered.
    """
    for q in ("Personal link", "Link", "Please share a link to your work",
              "Any relevant links?", "Portfolio URL"):
        got = resolver.resolve(q)
        assert got is not None, q
        assert got[0] == "https://example.com/", q


def test_linkedin_link_still_beats_the_generic_link_rule(resolver):
    """Ordering is load-bearing: the specific rule must win."""
    answer, rule = resolver.resolve("LinkedIn profile link")
    assert rule == "linkedin"
    assert answer == "https://www.linkedin.com/in/example-user/"


def test_specific_link_words_still_win(resolver):
    for q in ("GitHub link", "Blog link", "Portfolio website"):
        assert resolver.resolve(q)[1] == "website", q
