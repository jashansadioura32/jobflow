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


# ---------------- standard screening questions ----------------

@pytest.mark.parametrize("question,expected,rule", [
    ("Do you have any criminal records?", "No", "criminal_record"),
    ("Have you ever been convicted of a felony?", "No", "criminal_record"),
    ("Are you comfortable with a background check?", "Yes", "background_check"),
    ("Will you consent to a drug test?", "Yes", "drug_test"),
    ("Are you willing to work night shifts?", "Yes", "night_shift"),
    ("Can you work weekends?", "Yes", "weekends"),
    ("Are you willing to relocate?", "Yes", "relocate"),
    ("Are you willing to travel?", "Yes", "travel"),
    ("Willing to work overtime?", "Yes", "overtime"),
    ("Have you applied to this organization in the last 6 months?",
     "No", "applied_before"),
    ("Have you previously employed with us?", "No", "previously_employed"),
    ("Do you have any relatives working here?", "No", "related_to_employee"),
    ("Are you bound by a non-compete agreement?", "No", "non_compete"),
    ("Are you currently employed?", "Yes", "currently_employed"),
    ("Do you have a valid driving licence?", "Yes", "driving_licence"),
    ("Do you have your own laptop?", "Yes", "own_equipment"),
    ("Can you join immediately?", "Yes", "start_immediately"),
])
def test_screening_questions_resolve(resolver, question, expected, rule):
    got = resolver.resolve(question)
    assert got is not None, f"no rule matched {question!r}"
    assert got == (expected, rule)


def test_screening_answers_come_from_config_not_a_guess(tmp_path):
    """Changing the config must change the answer -- these are facts."""
    from jobflow.core.models import ScreeningAnswers
    resume = tmp_path / "r.pdf"
    resume.write_bytes(b"%PDF-1.4")
    p = Profile(
        identity=Identity(first_name="A", last_name="B", email="a@b.com",
                          phone="9999999999"),
        location=Location(city="X", country="Y"),
        professional=Professional(years_of_experience=1, resume_path=resume),
        compensation=Compensation(current_ctc=1, desired_salary=1,
                                  notice_period_days=0),
        eligibility=Eligibility(),
        screening=ScreeningAnswers(willing_relocate="No", criminal_record="Yes"),
    )
    r = AnswerResolver(p)
    assert r.resolve("Are you willing to relocate?")[0] == "No"
    assert r.resolve("Do you have any criminal records?")[0] == "Yes"


def test_blank_screening_answer_escalates(tmp_path):
    """An empty value means "ask me", not "answer anyway"."""
    from jobflow.core.models import ScreeningAnswers
    resume = tmp_path / "r.pdf"
    resume.write_bytes(b"%PDF-1.4")
    p = Profile(
        identity=Identity(first_name="A", last_name="B", email="a@b.com",
                          phone="9999999999"),
        location=Location(city="X", country="Y"),
        professional=Professional(years_of_experience=1, resume_path=resume),
        compensation=Compensation(current_ctc=1, desired_salary=1,
                                  notice_period_days=0),
        eligibility=Eligibility(),
        screening=ScreeningAnswers(criminal_record=""),
    )
    assert AnswerResolver(p).resolve("Do you have any criminal records?") is None


def test_screening_values_are_validated_at_load_time(tmp_path):
    """A typo must fail loudly at startup, not silently mid-application."""
    import pydantic
    from jobflow.core.models import ScreeningAnswers

    for bad in ("Ye", "maybe", "y", "true"):
        with pytest.raises(pydantic.ValidationError):
            ScreeningAnswers(criminal_record=bad)

    # Case and spacing are forgiven, since they are harmless.
    assert ScreeningAnswers(criminal_record=" no ").criminal_record == "No"


def test_applied_before_beats_currently_employed(resolver):
    """Ordering: "applied before" and "currently employed" both mention work."""
    assert resolver.resolve(
        "Have you applied to this company before?")[1] == "applied_before"
    assert resolver.resolve("Are you currently employed?")[1] == "currently_employed"


def test_travel_allowance_is_never_answered_yes(resolver):
    """Regression: a money question must not get a yes/no answer.

    "What travel allowance do you expect?" contains both "travel" and
    "do you", so a willingness-phrasing check alone let it through and put
    the word "Yes" into a field expecting a number.
    """
    for q in ("Travel allowance expected",
              "Travel reimbursement amount",
              "What travel allowance do you expect?",
              "How much travel expense do you claim?",
              "Travel budget required"):
        assert resolver.resolve(q) is None, q


def test_willingness_to_travel_still_resolves(resolver):
    for q in ("Are you willing to travel?",
              "Can you travel for work?",
              "Are you comfortable travelling 50%?",
              "Open to travel?"):
        assert resolver.resolve(q) == ("Yes", "travel"), q


# ---------------- custom answers from config ----------------

def _profile_with_custom(tmp_path, mapping):
    from jobflow.core.models import CustomAnswers
    resume = tmp_path / "r.pdf"
    resume.write_bytes(b"%PDF-1.4")
    return Profile(
        identity=Identity(first_name="A", last_name="B", email="a@b.com",
                          phone="9999999999"),
        location=Location(city="X", country="Y"),
        professional=Professional(years_of_experience=6, resume_path=resume),
        compensation=Compensation(current_ctc=1, desired_salary=1,
                                  notice_period_days=0),
        eligibility=Eligibility(),
        custom=CustomAnswers(answers=mapping),
    )


def test_custom_answer_fills_a_gap(tmp_path):
    r = AnswerResolver(_profile_with_custom(
        tmp_path, {"favourite programming language": "Python"}))
    assert r.resolve("What is your favourite programming language?") == \
        ("Python", "custom")


def test_custom_answer_overrides_a_built_in_rule(tmp_path):
    """The escape hatch must also fix a rule that answers wrongly."""
    r = AnswerResolver(_profile_with_custom(tmp_path, {"gender": "Decline"}))
    assert r.resolve("Gender") == ("Decline", "custom")


def test_longer_phrases_win_regardless_of_file_order(tmp_path):
    r = AnswerResolver(_profile_with_custom(tmp_path, {
        "python": "general",
        "years of python": "6",
    }))
    assert r.resolve("How many years of Python?")[0] == "6"
    assert r.resolve("Do you know Python?")[0] == "general"


def test_unmatched_question_still_returns_none(tmp_path):
    r = AnswerResolver(_profile_with_custom(tmp_path, {"python": "6"}))
    assert r.resolve("What is your favourite colour?") is None


def test_blank_custom_value_is_ignored(tmp_path):
    """An empty value means "not answered", so the rules still get a turn.

    The notice rule is used rather than gender: this fixture's demographics
    are blank, so the gender rule would return None too and the test could
    not tell "custom was skipped" from "nothing matched".
    """
    r = AnswerResolver(_profile_with_custom(tmp_path, {"notice period": "  "}))
    assert r.resolve("What is your notice period?") == ("0", "notice_days")


def test_empty_custom_section_loads(tmp_path):
    """A fully commented-out `answers:` parses as None and must not error."""
    from jobflow.core.models import CustomAnswers
    assert CustomAnswers(answers=None).answers == {}
    assert CustomAnswers().match("anything") is None


# ---------------- passport and related screening ----------------

def test_passport_questions_resolve(resolver):
    assert resolver.resolve("Do you have a valid passport?") == ("Yes", "passport")
    assert resolver.resolve("Do you hold a passport?") == ("Yes", "passport")


def test_passport_number_is_not_answered_yes(resolver):
    """A number field must never receive the word "Yes"."""
    for q in ("Passport number", "Passport no.", "Passport expiry date"):
        got = resolver.resolve(q)
        assert got is None or got[1] != "passport", q


def test_notice_buyout_beats_the_generic_notice_rule(resolver):
    """Regression: "notice" matched first and answered a yes/no with "0".

    A form asking whether the notice period can be bought out received the
    number of notice days, which is nonsense in a yes/no field.
    """
    for q in ("Is your notice period buyable?",
              "Can your notice period be bought out?",
              "Notice period buyout available?",
              "Do you have a buyout option?"):
        answer, rule = resolver.resolve(q)
        assert rule == "notice_buyout", q
        assert answer in {"Yes", "No"}, q


def test_plain_notice_questions_still_return_days(resolver):
    assert resolver.resolve("What is your notice period?") == ("0", "notice_days")
    assert resolver.resolve("Notice period in months")[1] == "notice_months"
