"""Tests for deterministic screening.

These cover the cases that silently cost applications in practice:
substring false positives, "lowest credible years" extraction, and the
masters buffer.

Author: Jashan Sadioura
"""

from __future__ import annotations

import pytest

from jobflow.core.models import (
    Compensation, Decision, Defaults, Demographics, Eligibility,
    ExperienceGate, Exclusions, Identity, JobPosting, Location,
    Professional, Profile, SearchConfig, SearchFilters, SearchTier, SkipReason,
)
from jobflow.core.screening import (
    DeterministicScreener, extract_years_required, _contains_phrase,
)


@pytest.fixture
def profile(tmp_path):
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"%PDF-1.4 test")
    return Profile(
        identity=Identity(
            first_name="Alex", last_name="Doe",
            email="test@example.com", phone="9999999999",
        ),
        location=Location(city="Springfield", country="India"),
        professional=Professional(
            years_of_experience=6, has_masters=True, resume_path=resume,
        ),
        compensation=Compensation(
            current_ctc=1200000, desired_salary=1500000, notice_period_days=0,
        ),
        eligibility=Eligibility(has_security_clearance=False),
        demographics=Demographics(),
        defaults=Defaults(),
    )


@pytest.fixture
def config():
    return SearchConfig(
        tiers=[SearchTier(name="core", terms=["Data Quality Lead"])],
        filters=SearchFilters(location="India"),
        exclusions=Exclusions(
            description_blockers=["US Citizen", "must be located in the US"],
            clearance_blockers=["security clearance", "secret clearance", "top secret", "polygraph"],
            companies=["Crossover"],
        ),
        experience_gate=ExperienceGate(enabled=True, tolerance_years=2),
    )


def job(**kw):
    base = dict(job_id="1", title="Data Quality Lead", company="Acme", description="")
    return JobPosting(**{**base, **kw})


# ---------------- years extraction ----------------

@pytest.mark.parametrize("text,expected", [
    ("5+ years of experience required", 5),
    ("3-5 years of experience", 3),
    ("minimum of 7 years", 7),
    ("at least 4 yrs experience", 4),
    ("six years of relevant experience", 6),
    ("no requirement stated", None),
    ("", None),
    # Lowest credible wins: a "3 required / 8 preferred" post is a 3y role.
    ("3+ years required, 8+ years preferred", 3),
    # Company boilerplate must not be read as a requirement.
    ("We have 30 years in business", None),
])
def test_extract_years(text, expected):
    assert extract_years_required(text) == expected


# ---------------- phrase matching ----------------

def test_phrase_match_respects_word_boundaries():
    # The bug this guards against: bare "secret" matching unrelated text.
    assert not _contains_phrase("reporting to the company secretary", "secret")
    assert not _contains_phrase("our trade secrets policy applies", "secret")
    assert _contains_phrase("requires a secret clearance", "secret clearance")
    assert _contains_phrase("top secret required", "top secret")


def test_company_secretary_is_not_skipped(profile, config):
    """A governance role mentioning a Company Secretary must survive."""
    s = DeterministicScreener(profile, config)
    ev = s.screen(job(description="Partner with the Company Secretary on data governance."))
    assert ev.decision is Decision.APPLY


def test_real_clearance_is_skipped(profile, config):
    s = DeterministicScreener(profile, config)
    ev = s.screen(job(description="Must hold an active secret clearance."))
    assert ev.decision is Decision.SKIP
    assert ev.skip_reason is SkipReason.CLEARANCE_REQUIRED


# ---------------- blockers ----------------

def test_work_authorization_blocker(profile, config):
    s = DeterministicScreener(profile, config)
    ev = s.screen(job(description="Open to US Citizen applicants only."))
    assert ev.decision is Decision.SKIP
    assert ev.skip_reason is SkipReason.BLOCKED_PHRASE


def test_already_applied(profile, config):
    s = DeterministicScreener(profile, config)
    ev = s.screen(job(job_id="42"), applied_ids={"42"})
    assert ev.decision is Decision.SKIP
    assert ev.skip_reason is SkipReason.ALREADY_APPLIED


def test_excluded_company(profile, config):
    s = DeterministicScreener(profile, config)
    ev = s.screen(job(company="crossover"))  # case-insensitive
    assert ev.decision is Decision.SKIP
    assert ev.skip_reason is SkipReason.COMPANY_EXCLUDED


# ---------------- experience gate ----------------

def test_experience_within_masters_buffer(profile, config):
    """6y experience + 2y MBA buffer => an 8y posting still passes."""
    assert profile.experience_ceiling == 8
    s = DeterministicScreener(profile, config)
    assert s.screen(job(description="8+ years of experience")).decision is Decision.APPLY


def test_experience_above_ceiling(profile, config):
    s = DeterministicScreener(profile, config)
    ev = s.screen(job(description="12+ years of experience required"))
    assert ev.decision is Decision.SKIP
    assert ev.skip_reason is SkipReason.EXPERIENCE_TOO_HIGH


def test_missing_experience_is_not_a_blocker(profile, config):
    s = DeterministicScreener(profile, config)
    assert s.screen(job(description="Great team, no years listed.")).decision is Decision.APPLY


# ---------------- audit trail ----------------

def test_every_evaluation_carries_evidence(profile, config):
    s = DeterministicScreener(profile, config)
    for d in ["5+ years experience", "US Citizen only", "top secret required"]:
        ev = s.screen(job(description=d))
        assert ev.findings, "evaluation must always carry findings"
        if ev.decision is Decision.SKIP:
            assert ev.blockers, "a skip must record a blocker finding"


def test_salary_conversions(profile):
    c = profile.compensation
    assert c.current_lakhs == "12.00"
    assert c.desired_lakhs == "15.00"
    # Integer rupees — forms frequently reject decimals.
    assert c.desired_monthly == "125000"
    assert "." not in c.desired_monthly


# ---------------- skip evidence ----------------

def test_experience_skip_records_the_matched_phrase(profile, config):
    """A bare "requires 12y" in the log cannot be checked afterwards.

    Without the matched text there is no way to tell a real requirement
    from a misparse of "12 years of excellence in the market".
    """
    from jobflow.core.screening import DeterministicScreener

    posting = JobPosting(
        job_id="900", title="Senior Product Manager", company="Acme",
        description="We need at least 12 years of experience in product.",
    )
    ev = DeterministicScreener(profile, config).screen(posting)

    assert ev.skip_reason is SkipReason.EXPERIENCE_TOO_HIGH
    detail = [f.detail for f in ev.findings if f.dimension == "experience"][0]
    assert "12 years of experience" in detail


def test_experience_evidence_lists_other_figures_seen(profile, config):
    """The lowest figure wins; the rest are recorded to explain the choice."""
    from jobflow.core.screening import extract_years_evidence

    matches = extract_years_evidence(
        "10+ years preferred, 4+ years required, 3-12 years considered")
    numbers = [n for n, _ in matches]

    assert numbers == sorted(numbers)      # lowest first
    assert numbers[0] == 3
    assert all(phrase for _, phrase in matches)


def test_company_age_is_not_mistaken_for_a_requirement(profile, config):
    from jobflow.core.screening import extract_years_required

    assert extract_years_required(
        "The company has 30 years of excellence in the market.") is None
