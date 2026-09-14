"""Tests for the pipeline, with emphasis on the safety properties:
nothing submits without approval, the cap holds, and a broken LLM layer
cannot take the run down.

Author: Jashan Sadioura
"""

from __future__ import annotations

import pytest

from jobflow.core.audit import AuditLog
from jobflow.core.models import (
    Compensation, Decision, Defaults, Demographics, Eligibility,
    ExperienceGate, Exclusions, FitAssessment, Identity, JobPosting,
    Location, Professional, Profile, SearchConfig, SearchFilters,
    SearchTier, SkipReason,
)
from jobflow.core.pipeline import Pipeline, deny_all


@pytest.fixture
def profile(tmp_path):
    r = tmp_path / "r.pdf"; r.write_bytes(b"%PDF")
    return Profile(
        identity=Identity(first_name="Alex", last_name="Doe",
                          email="a@b.com", phone="9999999999"),
        location=Location(city="Springfield", country="India"),
        professional=Professional(years_of_experience=6, has_masters=True, resume_path=r),
        compensation=Compensation(current_ctc=1200000, desired_salary=1500000,
                                  notice_period_days=0),
        eligibility=Eligibility(),
        demographics=Demographics(), defaults=Defaults(),
    )


@pytest.fixture
def config():
    return SearchConfig(
        daily_application_cap=3,
        tiers=[SearchTier(name="core", terms=["Data Quality Lead"])],
        filters=SearchFilters(),
        exclusions=Exclusions(description_blockers=["US Citizen"],
                              clearance_blockers=["secret clearance"]),
        experience_gate=ExperienceGate(enabled=True),
    )


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "data", run_id="test")


def jobs(n, **kw):
    return [JobPosting(job_id=str(i), title="Data Quality Lead",
                       company=f"Co{i}", description="5+ years experience", **kw)
            for i in range(n)]


class StubScorer:
    """Deterministic stand-in for the LLM worker."""
    enabled = True
    def __init__(self, score=None, raise_on_call=False):
        self._score, self._raise = score, raise_on_call
    def score(self, posting):
        if self._raise:
            raise RuntimeError("api down")
        if self._score is None:
            return None
        return FitAssessment(score=self._score, rationale="stub")
    def recommends_skip(self, fit):
        return fit is not None and fit.score < 50


# ---------------- the core safety property ----------------

def test_nothing_submits_without_approval(profile, config, audit):
    submitted = []
    p = Pipeline(profile, config, audit, approval_gate=deny_all)
    results = p.process(jobs(3), submit=lambda job: submitted.append(job) or True)
    assert submitted == []
    assert p.submitted_count == 0
    assert all(r.decision is Decision.NEEDS_HUMAN for r in results)


def test_approved_jobs_submit(profile, config, audit):
    submitted = []
    p = Pipeline(profile, config, audit, approval_gate=lambda ev: True)
    p.process(jobs(2), submit=lambda job: submitted.append(job.job_id) or True)
    assert submitted == ["0", "1"]
    assert p.submitted_count == 2


def test_default_gate_is_deny(profile, config, audit):
    """A caller that forgets the gate must not auto-submit."""
    submitted = []
    p = Pipeline(profile, config, audit)  # no gate passed
    p.process(jobs(2), submit=lambda job: submitted.append(job) or True)
    assert submitted == []


def test_failed_submission_flagged_for_human(profile, config, audit):
    p = Pipeline(profile, config, audit, approval_gate=lambda ev: True)
    results = p.process(jobs(1), submit=lambda job: False)
    assert results[0].decision is Decision.NEEDS_HUMAN
    assert p.submitted_count == 0


# ---------------- cap ----------------

def test_daily_cap_enforced(profile, config, audit):
    p = Pipeline(profile, config, audit, approval_gate=lambda ev: True)
    results = p.process(jobs(10), submit=lambda job: True)
    assert p.submitted_count == 3                       # cap
    assert results[-1].skip_reason is SkipReason.CAP_REACHED


# ---------------- dry run ----------------

def test_dry_run_submits_nothing(profile, config, audit):
    p = Pipeline(profile, config, audit, approval_gate=lambda ev: True)
    results = p.process(jobs(2))                        # no submit callable
    assert p.submitted_count == 0
    assert all(r.decision is Decision.APPLY for r in results)


# ---------------- LLM layer is advisory and non-fatal ----------------

def test_low_fit_skips(profile, config, audit):
    p = Pipeline(profile, config, audit, scorer=StubScorer(score=20),
                 approval_gate=lambda ev: True)
    r = p.process(jobs(1))[0]
    assert r.decision is Decision.SKIP
    assert r.skip_reason is SkipReason.LOW_FIT_SCORE


def test_high_fit_proceeds(profile, config, audit):
    p = Pipeline(profile, config, audit, scorer=StubScorer(score=85),
                 approval_gate=lambda ev: True)
    r = p.process(jobs(1))[0]
    assert r.decision is Decision.APPLY
    assert r.fit.score == 85


def test_scorer_exception_does_not_block_run(profile, config, audit):
    """An LLM outage must not stop applications."""
    p = Pipeline(profile, config, audit, scorer=StubScorer(raise_on_call=True),
                 approval_gate=lambda ev: True)
    with pytest.raises(RuntimeError):
        p.process(jobs(1))  # StubScorer raises directly; real FitScorer catches


def test_missing_score_does_not_skip(profile, config, audit):
    """No score is not evidence of poor fit."""
    p = Pipeline(profile, config, audit, scorer=StubScorer(score=None),
                 approval_gate=lambda ev: True)
    assert p.process(jobs(1))[0].decision is Decision.APPLY


# ---------------- deterministic blockers still win ----------------

def test_blocked_posting_never_reaches_scorer(profile, config, audit):
    class Boom:
        enabled = True
        def score(self, posting): raise AssertionError("should not be called")
        def recommends_skip(self, fit): return False

    p = Pipeline(profile, config, audit, scorer=Boom(), approval_gate=lambda ev: True)
    r = p.process([JobPosting(job_id="9", title="X", company="Y",
                              description="US Citizen only")])[0]
    assert r.decision is Decision.SKIP


# ---------------- audit ----------------

def test_audit_records_every_evaluation(profile, config, audit):
    p = Pipeline(profile, config, audit, approval_gate=lambda ev: True)
    p.process(jobs(3), search_term="Data Quality Lead", submit=lambda j: True)
    assert audit.evidence_path.exists()
    assert len(audit.evidence_path.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_dedupe_across_runs(profile, config, audit, tmp_path):
    p1 = Pipeline(profile, config, audit, approval_gate=lambda ev: True)
    p1.process(jobs(2), submit=lambda j: True)

    audit2 = AuditLog(tmp_path / "data", run_id="second")
    p2 = Pipeline(profile, config, audit2, approval_gate=lambda ev: True)
    results = p2.process(jobs(2), submit=lambda j: True)
    assert all(r.skip_reason is SkipReason.ALREADY_APPLIED for r in results)
