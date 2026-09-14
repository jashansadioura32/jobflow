"""Tests for the bounded LLM worker.

The contract under test: malformed, out-of-range, or unavailable model output
must never reach the decision path, and must never raise.

Author: Jashan Sadioura
"""

from __future__ import annotations

import pytest

from jobflow.core.models import (
    Compensation, Defaults, Demographics, Eligibility, Identity,
    JobPosting, Location, Professional, Profile,
)
from jobflow.workers.fit_scorer import FitScorer, _extract_json


@pytest.fixture
def profile(tmp_path):
    r = tmp_path / "r.pdf"; r.write_bytes(b"%PDF")
    return Profile(
        identity=Identity(first_name="Alex", last_name="Doe",
                          email="a@b.com", phone="9999999999"),
        location=Location(city="Springfield", country="India"),
        professional=Professional(years_of_experience=6, resume_path=r,
                                  headline="AI & Data Product Manager",
                                  summary="Data quality and AI products."),
        compensation=Compensation(current_ctc=1200000, desired_salary=1500000,
                                  notice_period_days=0),
        eligibility=Eligibility(), demographics=Demographics(), defaults=Defaults(),
    )


def posting(desc="5+ years of data governance experience"):
    return JobPosting(job_id="1", title="Data Quality Lead",
                      company="Acme", description=desc)


class FakeClient:
    """Minimal stand-in for an OpenAI-compatible client."""
    def __init__(self, content: str | None = None, raise_exc: Exception | None = None):
        self._content, self._raise = content, raise_exc
        outer = self

        class _Completions:
            def create(self, **kw):
                if outer._raise:
                    raise outer._raise
                class _Msg:  message = type("M", (), {"content": outer._content})()
                return type("R", (), {"choices": [_Msg()]})()

        self.chat = type("C", (), {"completions": _Completions()})()


# ---------------- JSON extraction ----------------

@pytest.mark.parametrize("raw", [
    '{"score": 80, "rationale": "good"}',
    '```json\n{"score": 80, "rationale": "good"}\n```',
    'Here is my assessment:\n{"score": 80, "rationale": "good"}',
])
def test_extract_json_tolerates_wrapping(raw):
    assert _extract_json(raw)["score"] == 80


@pytest.mark.parametrize("raw", ["not json at all", "", "[1,2,3]"])
def test_extract_json_rejects_garbage(raw):
    assert _extract_json(raw) is None


# ---------------- happy path ----------------

def test_valid_response(profile):
    client = FakeClient('{"score": 82, "rationale": "Strong governance match",'
                        ' "matched_strengths": ["dbt"], "gaps": ["k8s"]}')
    fit = FitScorer(profile, client=client).score(posting())
    assert fit.score == 82
    assert fit.matched_strengths == ["dbt"]


# ---------------- failure modes all return None ----------------

@pytest.mark.parametrize("content", [
    "totally unparseable",
    '{"score": 150, "rationale": "out of range"}',
    '{"score": -5, "rationale": "negative"}',
    '{"score": "high", "rationale": "not numeric"}',
    '{"rationale": "no score key"}',
    '{"score": 80}',                      # missing rationale
    '{"score": 80, "rationale": "   "}',  # blank rationale
])
def test_bad_output_returns_none(profile, content):
    assert FitScorer(profile, client=FakeClient(content)).score(posting()) is None


def test_api_error_fails_open(profile):
    """An outage must return None, not raise."""
    scorer = FitScorer(profile, client=FakeClient(raise_exc=RuntimeError("503")))
    assert scorer.score(posting()) is None


def test_disabled_without_client(profile):
    scorer = FitScorer(profile, client=None)
    assert not scorer.enabled
    assert scorer.score(posting()) is None


def test_empty_description_skips_call(profile):
    called = {"n": 0}
    class Counting(FakeClient):
        def __init__(self): super().__init__('{"score":50,"rationale":"x"}')
    scorer = FitScorer(profile, client=Counting())
    assert scorer.score(posting(desc="")) is None


# ---------------- threshold semantics ----------------

def test_recommends_skip_threshold(profile):
    scorer = FitScorer(profile, client=FakeClient(), min_score=50)
    from jobflow.core.models import FitAssessment
    assert scorer.recommends_skip(FitAssessment(score=30, rationale="low"))
    assert not scorer.recommends_skip(FitAssessment(score=70, rationale="fine"))
    # Absence of a score is not evidence of poor fit.
    assert not scorer.recommends_skip(None)
