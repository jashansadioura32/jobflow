"""Tests for the LLM question-answering worker.

The worker exists to answer form questions the profile cannot map, but it
writes into real applications, so most of these tests are about what it
*refuses* to do: a discarded answer falls back to a predictable guess, while
a bad one reaches an employer.

Author: Jashan Sadioura
"""

from __future__ import annotations

import json

import pytest

from jobflow.core.models import (
    Compensation, Defaults, Demographics, Eligibility, Identity, JobPosting,
    Location, Professional, Profile,
)
from jobflow.workers.question_answerer import QuestionAnswerer


class FakeClient:
    """Minimal stand-in for an OpenAI client, recording what it was asked."""

    def __init__(self, payload, raise_on_call=False):
        self._payload = payload
        self._raise = raise_on_call
        self.prompts: list[str] = []
        self.chat = self                      # client.chat.completions...
        self.completions = self

    def create(self, model, messages, temperature=0, max_tokens=None):
        if self._raise:
            raise RuntimeError("API unavailable")
        self.prompts.append(messages[-1]["content"])
        body = (self._payload if isinstance(self._payload, str)
                else json.dumps(self._payload))
        return type("R", (), {"choices": [
            type("C", (), {"message": type("M", (), {"content": body})()})()
        ]})()


@pytest.fixture
def profile(tmp_path):
    r = tmp_path / "cv.pdf"
    r.write_bytes(b"%PDF-1.4")
    return Profile(
        identity=Identity(first_name="Alex", last_name="Doe",
                          email="test@example.com", phone="9999999999"),
        location=Location(city="Springfield", country="India"),
        professional=Professional(years_of_experience=6, has_masters=True,
                                  resume_path=r, headline="Data PM",
                                  recent_employer="Acme Corp",
                                  summary="Data and AI product manager."),
        compensation=Compensation(current_ctc=1200000, desired_salary=1500000,
                                  notice_period_days=0),
        eligibility=Eligibility(), demographics=Demographics(),
        defaults=Defaults(),
    )


def _answerer(profile, payload, **kw):
    return QuestionAnswerer(profile, client=FakeClient(payload, **kw))


# ---------------- the happy path ----------------

def test_answers_a_free_text_question(profile):
    a = _answerer(profile, {"answer": "I led a data quality programme.",
                            "confident": True})
    assert a.answer("Describe a project you are proud of") == \
        "I led a data quality programme."


def test_choice_answer_is_returned_as_the_exact_option(profile):
    """Selecting a dropdown needs the option string verbatim."""
    a = _answerer(profile, {"answer": "3-5 years", "confident": True})
    got = a.answer("How much experience?", options=["0-2 years", "3-5 years"])
    assert got == "3-5 years"


def test_choice_match_is_case_insensitive_but_returns_the_option(profile):
    a = _answerer(profile, {"answer": "YES", "confident": True})
    assert a.answer("Willing to relocate?", options=["Yes", "No"]) == "Yes"


def test_the_job_description_reaches_the_prompt(profile):
    client = FakeClient({"answer": "Yes", "confident": True})
    a = QuestionAnswerer(profile, client=client)
    posting = JobPosting(job_id="1", title="Data Lead", company="Acme",
                         description="We use dbt and Snowflake daily.")
    a.answer("Do you know dbt?", posting=posting)
    assert "dbt and Snowflake" in client.prompts[0]
    assert "Alex Doe" in client.prompts[0]


# ---------------- what it refuses to do ----------------

def test_declines_when_the_model_is_not_confident(profile):
    """The model saying "I cannot support this" must not become an answer."""
    a = _answerer(profile, {"answer": "Yes, 10 years", "confident": False})
    assert a.answer("Do you have 10 years of Kubernetes?") is None


def test_discards_an_over_long_answer(profile):
    """Forms truncate silently; a 2000-character essay is not an answer."""
    a = _answerer(profile, {"answer": "x" * 2000, "confident": True})
    assert a.answer("Tell us about yourself") is None


def test_discards_a_choice_answer_that_is_not_on_offer(profile):
    """Selecting an option that does not exist fails silently in the browser."""
    a = _answerer(profile, {"answer": "Maybe", "confident": True})
    assert a.answer("Relocate?", options=["Yes", "No"]) is None


def test_malformed_json_is_discarded(profile):
    a = _answerer(profile, "not json at all")
    assert a.answer("Anything?") is None


def test_json_inside_a_code_fence_is_accepted(profile):
    """Models emit fenced JSON even when told not to."""
    a = _answerer(profile, '```json\n{"answer": "Yes", "confident": true}\n```')
    assert a.answer("Willing to travel?") == "Yes"


def test_empty_answer_is_discarded(profile):
    a = _answerer(profile, {"answer": "   ", "confident": True})
    assert a.answer("Anything?") is None


def test_api_failure_returns_none(profile):
    """Fails closed: the caller falls back to its deterministic guess."""
    a = _answerer(profile, {"answer": "Yes", "confident": True}, raise_on_call=True)
    assert a.answer("Anything?") is None


def test_disabled_without_a_client(profile):
    a = QuestionAnswerer(profile, client=None)
    assert a.enabled is False
    assert a.answer("Anything?") is None


def test_blank_question_is_not_sent(profile):
    client = FakeClient({"answer": "Yes", "confident": True})
    assert QuestionAnswerer(profile, client=client).answer("  ") is None
    assert client.prompts == []
