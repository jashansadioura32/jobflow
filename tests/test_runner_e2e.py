"""End-to-end runner tests through FakeBrowser.

Exercises the whole loop — sign-in, search, description load, screening,
gate, submit — with no network and no real account, and pins the safety
properties at the composed level rather than per component.

Author: Jashan Sadioura
"""

from __future__ import annotations

import pytest

from jobflow.adapters.browser import Element
from jobflow.adapters.fake_browser import FakeBrowser, FakeField, FakePage
from jobflow.adapters.linkedin import SEL
from jobflow.adapters.runner import LinkedInRunner
from jobflow.core.audit import AuditLog
from jobflow.core.models import (
    Compensation, Decision, Defaults, Demographics, Eligibility, Identity,
    Location, Professional, Profile, SearchConfig, SearchFilters, SearchTier,
    SkipReason,
)


@pytest.fixture
def profile(tmp_path):
    r = tmp_path / "Test_Resume.pdf"
    r.write_bytes(b"%PDF-1.4")
    return Profile(
        identity=Identity(first_name="Alex", last_name="Doe",
                          email="test@example.com",
                          phone="9999999999"),
        location=Location(city="Springfield", country="India"),
        professional=Professional(years_of_experience=6, has_masters=True,
                                  resume_path=r, headline="AI & Data PM"),
        compensation=Compensation(current_ctc=1200000, desired_salary=1500000,
                                  notice_period_days=0),
        eligibility=Eligibility(), demographics=Demographics(), defaults=Defaults(),
    )


@pytest.fixture
def config():
    return SearchConfig(
        daily_application_cap=5,
        applications_per_term=3,
        tiers=[SearchTier(name="core", terms=["Data Quality Lead"])],
        filters=SearchFilters(location="India", easy_apply_only=True),
    )


def _wire(browser, config, job_id, description, *, answerable=True):
    """Script a signed-in session and one search page.

    Single-page by design, mirroring the real flow: clicking a result card
    loads the job into the detail pane on the *same* page, so the cards,
    the description, the Easy Apply button and the modal all live together
    rather than on a separate /jobs/view/<id> page.
    """
    browser.add_page(FakePage(
        url="https://www.linkedin.com/feed/",
        elements={SEL["feed_marker"]: [Element(handle=None)]},
    ))

    from jobflow.adapters.linkedin import LinkedInSearch
    search_url = LinkedInSearch(browser, config).build_url("Data Quality Lead")

    label = "What is your current city?" if answerable else "Describe a conflict"
    field = FakeField(selector="#f1", label=label, required=True)
    page = FakePage(
        url=search_url,
        elements={
            SEL["result_card"]: [Element(
                handle=None,
                attributes={"data-job-id": job_id, "data-selector": "#card0"})],
            "a.job-card-list__title": [Element(
                handle=None, text="Data Quality Lead",
                attributes={"href": f"/jobs/view/{job_id}/", "owner": "#card0"})],
            # The detail pane, populated once the card is clicked.
            "div.jobs-box__html-content": [Element(handle=None, text=description)],
            SEL["easy_apply_button"]: [Element(handle=None, text="Easy Apply")],
            SEL["modal"]: [Element(handle=None)],
            SEL["form_group"]: [Element(
                handle=None, text=label,
                attributes={"data-selector": "#f1", "aria-label": label,
                            "required": "true"})],
            # "owner" ties the input to its form group, so the fake can tell
            # a scoped lookup from a document-wide one.
            SEL["text_input"]: [Element(
                handle=field,
                attributes={"data-selector": "#f1", "aria-label": label,
                            "owner": "#f1"})],
            SEL["field_label"].split(", ")[0]: [Element(
                handle=None, text=label, attributes={"owner": "#f1"})],
            SEL["modal_submit"]: [Element(handle=None, text="Submit application")],
        },
        fields=[field],
    )
    browser.add_page(page)
    return page


def _human_clicks_submit(browser, detail):
    """Script the person submitting: the modal closes, success marker appears."""
    browser.closes_after[SEL["modal"]] = 1
    detail.elements[SEL["post_apply_marker"]] = [Element(handle=None, text="Applied")]


def _human_closes_modal(browser, detail):
    """Script the person skipping: the modal closes with no success marker."""
    browser.closes_after[SEL["modal"]] = 1
    detail.elements[SEL["post_apply_marker"]] = []


def test_full_loop_submits_when_human_clicks_submit(profile, config, tmp_path):
    b = FakeBrowser()
    detail = _wire(b, config, "555", "We need 5+ years of data governance experience.")
    _human_clicks_submit(b, detail)
    audit = AuditLog(tmp_path / "data", run_id="e2e")

    # Screening decides which postings are opened; the person decides in the
    # browser, so the terminal gate passes everything through -- as cmd_apply does.
    # --review: the person clicks Submit, JobFlow never does.
    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=False,
                            auto_submit=False)
    results = runner.run(prompt=lambda _: "")

    assert len(results) == 1
    assert results[0].decision is Decision.APPLY
    assert runner.pipeline.submitted_count == 1
    assert b.typed["#f1"] == "Springfield"
    assert "Submit application" not in b.clicks


def test_closing_the_modal_skips_without_submitting(profile, config, tmp_path):
    """The person's way to decline is closing the dialog, not a terminal prompt."""
    b = FakeBrowser()
    detail = _wire(b, config, "556", "5+ years of experience required.")
    _human_closes_modal(b, detail)
    audit = AuditLog(tmp_path / "data", run_id="e2e-skip")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=False,
                            auto_submit=False)
    results = runner.run(prompt=lambda _: "")

    assert results[0].decision is Decision.NEEDS_HUMAN
    assert runner.pipeline.submitted_count == 0


def test_live_leaves_an_unfinished_form_open_instead_of_discarding(
        profile, config, tmp_path):
    """A form the automation could not finish is the one to show a person.

    Discarding it in live mode destroyed the only evidence of what broke,
    and silently left the application saved on LinkedIn.
    """
    b = FakeBrowser()
    # answerable=False leaves a required question unanswered.
    detail = _wire(b, config, "570", "5+ years of experience.", answerable=False)
    _human_clicks_submit(b, detail)
    audit = AuditLog(tmp_path / "data", run_id="e2e-open")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=False)
    results = runner.run(prompt=lambda _: "")

    # The person submitted it themselves after fixing the question.
    assert results[0].decision is Decision.APPLY
    assert runner.pipeline.submitted_count == 1


def test_dry_run_still_discards_an_unfinished_form(profile, config, tmp_path):
    """Dry run must not block on a person: it discards and moves on."""
    b = FakeBrowser()
    _wire(b, config, "571", "5+ years of experience.", answerable=False)
    audit = AuditLog(tmp_path / "data", run_id="e2e-dry-discard")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=True)
    results = runner.run(prompt=lambda _: "")

    assert results[0].decision is Decision.NEEDS_HUMAN
    assert runner.pipeline.submitted_count == 0


def test_no_decision_within_timeout_skips_the_posting(profile, config, tmp_path):
    """An unattended run must not leave an application hanging open forever."""
    b = FakeBrowser()
    _wire(b, config, "557", "5+ years of experience required.")
    # Nothing scripted to close: the modal stays up past the timeout.
    b.closes_after[SEL["modal"]] = 99
    audit = AuditLog(tmp_path / "data", run_id="e2e-timeout")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=False,
                            auto_submit=False, review_timeout=0.01)
    results = runner.run(prompt=lambda _: "")

    assert results[0].decision is Decision.NEEDS_HUMAN
    assert runner.pipeline.submitted_count == 0


def test_full_loop_dry_run_submits_nothing(profile, config, tmp_path):
    b = FakeBrowser()
    _wire(b, config, "556", "5+ years of experience required.")
    audit = AuditLog(tmp_path / "data", run_id="e2e-dry")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=True)
    results = runner.run(prompt=lambda _: "")

    # The form is still filled and reviewed, but Submit is never clicked.
    assert results[0].decision is Decision.APPLY
    assert "Submit application" not in b.clicks


def test_declined_at_gate_is_not_submitted(profile, config, tmp_path):
    b = FakeBrowser()
    _wire(b, config, "557", "5+ years of experience.")
    audit = AuditLog(tmp_path / "data", run_id="e2e-deny")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: False, dry_run=False)
    results = runner.run(prompt=lambda _: "")

    assert results[0].decision is Decision.NEEDS_HUMAN
    assert runner.pipeline.submitted_count == 0


def test_unanswerable_question_blocks_submission_when_reviewing(profile, config, tmp_path):
    """With a person in the loop, an unmappable question is never guessed."""
    b = FakeBrowser()
    _wire(b, config, "558", "5+ years of experience.", answerable=False)
    audit = AuditLog(tmp_path / "data", run_id="e2e-unans")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=False,
                            auto_submit=False, review_timeout=0.01)
    results = runner.run(prompt=lambda _: "")

    assert runner.pipeline.submitted_count == 0


def test_auto_submit_guesses_an_unmappable_question_and_sends(profile, config, tmp_path):
    """Auto-submit has nobody to escalate to, so it guesses and sends.

    This is a deliberate trade: an answer the profile does not cover reaches
    an employer. The guess is recorded so it can be audited afterwards.
    """
    b = FakeBrowser()
    _wire(b, config, "572", "5+ years of experience.", answerable=False)
    audit = AuditLog(tmp_path / "data", run_id="e2e-guess")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=False,
                            auto_submit=True)
    results = runner.run(prompt=lambda _: "")

    assert results[0].decision is Decision.APPLY
    assert runner.pipeline.submitted_count == 1
    assert "Submit application" in b.clicks      # JobFlow clicked it itself


def test_dry_run_never_guesses(profile, config, tmp_path):
    """Guessing is for unattended sending only; a dry run still escalates."""
    b = FakeBrowser()
    _wire(b, config, "573", "5+ years of experience.", answerable=False)
    audit = AuditLog(tmp_path / "data", run_id="e2e-dry-noguess")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=True,
                            auto_submit=True)
    results = runner.run(prompt=lambda _: "")

    assert results[0].decision is Decision.NEEDS_HUMAN
    assert runner.pipeline.submitted_count == 0
    assert results[0].decision is Decision.NEEDS_HUMAN


def test_screening_blocks_before_any_form_interaction(profile, config, tmp_path):
    b = FakeBrowser()
    _wire(b, config, "559", "Open to US Citizen applicants only.")
    audit = AuditLog(tmp_path / "data", run_id="e2e-block")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=False)
    # Default exclusions are empty in this config, so add the blocker.
    config.exclusions.description_blockers = ["US Citizen"]
    results = runner.run(prompt=lambda _: "")

    assert results[0].decision is Decision.SKIP
    assert results[0].skip_reason is SkipReason.BLOCKED_PHRASE
    assert b.typed == {}          # never touched the form


def test_aborts_when_not_signed_in(profile, config, tmp_path):
    b = FakeBrowser()
    b.add_page(FakePage(url="https://www.linkedin.com/feed/",
                        elements={SEL["signin_link"]: [Element(handle=None)]}))
    audit = AuditLog(tmp_path / "data", run_id="e2e-nologin")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=False)
    assert runner.run(prompt=lambda _: "") == []


def test_audit_written_for_every_posting(profile, config, tmp_path):
    b = FakeBrowser()
    _wire(b, config, "560", "5+ years of experience.")
    audit = AuditLog(tmp_path / "data", run_id="e2e-audit")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=True)
    runner.run(prompt=lambda _: "")

    assert audit.evidence_path.exists()
    assert audit.evidence_path.read_text(encoding="utf-8").strip()


# ---------------- the LLM answerer is wired only when unattended ----------

class _StubScorer:
    enabled = True
    client = object()
    model = "gpt-4o-mini"

    def score(self, posting):
        return None

    def recommends_skip(self, fit):
        return False


def test_ai_answerer_is_used_only_when_nobody_is_watching(profile, config, tmp_path):
    """Under --review a person answers the question, which beats a guess."""
    audit = AuditLog(tmp_path / "data", run_id="ai-wiring")

    unattended = LinkedInRunner(FakeBrowser(), profile, config, audit,
                                scorer=_StubScorer(), dry_run=False,
                                auto_submit=True)
    assert unattended.filler.answerer is not None

    reviewing = LinkedInRunner(FakeBrowser(), profile, config, audit,
                               scorer=_StubScorer(), dry_run=False,
                               auto_submit=False)
    assert reviewing.filler.answerer is None

    dry = LinkedInRunner(FakeBrowser(), profile, config, audit,
                         scorer=_StubScorer(), dry_run=True, auto_submit=True)
    assert dry.filler.answerer is None


def test_no_answerer_without_an_llm_scorer(profile, config, tmp_path):
    """--use-llm is what supplies the client; without it there is no answerer."""
    audit = AuditLog(tmp_path / "data", run_id="ai-none")
    runner = LinkedInRunner(FakeBrowser(), profile, config, audit,
                            dry_run=False, auto_submit=True)
    assert runner.filler.answerer is None
