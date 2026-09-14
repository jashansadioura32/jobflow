"""Tests for the Quit button, stop reasons, and the closing summary.

The quit signal is an Event, so these drive it directly rather than opening
a window: a test suite must never depend on a display being present.

Author: Jashan Sadioura
"""

from __future__ import annotations

import pytest

from jobflow.adapters.fake_browser import FakeBrowser
from jobflow.adapters.quit_button import NullQuitButton, QuitButton
from jobflow.adapters.runner import LinkedInRunner
from jobflow.core.audit import AuditLog
from jobflow.core.models import (
    Compensation, Decision, Defaults, Demographics, Eligibility, Identity,
    Location, Professional, Profile, SearchConfig, SearchFilters, SearchTier,
)

from tests.test_runner_e2e import _wire  # reuse the single-page fake


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


# ---------------- the quit signal ----------------

def test_quit_button_starts_unset():
    assert QuitButton().quit_requested is False


def test_request_quit_sets_the_flag():
    q = QuitButton()
    q.request_quit()
    assert q.quit_requested is True


def test_null_quit_button_never_quits():
    assert NullQuitButton().quit_requested is False


def test_quit_button_stop_is_safe_without_a_window():
    """stop() must not raise when the window never opened."""
    QuitButton().stop()


# ---------------- stopping a run ----------------

def test_quit_before_the_run_stops_it_immediately(profile, config, tmp_path):
    b = FakeBrowser()
    _wire(b, config, "600", "5+ years of experience.")
    audit = AuditLog(tmp_path / "data", run_id="quit-early")

    q = QuitButton()
    q.request_quit()
    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=True,
                            quit_button=q)
    results = runner.run(prompt=lambda _: "")

    assert results == []
    assert runner.stop_reason == "you clicked Quit"


def test_a_completed_run_reports_that_it_finished(profile, config, tmp_path):
    b = FakeBrowser()
    _wire(b, config, "601", "5+ years of experience.")
    audit = AuditLog(tmp_path / "data", run_id="quit-none")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=True)
    runner.run(prompt=lambda _: "")

    assert runner.stop_reason == "all search terms completed"


def test_daily_cap_is_reported_as_the_stop_reason(profile, tmp_path):
    """The cap counts *submissions*, so it only binds on a live run."""
    # _wire scripts the page for "Data Quality Lead", so the terms must be
    # that: a different term builds a different URL and the fake returns
    # no cards at all.
    cfg = SearchConfig(
        daily_application_cap=1,
        applications_per_term=3,
        tiers=[SearchTier(name="core",
                          terms=["Data Quality Lead", "Data Governance Manager"])],
        filters=SearchFilters(location="India", easy_apply_only=True),
    )
    b = FakeBrowser()
    _wire(b, cfg, "602", "5+ years of experience.")
    audit = AuditLog(tmp_path / "data", run_id="quit-cap")

    runner = LinkedInRunner(b, profile, cfg, audit,
                            approval_gate=lambda ev: True, dry_run=False,
                            auto_submit=True)
    runner.run(prompt=lambda _: "")

    assert runner.pipeline.submitted_count == 1
    assert "daily cap of 1" in runner.stop_reason


# ---------------- the summary ----------------

def test_summary_reports_counts_reason_and_signoff(profile, config, tmp_path):
    b = FakeBrowser()
    _wire(b, config, "603", "5+ years of experience.")
    audit = AuditLog(tmp_path / "data", run_id="quit-summary")

    q = QuitButton()
    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=True,
                            quit_button=q)
    results = runner.run(prompt=lambda _: "")
    q.request_quit()
    runner.stop_reason = "you clicked Quit"

    text = "\n".join(runner.summary_lines(results))

    assert "JobFlow run summary" in text
    assert "Stopped because : you clicked Quit" in text
    assert "Postings evaluated" in text
    assert "pass it on" in text
    # The summary greets whoever is running it, from their own profile --
    # a shared repo must not print the author's name on a user's screen.
    assert "Good luck out there, Alex." in text


def test_summary_greets_the_configured_user_not_the_author(config, tmp_path):
    """Someone else's run must greet them, not the person who wrote this."""
    resume = tmp_path / "r.pdf"
    resume.write_bytes(b"%PDF-1.4")
    other = Profile(
        identity=Identity(first_name="Priya", last_name="Sharma",
                          email="priya@example.com", phone="9000000000"),
        location=Location(city="Pune", country="India"),
        professional=Professional(years_of_experience=4, resume_path=resume),
        compensation=Compensation(current_ctc=1, desired_salary=1,
                                  notice_period_days=0),
        eligibility=Eligibility(), demographics=Demographics(), defaults=Defaults(),
    )
    audit = AuditLog(tmp_path / "data", run_id="other-user")
    runner = LinkedInRunner(FakeBrowser(), other, config, audit, dry_run=True)

    text = "\n".join(runner.summary_lines([]))

    assert "Good luck out there, Priya." in text
    # Only the greeting and sign-off are checked: the summary also prints the
    # audit log paths, which contain whatever the machine's home directory is.
    greeting = [ln for ln in runner.summary_lines([]) if "Good luck" in ln]
    assert greeting == ["  Good luck out there, Priya."]
    assert not any("example-user" in ln for ln in runner.summary_lines([]))


def test_summary_is_ascii_only(profile, config, tmp_path):
    """Windows consoles default to cp1252 and mangle anything else."""
    b = FakeBrowser()
    _wire(b, config, "604", "5+ years of experience.")
    audit = AuditLog(tmp_path / "data", run_id="quit-ascii")

    runner = LinkedInRunner(b, profile, config, audit,
                            approval_gate=lambda ev: True, dry_run=True)
    results = runner.run(prompt=lambda _: "")

    "\n".join(runner.summary_lines(results)).encode("ascii")


def test_summary_works_for_an_empty_run(profile, config, tmp_path):
    """A run that stopped before evaluating anything still reports cleanly."""
    audit = AuditLog(tmp_path / "data", run_id="quit-empty")
    runner = LinkedInRunner(FakeBrowser(), profile, config, audit, dry_run=True)

    text = "\n".join(runner.summary_lines([]))
    assert "Postings evaluated                 : 0" in text


# ---------------- the summary window ----------------

def test_show_summary_reports_failure_without_a_display(monkeypatch):
    """No display must mean "not shown", never a crash.

    The caller prints the summary first and uses the return value only to
    decide whether the window appeared, so a False here costs nothing.
    """
    import jobflow.adapters.quit_button as qb

    class _NoTk:
        def Tk(self):
            raise RuntimeError("no display")

    monkeypatch.setitem(__import__("sys").modules, "tkinter", _NoTk())
    assert qb.show_summary(["a", "b"]) is False


def test_show_summary_survives_a_broken_toolkit(monkeypatch):
    """A Tk that imports but fails to build a window must not raise."""
    import sys
    import jobflow.adapters.quit_button as qb

    class _BrokenTk:
        def Tk(self):
            raise Exception("display refused")

    monkeypatch.setitem(sys.modules, "tkinter", _BrokenTk())
    assert qb.show_summary(["summary"]) is False


def test_summary_lines_render_as_joinable_text(profile, config, tmp_path):
    """Whatever the window does, the lines must form printable text.

    The terminal copy is the one that always happens; the window is a
    convenience on top of it.
    """
    audit = AuditLog(tmp_path / "data", run_id="summary-text")
    runner = LinkedInRunner(FakeBrowser(), profile, config, audit, dry_run=True)

    lines = runner.summary_lines([])
    text = "\n".join(lines)

    assert isinstance(lines, list)
    assert all(isinstance(line, str) for line in lines)
    assert "JobFlow run summary" in text
