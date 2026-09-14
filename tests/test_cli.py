"""Tests for the command line interface.

This is the file every user actually runs, so these cover the contract they
depend on: which flags exist, what exit code each command returns, and that
a configuration problem is reported rather than raised as a traceback.

The browser is never launched here -- `apply` is covered by its argument
parsing and its refusal paths only.

Author: Jashan Sadioura
"""

from __future__ import annotations

import pytest

from jobflow.cli import build_parser, main

VALID_PROFILE = """
identity:
  first_name: "Alex"
  last_name: "Doe"
  email: "alex@example.com"
  phone: "9999999999"
location:
  city: "Springfield"
  country: "India"
professional:
  years_of_experience: 6
  resume_path: "{resume}"
compensation:
  current_ctc: 1200000
  desired_salary: 1500000
  notice_period_days: 0
eligibility: {{}}
"""

VALID_SEARCH = """
daily_application_cap: 5
applications_per_term: 10
tiers:
  - name: "core"
    terms: ["Data Lead", "Analytics Manager"]
filters:
  location: "India"
"""


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Point the CLI at a throwaway config directory."""
    resume = tmp_path / "cv.pdf"
    resume.write_bytes(b"%PDF-1.4")
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "profile.yaml").write_text(
        VALID_PROFILE.format(resume=resume.as_posix()), encoding="utf-8")
    (cfg / "search.yaml").write_text(VALID_SEARCH, encoding="utf-8")

    import jobflow.core.config as config_mod
    monkeypatch.setattr(config_mod, "CONFIG_DIR", cfg)

    import jobflow.cli as cli_mod
    monkeypatch.setattr(cli_mod, "DATA_DIR", tmp_path / "data")
    return cfg


# ---------------- argument parsing ----------------

def test_every_command_is_registered():
    p = build_parser()
    for cmd in ("validate", "terms", "screen", "apply", "report"):
        assert p.parse_args([cmd] if cmd != "screen" else [cmd, "f.json"])


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_an_unknown_command_exits():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nonsense"])


@pytest.mark.parametrize("argv", [["-v", "apply"], ["apply", "-v"]])
def test_verbose_works_on_either_side_of_the_subcommand(argv):
    """argparse accepts -v in one position only unless it is inherited."""
    a = build_parser().parse_args(argv)
    assert bool(a.verbose or getattr(a, "_root_verbose", False)) is True


def test_apply_defaults_are_the_safe_ones():
    a = build_parser().parse_args(["apply"])
    assert a.live is False           # dry run unless asked
    assert a.review is False
    assert a.saved_profile is False
    assert a.use_llm is False


def test_apply_flags_parse():
    a = build_parser().parse_args([
        "apply", "--live", "--review", "--review-timeout", "300",
        "--use-llm", "--profile-dir", "/tmp/p", "--saved-profile",
        "--keep-open", "--yes",
    ])
    assert a.live and a.review and a.saved_profile and a.keep_open and a.yes
    assert a.review_timeout == 300
    assert a.profile_dir == "/tmp/p"


# ---------------- exit codes ----------------

def test_validate_succeeds_on_good_config(config_dir, capsys):
    assert main(["validate"]) == 0
    out = capsys.readouterr().out
    assert "Configuration is valid" in out
    assert "Alex Doe" in out


def test_terms_lists_the_search_plan(config_dir, capsys):
    assert main(["terms"]) == 0
    out = capsys.readouterr().out
    assert "Data Lead" in out
    assert "Analytics Manager" in out
    assert "core" in out


def test_validate_reports_a_broken_config_without_a_traceback(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "profile.yaml").write_text("identity: {}", encoding="utf-8")
    (cfg / "search.yaml").write_text("tiers: []", encoding="utf-8")
    import jobflow.core.config as config_mod
    monkeypatch.setattr(config_mod, "CONFIG_DIR", cfg)

    assert main(["validate"]) == 1
    assert "Configuration error" in capsys.readouterr().err


def test_screen_rejects_a_missing_file(config_dir, capsys):
    assert main(["screen", "does-not-exist.json"]) == 1
    assert "No such file" in capsys.readouterr().err


def test_screen_rejects_invalid_json(config_dir, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main(["screen", str(bad)]) == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_screen_rejects_a_json_object(config_dir, tmp_path, capsys):
    """The file must be a list of postings, not a single object."""
    bad = tmp_path / "obj.json"
    bad.write_text('{"job_id": "1"}', encoding="utf-8")
    assert main(["screen", str(bad)]) == 1
    assert "array of postings" in capsys.readouterr().err


def test_screen_evaluates_postings(config_dir, tmp_path, capsys):
    good = tmp_path / "jobs.json"
    good.write_text(
        '[{"job_id": "1", "title": "Data Lead", "company": "Acme",'
        ' "description": "3+ years of experience."}]', encoding="utf-8")

    assert main(["screen", str(good)]) == 0
    out = capsys.readouterr().out
    assert "1 evaluated" in out


def test_report_runs_on_an_empty_log(config_dir, capsys):
    assert main(["report"]) == 0
    assert "Applications on record" in capsys.readouterr().out
