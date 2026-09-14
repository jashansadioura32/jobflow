"""Tests for configuration loading.

Config loading is the first thing that runs on every command, and its error
messages are the first thing a new user sees. These pin the messages as much
as the behaviour: "Missing config file" with no further guidance is how a
fresh clone becomes a support question.

Author: Jashan Sadioura
"""

from __future__ import annotations

import pytest

from jobflow.core.config import (
    ConfigError, load_all, load_profile, load_search_config,
)

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
    terms: ["Data Lead"]
filters:
  location: "India"
"""


@pytest.fixture
def resume(tmp_path):
    f = tmp_path / "cv.pdf"
    f.write_bytes(b"%PDF-1.4")
    return f


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


# ---------------- the happy path ----------------

def test_loads_a_valid_profile(tmp_path, resume):
    p = _write(tmp_path / "profile.yaml",
               VALID_PROFILE.format(resume=resume.as_posix()))
    profile = load_profile(p)
    assert profile.identity.first_name == "Alex"
    assert profile.professional.years_of_experience == 6


def test_loads_a_valid_search_config(tmp_path):
    s = _write(tmp_path / "search.yaml", VALID_SEARCH)
    cfg = load_search_config(s)
    assert cfg.daily_application_cap == 5
    assert cfg.ordered_terms == [("Data Lead", "core")]


def test_load_all_returns_both(tmp_path, resume):
    p = _write(tmp_path / "profile.yaml",
               VALID_PROFILE.format(resume=resume.as_posix()))
    s = _write(tmp_path / "search.yaml", VALID_SEARCH)
    profile, cfg = load_all(p, s)
    assert profile.identity.last_name == "Doe"
    assert cfg.applications_per_term == 10


# ---------------- the messages a new user sees ----------------

def test_missing_file_points_at_the_example(tmp_path):
    """A fresh clone has only the .example files; say what to copy."""
    _write(tmp_path / "profile.example.yaml", "identity: {}")

    with pytest.raises(ConfigError) as e:
        load_profile(tmp_path / "profile.yaml")

    msg = str(e.value)
    assert "Missing config file: profile.yaml" in msg
    assert "profile.example.yaml profile.yaml" in msg     # the copy command
    assert "Windows" in msg                                # both platforms
    assert str(tmp_path) in msg                            # where to do it


def test_missing_file_without_an_example_is_still_clear(tmp_path):
    with pytest.raises(ConfigError) as e:
        load_profile(tmp_path / "profile.yaml")
    assert "Missing config file" in str(e.value)


def test_broken_yaml_names_the_file(tmp_path):
    p = _write(tmp_path / "profile.yaml", "identity:\n  first_name: 'x'\n bad indent\n")
    with pytest.raises(ConfigError) as e:
        load_profile(p)
    assert "profile.yaml is not valid YAML" in str(e.value)


def test_a_yaml_list_is_rejected(tmp_path):
    """A top-level list is valid YAML but not a config."""
    p = _write(tmp_path / "profile.yaml", "- one\n- two\n")
    with pytest.raises(ConfigError) as e:
        load_profile(p)
    assert "must contain a YAML mapping" in str(e.value)


def test_validation_errors_name_every_bad_field(tmp_path):
    """The message must say which lines to fix, not just that it failed."""
    p = _write(tmp_path / "profile.yaml", """
identity:
  first_name: "Alex"
  last_name: "Doe"
  email: "not-an-email"
  phone: "123"
location:
  city: "Springfield"
  country: "India"
professional:
  years_of_experience: 6
  resume_path: "/nowhere/missing.pdf"
compensation:
  current_ctc: 1200000
  desired_salary: 1500000
  notice_period_days: 0
eligibility: {}
""")
    with pytest.raises(ConfigError) as e:
        load_profile(p)

    msg = str(e.value)
    assert "failed validation" in msg
    assert "identity.email" in msg          # bad address
    assert "resume_path" in msg             # missing file


def test_a_missing_resume_is_caught_at_load_time(tmp_path):
    """Checked at startup rather than mid-run, when a form needs it."""
    p = _write(tmp_path / "profile.yaml",
               VALID_PROFILE.format(resume="/definitely/not/here.pdf"))
    with pytest.raises(ConfigError) as e:
        load_profile(p)
    assert "resume" in str(e.value).lower()


def test_search_config_requires_at_least_one_tier(tmp_path):
    s = _write(tmp_path / "search.yaml", "daily_application_cap: 5\ntiers: []\n")
    with pytest.raises(ConfigError) as e:
        load_search_config(s)
    assert "tiers" in str(e.value)
