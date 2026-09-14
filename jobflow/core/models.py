"""Typed configuration and domain models.

Every value that reaches an application form passes through these models, so
validation happens once at load time rather than being rediscovered halfway
through a run.

Author: Jashan Sadioura
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


# --------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------

class Identity(BaseModel):
    first_name: str = Field(min_length=1)
    middle_name: str = ""
    last_name: str = Field(min_length=1)
    email: EmailStr
    phone: str = Field(min_length=6)
    phone_country_code: str = "+91"
    linkedin_url: str = ""
    website: str = ""

    @field_validator("phone")
    @classmethod
    def _digits_only(cls, v: str) -> str:
        digits = "".join(c for c in v if c.isdigit())
        if not 6 <= len(digits) <= 15:
            raise ValueError(
                f"phone must contain 6-15 digits, got {len(digits)} in {v!r}"
            )
        return digits

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.middle_name, self.last_name]
        return " ".join(p for p in parts if p)


class Location(BaseModel):
    city: str
    street: str = ""
    state: str = ""
    zipcode: str = ""
    country: str


class Professional(BaseModel):
    headline: str = ""
    years_of_experience: int = Field(ge=0, le=60)
    has_masters: bool = False
    recent_employer: str = ""
    resume_path: Path
    summary: str = ""

    @field_validator("resume_path")
    @classmethod
    def _resume_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(
                f"resume not found at {v}\n"
                "Fix professional.resume_path in config/profile.yaml."
            )
        if not v.is_file():
            raise ValueError(f"resume_path is not a file: {v}")
        if v.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            raise ValueError(f"resume must be PDF or Word, got {v.suffix!r}")
        return v


class Compensation(BaseModel):
    current_ctc: int = Field(ge=0)
    desired_salary: int = Field(ge=0)
    currency: str = "INR"
    notice_period_days: int = Field(ge=0, le=365)

    # Salary questions arrive in inconsistent units; precompute each form so
    # the answer layer never does arithmetic inline.
    @property
    def desired_lakhs(self) -> str:
        return f"{self.desired_salary / 100_000:.2f}"

    @property
    def current_lakhs(self) -> str:
        return f"{self.current_ctc / 100_000:.2f}"

    @property
    def desired_monthly(self) -> str:
        # Integer rupees: many forms reject decimals.
        return str(round(self.desired_salary / 12))

    @property
    def current_monthly(self) -> str:
        return str(round(self.current_ctc / 12))

    @property
    def notice_period_months(self) -> str:
        return str(round(self.notice_period_days / 30))

    @property
    def notice_period_weeks(self) -> str:
        return str(round(self.notice_period_days / 7))


class Eligibility(BaseModel):
    requires_visa_sponsorship: bool = False
    work_authorization: str = ""
    has_security_clearance: bool = False


class Demographics(BaseModel):
    gender: str = ""
    ethnicity: str = ""
    disability_status: str = ""
    veteran_status: str = ""


class Defaults(BaseModel):
    confidence_level: int = Field(default=8, ge=1, le=10)


class Profile(BaseModel):
    identity: Identity
    location: Location
    professional: Professional
    compensation: Compensation
    eligibility: Eligibility
    demographics: Demographics = Demographics()
    defaults: Defaults = Defaults()

    @property
    def experience_ceiling(self) -> int:
        """Highest 'years required' still worth applying to."""
        buffer = 2 if self.professional.has_masters else 0
        return self.professional.years_of_experience + buffer


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

class SearchTier(BaseModel):
    name: str
    terms: list[str] = Field(min_length=1)


class SearchFilters(BaseModel):
    location: str = ""
    date_posted: Literal[
        "Any time", "Past month", "Past week", "Past 24 hours"
    ] = "Past week"
    easy_apply_only: bool = True
    experience_levels: list[str] = []
    job_types: list[str] = []
    workplace_types: list[str] = []


class Exclusions(BaseModel):
    description_blockers: list[str] = []
    clearance_blockers: list[str] = []
    companies: list[str] = []


class ExperienceGate(BaseModel):
    enabled: bool = True
    tolerance_years: int = Field(default=2, ge=0)


class SearchConfig(BaseModel):
    daily_application_cap: int = Field(default=40, ge=1)
    applications_per_term: int = Field(default=30, ge=1)
    tiers: list[SearchTier] = Field(min_length=1)
    filters: SearchFilters = SearchFilters()
    exclusions: Exclusions = Exclusions()
    experience_gate: ExperienceGate = ExperienceGate()

    @property
    def ordered_terms(self) -> list[tuple[str, str]]:
        """(term, tier_name) pairs in tier order — priority is positional."""
        return [(t, tier.name) for tier in self.tiers for t in tier.terms]


# --------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------

class Decision(str, Enum):
    """Outcome of evaluating one posting."""
    APPLY = "apply"
    SKIP = "skip"
    NEEDS_HUMAN = "needs_human"


class SkipReason(str, Enum):
    ALREADY_APPLIED = "already_applied"
    BLOCKED_PHRASE = "blocked_phrase"
    CLEARANCE_REQUIRED = "clearance_required"
    EXPERIENCE_TOO_HIGH = "experience_too_high"
    COMPANY_EXCLUDED = "company_excluded"
    LOW_FIT_SCORE = "low_fit_score"
    NO_EASY_APPLY = "no_easy_apply"
    CAP_REACHED = "cap_reached"


class JobPosting(BaseModel):
    job_id: str
    title: str
    company: str
    location: str = ""
    url: str = ""
    description: str = ""
    workplace_type: str = ""
    experience_required: int | None = None

    @property
    def description_lower(self) -> str:
        return self.description.lower()


class Finding(BaseModel):
    """One piece of evidence behind a decision.

    Deterministic checks and LLM workers emit the same shape, so the audit
    log reads uniformly regardless of which layer produced the finding.
    """
    dimension: str
    passed: bool
    detail: str
    source: Literal["deterministic", "llm"] = "deterministic"
    severity: Literal["info", "warning", "blocker"] = "info"


class FitAssessment(BaseModel):
    """Advisory score from the LLM layer. Never blocks on its own."""
    score: int = Field(ge=0, le=100)
    rationale: str
    matched_strengths: list[str] = []
    gaps: list[str] = []

    @model_validator(mode="after")
    def _require_rationale(self) -> "FitAssessment":
        if not self.rationale.strip():
            raise ValueError("fit assessment must include a rationale")
        return self


class Evaluation(BaseModel):
    """Complete record for one posting: the decision and why."""
    posting: JobPosting
    decision: Decision
    skip_reason: SkipReason | None = None
    findings: list[Finding] = []
    fit: FitAssessment | None = None

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocker"]

    def summary_line(self) -> str:
        head = f"[{self.decision.value.upper()}] {self.posting.title} @ {self.posting.company}"
        if self.skip_reason:
            head += f" — {self.skip_reason.value}"
        if self.fit:
            head += f" (fit {self.fit.score})"
        return head


# ---------------------------------------------------------------------------
#
# Thanks for using JobFlow.
#
# Built because an application carries your name, so the tool should stop and
# ask rather than guess. If it helped your search, pass it on to someone else
# who is looking.
#
# Jashan Sadioura  ·  https://www.linkedin.com/in/sadioura-jashan/
#
# ---------------------------------------------------------------------------
