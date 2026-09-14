"""Evaluation pipeline.

Ordering is the design: cheap certain checks, then optional judgement, then
a human gate. Nothing is submitted without an explicit approval callback.

Author: Jashan Sadioura
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Protocol

from jobflow.core.audit import AuditLog
from jobflow.core.models import (
    Decision, Evaluation, Finding, JobPosting, Profile, SearchConfig, SkipReason,
)
from jobflow.core.screening import DeterministicScreener

log = logging.getLogger(__name__)


class Scorer(Protocol):
    """Structural type for the optional LLM layer."""
    enabled: bool
    def score(self, posting: JobPosting): ...
    def recommends_skip(self, fit) -> bool: ...


# Returns True to submit. The default refuses, so a caller that forgets to
# supply a gate cannot silently auto-submit.
ApprovalGate = Callable[[Evaluation], bool]


def deny_all(_: Evaluation) -> bool:
    return False


class Pipeline:
    def __init__(
        self,
        profile: Profile,
        config: SearchConfig,
        audit: AuditLog,
        scorer: Scorer | None = None,
        approval_gate: ApprovalGate = deny_all,
    ) -> None:
        self.profile = profile
        self.config = config
        self.audit = audit
        self.scorer = scorer
        self.approval_gate = approval_gate
        self.screener = DeterministicScreener(profile, config)
        self._applied_ids = audit.applied_job_ids()
        self._submitted_this_run = 0

    @property
    def submitted_count(self) -> int:
        return self._submitted_this_run

    @property
    def cap_reached(self) -> bool:
        return self._submitted_this_run >= self.config.daily_application_cap

    def evaluate(self, posting: JobPosting) -> Evaluation:
        """Screen, then optionally score. No submission happens here."""
        ev = self.screener.screen(posting, self._applied_ids)
        if ev.decision is Decision.SKIP:
            return ev

        if self.scorer is not None and self.scorer.enabled:
            fit = self.scorer.score(posting)
            if fit is not None:
                ev.fit = fit
                ev.findings.append(
                    Finding(
                        dimension="fit_assessment",
                        passed=not self.scorer.recommends_skip(fit),
                        detail=f"score {fit.score}: {fit.rationale}",
                        source="llm",
                        severity="warning" if self.scorer.recommends_skip(fit) else "info",
                    )
                )
                if self.scorer.recommends_skip(fit):
                    ev.decision = Decision.SKIP
                    ev.skip_reason = SkipReason.LOW_FIT_SCORE
        return ev

    def process(
        self,
        postings: Iterable[JobPosting],
        search_term: str = "",
        submit: Callable[[JobPosting], bool | str] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[Evaluation]:
        """Evaluate postings, gating every submission through a human.

        `submit` performs the actual application. It may return a plain bool,
        or a string naming why it failed -- the reason is recorded in the
        evidence log, so "why did this not apply?" is answerable later
        without re-running anything.
        """
        results: list[Evaluation] = []

        for posting in postings:
            # Checked between postings, never mid-application: stopping
            # halfway through a submission would leave a partly filled form
            # in an unknown state.
            if should_stop is not None and should_stop():
                break
            if self.cap_reached:
                ev = Evaluation(
                    posting=posting,
                    decision=Decision.SKIP,
                    skip_reason=SkipReason.CAP_REACHED,
                    findings=[Finding(
                        dimension="daily_cap",
                        passed=False,
                        detail=f"reached cap of {self.config.daily_application_cap}",
                        severity="blocker",
                    )],
                )
                self.audit.record(ev, search_term)
                results.append(ev)
                break

            ev = self.evaluate(posting)

            if ev.decision is Decision.APPLY:
                if not self.approval_gate(ev):
                    ev.decision = Decision.NEEDS_HUMAN
                    ev.findings.append(Finding(
                        dimension="human_approval",
                        passed=False,
                        detail="declined or deferred by reviewer",
                        severity="info",
                    ))
                elif submit is not None:
                    outcome = submit(ev.posting)
                    # A string is a failure reason; a bare bool carries none.
                    ok = outcome is True
                    reason = outcome if isinstance(outcome, str) else "submission failed"
                    ev.findings.append(Finding(
                        dimension="submission",
                        passed=ok,
                        detail="submitted" if ok else reason,
                        severity="info" if ok else "blocker",
                    ))
                    if ok:
                        self._submitted_this_run += 1
                        self._applied_ids.add(posting.job_id)
                    else:
                        ev.decision = Decision.NEEDS_HUMAN
                else:
                    ev.findings.append(Finding(
                        dimension="submission",
                        passed=True,
                        detail="dry run: approved but not submitted",
                        severity="info",
                    ))
                    self._applied_ids.add(posting.job_id)

            self.audit.record(ev, search_term)
            results.append(ev)
            log.info(ev.summary_line())

        return results


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
