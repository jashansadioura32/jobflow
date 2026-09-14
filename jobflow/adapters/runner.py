"""End-to-end run: search, screen, review, apply.

Composes the adapter with the pipeline. Terms are worked in tier order so
the closest-matching roles consume the daily budget first.

Author: Jashan Sadioura
"""

from __future__ import annotations

import logging

from jobflow.adapters.browser import Browser
from jobflow.adapters.linkedin import EasyApplyFiller, LinkedInSearch, LinkedInSession
from jobflow.core.audit import AuditLog
from jobflow.core.models import (
    Decision, Finding, JobPosting, Profile, SearchConfig,
)
from jobflow.core.pipeline import ApprovalGate, Pipeline, deny_all

log = logging.getLogger(__name__)


class LinkedInRunner:
    def __init__(
        self,
        browser: Browser,
        profile: Profile,
        config: SearchConfig,
        audit: AuditLog,
        scorer=None,
        approval_gate: ApprovalGate = deny_all,
        dry_run: bool = True,
        review_timeout: float = 600.0,
        auto_submit: bool = True,
        quit_button=None,
    ) -> None:
        self.browser = browser
        self.profile = profile
        self.config = config
        self.audit = audit
        self.dry_run = dry_run
        # How long a filled application waits for a decision in the browser.
        self.review_timeout = review_timeout
        self.auto_submit = auto_submit
        self.session = LinkedInSession(browser)
        self.search = LinkedInSearch(browser, config)
        # Guessing only makes sense when nobody is watching to be asked.
        unattended = auto_submit and not dry_run
        # The LLM answerer reuses the scorer's client rather than building a
        # second one, and is only supplied when unattended: under --review a
        # person answers the question, which beats a generated answer.
        answerer = None
        if unattended and scorer is not None and getattr(scorer, "enabled", False):
            from jobflow.core.resume import ResumeText
            from jobflow.workers.question_answerer import QuestionAnswerer
            answerer = QuestionAnswerer(
                profile, client=scorer.client, model=scorer.model,
                # Read lazily and once: extraction is wasted on a run where
                # every question maps cleanly.
                resume_text=ResumeText(profile.professional.resume_path),
            )
        self.filler = EasyApplyFiller(browser, profile,
                                      guess_unmapped=unattended,
                                      answerer=answerer)
        self.pipeline = Pipeline(
            profile, config, audit, scorer=scorer, approval_gate=approval_gate
        )
        self.limit_reached = False
        if quit_button is None:
            from jobflow.adapters.quit_button import NullQuitButton
            quit_button = NullQuitButton()
        self.quit_button = quit_button
        # Why the run ended, for the closing summary.
        self.stop_reason = "all search terms completed"

    # -- submission ------------------------------------------------------

    def _submit(self, posting: JobPosting) -> bool | str:
        """Fill the application and send it.

        In live mode with `auto_submit` the filler clicks Submit itself and
        returns True when LinkedIn accepted it. With `auto_submit` off the
        form is left open on the Submit step for a person to decide.

        In dry-run the modal is filled and then discarded, so the answer
        mapping can be checked without sending anything.
        """
        send = self.auto_submit and not self.dry_run
        # Gives the LLM worker the job description as context for any
        # question the profile cannot answer.
        self.filler.current_posting = posting
        report = self.filler.run(submit=send)

        if report.unanswered:
            log.warning(
                "%s: %d unanswered question(s): %s",
                posting.job_id, len(report.unanswered),
                "; ".join(report.unanswered[:3]),
            )
        if report.ai_answered:
            # Written by a model, not taken from your profile. Listed
            # separately from crude guesses: different failure modes,
            # different scrutiny.
            log.warning("%s: AI answered %d question(s): %s",
                        posting.job_id, len(report.ai_answered),
                        "; ".join(f"{k} -> {v}"
                                  for k, v in report.ai_answered.items()))
        if report.guessed:
            # Guesses are the answers you did not choose; name every one.
            log.warning("%s: guessed %d answer(s): %s",
                        posting.job_id, len(report.guessed),
                        "; ".join(f"{k} -> {v}" for k, v in report.guessed.items()))
        if report.aborted_reason:
            log.warning("%s: %s", posting.job_id, report.aborted_reason)

        if self.filler.daily_limit_reached():
            log.warning("LinkedIn daily application limit reached.")
            self.limit_reached = True

        # A form that could not be completed honestly is never offered for
        # submission -- that is the property the whole design protects.
        if report.needs_human or not report.reached_submit:
            reason = (report.aborted_reason
                      or (f"{len(report.unanswered)} unanswered: "
                          + "; ".join(report.unanswered[:3])
                          if report.unanswered else "did not reach the submit step"))
            if self.dry_run:
                self.filler.dismiss()
            else:
                # Live: leave it open. Discarding hides what went wrong, and
                # a form the automation could not finish is exactly the one
                # a person should be looking at.
                log.warning("%s: %s -- leaving the dialog open for you.",
                            posting.job_id, reason)
                outcome = self.filler.wait_for_human(timeout=self.review_timeout)
                if outcome == "submitted":
                    log.info("  Submitted by you.")
                    return True
                self.filler.dismiss()
            return reason

        if self.dry_run:
            self.filler.dismiss()
            return True

        if self.auto_submit:
            if report.submitted:
                log.info("  Submitted.")
                return True
            self.filler.dismiss()
            return "reached submit but the click did not register"

        log.info("")
        log.info("  Review it in the browser: click Submit to apply, or close")
        log.info("  the dialog to skip. Waiting...")
        outcome = self.filler.wait_for_human(timeout=self.review_timeout)

        if outcome == "timeout":
            log.warning("%s: no decision after %.0fs; skipping.",
                        posting.job_id, self.review_timeout)
            self.filler.dismiss()
            return f"no decision within {self.review_timeout:.0f}s"

        log.info("  %s.", "Submitted" if outcome == "submitted" else "Skipped")
        return True if outcome == "submitted" else "closed without submitting"

    # -- run -------------------------------------------------------------

    def run(self, prompt=input) -> list:
        # Stated up front: the difference between filling forms and sending
        # them is not something a person should have to infer mid-run.
        if self.dry_run:
            log.info("=" * 68)
            log.info("DRY RUN - every application is filled and then DISCARDED.")
            log.info("Nothing is submitted. Re-run with --live to apply.")
            log.info("=" * 68)
        elif self.auto_submit:
            log.info("=" * 68)
            log.info("LIVE + AUTO-SUBMIT - applications are filled and SENT")
            log.info("automatically under your name, with no review step.")
            log.info("Unmapped required questions are GUESSED, not skipped.")
            log.info("=" * 68)
        else:
            log.info("=" * 68)
            log.info("LIVE - each filled application waits for you to click")
            log.info("Submit in the browser. Close the dialog to skip it.")
            log.info("=" * 68)

        if not self.session.ensure_signed_in(prompt=prompt):
            log.error("Not signed in; aborting.")
            return []

        all_results: list = []

        for term, tier in self.config.ordered_terms:
            if self.quit_button.quit_requested:
                self.stop_reason = "you clicked Quit"
                break
            if self.pipeline.cap_reached:
                self.stop_reason = (
                    f"daily cap of {self.config.daily_application_cap} reached")
                break
            if self.limit_reached:
                self.stop_reason = "LinkedIn's own daily application limit was reached"
                break

            log.info("")
            log.info("=== %s  (tier: %s) ===", term, tier)

            found = self.search.collect(
                term, limit=self.config.applications_per_term
            )
            if not found:
                continue

            # Descriptions are fetched lazily, one posting at a time, so a
            # run that hits the cap early has not paid for the rest. Each
            # posting is opened by clicking its card, which loads the detail
            # pane the Easy Apply button lives in.
            enriched: list[JobPosting] = []
            # Applied already, or opened before and found to be external-apply:
            # either way there is nothing to gain by clicking into it again.
            # Jobs screened out by the rules are deliberately NOT remembered,
            # so raising your experience ceiling brings them straight back.
            known = self.audit.skip_on_sight_ids()
            for p, card in found:
                if p.job_id in known:
                    log.info("Skipping %s: already applied or cannot be "
                             "applied to via Easy Apply.", p.job_id)
                    continue
                try:
                    enriched.append(self.search.load_description(p, card))
                except Exception as e:
                    log.warning("Could not load %s: %s", p.job_id, e)

            results = self.pipeline.process(
                enriched, search_term=term, submit=self._submit,
                should_stop=lambda: self.quit_button.quit_requested,
            )
            all_results.extend(results)

            if self.quit_button.quit_requested:
                self.stop_reason = "you clicked Quit"
                break
            if self.limit_reached:
                self.stop_reason = "LinkedIn's own daily application limit was reached"
                break
            if self.pipeline.cap_reached:
                self.stop_reason = (
                    f"daily cap of {self.config.daily_application_cap} reached")
                break

        log.info("")
        verb = "completed (dry run, nothing sent)" if self.dry_run else "submitted"
        log.info("Run complete: %d evaluated, %d %s.",
                 len(all_results), self.pipeline.submitted_count, verb)
        return all_results

    # -- summary ---------------------------------------------------------

    def summary_lines(self, results: list) -> list[str]:
        """The closing report: what happened, why it ended, and a sign-off."""
        applied = sum(r.decision is Decision.APPLY for r in results)
        skipped = sum(r.decision is Decision.SKIP for r in results)
        pending = sum(r.decision is Decision.NEEDS_HUMAN for r in results)
        sent = self.pipeline.submitted_count
        word = "filled (dry run, nothing sent)" if self.dry_run else "submitted"

        bar = "=" * 68
        lines = [
            "",
            bar,
            "  JobFlow run summary",
            bar,
            f"  Stopped because : {self.stop_reason}",
            "",
            f"  Applications {word:<22}: {sent}",
            f"  Postings evaluated                 : {len(results)}",
            f"  Passed screening                   : {applied}",
            f"  Skipped by the rules               : {skipped}",
            f"  Needing your attention             : {pending}",
        ]

        reasons = {k: v for k, v in self.audit.summarize().items()
                   if k not in {"apply", "needs_human"}}
        if reasons:
            lines.append("")
            lines.append("  Why postings were skipped:")
            for reason, count in sorted(reasons.items()):
                lines.append(f"    {count:3}  {reason}")

        lines += [
            "",
            f"  Evidence log : {self.audit.evidence_path}",
            f"  Applications : {self.audit.applications_path}",
            bar,
            "",
            f"  Good luck out there, {self.profile.identity.first_name}.",
            "",
            "  If JobFlow helped, pass it on to someone else who is job",
            "  hunting - that's the only thanks the author is after.",
            bar,
        ]
        return lines


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
