"""JobFlow command line interface.

  jobflow validate         check configuration and exit
  jobflow terms            print the search plan in priority order
  jobflow screen <file>    evaluate postings from a JSON file (no browser)
  jobflow report           summarize the audit log

Author: Jashan Sadioura
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from jobflow.core.audit import AuditLog
from jobflow.core.config import ConfigError, load_all
from jobflow.core.models import Decision, Evaluation, JobPosting
from jobflow.core.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )


def _print_evaluation(ev: Evaluation) -> None:
    print(f"\n{'=' * 72}")
    print(f"{ev.posting.title}")
    print(f"{ev.posting.company} — {ev.posting.location or 'location not listed'}")
    if ev.posting.url:
        print(f"{ev.posting.url}")
    print(f"{'-' * 72}")
    for f in ev.findings:
        mark = {"blocker": "BLOCK", "warning": "WARN ", "info": "ok   "}[f.severity]
        tag = "llm" if f.source == "llm" else "det"
        print(f"  [{mark}] ({tag}) {f.dimension}: {f.detail}")
    if ev.fit:
        print(f"{'-' * 72}")
        print(f"  fit score: {ev.fit.score}/100 — {ev.fit.rationale}")
        if ev.fit.matched_strengths:
            print(f"  strengths: {', '.join(ev.fit.matched_strengths)}")
        if ev.fit.gaps:
            print(f"  gaps     : {', '.join(ev.fit.gaps)}")
    print(f"{'=' * 72}")


def interactive_gate(ev: Evaluation) -> bool:
    """Ask before every submission. Anything but an explicit 'y' declines."""
    _print_evaluation(ev)
    try:
        reply = input("Submit this application? [y/N/q] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nDeclined (no input).")
        return False
    if reply == "q":
        raise KeyboardInterrupt("user quit at review gate")
    return reply == "y"


def cmd_validate(args) -> int:
    try:
        profile, search = load_all()
    except ConfigError as e:
        print(f"Configuration error:\n{e}", file=sys.stderr)
        return 1

    print("Configuration is valid.\n")
    print(f"  candidate      : {profile.identity.full_name}")
    print(f"  email          : {profile.identity.email}")
    print(f"  phone          : {profile.identity.phone_country_code} {profile.identity.phone}")
    print(f"  location       : {profile.location.city}, {profile.location.country}")
    print(f"  experience     : {profile.professional.years_of_experience}y "
          f"(ceiling {profile.experience_ceiling}y"
          f"{', incl. masters buffer' if profile.professional.has_masters else ''})")
    print(f"  resume         : {profile.professional.resume_path.name}")
    print(f"  current CTC    : {profile.compensation.current_ctc:,} "
          f"({profile.compensation.current_lakhs} lakhs)")
    print(f"  desired salary : {profile.compensation.desired_salary:,} "
          f"({profile.compensation.desired_lakhs} lakhs)")
    print(f"  notice period  : {profile.compensation.notice_period_days} days")
    print()
    print(f"  search terms   : {len(search.ordered_terms)} across {len(search.tiers)} tiers")
    print(f"  location filter: {search.filters.location}")
    print(f"  date posted    : {search.filters.date_posted}")
    print(f"  daily cap      : {search.daily_application_cap}")
    return 0


def cmd_terms(args) -> int:
    try:
        _, search = load_all()
    except ConfigError as e:
        print(f"Configuration error:\n{e}", file=sys.stderr)
        return 1
    print("Search plan (evaluated in this order):\n")
    n = 0
    for tier in search.tiers:
        print(f"  {tier.name}")
        for term in tier.terms:
            n += 1
            print(f"    {n:2}. {term}")
        print()
    print(f"{n} terms; up to {search.applications_per_term} applications each, "
          f"capped at {search.daily_application_cap} per run.")
    return 0


def cmd_screen(args) -> int:
    """Evaluate postings from a JSON file. No browser, no submissions."""
    try:
        profile, search = load_all()
    except ConfigError as e:
        print(f"Configuration error:\n{e}", file=sys.stderr)
        return 1

    path = Path(args.file)
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 1

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"{path.name} is not valid JSON: {e}", file=sys.stderr)
        return 1
    if not isinstance(raw, list):
        print("Expected a JSON array of postings.", file=sys.stderr)
        return 1

    try:
        postings = [JobPosting.model_validate(r) for r in raw]
    except Exception as e:
        print(f"Invalid posting in {path.name}: {e}", file=sys.stderr)
        return 1

    audit = AuditLog(DATA_DIR)
    scorer = None
    if args.use_llm:
        try:
            from openai import OpenAI
            from jobflow.workers.fit_scorer import FitScorer
            scorer = FitScorer(profile, client=OpenAI(), model=args.model,
                               min_score=args.min_score)
            print(f"Fit scoring enabled ({args.model}, threshold {args.min_score}).")
        except Exception as e:
            print(f"Fit scoring unavailable, continuing without it: {e}")

    gate = interactive_gate if args.interactive else (lambda ev: False)
    pipeline = Pipeline(profile, search, audit, scorer=scorer, approval_gate=gate)

    try:
        results = pipeline.process(postings, search_term=args.term)
    except KeyboardInterrupt:
        print("\nStopped at review gate.")
        return 130

    applied = sum(r.decision is Decision.APPLY for r in results)
    skipped = sum(r.decision is Decision.SKIP for r in results)
    pending = sum(r.decision is Decision.NEEDS_HUMAN for r in results)

    print(f"\n{len(results)} evaluated: {applied} would apply, "
          f"{skipped} skipped, {pending} awaiting review")
    print(f"Evidence log: {audit.evidence_path}")

    if skipped:
        print("\nSkip reasons:")
        for reason, count in sorted(audit.summarize().items()):
            if reason not in {"apply", "needs_human"}:
                print(f"  {count:3}  {reason}")
    _signoff()
    return 0



def cmd_apply(args) -> int:
    """Search LinkedIn, screen results, and apply with a human gate."""
    try:
        profile, search = load_all()
    except ConfigError as e:
        print("Configuration error:\n" + str(e), file=sys.stderr)
        return 1

    try:
        from jobflow.adapters.browser import SeleniumBrowser
        from jobflow.adapters.runner import LinkedInRunner
    except ImportError as e:
        print("Browser support requires selenium: "
              "pip install -e '.[browser]'\n" + str(e), file=sys.stderr)
        return 1

    if args.live and not args.yes:
        if args.review:
            print("LIVE MODE — each application is filled and left open on LinkedIn.")
            print("You click Submit yourself; closing the dialog skips the posting.")
        else:
            print("LIVE MODE + AUTO-SUBMIT — applications are SENT automatically")
            print("under your name, with no review. Questions that cannot be")
            print("mapped from your profile are GUESSED rather than skipped.")
            print("Use --review to approve each one in the browser instead.")
        print("Automating applications may conflict with the platform's terms of service.")
        if input("Type 'live' to continue: ").strip().lower() != "live":
            print("Aborted.")
            return 1

    audit = AuditLog(DATA_DIR)
    scorer = None
    if args.use_llm:
        try:
            from openai import OpenAI
            from jobflow.workers.fit_scorer import FitScorer
            scorer = FitScorer(profile, client=OpenAI(), model=args.model,
                               min_score=args.min_score)
            print(f"Fit scoring enabled ({args.model}, threshold {args.min_score}).")
        except Exception as e:
            print(f"Fit scoring unavailable, continuing without it: {e}")

    from jobflow.adapters.quit_button import QuitButton
    quit_button = QuitButton().start()

    browser = SeleniumBrowser.launch(
        headless=False,                    # sign-in and review need a visible window
        profile_dir=args.profile_dir,
        # A dedicated profile is used in place: the session persists, so
        # sign-in is a one-time cost rather than every run.
        clone_profile=not args.saved_profile,
    )
    # Built before the try so the summary survives a Ctrl+C or a crash: a
    # run that stopped early is exactly the one whose numbers you want.
    runner = LinkedInRunner(
        browser, profile, search, audit,
        scorer=scorer,
        approval_gate=lambda ev: True,
        dry_run=not args.live,
        review_timeout=args.review_timeout,
        auto_submit=not args.review,
        quit_button=quit_button,
    )
    results: list = []
    code = 0
    try:
        results = runner.run()
    except KeyboardInterrupt:
        runner.stop_reason = "you pressed Ctrl+C"
        code = 130
    finally:
        quit_button.stop()
        if not args.keep_open:
            browser.quit()

    summary = runner.summary_lines(results)
    # Printed first, always: a popup that cannot open (no display, remote
    # session) must never be the only copy of the summary.
    for line in summary:
        print(line)

    from jobflow.adapters.quit_button import show_summary
    show_summary(summary)
    return code


def _signoff() -> None:
    """Closing note shown at the end of a run.

    Deliberately carries no author name or link: this prints on the screen of
    whoever runs the tool, and a shared repo should not greet its users with
    someone else's contact details. Attribution lives in the README, the
    LICENSE, and the module headers.
    """
    print()
    # ASCII only: Windows consoles default to cp1252 and mangle em dashes.
    print("Good luck out there. If JobFlow helped, pass it on to someone")
    print("else who is job hunting - that's the only thanks the author is after.")


def cmd_report(args) -> int:
    audit = AuditLog(DATA_DIR)
    ids = audit.applied_job_ids()
    print(f"Applications on record: {len(ids)}")
    print(f"Log: {audit.applications_path}")
    if not audit.applications_path.exists():
        print("No applications yet.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jobflow",
        description="Screen and apply to job postings with a human in the loop.",
    )
    # -v is defined once, on a parent every subcommand inherits, so it works
    # before or after the subcommand name. Defining it on both the top-level
    # parser and the subparser does not work: they share one namespace, and
    # the subparser's default silently overwrites a flag given earlier.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true",
                        help="verbose logging")

    p.add_argument("-v", "--verbose", action="store_true",
                   help=argparse.SUPPRESS, dest="_root_verbose")

    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="check configuration",
                   parents=[common]).set_defaults(func=cmd_validate)
    sub.add_parser("terms", help="print the search plan",
                   parents=[common]).set_defaults(func=cmd_terms)

    s = sub.add_parser("screen", help="evaluate postings from a JSON file",
                       parents=[common])
    s.add_argument("file", help="JSON array of postings")
    s.add_argument("--term", default="", help="search term to record in the log")
    s.add_argument("--interactive", action="store_true",
                   help="prompt before each application")
    s.add_argument("--use-llm", action="store_true", help="enable fit scoring")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--min-score", type=int, default=50)
    s.set_defaults(func=cmd_screen)

    a = sub.add_parser("apply", help="search and apply on LinkedIn",
                       parents=[common])
    a.add_argument("--live", action="store_true",
                   help="actually submit (default is a dry run)")
    a.add_argument("--yes", action="store_true", help="skip the live-mode confirmation")
    a.add_argument("--review", action="store_true",
                   help="review each application in the browser and click Submit "
                        "yourself, instead of submitting automatically")
    a.add_argument("--review-timeout", type=float, default=600.0,
                   help="with --review, seconds an application waits for your "
                        "decision before being skipped (default 600)")
    a.add_argument("--use-llm", action="store_true", help="enable fit scoring")
    a.add_argument("--model", default="gpt-4o-mini")
    a.add_argument("--min-score", type=int, default=50)
    a.add_argument("--profile-dir", default=None,
                   help="optional Chrome profile override; defaults to a copy of the logged-in Default Chrome profile")
    a.add_argument("--saved-profile", action="store_true",
                   help="use --profile-dir in place instead of copying it, so the "
                        "sign-in persists between runs (sign in once)")
    a.add_argument("--keep-open", action="store_true",
                   help="leave the browser open when the run ends")
    a.set_defaults(func=cmd_apply)

    sub.add_parser("report", help="summarize the audit log",
                   parents=[common]).set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # -v is accepted on either side of the subcommand; either one enables it.
    args.verbose = bool(args.verbose or getattr(args, "_root_verbose", False))
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


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
