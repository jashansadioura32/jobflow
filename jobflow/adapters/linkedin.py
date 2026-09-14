"""LinkedIn adapter.

Three responsibilities, kept separate so each is testable:
  * `LinkedInSession`  — sign-in state; never handles a password.
  * `LinkedInSearch`   — search URLs, first-page collection, posting extraction.
  * `EasyApplyFiller`  — multi-step form traversal via AnswerResolver.

Selectors live in one block at the top. They are the first thing to break
when LinkedIn ships a redesign, so they are not scattered through the logic.

Author: Jashan Sadioura
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urlencode

from jobflow.adapters.browser import Browser, Element
from jobflow.core.answers import AnswerResolver
from jobflow.core.models import JobPosting, Profile, SearchConfig

log = logging.getLogger(__name__)

BASE = "https://www.linkedin.com"

# --------------------------------------------------------------------------
# Selectors — update these first when the UI changes.
# --------------------------------------------------------------------------

SEL = {
    "signin_link": "a[href*='/login'], button.sign-in-form__submit-btn",
    "feed_marker": "div.feed-identity-module, button[aria-label*='Start a post']",
    "nav_me": "button[data-control-name='nav.settings'], img.global-nav__me-photo",

    "result_card": "li.jobs-search-results__list-item, div.job-card-container",
    "card_title": "a.job-card-list__title, a.job-card-container__link",
    "card_company": "span.job-card-container__primary-description, "
                    "div.artdeco-entity-lockup__subtitle",
    "card_location": "li.job-card-container__metadata-item, "
                     "div.artdeco-entity-lockup__caption",

    # The card's own anchor. Clicking it loads the job into the detail pane
    # beside the results; navigating to /jobs/view/<id> instead lands on a
    # different page whose Easy Apply button is not the one we drive.
    "card_link": "a.job-card-list__title, a.job-card-container__link, a",
    # Present on a card LinkedIn already considers applied to.
    "card_applied_marker": "li.job-card-container__footer-job-state, "
                           "span.job-card-container__footer-job-state",

    "detail_title": "h1.job-title, h1.t-24",
    "detail_company": "div.job-details-jobs-unified-top-card__company-name a, "
                      "a.app-aware-link[href*='/company/']",
    "detail_location": "div.job-details-jobs-unified-top-card__primary-description-container",
    # The description body inside the detail pane. jobs-box__html-content is
    # what the pane actually renders; the others are older/standalone markup.
    "detail_description": "div.jobs-box__html-content, "
                          "div.jobs-description__content, "
                          "article.jobs-description__container",
    "see_more": "button.jobs-description__footer-button",

    # aria-label carries the job title ("Easy Apply to Data Lead at X"), so
    # it is matched by substring, never equality.
    "easy_apply_button": "button.jobs-apply-button[aria-label*='Easy Apply'], "
                         "button.jobs-apply-button[aria-label*='Easy apply'], "
                         "button.jobs-apply-button",
    # LinkedIn also serves an SDUI apply flow whose container differs.
    "modal": "div.jobs-easy-apply-modal, div[data-test-modal][role='dialog'], "
             "div.artdeco-modal[role='dialog']",
    # Labels vary by locale and step; match on the stable prefix instead.
    "modal_next": "button[aria-label*='next step'], button[aria-label*='Continue']",
    "modal_review": "button[aria-label*='Review your application'], "
                    "button[aria-label*='Review']",
    "modal_submit": "button[aria-label*='Submit application']",
    "modal_dismiss": "button[aria-label='Dismiss'], button.artdeco-modal__dismiss",
    # Shown after LinkedIn accepts an application; distinguishes a real
    # submission from the modal simply being closed.
    "post_apply_marker": "div.artdeco-inline-feedback--success, "
                         "div.jobs-post-apply-status-module",
    "modal_discard": "button[data-control-name='discard_application_confirm_btn']",

    # data-test-* attributes survive redesigns far better than class names;
    # the classes are kept as a fallback for older markup.
    "form_group": "div[data-test-form-element], "
                  "div.jobs-easy-apply-form-section__grouping, "
                  "div.fb-dash-form-element",
    "field_label": "label, legend, span.fb-dash-form-element__label",
    "text_input": "input[type='text'], input[type='email'], input[type='tel'], textarea",
    "select_input": "select",
    "radio_input": "input[type='radio']",
    "file_input": "input[type='file'], input[name='file']",
    "error_marker": "div.artdeco-inline-feedback--error, "
                    "div.fb-dash-form-element__error-text",
    "limit_marker": "div.artdeco-inline-feedback__message",
}

_JOB_ID_PATTERNS = [
    re.compile(r"/jobs/view/(\d+)"),
    re.compile(r"currentJobId=(\d+)"),
]

DATE_POSTED_PARAM = {
    "Past 24 hours": "r86400",
    "Past week": "r604800",
    "Past month": "r2592000",
    "Any time": "",
}

EXPERIENCE_PARAM = {
    "Internship": "1", "Entry level": "2", "Associate": "3",
    "Mid-Senior level": "4", "Director": "5", "Executive": "6",
}

JOB_TYPE_PARAM = {
    "Full-time": "F", "Part-time": "P", "Contract": "C",
    "Temporary": "T", "Internship": "I", "Volunteer": "V", "Other": "O",
}

WORKPLACE_PARAM = {"On-site": "1", "Remote": "2", "Hybrid": "3"}


def extract_job_id(url: str) -> str | None:
    for pattern in _JOB_ID_PATTERNS:
        m = pattern.search(url or "")
        if m:
            return m.group(1)
    return None


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------

class LinkedInSession:
    """Sign-in state.

    Deliberately never accepts or stores a password. Automated credential
    submission is the single biggest trigger for checkpoints and account
    restriction, so sign-in is manual and the session is reused from a
    persistent browser profile.
    """

    def __init__(self, browser: Browser) -> None:
        self.browser = browser

    def is_signed_in(self) -> bool:
        url = self.browser.current_url()
        if "/login" in url or "/checkpoint" in url or "/authwall" in url:
            return False
        if self.browser.find(SEL["feed_marker"]) or self.browser.find(SEL["nav_me"]):
            return True
        return self.browser.find(SEL["signin_link"]) is None

    def ensure_signed_in(self, prompt=input) -> bool:
        """Open the feed and, if needed, wait for a manual sign-in.

        `prompt` is injected so tests can drive it without stdin.
        """
        self.browser.goto(f"{BASE}/feed/")
        if self.is_signed_in():
            log.info("Existing LinkedIn session found.")
            return True

        print("\n" + "=" * 68)
        print("Sign in to LinkedIn in the browser window that just opened.")
        print("Complete any 2FA or checkpoint, then return here.")
        print("=" * 68)
        try:
            prompt("Press Enter once you are signed in (or type 'q' to abort): ")
        except (EOFError, KeyboardInterrupt):
            return False

        self.browser.goto(f"{BASE}/feed/")
        ok = self.is_signed_in()
        log.info("Sign-in %s.", "confirmed" if ok else "not detected")
        return ok


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

class LinkedInSearch:
    """Builds search URLs and extracts postings."""

    def __init__(self, browser: Browser, config: SearchConfig) -> None:
        self.browser = browser
        self.config = config
        # Pause after each click, as the reference bot's click_gap does:
        # LinkedIn renders asynchronously and dislikes rapid-fire input.
        self.click_gap = 1.0

    def build_url(self, term: str) -> str:
        f = self.config.filters
        params: dict[str, str] = {"keywords": term}
        if f.location:
            params["location"] = f.location
        if f.easy_apply_only:
            params["f_AL"] = "true"
        if (dp := DATE_POSTED_PARAM.get(f.date_posted, "")):
            params["f_TPR"] = dp
        if f.experience_levels:
            codes = [EXPERIENCE_PARAM[l] for l in f.experience_levels if l in EXPERIENCE_PARAM]
            if codes:
                params["f_E"] = ",".join(codes)
        if f.job_types:
            codes = [JOB_TYPE_PARAM[t] for t in f.job_types if t in JOB_TYPE_PARAM]
            if codes:
                params["f_JT"] = ",".join(codes)
        if f.workplace_types:
            codes = [WORKPLACE_PARAM[w] for w in f.workplace_types if w in WORKPLACE_PARAM]
            if codes:
                params["f_WT"] = ",".join(codes)
        return f"{BASE}/jobs/search/?{urlencode(params)}"

    def _card_text(self, card: Element, selector: str) -> str:
        """Read text from inside one result card.

        Scoped to `card`: a document-wide lookup returns the first card's
        company and location for every posting in the list.
        """
        el = None
        for part in selector.split(", "):
            el = self.browser.find_within(card, part.strip())
            if el:
                break
        return self.browser.text_of(el).strip() if el else ""

    def collect(self, term: str, limit: int) -> list[tuple[JobPosting, Element]]:
        """Collect up to `limit` postings from the first page of results.

        Returns (posting, card) pairs. The card element is kept because the
        job is opened by *clicking* it, which loads the detail pane beside
        the results. Navigating to /jobs/view/<id> instead lands on a
        different page whose Easy Apply button is not the one we drive --
        that mistake is why applications never started.

        Deliberately single-page: no pagination, no "start=" offset.
        """
        collected: list[tuple[JobPosting, Element]] = []
        seen: set[str] = set()

        self.browser.goto(self.build_url(term))
        self.browser.wait_for(SEL["result_card"], timeout=15.0)
        cards = self.browser.find_all(SEL["result_card"])
        if not cards:
            log.info("No results for %r.", term)
            return collected

        for card in cards:
            if len(collected) >= limit:
                break
            link = None
            for part in SEL["card_link"].split(", "):
                link = self.browser.find_within(card, part.strip())
                if link:
                    break
            href = (link.attr("href") if link else "") or ""
            job_id = extract_job_id(href) or card.attr("data-job-id") \
                or card.attr("data-occludable-job-id")
            if not job_id:
                # Never silent: a card whose id cannot be read is dropped
                # before any audit record exists, so this log line is the
                # only evidence it was ever seen. Promoted cards at the top
                # of the results use different markup, which is exactly
                # where an unreadable id would cost the best matches.
                preview = " ".join(card.text.split())[:60]
                log.warning("Skipping a card with no readable job id "
                            "(markup may have changed): %r", preview)
                continue
            if job_id in seen:
                continue
            seen.add(job_id)

            # LinkedIn marks cards it already considers applied to -- but the
            # same footer element also carries "Viewed", "Promoted", "Early
            # applicant" and "Actively reviewing". Matching the container
            # alone silently dropped good jobs, and promoted listings cluster
            # at the top of results, so the best matches were the ones lost.
            # The text is what distinguishes them, so the text is what we read.
            if (state := self.browser.find_within(card, SEL["card_applied_marker"])):
                state_text = self.browser.text_of(state).strip().lower()
                if "applied" in state_text:
                    log.info("Skipping %s: LinkedIn says applied (%r).",
                             job_id, state_text)
                    continue

            title = (self.browser.text_of(link).strip() if link else "") or card.text.strip()
            posting = JobPosting(
                job_id=job_id,
                title=title.split("\n")[0][:200],
                company=self._card_text(card, SEL["card_company"])[:200],
                location=self._card_text(card, SEL["card_location"])[:200],
                url=href if href.startswith("http") else f"{BASE}/jobs/view/{job_id}/",
            )
            collected.append((posting, card))

        log.info("Collected %d postings for %r.", len(collected), term)
        return collected

    def open_posting(self, card: Element) -> bool:
        """Click a result card so the job loads in the detail pane.

        Stays on the search page throughout: the pane is where the Easy
        Apply button and its modal live.
        """
        link = None
        for part in SEL["card_link"].split(", "):
            link = self.browser.find_within(card, part.strip())
            if link:
                break
        target = link or card
        self.browser.scroll_into_view(target)
        self.browser.click(target)
        self.browser.sleep(self.click_gap)
        return self.browser.wait_for(SEL["detail_description"], timeout=15.0) is not None

    def load_description(self, posting: JobPosting, card: Element) -> JobPosting:
        """Open the posting in the detail pane and attach its description."""
        if not self.open_posting(card):
            log.warning("%s: detail pane did not load.", posting.job_id)

        if (more := self.browser.find(SEL["see_more"])):
            self.browser.click(more)
            self.browser.sleep(self.click_gap)

        desc_el = None
        for part in SEL["detail_description"].split(", "):
            desc_el = self.browser.find(part.strip())
            if desc_el:
                break
        description = self.browser.text_of(desc_el) if desc_el else ""

        data = posting.model_dump()
        data["description"] = description
        if not data.get("company"):
            company_el = None
            for part in SEL["detail_company"].split(", "):
                company_el = self.browser.find(part.strip())
                if company_el:
                    break
            data["company"] = self.browser.text_of(company_el).strip() if company_el else ""
        return JobPosting.model_validate(data)

    def has_easy_apply(self) -> bool:
        return self.browser.find(SEL["easy_apply_button"]) is not None


# --------------------------------------------------------------------------
# Easy Apply
# --------------------------------------------------------------------------

@dataclass
class FillReport:
    """Outcome of filling one form, for the audit trail."""
    answered: dict[str, str] = field(default_factory=dict)
    # Answers that were guessed rather than resolved from the profile, kept
    # separate so the evidence log can show exactly what was invented.
    guessed: dict[str, str] = field(default_factory=dict)
    unanswered: list[str] = field(default_factory=list)
    uploaded_resume: bool = False
    steps: int = 0
    reached_submit: bool = False
    submitted: bool = False
    aborted_reason: str | None = None

    @property
    def needs_human(self) -> bool:
        return bool(self.unanswered) or self.aborted_reason is not None


class EasyApplyFiller:
    """Traverses the multi-step Easy Apply modal.

    Refuses to submit when any required question is unanswered. Guessing on
    a form that reaches an employer is the failure mode this design exists
    to prevent.
    """

    MAX_STEPS = 12   # guards against a navigation loop

    def __init__(
        self,
        browser: Browser,
        profile: Profile,
        resolver: AnswerResolver | None = None,
        guess_unmapped: bool = False,
    ) -> None:
        self.browser = browser
        self.profile = profile
        self.resolver = resolver or AnswerResolver(profile)
        # With no human in the loop, refusing to answer blocks the whole
        # application. Guessing is opt-in and every guess is recorded.
        self.guess_unmapped = guess_unmapped

    # -- field helpers ---------------------------------------------------

    def _label_for(self, group: Element) -> str:
        """Read the question text from inside this group.

        Scoped deliberately: a document-wide label lookup returns the first
        label on the page for every question, which silently gives every
        field the first question's answer.
        """
        for part in SEL["field_label"].split(", "):
            el = self.browser.find_within(group, part.strip())
            if el:
                text = self.browser.text_of(el).strip()
                if text:
                    return text
        return group.attr("aria-label") or group.text.strip()

    def _fill_group(self, group: Element, report: FillReport) -> None:
        """Answer the one question inside `group`.

        Every lookup here is scoped to the group. Searching the document
        instead is the bug this method is written against.
        """
        label = self._label_for(group) or group.attr("aria-label")
        if not label:
            return

        required = bool(group.attr("required") or group.attr("aria-required"))

        # File input: the resume.
        if (file_el := self.browser.find_within(group, SEL["file_input"])):
            path = str(self.profile.professional.resume_path)
            self.browser.upload_file(file_el, path)
            report.uploaded_resume = True
            report.answered[label] = f"<resume: {self.profile.professional.resume_path.name}>"
            return

        # Select: constrain the answer to the offered options.
        if (sel_el := self.browser.find_within(group, SEL["select_input"])):
            options = self.browser.options_for(sel_el)
            resolved = self.resolver.resolve_choice(label, options)
            if not resolved and self.guess_unmapped:
                resolved = self.resolver.guess_choice(options)
            if resolved:
                value, rule = resolved
                self.browser.select_option(sel_el, value)
                report.answered[label] = value
                if rule == "guess":
                    report.guessed[label] = value
            elif required or options:
                report.unanswered.append(label)
            return

        # Radio group.
        radios = self.browser.find_all_within(group, SEL["radio_input"])
        if radios:
            options = [r.attr("value") or r.text for r in radios]
            resolved = self.resolver.resolve_choice(label, options)
            if not resolved and self.guess_unmapped:
                resolved = self.resolver.guess_choice(options)
            if resolved:
                value, rule = resolved
                for r in radios:
                    if (r.attr("value") or r.text) == value:
                        self.browser.click(r)
                        report.answered[label] = value
                        if rule == "guess":
                            report.guessed[label] = value
                        return
            report.unanswered.append(label)
            return

        # Free text.
        if (text_el := self.browser.find_within(group, SEL["text_input"])):
            resolved = self.resolver.resolve(label)
            if not resolved and self.guess_unmapped and required:
                resolved = self.resolver.guess_text(label)
            if resolved:
                value, rule = resolved
                self.browser.type_text(text_el, value)
                report.answered[label] = value
                if rule == "guess":
                    report.guessed[label] = value
            else:
                report.unanswered.append(label)

    def _modal_button(self, modal: Element | None, text: str,
                      css_fallback: str) -> Element | None:
        """Find a modal control by its visible text, falling back to CSS.

        Text first because that is what LinkedIn keeps stable: aria-labels
        and class names change between redesigns and locales, and matching
        them by attribute is why the traversal never reached Submit.
        """
        if modal is not None:
            if (el := self.browser.find_by_text_within(modal, text)):
                return el
        if (el := self.browser.find_by_text(text)):
            return el
        return self.browser.find(css_fallback)

    def fill_current_step(self, report: FillReport) -> None:
        for group in self.browser.find_all(SEL["form_group"]):
            self._fill_group(group, report)

    # -- traversal -------------------------------------------------------

    def run(self, submit: bool = False) -> FillReport:
        """Open the modal, fill every step, and stop before submitting.

        `submit=True` clicks Submit only when no required question was left
        unanswered. The human gate upstream decides whether to pass True.
        """
        report = FillReport()

        # Wait rather than checking once: the pane renders asynchronously, so
        # a bare find cannot tell a missing button from an unrendered one.
        button = self.browser.wait_for(SEL["easy_apply_button"], timeout=10.0)
        if button is None:
            report.aborted_reason = "no Easy Apply button"
            return report

        # An "Apply" button that is not Easy Apply hands off to the company's
        # own site. Clicking it navigates away or opens a tab, and no modal
        # ever appears -- so it is identified before the click, not after.
        label = (button.attr("aria-label") + " " + button.text).lower()
        if "easy apply" not in label:
            report.aborted_reason = "external apply (not Easy Apply)"
            return report

        self.browser.scroll_into_view(button)
        self.browser.click(button)
        self.browser.sleep(1.0)

        # Kept, not discarded: every control lookup below is scoped to this
        # element so a same-worded button elsewhere on the page cannot match.
        modal = self.browser.wait_for(SEL["modal"], timeout=10.0)
        if modal is None:
            report.aborted_reason = (
                "Easy Apply modal did not open (external apply, or the modal "
                "selector is stale)"
            )
            return report

        for _ in range(self.MAX_STEPS):
            report.steps += 1
            self.fill_current_step(report)

            # Scoped to the modal: artdeco-inline-feedback--error is a generic
            # LinkedIn error style used all over the page, so a document-wide
            # find aborted good applications over an error somewhere else.
            if (err := self.browser.find_within(modal, SEL["error_marker"])):
                detail = self.browser.text_of(err).strip()
                report.aborted_reason = (
                    f"form reported a validation error: {detail[:120]}"
                    if detail else "form reported a validation error")
                return report

            if (submit_btn := self._modal_button(modal, "Submit application",
                                                 SEL["modal_submit"])):
                report.reached_submit = True
                if report.unanswered:
                    report.aborted_reason = (
                        f"{len(report.unanswered)} unanswered question(s)"
                    )
                    return report
                if submit:
                    self.browser.click(submit_btn)
                    report.submitted = True
                return report

            # Review comes before Next: on the last step both may be present.
            nxt = (self._modal_button(modal, "Review", SEL["modal_review"])
                   or self._modal_button(modal, "Next", SEL["modal_next"])
                   or self._modal_button(modal, "Continue to next step",
                                         SEL["modal_next"]))
            if nxt is None:
                report.aborted_reason = "no next/review/submit control found"
                return report
            self.browser.scroll_into_view(nxt)
            self.browser.click(nxt)
            self.browser.sleep(1.0)

        report.aborted_reason = f"exceeded {self.MAX_STEPS} steps"
        return report

    def wait_for_human(self, timeout: float = 600.0) -> str:
        """Leave the filled modal open and wait for the person to act.

        Returns "submitted" if the application was sent, "dismissed" if the
        modal was closed without sending, or "timeout" if neither happened
        in time. Submission is inferred from the modal closing while the
        post-apply confirmation is showing: LinkedIn replaces the form with
        it, so its presence is the only signal that Submit was clicked
        rather than the modal being discarded.
        """
        if not self.browser.wait_while_present(SEL["modal"], timeout=timeout):
            return "timeout"
        if self.browser.find(SEL["post_apply_marker"]) is not None:
            return "submitted"
        return "dismissed"

    def dismiss(self) -> None:
        """Close the modal and discard the draft.

        Dismissing an Easy Apply modal makes LinkedIn ask whether to save the
        application for later. Failing to click "Discard" leaves it saved, so
        every skipped job silently piles up in "My Jobs". The confirm button
        is matched by its visible text: its classes and data-control-name
        attribute change, the word does not.
        """
        self.browser.press_escape()
        self.browser.sleep(0.5)

        for word in ("Discard", "Discard application"):
            if (discard := self.browser.find_by_text(word)):
                self.browser.click(discard)
                self.browser.sleep(0.5)
                return

        # Fall back to the older attribute-based control, then the X.
        if (discard := self.browser.find(SEL["modal_discard"])):
            self.browser.click(discard)
            return
        if (x := self.browser.find(SEL["modal_dismiss"])):
            self.browser.click(x)

    def daily_limit_reached(self) -> bool:
        el = self.browser.find(SEL["limit_marker"])
        if el is None:
            return False
        return "exceeded the daily" in self.browser.text_of(el).lower()


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
