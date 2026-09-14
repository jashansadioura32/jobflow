"""Tests for the LinkedIn adapter, driven through FakeBrowser.

The properties that matter here: URLs encode the configured filters, the
collector reads one page and dedupes, and the filler never submits a form
with an unanswered required question.

Author: Jashan Sadioura
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jobflow.adapters.browser import Element, SeleniumBrowser
from jobflow.adapters.fake_browser import FakeBrowser, FakeField, FakePage
from jobflow.adapters.linkedin import (
    SEL, EasyApplyFiller, LinkedInSearch, LinkedInSession, extract_job_id,
)
from jobflow.core.models import (
    Compensation, Defaults, Demographics, Eligibility, Identity, JobPosting,
    Location, Professional, Profile, SearchConfig, SearchFilters, SearchTier,
)


@pytest.fixture
def profile(tmp_path):
    r = tmp_path / "Test_Resume.pdf"
    r.write_bytes(b"%PDF-1.4")
    return Profile(
        identity=Identity(
            first_name="Alex", last_name="Doe",
            email="test@example.com", phone="9999999999",
            phone_country_code="+91",
            linkedin_url="https://www.linkedin.com/in/example-user/",
            website="https://example.com/",
        ),
        location=Location(city="Springfield", street="Springfield", state="Example State",
                          zipcode="500001", country="India"),
        professional=Professional(
            headline="AI & Data Product Manager", years_of_experience=6,
            has_masters=True, recent_employer="Acme Corp", resume_path=r,
            summary="AI and data products.",
        ),
        compensation=Compensation(current_ctc=1200000, desired_salary=1500000,
                                  notice_period_days=0),
        eligibility=Eligibility(requires_visa_sponsorship=False),
        demographics=Demographics(gender="Male", ethnicity="Asian",
                                  disability_status="No", veteran_status="No"),
        defaults=Defaults(),
    )


@pytest.fixture
def config():
    return SearchConfig(
        tiers=[SearchTier(name="core", terms=["Data Quality Lead"])],
        filters=SearchFilters(
            location="India", date_posted="Past week", easy_apply_only=True,
            experience_levels=["Mid-Senior level", "Director"],
        ),
    )


# ---------------- job id ----------------

@pytest.mark.parametrize("url,expected", [
    ("https://www.linkedin.com/jobs/view/4413645554/", "4413645554"),
    ("/jobs/search/?currentJobId=123456&keywords=x", "123456"),
    ("https://example.com/nope", None),
    ("", None),
])
def test_extract_job_id(url, expected):
    assert extract_job_id(url) == expected


# ---------------- search URL ----------------

def test_build_url_encodes_filters(config):
    url = LinkedInSearch(FakeBrowser(), config).build_url("Data Quality Lead")
    assert "keywords=Data+Quality+Lead" in url
    assert "location=India" in url
    assert "f_AL=true" in url            # easy apply
    assert "f_TPR=r604800" in url        # past week
    assert "f_E=4%2C5" in url            # mid-senior + director


def test_build_url_never_paginates(config):
    """The search is single-page by design: no offset is ever sent."""
    url = LinkedInSearch(FakeBrowser(), config).build_url("x")
    assert "start=" not in url


def test_build_url_omits_empty_filters():
    cfg = SearchConfig(
        tiers=[SearchTier(name="t", terms=["x"])],
        filters=SearchFilters(location="", date_posted="Any time",
                              easy_apply_only=False),
    )
    url = LinkedInSearch(FakeBrowser(), cfg).build_url("x")
    assert "location=" not in url
    assert "f_AL" not in url
    assert "f_TPR" not in url


def _user_data_dir(opts) -> str:
    for arg in opts.arguments:
        if arg.startswith("--user-data-dir="):
            return arg.split("=", 1)[1]
    raise AssertionError("no --user-data-dir argument")


def test_chrome_options_target_the_default_profile():
    opts = SeleniumBrowser.build_chrome_options()
    assert any(arg == "--profile-directory=Default" for arg in opts.arguments)


def test_chrome_options_clone_rather_than_lock_the_live_profile():
    """Chrome locks a user-data dir to one process, so we must not point at it.

    Regression test: targeting the real profile crashed the run whenever the
    user had Chrome open.
    """
    live_root = SeleniumBrowser._default_user_data_dir()
    used = _user_data_dir(SeleniumBrowser.build_chrome_options())
    assert used != live_root
    assert "jobflow-chrome-" in used


def test_clone_can_be_disabled_for_a_dedicated_profile(tmp_path):
    opts = SeleniumBrowser.build_chrome_options(profile_dir=str(tmp_path),
                                                clone_profile=False)
    assert _user_data_dir(opts) == str(tmp_path)


def test_default_profile_path_is_normalised_to_its_parent(tmp_path):
    root, name = SeleniumBrowser._resolve_profile(str(tmp_path / "Default"))
    assert root == str(tmp_path)
    assert name == "Default"


def test_clone_copies_session_files_and_leaves_the_source_untouched(tmp_path):
    src_root = tmp_path / "User Data"
    (src_root / "Default" / "Network").mkdir(parents=True)
    # Chrome 96+ keeps cookies here, not at the profile root.
    (src_root / "Default" / "Network" / "Cookies").write_text("cookie-data",
                                                              encoding="utf-8")
    (src_root / "Local State").write_text("state-data", encoding="utf-8")
    # Cache is deliberately not copied.
    (src_root / "Default" / "Cache").mkdir()
    (src_root / "Default" / "Cache" / "big").write_text("x" * 1000, encoding="utf-8")

    dest = Path(SeleniumBrowser._clone_profile(str(src_root), "Default"))

    assert (dest / "Default" / "Network" / "Cookies").read_text(
        encoding="utf-8") == "cookie-data"
    assert (dest / "Local State").read_text(encoding="utf-8") == "state-data"
    assert not (dest / "Default" / "Cache").exists()
    assert (src_root / "Default" / "Network" / "Cookies").exists()


def test_clone_survives_a_missing_profile(tmp_path):
    """A missing profile costs a manual sign-in, never a crash."""
    dest = Path(SeleniumBrowser._clone_profile(str(tmp_path / "nope"), "Default"))
    assert dest.is_dir()


# ---------------- session ----------------

def test_session_detects_signed_in():
    b = FakeBrowser()
    b.add_page(FakePage(url="https://www.linkedin.com/feed/",
                        elements={SEL["feed_marker"]: [Element(handle=None)]}))
    assert LinkedInSession(b).ensure_signed_in(prompt=lambda _: "") is True


def test_session_detects_signed_out_then_manual_login():
    b = FakeBrowser()
    login = FakePage(url="https://www.linkedin.com/feed/",
                     elements={SEL["signin_link"]: [Element(handle=None)]})
    b.add_page(login)

    def fake_prompt(_):
        # Simulate the user signing in while we wait.
        b.add_page(FakePage(url="https://www.linkedin.com/feed/",
                            elements={SEL["feed_marker"]: [Element(handle=None)]}))
        return ""

    assert LinkedInSession(b).ensure_signed_in(prompt=fake_prompt) is True


def test_session_never_accepts_a_password():
    """No method on the session may take a credential parameter.

    Checks the signatures rather than the source text, so the prose in the
    docstring explaining this choice does not trip the assertion.
    """
    import inspect
    banned = {"password", "passwd", "pwd", "secret", "credentials", "username"}
    for name, method in inspect.getmembers(LinkedInSession, inspect.isfunction):
        params = set(inspect.signature(method).parameters)
        assert not (params & banned), f"{name} accepts a credential: {params & banned}"


# ---------------- collection ----------------

def _search_page(url, job_ids):
    """Build a results page where each card owns its own title link.

    Cards carry a "data-selector" and their title an "owner" naming it, so
    the fake resolves a scoped lookup per card rather than handing every
    card the first card's link.
    """
    cards = [
        Element(handle=None,
                attributes={"data-job-id": j, "data-selector": f"#card{i}"})
        for i, j in enumerate(job_ids)
    ]
    titles = [
        Element(handle=None, text="Data Quality Lead",
                attributes={"href": f"/jobs/view/{j}/", "owner": f"#card{i}"})
        for i, j in enumerate(job_ids)
    ]
    return FakePage(url=url, elements={
        SEL["result_card"]: cards,
        "a.job-card-list__title": titles,
    })


def test_collect_stops_when_no_results(config):
    b = FakeBrowser()
    assert LinkedInSearch(b, config).collect("x", limit=10) == []


def test_collect_dedupes_and_respects_limit(config):
    b = FakeBrowser()
    s = LinkedInSearch(b, config)
    # A repeated card on the one page is still collected only once.
    b.add_page(_search_page(s.build_url("x"), ["111", "111", "111"]))
    got = s.collect("x", limit=10)
    assert [p.job_id for p, _card in got] == ["111"]


def test_collect_reads_only_the_first_page(config):
    """Pagination is removed: a second page of results is never fetched."""
    b = FakeBrowser()
    s = LinkedInSearch(b, config)
    b.add_page(_search_page(s.build_url("x"), ["111", "222"]))
    got = s.collect("x", limit=25)
    assert [p.job_id for p, _card in got] == ["111", "222"]
    # One navigation only -- no offset pages were requested.
    assert len(b.visited) == 1
    assert "start=" not in b.visited[0]


# ---------------- form filling ----------------

def _input_el(field, tag=""):
    """Build an input element owned by its form group.

    "owner" is what lets FakeBrowser distinguish a scoped lookup from a
    document-wide one; without it a multi-question form silently gives every
    question the first question's answer.
    """
    return Element(handle=field, tag=tag,
                   attributes={"data-selector": field.selector,
                               "aria-label": field.label,
                               "owner": field.selector})


def _form_page(fields, *, with_submit=True, with_next=False, error=False):
    els = {
        SEL["easy_apply_button"]: [Element(handle=None, text="Easy Apply")],
        SEL["modal"]: [Element(handle=None)],
        SEL["form_group"]: [
            Element(handle=None, text=f.label,
                    attributes={"data-selector": f.selector,
                                "aria-label": f.label,
                                "required": "true" if f.required else ""})
            for f in fields
        ],
    }
    if with_submit:
        els[SEL["modal_submit"]] = [Element(handle=None, text="Submit application")]
    if with_next:
        els[SEL["modal_next"]] = [Element(handle=None, text="Continue")]
    if error:
        els[SEL["error_marker"]] = [Element(handle=None, text="required")]
    return FakePage(url="https://www.linkedin.com/jobs/view/1/",
                    elements=els, fields=fields)


def test_fills_answerable_text_fields(profile):
    fields = [FakeField(selector="#phone", label="Mobile phone number", required=True)]
    b = FakeBrowser()
    page = _form_page(fields)
    # The single text input resolves to this field.
    page.elements[SEL["text_input"]] = [_input_el(fields[0])]
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=False)
    assert report.answered["Mobile phone number"] == "9999999999"
    assert report.unanswered == []
    assert report.reached_submit is True
    assert report.submitted is False          # submit=False


def test_refuses_to_submit_with_unanswered_question(profile):
    """The central safety property of the filler."""
    fields = [FakeField(selector="#q", label="Describe a conflict with a coworker",
                        required=True)]
    b = FakeBrowser()
    page = _form_page(fields)
    page.elements[SEL["text_input"]] = [_input_el(fields[0])]
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=True)   # submit requested
    assert report.unanswered == ["Describe a conflict with a coworker"]
    assert report.submitted is False
    assert report.needs_human is True
    assert "unanswered" in report.aborted_reason


def test_submits_when_everything_answered(profile):
    fields = [FakeField(selector="#city", label="What is your current city?")]
    b = FakeBrowser()
    page = _form_page(fields)
    page.elements[SEL["text_input"]] = [_input_el(fields[0])]
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=True)
    assert report.answered["What is your current city?"] == "Springfield"
    assert report.submitted is True


def _modal_page(url="https://www.linkedin.com/jobs/view/1/"):
    return FakePage(url=url, elements={SEL["modal"]: [Element(handle=None)]})


def test_wait_for_human_reports_a_submission(profile):
    b = FakeBrowser()
    page = _modal_page()
    page.elements[SEL["post_apply_marker"]] = [Element(handle=None, text="Applied")]
    b.add_page(page)
    b.goto(page.url)
    b.closes_after[SEL["modal"]] = 1

    assert EasyApplyFiller(b, profile).wait_for_human(timeout=1) == "submitted"


def test_wait_for_human_reports_a_dismissal(profile):
    """Modal gone with no success marker means the person closed it."""
    b = FakeBrowser()
    page = _modal_page()
    b.add_page(page)
    b.goto(page.url)
    b.closes_after[SEL["modal"]] = 1

    assert EasyApplyFiller(b, profile).wait_for_human(timeout=1) == "dismissed"


def test_wait_for_human_reports_a_timeout(profile):
    b = FakeBrowser()
    page = _modal_page()
    b.add_page(page)
    b.goto(page.url)
    b.closes_after[SEL["modal"]] = 99      # never closes

    assert EasyApplyFiller(b, profile).wait_for_human(timeout=0.01) == "timeout"


def test_select_constrained_to_offered_options(profile):
    fields = [FakeField(selector="#gender", label="Gender", kind="select",
                        options=["Male", "Female", "Decline"])]
    b = FakeBrowser()
    page = _form_page(fields)
    page.elements[SEL["select_input"]] = [_input_el(fields[0], tag="select")]
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=False)
    assert report.answered["Gender"] == "Male"
    assert b.selected["#gender"] == "Male"


def test_unmatched_select_escalates(profile):
    fields = [FakeField(selector="#x", label="Gender", kind="select",
                        options=["Option A", "Option B"])]
    b = FakeBrowser()
    page = _form_page(fields)
    page.elements[SEL["select_input"]] = [_input_el(fields[0], tag="select")]
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=True)
    assert "Gender" in report.unanswered
    assert report.submitted is False


def test_resume_uploaded(profile):
    b = FakeBrowser()
    cv = FakeField(selector="#cv", label="Upload resume", kind="file")
    page = _form_page([cv])
    page.elements[SEL["file_input"]] = [_input_el(cv)]
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=False)
    assert report.uploaded_resume is True
    assert b.uploads and b.uploads[0].endswith("Test_Resume.pdf")


def test_each_question_on_a_form_gets_its_own_answer(profile):
    """Regression: every lookup in _fill_group must be scoped to its group.

    With document-wide finds, all three questions resolved against the first
    input on the page, so the phone number was written into every field and
    the other two labels were never seen. The fake could not catch it until
    it modelled ownership, which is why this test exists.
    """
    phone = FakeField(selector="#phone", label="Mobile phone number", required=True)
    city = FakeField(selector="#city", label="What is your current city?", required=True)
    notice = FakeField(selector="#notice", label="Notice period in days", required=True)
    fields = [phone, city, notice]

    b = FakeBrowser()
    page = _form_page(fields)
    page.elements[SEL["text_input"]] = [_input_el(f) for f in fields]
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=False)

    assert report.answered["Mobile phone number"] == "9999999999"
    assert report.answered["What is your current city?"] == "Springfield"
    assert report.answered["Notice period in days"] == "0"
    assert report.unanswered == []
    # Each field received its own value, not the first field's.
    assert b.typed["#phone"] == "9999999999"
    assert b.typed["#city"] == "Springfield"
    assert b.typed["#notice"] == "0"


def test_each_result_card_reads_its_own_company_and_title():
    """Regression: card lookups must be scoped to the card.

    Document-wide finds gave every posting the first card's company and
    location, which is why a real run logged empty companies.
    """
    cfg = SearchConfig(
        tiers=[SearchTier(name="core", terms=["Data Quality Lead"])],
        filters=SearchFilters(location="India", easy_apply_only=True),
    )
    b = FakeBrowser()
    url = LinkedInSearch(b, cfg).build_url("Data Quality Lead")
    cards = [
        Element(handle=None, attributes={"data-selector": "#c1", "data-job-id": "111"}),
        Element(handle=None, attributes={"data-selector": "#c2", "data-job-id": "222"}),
    ]
    page = FakePage(url=url, elements={
        SEL["result_card"]: cards,
        "a.job-card-list__title": [
            Element(handle=None, text="Data Quality Lead",
                    attributes={"href": "/jobs/view/111/", "owner": "#c1"}),
            Element(handle=None, text="Master Data Specialist",
                    attributes={"href": "/jobs/view/222/", "owner": "#c2"}),
        ],
        "span.job-card-container__primary-description": [
            Element(handle=None, text="Acme Data", attributes={"owner": "#c1"}),
            Element(handle=None, text="Tata Consultancy Services",
                    attributes={"owner": "#c2"}),
        ],
    })
    b.add_page(page)

    postings = [p for p, _card in
                LinkedInSearch(b, cfg).collect("Data Quality Lead", limit=2)]

    assert [p.job_id for p in postings] == ["111", "222"]
    assert postings[0].company == "Acme Data"
    assert postings[1].company == "Tata Consultancy Services"
    assert postings[1].title == "Master Data Specialist"


def _text_only_form_page(fields, *, steps=("Next", "Submit application")):
    """A modal whose controls are findable ONLY by visible text.

    The real Easy Apply modal is like this: its aria-labels and class names
    do not match our CSS selectors, so a fixture that registers buttons
    under those selectors proves nothing. Traversal that relies on CSS
    fails here, which is the point.
    """
    els = {
        SEL["easy_apply_button"]: [Element(handle=None, text="Easy Apply")],
        SEL["modal"]: [Element(handle=None, attributes={"data-selector": "#modal"})],
        SEL["form_group"]: [
            Element(handle=None, text=f.label,
                    attributes={"data-selector": f.selector,
                                "aria-label": f.label,
                                "required": "true" if f.required else ""})
            for f in fields
        ],
    }
    # Registered under keys that match no SEL entry: text is the only route.
    for i, word in enumerate(steps):
        els[f"div.unmatched-control-{i}"] = [
            Element(handle=None, text=word, attributes={"data-selector": f"#btn{i}"})
        ]
    return FakePage(url="https://www.linkedin.com/jobs/search/",
                    elements=els, fields=fields)


def test_traversal_finds_controls_by_text_not_css(profile):
    """Regression: Next/Review/Submit are matched by their visible text.

    Matching them by aria-label meant the loop never advanced past step one,
    so reached_submit stayed False and every application was discarded --
    which looked like "it saves instead of applying".
    """
    field = FakeField(selector="#city", label="What is your current city?")
    b = FakeBrowser()
    page = _text_only_form_page([field])
    page.elements[SEL["text_input"]] = [_input_el(field)]
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=False)

    # Reached Submit even though no control matches a CSS selector: the
    # only route to them is their visible text.
    assert report.reached_submit is True
    assert report.aborted_reason is None
    assert report.answered["What is your current city?"] == "Springfield"


def test_traversal_reaches_submit_through_a_review_step(profile):
    field = FakeField(selector="#city", label="What is your current city?")
    b = FakeBrowser()
    page = _text_only_form_page([field], steps=("Review", "Submit application"))
    page.elements[SEL["text_input"]] = [_input_el(field)]
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=True)

    assert report.reached_submit is True
    assert report.submitted is True


def test_dismiss_discards_rather_than_saving(profile):
    """Regression: a dismissed application must not be left saved.

    LinkedIn asks "save this application?" when a modal is closed. Missing
    the Discard button leaves every skipped job saved in My Jobs, which is
    what a real run did.
    """
    b = FakeBrowser()
    page = FakePage(url="https://www.linkedin.com/jobs/search/", elements={
        SEL["modal"]: [Element(handle=None)],
        "button.discard": [Element(handle=None, text="Discard",
                                   attributes={"data-selector": "#discard"})],
        "button.save": [Element(handle=None, text="Save",
                                attributes={"data-selector": "#save"})],
    })
    b.add_page(page)
    b.goto(page.url)

    EasyApplyFiller(b, profile).dismiss()

    assert "<escape>" in b.clicks
    assert "#discard" in b.clicks
    assert "#save" not in b.clicks      # never saves


def test_external_apply_is_skipped_without_clicking(profile):
    """An "Apply" button that is not Easy Apply hands off to the employer's site.

    Clicking it navigates away from LinkedIn, so it must be identified from
    the button itself rather than by waiting for a modal that never opens.
    """
    b = FakeBrowser()
    page = FakePage(url="https://www.linkedin.com/jobs/view/1/", elements={
        SEL["easy_apply_button"]: [
            Element(handle=None, text="Apply",
                    attributes={"aria-label": "Apply to Master Data Specialist"})
        ],
    })
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=False)
    assert report.aborted_reason == "external apply (not Easy Apply)"
    assert report.submitted is False
    assert b.clicks == []          # never clicked, so never navigated away


def test_easy_apply_button_is_recognised_by_its_label(profile):
    fields = [FakeField(selector="#city", label="What is your current city?")]
    b = FakeBrowser()
    page = _form_page(fields)
    page.elements[SEL["text_input"]] = [_input_el(fields[0])]
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=False)
    assert report.aborted_reason is None
    assert report.reached_submit is True


def test_missing_easy_apply_button_aborts(profile):
    b = FakeBrowser()
    b.add_page(FakePage(url="https://www.linkedin.com/jobs/view/1/"))
    b.goto("https://www.linkedin.com/jobs/view/1/")
    report = EasyApplyFiller(b, profile).run()
    assert report.aborted_reason == "no Easy Apply button"
    assert report.submitted is False


def test_validation_error_aborts(profile):
    b = FakeBrowser()
    page = _form_page([], error=True)
    b.add_page(page)
    b.goto(page.url)
    report = EasyApplyFiller(b, profile).run(submit=True)
    assert report.submitted is False
    assert "validation error" in report.aborted_reason


def test_step_loop_is_bounded(profile):
    """A modal that always offers 'Next' must not loop forever."""
    b = FakeBrowser()
    page = _form_page([], with_submit=False, with_next=True)
    b.add_page(page)
    b.goto(page.url)
    report = EasyApplyFiller(b, profile).run(submit=True)
    # Stops as soon as the same control reappears, rather than clicking it
    # MAX_STEPS times: a form that does not advance is a loop, and burning
    # twelve clicks on LinkedIn's Reapply dialog is how that showed up.
    assert report.steps < EasyApplyFiller.MAX_STEPS
    assert "did not advance" in report.aborted_reason
    assert report.submitted is False


# ---------------- card footer states ----------------

def _card_page(cfg, states):
    """A results page whose cards carry the given footer-state text."""
    b = FakeBrowser()
    url = LinkedInSearch(b, cfg).build_url("Data Quality Lead")
    cards, footers, titles = [], [], []
    for i, state in enumerate(states):
        cards.append(Element(handle=None, attributes={
            "data-job-id": f"{100+i}", "data-selector": f"#c{i}"}))
        titles.append(Element(handle=None, text=f"Job {i}", attributes={
            "href": f"/jobs/view/{100+i}/", "owner": f"#c{i}"}))
        if state:
            footers.append(Element(handle=None, text=state,
                                   attributes={"owner": f"#c{i}"}))
    # Registered under the full selector string: FakeBrowser.find_all does an
    # exact dict-key lookup, so a single comma-separated part never matches.
    page = FakePage(url=url, elements={
        SEL["result_card"]: cards,
        "a.job-card-list__title": titles,
        SEL["card_applied_marker"]: footers,
    })
    b.add_page(page)
    return b


def test_only_applied_cards_are_skipped(config):
    """Regression: "Promoted"/"Viewed" cards were dropped as if applied.

    The footer element is shared by every card state, so matching the
    container alone silently discarded good jobs -- and promoted listings
    sit at the top of the results, so the best matches were lost first.
    """
    b = _card_page(config, ["Promoted", "Viewed", "Applied", "Early applicant"])
    got = LinkedInSearch(b, config).collect("Data Quality Lead", limit=10)

    # Everything except the genuinely-applied card survives.
    assert [p.job_id for p, _c in got] == ["100", "101", "103"]


def test_applied_cards_are_still_skipped(config):
    b = _card_page(config, ["Applied", "Applied 2 weeks ago"])
    assert LinkedInSearch(b, config).collect("Data Quality Lead", limit=10) == []


def test_cards_without_a_footer_state_are_kept(config):
    b = _card_page(config, [None, None])
    got = LinkedInSearch(b, config).collect("Data Quality Lead", limit=10)
    assert len(got) == 2


# ---------------- typeahead / autocomplete fields ----------------

def test_location_field_selects_from_the_dropdown(profile):
    """Regression: typing a city is not the same as choosing one.

    LinkedIn's location field is an autocomplete. Typing "Springfield"
    leaves the box looking filled while the form holds no value at all, so
    the application is rejected. The suggestion must be committed with
    Down+Enter after the dropdown renders.
    """
    field = FakeField(selector="#city", label="What is your current city?",
                      required=True)
    b = FakeBrowser()
    page = _form_page([field])
    page.elements[SEL["text_input"]] = [_input_el(field)]
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=False)

    assert report.answered["What is your current city?"] == "Springfield"
    assert "#city" in b.typeahead_committed


def test_plain_text_fields_are_not_treated_as_typeaheads(profile):
    """A phone box has no dropdown; pressing Enter in one submits the form."""
    field = FakeField(selector="#phone", label="Mobile phone number",
                      required=True)
    b = FakeBrowser()
    page = _form_page([field])
    page.elements[SEL["text_input"]] = [_input_el(field)]
    b.add_page(page)
    b.goto(page.url)

    EasyApplyFiller(b, profile).run(submit=False)

    assert b.typeahead_committed == []


@pytest.mark.parametrize("label,expected", [
    ("What is your current city?", True),
    ("City", True),
    ("Current location", True),
    ("Street address", True),
    ("Home town", True),
    ("Mobile phone number", False),
    ("Years of experience", False),
    ("Email address", False),
])
def test_typeahead_detection(label, expected):
    assert EasyApplyFiller._is_typeahead(label, "unmapped") is expected


def test_typeahead_detected_by_rule_name_even_for_odd_labels():
    """The rule name catches fields the label wording would miss."""
    assert EasyApplyFiller._is_typeahead("Where are you based?", "city") is True
    assert EasyApplyFiller._is_typeahead("Where are you based?", "phone") is False


# ---------------- already-applied / Reapply ----------------

def _apply_button_page(label_text, aria=""):
    return FakePage(url="https://www.linkedin.com/jobs/search/", elements={
        SEL["easy_apply_button"]: [
            Element(handle=None, text=label_text,
                    attributes={"aria-label": aria or label_text})
        ],
        SEL["modal"]: [Element(handle=None, attributes={"data-selector": "#m"})],
    })


@pytest.mark.parametrize("text,aria", [
    ("Reapply", "Reapply to Data Lead at Acme"),
    ("Easy Apply", "Easy Apply · Applied"),
    ("Applied", "Applied"),
])
def test_already_applied_jobs_are_not_reopened(profile, text, aria):
    """Regression: the Reapply dialog has no form and no Submit control.

    Clicking into it left the traversal with nothing to fill, so it fell to
    the Next/Review branch and clicked around the dialog until MAX_STEPS --
    the "clicking Reapply again and again" loop.
    """
    b = FakeBrowser()
    page = _apply_button_page(text, aria)
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=True)

    assert report.aborted_reason == "already applied (LinkedIn offers Reapply)"
    assert report.submitted is False
    assert b.clicks == []          # never even opened the dialog


def test_a_normal_easy_apply_button_is_still_clicked(profile):
    fields = [FakeField(selector="#city", label="What is your current city?")]
    b = FakeBrowser()
    page = _form_page(fields)
    page.elements[SEL["text_input"]] = [_input_el(fields[0])]
    b.add_page(page)
    b.goto(page.url)

    report = EasyApplyFiller(b, profile).run(submit=False)
    assert report.aborted_reason is None
    assert report.reached_submit is True
