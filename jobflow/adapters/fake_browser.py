"""In-memory Browser implementation for tests.

Pages are plain dicts of selector -> elements, so adapter logic can be
exercised deterministically: result collection, form traversal, unanswerable
questions, and submission, with no network and no real account.

Author: Jashan Sadioura
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jobflow.adapters.browser import Element


@dataclass
class FakeField:
    """A form field the fake renders and records writes to."""
    selector: str
    label: str
    kind: str = "text"              # text | select | file | radio
    options: list[str] = field(default_factory=list)
    required: bool = False
    value: str = ""


@dataclass
class FakePage:
    url: str
    elements: dict[str, list[Element]] = field(default_factory=dict)
    fields: list[FakeField] = field(default_factory=list)
    text: str = ""


class FakeBrowser:
    """Scripted Browser. Records every interaction for assertions."""

    def __init__(self, pages: dict[str, FakePage] | None = None) -> None:
        self.pages = pages or {}
        self._url = "about:blank"
        self.clicks: list[str] = []
        self.typed: dict[str, str] = {}
        self.selected: dict[str, str] = {}
        self.uploads: list[str] = []
        self.visited: list[str] = []
        self.quit_called = False
        self.waited: list[str] = []
        # Fields whose autocomplete dropdown was committed with Down+Enter.
        self.typeahead_committed: list[str] = []
        # selector -> polls it survives before "the human" closes it.
        self.closes_after: dict[str, int] = {}

    # -- scripting helpers ----------------------------------------------

    def add_page(self, page: FakePage) -> None:
        self.pages[page.url] = page

    @property
    def page(self) -> FakePage | None:
        return self.pages.get(self._url)

    def _field_for(self, element: Element) -> FakeField | None:
        page = self.page
        if not page:
            return None
        for f in page.fields:
            if f.selector == element.attr("data-selector"):
                return f
        return None

    # -- Browser protocol -----------------------------------------------

    def goto(self, url: str) -> None:
        self._url = url
        self.visited.append(url)

    def current_url(self) -> str:
        return self._url

    def find(self, selector: str) -> Element | None:
        found = self.find_all(selector)
        return found[0] if found else None

    def find_all(self, selector: str) -> list[Element]:
        page = self.page
        if not page:
            return []
        if selector in page.elements:
            return page.elements[selector]
        # Form fields are addressable by their own selector.
        return [
            Element(
                handle=f,
                tag="select" if f.kind == "select" else "input",
                text=f.label,
                attributes={
                    "data-selector": f.selector,
                    "aria-label": f.label,
                    "type": f.kind,
                    "required": "true" if f.required else "",
                },
            )
            for f in page.fields
            if f.selector == selector
        ]

    def find_within(self, root: Element, selector: str) -> Element | None:
        found = self.find_all_within(root, selector)
        return found[0] if found else None

    def find_all_within(self, root: Element, selector: str) -> list[Element]:
        """Find inside one group only.

        The fake models scoping with an explicit `owner` attribute: an
        element belongs to the group whose `data-selector` it names. Without
        this, a document-wide lookup and a scoped one are indistinguishable
        here, which is exactly how the real filler shipped a bug where every
        question on a form received the first question's answer.
        """
        key = root.attr("data-selector")
        return [
            el for el in self.find_all(selector)
            if el.attr("owner") == key
        ]

    def wait_for(self, selector: str, timeout: float = 10.0) -> Element | None:
        return self.find(selector)

    def wait_for_within(
        self, root: Element, selector: str, timeout: float = 10.0
    ) -> Element | None:
        self.waited.append(selector)
        return self.find_within(root, selector)

    def wait_while_present(self, selector: str, timeout: float = 600.0) -> bool:
        """Simulate a human closing something, without real waiting.

        `closes_after` scripts how many polls a selector survives, so tests
        can drive both the "human acted" and "timed out" paths.
        """
        self.waited.append(selector)
        remaining = self.closes_after.get(selector)
        if remaining is None:
            # Nothing scripted: treat it as already gone.
            return self.find(selector) is None
        if remaining <= 1:
            self.closes_after[selector] = 0
            page = self.page
            if page and selector in page.elements:
                page.elements[selector] = []
            return True
        self.closes_after[selector] = remaining - 1
        return False

    def click(self, element: Element) -> None:
        self.clicks.append(element.attr("data-selector") or element.text or "?")

    def type_text(self, element: Element, text: str) -> None:
        key = element.attr("data-selector") or element.attr("aria-label")
        self.typed[key] = text
        f = self._field_for(element)
        if f:
            f.value = text

    def select_option(self, element: Element, value: str) -> None:
        key = element.attr("data-selector") or element.attr("aria-label")
        self.selected[key] = value
        f = self._field_for(element)
        if f:
            f.value = value

    def options_for(self, element: Element) -> list[str]:
        f = self._field_for(element)
        return list(f.options) if f else []

    def upload_file(self, element: Element, path: str) -> None:
        self.uploads.append(path)

    def text_of(self, element: Element) -> str:
        return element.text

    def scroll_into_view(self, element: Element) -> None:
        pass

    def press_escape(self) -> None:
        self.clicks.append("<escape>")

    def commit_typeahead(self, element: Element, pause: float = 2.0) -> None:
        """Record that a dropdown suggestion was accepted for this field."""
        key = element.attr("data-selector") or element.attr("aria-label")
        self.typeahead_committed.append(key)

    def find_by_text(self, text: str) -> Element | None:
        """Match an element by visible text, as the real browser's XPath does."""
        page = self.page
        if not page:
            return None
        for elements in page.elements.values():
            for el in elements:
                if el.text.strip() == text or el.attr("aria-label") == text:
                    return el
        return None

    def find_by_text_within(self, root: Element, text: str) -> Element | None:
        """Match by text inside one container, using the same ownership model."""
        page = self.page
        if not page:
            return None
        key = root.attr("data-selector")
        for elements in page.elements.values():
            for el in elements:
                if el.text.strip() != text and el.attr("aria-label") != text:
                    continue
                # An element with no owner is treated as belonging to the
                # container, matching how a real modal nests its buttons.
                if el.attr("owner") in ("", key):
                    return el
        return None

    def sleep(self, seconds: float) -> None:
        pass

    def quit(self) -> None:
        self.quit_called = True


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
