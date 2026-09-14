"""Browser abstraction.

All DOM interaction goes through the `Browser` protocol. The adapter logic
above it — result collection, form traversal, answer resolution — is written
against this interface, so it can be tested against a fake without a real
browser or a real account.

`SeleniumBrowser` is the only place Selenium is imported.

Author: Jashan Sadioura
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


@dataclass
class Element:
    """A page element, identified opaquely by handle.

    Concrete browsers map `handle` to whatever they need (a WebElement, a
    dict, an index). Adapter code never inspects it.
    """
    handle: object
    tag: str = ""
    text: str = ""
    attributes: dict[str, str] = field(default_factory=dict)

    def attr(self, name: str, default: str = "") -> str:
        return self.attributes.get(name, default)


class ElementNotFound(Exception):
    """Raised when a required element is absent."""


@runtime_checkable
class Browser(Protocol):
    """Minimal surface the adapter needs. Kept deliberately small."""

    def goto(self, url: str) -> None: ...
    def current_url(self) -> str: ...
    def find(self, selector: str) -> Element | None: ...
    def find_all(self, selector: str) -> list[Element]: ...
    def find_within(self, root: Element, selector: str) -> Element | None: ...
    def find_all_within(self, root: Element, selector: str) -> list[Element]: ...
    def click(self, element: Element) -> None: ...
    def type_text(self, element: Element, text: str) -> None: ...
    def select_option(self, element: Element, value: str) -> None: ...
    def options_for(self, element: Element) -> list[str]: ...
    def upload_file(self, element: Element, path: str) -> None: ...
    def text_of(self, element: Element) -> str: ...
    def wait_for(self, selector: str, timeout: float = 10.0) -> Element | None: ...
    def wait_while_present(self, selector: str, timeout: float = 600.0) -> bool: ...
    def scroll_into_view(self, element: Element) -> None: ...
    def sleep(self, seconds: float) -> None: ...
    def press_escape(self) -> None: ...
    def commit_typeahead(self, element: Element, pause: float = 2.0) -> None: ...
    def find_by_text(self, text: str) -> Element | None: ...
    def find_by_text_within(self, root: Element, text: str) -> Element | None: ...
    def quit(self) -> None: ...


class SeleniumBrowser:
    """Selenium-backed Browser.

    Selectors are CSS. Every method degrades to None/[] rather than raising,
    so callers handle absence explicitly instead of catching driver
    exceptions everywhere.
    """

    def __init__(self, driver, default_timeout: float = 10.0) -> None:
        self._driver = driver
        self._timeout = default_timeout

    @staticmethod
    def _default_user_data_dir() -> str:
        """Return the standard local Chrome user-data directory.

        On Windows this is the folder that contains the signed-in Default
        profile, so Selenium can reuse the browser profile where Gmail and
        LinkedIn sessions already exist.
        """
        import os
        from pathlib import Path

        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            return str(Path(local_appdata) / "Google" / "Chrome" / "User Data")

        return str(Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data")

    @staticmethod
    def _resolve_profile(profile_dir: str | None) -> tuple[str, str]:
        """Map arbitrary profile_dir input to (user-data-dir, profile-directory).

        No profile_dir means the standard Chrome user-data root. A path to a
        `Default` profile folder is normalised to its parent, since Chrome
        wants the root plus a profile name rather than the profile itself.
        Any other path is treated as a user-data root in its own right.
        """
        from pathlib import Path

        if not profile_dir:
            return SeleniumBrowser._default_user_data_dir(), "Default"

        p = Path(profile_dir).expanduser()
        if p.name == "Default":
            return str(p.parent), "Default"
        return str(p), "Default"

    # Session state lives in these; the rest of a Chrome profile is cache we
    # do not need and would spend gigabytes copying. Cookies moved to
    # Network/Cookies in Chrome 96 -- the old top-level path is long gone.
    _SESSION_FILES = ("Network/Cookies", "Login Data", "Web Data",
                      "Preferences", "Local State")

    # Chrome holds an exclusive lock on the live cookie DB, so a copy taken
    # while the browser is open usually misses it. That costs one manual
    # sign-in, not a failed run.
    _COOKIE_FILE = "Network/Cookies"

    @classmethod
    def _clone_profile(cls, user_data_dir: str, profile_directory: str) -> str:
        """Copy the signed-in profile into a temp user-data dir and return it.

        Chrome locks a user-data directory to one process, so pointing
        Selenium at the live profile fails outright whenever the user has
        Chrome open. Cloning keeps the LinkedIn session without contending
        for that lock, and leaves the real profile untouched -- a crashed
        run can never corrupt it.
        """
        import shutil
        import tempfile
        from pathlib import Path

        src_root = Path(user_data_dir)
        dest_root = Path(tempfile.mkdtemp(prefix="jobflow-chrome-"))
        src_profile = src_root / profile_directory
        dest_profile = dest_root / profile_directory
        dest_profile.mkdir(parents=True, exist_ok=True)

        if not src_profile.is_dir():
            log.warning("Chrome profile %s not found; starting a clean profile. "
                        "You will need to sign in once.", src_profile)
            return str(dest_root)

        copied = 0
        cookies_copied = False
        for name in cls._SESSION_FILES:
            # "Local State" sits at the root; the rest live in the profile.
            base_src, base_dest = (src_root, dest_root) if name == "Local State" \
                else (src_profile, dest_profile)
            src = base_src / name
            if not src.exists():
                continue
            dest = base_dest / name
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dest)
                copied += 1
                if name == cls._COOKIE_FILE:
                    cookies_copied = True
            except OSError as e:
                # A locked file is not fatal: worst case the clone lacks a
                # session and the user signs in manually.
                log.debug("Could not copy %s from the Chrome profile: %s", name, e)

        if not cookies_copied:
            log.info("Chrome is holding the cookie database open, so the saved "
                     "LinkedIn session could not be copied. Sign in once in the "
                     "window that opens; close Chrome first to carry it over.")
        log.debug("Cloned %d session file(s) to %s", copied, dest_root)
        return str(dest_root)

    @classmethod
    def build_chrome_options(
        cls,
        headless: bool = False,
        profile_dir: str | None = None,
        clone_profile: bool = True,
    ) -> "Options":
        """Create the Selenium Chrome options.

        Resolves the signed-in Chrome profile, then by default hands Selenium
        a throwaway *copy* of it rather than the profile itself: Chrome locks
        a user-data directory to one process, so targeting the live profile
        fails whenever the user has Chrome open. Pass `clone_profile=False`
        to point at the resolved directory directly, which requires that no
        other Chrome instance is using it.
        """
        from selenium.webdriver.chrome.options import Options

        opts = Options()
        if headless:
            opts.add_argument("--headless=new")

        user_data_dir, profile_directory = cls._resolve_profile(profile_dir)
        if clone_profile:
            user_data_dir = cls._clone_profile(user_data_dir, profile_directory)
        opts.add_argument(f"--user-data-dir={user_data_dir}")
        opts.add_argument(f"--profile-directory={profile_directory}")

        opts.add_argument("--window-size=1400,1000")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        return opts

    # -- construction ----------------------------------------------------

    @classmethod
    def launch(
        cls,
        headless: bool = False,
        profile_dir: str | None = None,
        default_timeout: float = 10.0,
        clone_profile: bool = True,
    ) -> "SeleniumBrowser":
        """Start Chrome from a copy of the signed-in Chrome profile.

        The copy carries the existing LinkedIn session across, so sign-in is
        not needed every run, while leaving the real profile free for the
        browser the user already has open. `profile_dir` overrides which
        profile is copied; `clone_profile=False` uses it in place.
        """
        from selenium import webdriver

        opts = cls.build_chrome_options(headless=headless, profile_dir=profile_dir,
                                        clone_profile=clone_profile)
        driver = webdriver.Chrome(options=opts)
        return cls(driver, default_timeout=default_timeout)

    # -- navigation ------------------------------------------------------

    def goto(self, url: str) -> None:
        self._driver.get(url)

    def current_url(self) -> str:
        return self._driver.current_url

    # -- finding ---------------------------------------------------------

    def _wrap(self, raw) -> Element:
        try:
            tag = raw.tag_name or ""
        except Exception:
            tag = ""
        try:
            text = raw.text or ""
        except Exception:
            text = ""
        attrs: dict[str, str] = {}
        for name in (
            "id", "name", "type", "value", "aria-label", "href",
            "class", "placeholder", "required", "aria-required",
        ):
            try:
                v = raw.get_attribute(name)
                if v:
                    attrs[name] = v
            except Exception:
                continue
        return Element(handle=raw, tag=tag, text=text, attributes=attrs)

    def find(self, selector: str) -> Element | None:
        from selenium.common.exceptions import NoSuchElementException, WebDriverException
        from selenium.webdriver.common.by import By
        try:
            return self._wrap(self._driver.find_element(By.CSS_SELECTOR, selector))
        except (NoSuchElementException, WebDriverException):
            return None

    def find_all(self, selector: str) -> list[Element]:
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.common.by import By
        try:
            return [
                self._wrap(e)
                for e in self._driver.find_elements(By.CSS_SELECTOR, selector)
            ]
        except WebDriverException:
            return []

    def find_within(self, root: Element, selector: str) -> Element | None:
        found = self.find_all_within(root, selector)
        return found[0] if found else None

    def find_all_within(self, root: Element, selector: str) -> list[Element]:
        """Search inside `root` only.

        Selenium scopes a CSS search to the element it is called on, so this
        is find_all rooted at a single form group rather than the document.
        Without it, a multi-question form gives every question the first
        question's answer.
        """
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.common.by import By
        try:
            return [
                self._wrap(e)
                for e in root.handle.find_elements(By.CSS_SELECTOR, selector)
            ]
        except (WebDriverException, AttributeError):
            return []

    def wait_for(self, selector: str, timeout: float | None = None) -> Element | None:
        from selenium.common.exceptions import TimeoutException, WebDriverException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
        try:
            raw = WebDriverWait(self._driver, timeout or self._timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            return self._wrap(raw)
        except (TimeoutException, WebDriverException):
            return None

    def wait_for_within(
        self, root: Element, selector: str, timeout: float = 10.0
    ) -> Element | None:
        """Poll for `selector` inside `root` until it appears or time runs out.

        LinkedIn renders the job detail pane asynchronously after a card is
        clicked, so a bare find right after the click cannot distinguish a
        missing element from one that has not arrived yet.
        """
        deadline = time.monotonic() + timeout
        while True:
            found = self.find_within(root, selector)
            if found is not None:
                return found
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.25)

    def wait_while_present(self, selector: str, timeout: float = 600.0) -> bool:
        """Block until `selector` disappears. True if it went, False on timeout.

        The long default is deliberate: this waits on a person reading an
        application, not on a page load.
        """
        from selenium.common.exceptions import TimeoutException, WebDriverException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
        try:
            WebDriverWait(self._driver, timeout).until(
                EC.invisibility_of_element_located((By.CSS_SELECTOR, selector))
            )
            return True
        except TimeoutException:
            return False
        except WebDriverException:
            # A closed window or dead session counts as gone, not as an error.
            return True

    # -- interaction -----------------------------------------------------

    def click(self, element: Element) -> None:
        from selenium.common.exceptions import (
            ElementClickInterceptedException, ElementNotInteractableException,
        )
        try:
            element.handle.click()
        except (ElementClickInterceptedException, ElementNotInteractableException):
            # Overlays and off-screen elements are common; JS click is the
            # reliable fallback.
            self._driver.execute_script("arguments[0].click();", element.handle)

    def type_text(self, element: Element, text: str) -> None:
        element.handle.clear()
        element.handle.send_keys(text)

    def select_option(self, element: Element, value: str) -> None:
        from selenium.webdriver.support.select import Select
        Select(element.handle).select_by_visible_text(value)

    def options_for(self, element: Element) -> list[str]:
        from selenium.webdriver.support.select import Select
        try:
            return [o.text.strip() for o in Select(element.handle).options if o.text.strip()]
        except Exception:
            return []

    def upload_file(self, element: Element, path: str) -> None:
        element.handle.send_keys(path)

    def text_of(self, element: Element) -> str:
        try:
            return element.handle.text or ""
        except Exception:
            return element.text

    def scroll_into_view(self, element: Element) -> None:
        try:
            self._driver.execute_script(
                "arguments[0].scrollIntoView({block:'center'});", element.handle
            )
        except Exception:
            pass

    def commit_typeahead(self, element: Element, pause: float = 2.0) -> None:
        """Pick the first suggestion from an autocomplete dropdown.

        LinkedIn's city and address fields are typeaheads: typing "Gurugram"
        leaves the box looking filled while the form holds no value at all,
        because nothing was chosen from the list. The value is only committed
        when a suggestion is selected, so after typing we wait for the
        dropdown to render, then press Down and Enter to take the first one.

        The pause is not optional -- the list is fetched over the network,
        and pressing Down before it appears selects nothing.
        """
        from selenium.webdriver.common.action_chains import ActionChains
        from selenium.webdriver.common.keys import Keys
        try:
            time.sleep(pause)
            actions = ActionChains(self._driver)
            actions.send_keys(Keys.ARROW_DOWN)
            actions.send_keys(Keys.ENTER)
            actions.perform()
        except Exception as e:
            # A field that was not a typeahead after all keeps the typed
            # text; failing here must not abandon the whole application.
            log.debug("Could not commit typeahead selection: %s", e)

    def press_escape(self) -> None:
        """Send Escape to the page, closing whatever modal is open."""
        from selenium.webdriver.common.action_chains import ActionChains
        from selenium.webdriver.common.keys import Keys
        try:
            ActionChains(self._driver).send_keys(Keys.ESCAPE).perform()
        except Exception:
            pass

    @staticmethod
    def _text_xpath(text: str, prefix: str) -> str:
        return (
            f'{prefix}button[normalize-space(.)="{text}"]'
            f' | {prefix}span[normalize-space(.)="{text}"]'
            f' | {prefix}*[@aria-label="{text}"]'
        )

    def find_by_text(self, text: str) -> Element | None:
        """Find a clickable element by its exact visible text.

        CSS cannot match on text, but LinkedIn's buttons ("Next", "Review",
        "Submit application", "Discard") are only reliably identified that
        way -- their classes and data-control-name attributes change, the
        words do not.
        """
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.common.by import By
        try:
            found = self._driver.find_elements(
                By.XPATH, self._text_xpath(text, "//"))
        except WebDriverException:
            return None
        return self._wrap(found[0]) if found else None

    def find_by_text_within(self, root: Element, text: str) -> Element | None:
        """Find by visible text inside `root` only.

        The modal's own Next/Review/Submit must not be confused with
        same-worded controls elsewhere on the page.
        """
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.common.by import By
        try:
            found = root.handle.find_elements(
                By.XPATH, self._text_xpath(text, ".//"))
        except (WebDriverException, AttributeError):
            return None
        return self._wrap(found[0]) if found else None

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def quit(self) -> None:
        try:
            self._driver.quit()
        except Exception:
            pass


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
