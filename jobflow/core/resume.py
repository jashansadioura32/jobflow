"""Resume text extraction.

Pulls the plain text out of the resume PDF so the LLM worker can answer
questions from what the CV actually says, rather than from the short summary
in profile.yaml.

Two honest limits worth knowing:
  * A PDF exports as flat text with the visual layout stripped, so columns
    and tables can interleave. It is usually good enough to answer "have you
    used Snowflake", and it is not a faithful rendering of the document.
  * A scanned or image-only PDF yields nothing at all. That degrades to the
    profile summary rather than failing the run.

Author: Jashan Sadioura
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Bounds token spend: a CV beyond this is padding, and the model reads the
# most relevant parts first anyway.
MAX_RESUME_CHARS = 12000


def extract_text(path: Path) -> str:
    """Return the resume's text, or "" when it cannot be read.

    Never raises: a missing parser, an unreadable file or an image-only PDF
    all yield "", and the caller falls back to the profile summary.
    """
    if not path or not Path(path).exists():
        log.debug("No resume at %s", path)
        return ""

    suffix = Path(path).suffix.lower()
    if suffix == ".txt":
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError as e:
            log.warning("Could not read resume %s: %s", path, e)
            return ""

    if suffix != ".pdf":
        log.info("Resume is a %s file; only .pdf and .txt can be read for "
                 "question answering.", suffix or "(no extension)")
        return ""

    try:
        from pypdf import PdfReader
    except ImportError:
        log.info("Install pypdf to let the AI read your resume: "
                 "pip install -e \".[llm]\"")
        return ""

    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                # One unreadable page must not lose the rest.
                continue
        text = "\n".join(pages).strip()
    except Exception as e:
        log.warning("Could not extract text from %s: %s", path, e)
        return ""

    if not text:
        log.warning("No text found in %s -- it may be a scanned image. The AI "
                    "will use your profile summary instead.", Path(path).name)
        return ""

    # Collapse the runs of blank lines PDF extraction tends to produce.
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned: list[str] = []
    blank = False
    for ln in lines:
        if ln.strip():
            cleaned.append(ln)
            blank = False
        elif not blank:
            cleaned.append("")
            blank = True
    text = "\n".join(cleaned).strip()

    if len(text) > MAX_RESUME_CHARS:
        text = text[:MAX_RESUME_CHARS] + "\n[resume truncated]"
    return text


class ResumeText:
    """Lazily-extracted resume text, read once and reused.

    Extraction costs a file read and a parse, so doing it per question would
    be wasteful on a run that fills dozens of forms.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._text: str | None = None

    @property
    def text(self) -> str:
        if self._text is None:
            self._text = extract_text(self.path)
            if self._text:
                log.info("Read %d characters from %s for question answering.",
                         len(self._text), Path(self.path).name)
        return self._text

    def __bool__(self) -> bool:
        return bool(self.text)

    def __str__(self) -> str:
        """The extracted text itself.

        Without this, str() falls back to the default repr and callers that
        interpolate the object hand the model a memory address instead of a
        CV -- silently, because __bool__ still reports True.
        """
        return self.text


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
