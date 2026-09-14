"""Audit logging.

Every evaluation is recorded with the evidence behind it, whether or not an
application followed. The point is being able to answer "why was this job
skipped?" weeks later without re-running anything.

Two sinks:
  * JSONL — full evidence, one object per evaluation, for analysis.
  * CSV    — flat application log, for spreadsheet review.

Author: Jashan Sadioura
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from jobflow.core.models import Decision, Evaluation

CSV_COLUMNS = [
    "timestamp", "job_id", "title", "company", "location",
    "decision", "skip_reason", "fit_score", "search_term", "url",
]


class AuditLog:
    """Append-only evaluation log."""

    def __init__(self, data_dir: Path, run_id: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.evidence_path = self.data_dir / f"evaluations_{self.run_id}.jsonl"
        self.applications_path = self.data_dir / "applications.csv"
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if self.applications_path.exists():
            return
        with self.applications_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)

    def record(self, ev: Evaluation, search_term: str = "") -> None:
        ts = datetime.now(timezone.utc).isoformat()

        record = {
            "timestamp": ts,
            "run_id": self.run_id,
            "search_term": search_term,
            "decision": ev.decision.value,
            "skip_reason": ev.skip_reason.value if ev.skip_reason else None,
            # The full description is far too large to keep per posting, but
            # dropping it entirely made skips unauditable: "requires 12y" with
            # no text behind it cannot be checked. A prefix is the compromise.
            "posting": {
                **ev.posting.model_dump(exclude={"description"}),
                "description_excerpt": (ev.posting.description or "")[:600],
            },
            "findings": [f.model_dump() for f in ev.findings],
            "fit": ev.fit.model_dump() if ev.fit else None,
        }
        with self.evidence_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with self.applications_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                ts,
                ev.posting.job_id,
                ev.posting.title,
                ev.posting.company,
                ev.posting.location,
                ev.decision.value,
                ev.skip_reason.value if ev.skip_reason else "",
                ev.fit.score if ev.fit else "",
                search_term,
                ev.posting.url,
            ])

    def applied_job_ids(self) -> set[str]:
        """Job IDs already applied to, across all previous runs."""
        if not self.applications_path.exists():
            return set()
        ids: set[str] = set()
        with self.applications_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("decision") == Decision.APPLY.value:
                    jid = (row.get("job_id") or "").strip()
                    if jid:
                        ids.add(jid)
        return ids

    def summarize(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        if not self.evidence_path.exists():
            return counts
        with self.evidence_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = rec.get("skip_reason") or rec.get("decision") or "unknown"
                counts[key] = counts.get(key, 0) + 1
        return counts


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
