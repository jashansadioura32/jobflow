# JobFlow

Screens LinkedIn jobs against rules you control, fills Easy Apply forms from
your profile, and keeps an audit log of every decision.

**This submits real applications under your name.** Read the
[Safety](#safety) section before your first live run.

---

## Requirements

- Python 3.11 or newer
- Google Chrome
- A LinkedIn account

## Install

```bash
git clone https://github.com/jashansadioura32/jobflow.git
cd jobflow

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -e ".[dev,browser]"
pip install -e ".[llm]"         # optional: AI fit scoring
```

## Configure

Copy both example files and edit them:

```bash
cd config
cp profile.example.yaml profile.yaml     # Windows: copy profile.example.yaml profile.yaml
cp search.example.yaml search.yaml       # Windows: copy search.example.yaml search.yaml
cd ..
```

**`config/profile.yaml`** — your name, contact details, salary, notice
period, and the absolute path to your resume PDF. Every value here is typed
into real application forms.

**`config/search.yaml`** — the job titles to search for, location filter, and
the rules for skipping postings.

Check it loaded correctly:

```bash
jobflow validate
```

This prints the values it will use and fails with a specific message if
anything is missing or malformed. It will not pass until
`professional.resume_path` points at a PDF that actually exists — that path
is the file uploaded to every application, so it is checked at startup rather
than mid-run.

Both `profile.yaml` and `search.yaml` are gitignored, so your details stay on
your machine.

## Run

**1. Screen sample postings — no browser, nothing sent:**

```bash
jobflow screen data/sample_postings.json
```

**2. Dry run against real LinkedIn — fills forms, sends nothing:**

```bash
jobflow apply --profile-dir ~/jobflow-chrome --saved-profile -v
```

Chrome opens. Sign in to LinkedIn once, then press Enter in the terminal.
JobFlow searches, screens, fills every form, and discards it. Use this to
check that your answers are mapping correctly before anything is sent.

The `--saved-profile` flag keeps your login in a dedicated Chrome profile, so
you only sign in on the first run.

**3. Live run — this sends applications:**

```bash
jobflow apply --profile-dir ~/jobflow-chrome --saved-profile --live
```

Type `live` at the confirmation prompt.

## Modes

| Command | What it does |
|---|---|
| `jobflow apply` | Dry run. Fills forms, discards them, sends nothing. |
| `jobflow apply --live` | Fills and **submits automatically**. Guesses at questions your profile cannot answer. |
| `jobflow apply --live --review` | Fills, then waits for **you** to click Submit in the browser. Never guesses. |

While a run is in progress, a small **Quit** window floats above the browser.
Click it to stop after the current application finishes. When the run ends —
by Quit, by hitting the daily cap, or by finishing — you get a summary of
what happened, in the terminal and in a window.

## Useful flags

| Flag | Effect |
|---|---|
| `-v` | Show each question and the answer chosen for it |
| `--review` | Approve each application yourself (see Modes) |
| `--review-timeout 300` | Seconds to wait for your decision, with `--review` |
| `--use-llm` | Score job-description fit with an LLM (needs `OPENAI_API_KEY`) |
| `--keep-open` | Leave Chrome open when the run ends |

Press Ctrl+C at any time. You still get a summary.

## Other commands

```bash
jobflow validate    # check your config and print the resolved values
jobflow terms       # print the search plan in priority order
jobflow report      # summarize the application log
```

## Where your data goes

| Path | Contents |
|---|---|
| `data/evaluations_*.jsonl` | Every decision, with the evidence behind it |
| `data/applications.csv` | One row per posting evaluated |

Both are gitignored. To find out why a job was skipped, search the evidence
log for its title — each record lists the rule that rejected it.

## Safety

**Start with a low cap.** Set `daily_application_cap: 5` in `search.yaml` for
your first live run. If an answer is mapped wrongly, you would rather find
out after five applications than forty.

**`--live` guesses.** With no human in the loop, a required question your
profile cannot answer gets a plausible guess rather than blocking the
application. Every guess is logged — run with `-v` and check them. Use
`--live --review` if you want to approve each application yourself.

**JobFlow never handles your password.** You sign in manually; the session is
reused from a Chrome profile on your machine.

**Automating applications may conflict with LinkedIn's terms of service**, and
accounts do get restricted. That risk is yours to accept.

## Troubleshooting

**`Missing config file: profile.yaml`** — you skipped the copy step in
[Configure](#configure).

**`no Easy Apply button`** in the log — the posting is an external-apply job,
or LinkedIn changed its markup. Selectors live in the `SEL` dict at the top of
`jobflow/adapters/linkedin.py`.

**Chrome won't start** — close all Chrome windows, or use `--saved-profile`
with a `--profile-dir` that no other Chrome instance is using.

**A question was answered wrongly** — add a rule to
`jobflow/core/answers.py`. Rules are ordered and the first match wins, so put
specific patterns before general ones.

## Tests

```bash
pytest
```

156 tests, no network and no LinkedIn account required — the whole
search-to-submit loop runs against an in-memory fake browser.

## License

MIT — see [LICENSE](LICENSE).
