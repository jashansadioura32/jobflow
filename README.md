# JobFlow

JobFlow applies to jobs on LinkedIn for you.

You tell it which job titles you want and what your details are. It searches
LinkedIn, throws away the jobs that don't suit you, fills in the Easy Apply
form, and sends it. It writes down every decision it made, so you can always
check why it applied to something or why it skipped it.

> **Please read this first.** JobFlow can send real job applications in your
> name. Follow the steps below in order — step 5 practises with the browser
> without sending anything, so you can see what it will do before it does it.

---

## What you need before you start

Three things:

1. **Python 3.11 or newer.** To check, open a terminal and type
   `python --version`. If you get an error or a number below 3.11, install it
   from [python.org](https://www.python.org/downloads/).
2. **Google Chrome.**
3. **A LinkedIn account.**

Everything below is typed into a terminal (Command Prompt or PowerShell on
Windows, Terminal on Mac).

---

## Step 1 — Download JobFlow

```bash
git clone https://github.com/jashansadioura32/jobflow.git
cd jobflow
```

You are now inside the JobFlow folder. Every other command on this page is
run from here.

## Step 2 — Create a private workspace for it

This keeps JobFlow's code separate from anything else on your computer, so it
can't break your other programs.

```bash
python -m venv .venv
```

Then switch into it:

```bash
.venv\Scripts\activate
```

On Mac or Linux, use this instead:

```bash
source .venv/bin/activate
```

You'll know it worked because `(.venv)` appears at the start of your terminal
line. **You need to do this every time you open a new terminal to use
JobFlow.**

## Step 3 — Install it

```bash
pip install -e ".[dev,browser]"
```

This takes a minute or two. Ignore any warnings as long as the last line
doesn't say "error".

## Step 4 — Tell JobFlow about you

JobFlow needs two files: one about you, one about the jobs you want. Ready-made
examples are included, so you just copy them and fill in your own details.

**On Windows:**

```bash
copy config\profile.example.yaml config\profile.yaml
copy config\search.example.yaml config\search.yaml
```

**On Mac or Linux:**

```bash
cp config/profile.example.yaml config/profile.yaml
cp config/search.example.yaml config/search.yaml
```

Now open those two new files in any text editor and replace the example values
with yours.

**`config/profile.yaml`** is about you — your name, email, phone, city,
current and expected salary, notice period, and where your CV is saved.
Whatever you write here gets typed into real application forms, so check it
carefully. One value needs particular care:

```yaml
resume_path: "C:/Users/YourName/Documents/my_cv.pdf"
```

That has to be the full path to a PDF that really exists on your computer, and
use forward slashes `/` even on Windows.

**`config/search.yaml`** is about the jobs — which titles to search for, which
country, and when to skip a job. You can change these any time.

Now check JobFlow understood everything:

```bash
jobflow validate
```

If it prints your details back to you, you're ready. If something is wrong, it
tells you exactly which line to fix. The most common message is that it can't
find your CV — that means the `resume_path` above is pointing at a file that
isn't there.

Your two files stay on your computer. They are never uploaded anywhere.

## Step 5 — Practise runs (nothing gets sent)

**First, without even opening a browser:**

```bash
jobflow screen data/sample_postings.json
```

This runs five made-up job adverts through your rules and shows you which ones
it would apply to and why it rejected the others. It's a quick way to see
whether your settings make sense.

**Then, with the real LinkedIn but still sending nothing:**

```bash
jobflow apply --profile-dir ~/jobflow-chrome --saved-profile -v
```

Chrome opens. Log in to LinkedIn yourself, then come back to the terminal and
press Enter. JobFlow now searches for your job titles, opens each one, and
fills in the whole form — then throws it away without sending it.

Watch what it types. The `-v` shows you every question it was asked and the
answer it chose. If it fills something in wrongly, this is where you find out,
with no harm done. **Don't skip this step.**

You only have to log in once. `--saved-profile` remembers you for next time.

## Step 6 — Apply for real

When the practice run looks right:

```bash
jobflow apply --profile-dir ~/jobflow-chrome --saved-profile --live
```

It asks you to type `live` to confirm. Then it applies to jobs for real.

**Before your first live run, lower the limit.** Open `config/search.yaml` and
set:

```yaml
daily_application_cap: 5
```

Five applications is enough to spot a mistake. You can raise it once you trust
it.

---

## Two ways to apply

**`--live` sends applications by itself.** If a form asks something your
profile doesn't cover, it picks a sensible-looking answer and carries on. It
tells you every time it did this — look for the word "guessed" in the output.

**`--live --review` puts you in charge.** It fills the form and then stops,
leaving it open on screen. You read it and click Submit yourself, or close the
box to skip that job. It never guesses in this mode.

Use `--review` until you trust it.

## Stopping a run

A small **Quit** button floats on top of your screen while JobFlow is working.
Click it and JobFlow finishes the application it's on, then stops — it never
abandons a form half-filled. Pressing Ctrl+C in the terminal also works.

It stops on its own when it reaches your daily limit, or when LinkedIn says
you've applied to enough jobs for one day.

However it ends, you get a summary — how many it applied to, how many it
skipped and why — both in the terminal and in a window.

## Handy extras

| Add this | To do this |
|---|---|
| `-v` | See every question and the answer JobFlow chose |
| `--review` | Check each application yourself before it's sent |
| `--review-timeout 300` | Wait 5 minutes for your decision instead of 10 |
| `--keep-open` | Leave Chrome open when it finishes |
| `--use-llm` | Let an AI rate how well each job matches you (needs an OpenAI key) |

Other things you can run:

```bash
jobflow validate    # check your details are correct
jobflow terms       # list the job titles it will search for, in order
jobflow report      # how many jobs you've applied to so far
```

## Checking what it did

JobFlow keeps two files in the `data` folder:

- **`applications.csv`** — one line per job, openable in Excel
- **`evaluations_*.jsonl`** — the full reasoning behind every decision

To find out why a job was skipped, search these for the job title. Each entry
names the exact rule that rejected it. Both files stay on your computer.

## When something goes wrong

**"Missing config file: profile.yaml"**
You missed Step 4. Copy the two example files.

**"resume not found"**
The `resume_path` in `config/profile.yaml` is pointing at a file that isn't
there. Check the spelling and use forward slashes `/`.

**Chrome won't open**
Close every Chrome window and try again. Chrome won't let two programs share
one profile.

**"no Easy Apply button" appears a lot**
Those jobs send you to the company's own website instead, which JobFlow can't
fill in. It skips them on purpose. If it happens on *every* job, LinkedIn has
probably changed its website — see the note at the bottom of this page.

**It answered a question wrongly**
Open `jobflow/core/answers.py` and add a rule for that question. The rules are
checked from top to bottom and the first match wins, so put specific ones
above general ones.

## Things you should know

**It applies in your name.** Every application carries your name and your CV.
Practise with Step 5 first.

**`--live` guesses.** When nobody is watching, an unanswerable question gets a
plausible guess rather than stopping the whole application. Every guess is
written down. Use `--review` if you'd rather decide yourself.

**JobFlow never sees your password.** You log in yourself, in your own browser.

**LinkedIn may not like this.** Automated applications can go against
LinkedIn's terms of service, and accounts do sometimes get restricted. That's
your decision to make.

**LinkedIn changes its website.** JobFlow finds buttons by looking for them on
the page, so a redesign can break it. If applications suddenly stop working,
that's usually why. The parts that need updating are all in one place, at the
top of `jobflow/adapters/linkedin.py`.

## For developers

```bash
pytest
```

173 tests. They run against a fake browser, so no internet connection and no
LinkedIn account are needed.

## Licence

MIT — free to use and change. See [LICENSE](LICENSE).
