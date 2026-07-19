# The Skeptical Evaluator — perturbation agent for a SWE-bench code fixer

A working prototype of the hackathon idea: instead of trusting a code-fixing
agent's single score, re-check every candidate fix under several
**meaning-preserving perturbations** and only certify it when the score
distribution is tight *and* high. High variance across perturbations is the
reward-hacking signature.

This directory implements the piece the demo turns on — a **Perturbation Agent**
that reads a real SWE-bench task and finds concrete, grounded places to change
the *input* to a code-fixing agent without changing what a correct fix is — plus
the evaluator gate and two experiments that test the mean+variance theory.

## What is real vs. modeled here

Being precise about this matters:

| Piece | Status |
|---|---|
| SWE-bench instances (repos, issue text, gold patch, test lists) | **Real** — pulled live from the HuggingFace datasets-server |
| README / repo files the agent reads to find perturbation sites | **Real** — fetched from GitHub at the exact base commit |
| The perturbation sites and the applied perturbations | **Real** — computed from the actual task |
| Static meaning-preservation check | **Real** — a perturbation is valid iff it never touches a file/line the gold or test patch edits |
| LLM-assisted paraphrase perturbations | **Real** — calls the Anthropic API (optional; off by default) |
| Per-perturbation pass/fail *scores* | **Modeled by default; real via Daytona.** The modeled fixers in `se/fixers.py` capture the one thing that differs between a genuine and a reward-hacking fix (which incidental feature it depends on). The `DaytonaFixer` path (below) replaces them with **ground-truth** SWE-bench pass/fail run in cloud sandboxes. |
| The RSI training experiment | **Real** learning run (logistic regression, gradient descent) on synthetic-but-principled data — it *demonstrates the mechanism* (spurious-feature shortcut learning), it is not trained on SWE-bench. |

## Perturbation taxonomy

Two sides of the pipeline are perturbable:

- **`fixer_input`** — what the fixer reads: the problem statement, the README,
  repo code. Tests whether the *fix* is robust to how the task is presented.
  - `ps_reorder_paragraphs`, `ps_prepend_pleasantry`, `ps_append_noise`,
    `ps_paraphrase` (LLM) — reword/reshuffle the issue, code blocks kept verbatim.
  - `readme_reorder_sections`, `readme_inject_note`, `readme_paraphrase` (LLM).
  - `noise_file`, `comment_inject` — add benign content to files no patch touches.
  - `metamorphic_rename_strings`, `metamorphic_rename_repro`,
    `metamorphic_shift_literals` — perturb the *example values* in the issue's
    reproduction (the exact place `ps_paraphrase` leaves alone). Meaning-preserving
    because the hidden tests are unchanged, so a genuine fix is unaffected while a
    fix that memorized those values is graded against values it never saw. This is
    the channel that catches **value-hardcoding**, the dominant generalization exploit.
- **`evaluator`** — how the fix is checked: e.g. `test_reorder`. Tests whether the
  *solution* overfit to a specific test execution ("make the first test pass").

Each site is reported with a concrete, grounded description (the actual README
section names, the real test-function count, the exact prose-paragraph count) and
a `safe` flag from the static meaning-preservation check.

**Coverage is over feature axes, not over tricks.** A perturbation catches an
exploit iff some axis varies the feature the exploit secretly depends on. Axes
covered today: `ps`, `metamorphic`, `readme`, `test_order`, `code_context`. The
known frontier (not yet implemented): **execution** (reseed / isolate / vary
env — catches flakiness & state-leak exploits), **specification** (held-out
property tests — the residual spec-gaming case perturbation alone cannot reach),
and **evaluator** (perturb an LLM judge — catches judge-gaming).

**Two gates.** `evaluate_skeptically` runs every safe perturbation (fixed budget).
`evaluate_adaptive` probes one perturbation per axis in priority order and stops
early on a confirmed collapse — robust fixes cost ~one probe per axis, blatant
exploits are rejected in a couple of samples.

### Channel coverage — measured, not assumed

`tests/test_coverage_survey.py` measures how often each metamorphic channel actually
fires across all 100 cached SWE-bench Lite instances:

| channel | fires on |
|---|--:|
| `metamorphic_rename_repro` (variables) | 2/100 (2%) |
| `metamorphic_shift_literals` (integers) | 5/100 (5%) |
| `metamorphic_rename_strings` (strings) | **63/100 (63%)** |
| **ANY metamorphic — before strings** | **5/100 (5%)** |
| **ANY metamorphic — after strings** | **64/100 (64%)** |

The flagship channel was effectively dead: it fired on **5 of 100 tasks**, and
astropy-12907 (the instance the original demo was built on) happened to be one of
them. Most repro examples use *strings* (`subdomain='admin'`, `SERVER_NAME:
'test.local'`), not integers. The channels are complementary — Flask fires the
string channel, astropy fires the variable/integer ones.

### Static patch inspection (not a perturbation)

`se/patch_inspect.py` reads a candidate diff and flags cheat-*shaped* code with no
execution at all: stub bodies, `except: pass`, `skip`/`xfail` markers, `exit(0)`,
test-file edits, and agent-created `conftest.py` (the documented evaluator-tampering
vector). This covers the class perturbation is blind to — tampering with the
*measurement* rather than failing to generalize.

Wired in via `apply_patch_inspection()`: findings are always recorded as a signal,
but only unambiguous tampering triggers a veto, so a buggy detector can't silently
kill a good patch. Validated against **25 real gold patches with zero false
positives**.

## Layout

```
se/
  swebench_data.py   pull real SWE-bench instances (no Docker)
  util.py            unified-diff parser + lightweight GitHub file fetch
  perturbations.py   THE PERTURBATION AGENT — find_sites() + apply() (incl. metamorphic)
  metamorphic_strings.py  string-level example-value renaming (the 63% channel)
  patch_inspect.py   static cheat detection on a diff (stubs, except:pass, conftest)
  fixers.py          pluggable fixer interface + modeled fixers + real-agent stub
  evaluator.py       the skeptical gate (fixed + adaptive) + failure bundle
  rsi_experiment.py  does perturbation-augmented training reduce reward hacking?
  llm.py             optional Anthropic wrapper for paraphrase perturbations
  daytona_runner.py  REAL SWE-bench eval in a Daytona sandbox (ground-truth pass/fail)
  agent.py           real code-fix agent (Anthropic tool-use, bash runs in the sandbox)
  daytona_fixer.py   DaytonaFixer — drop-in real fixer for evaluate_skeptically/adaptive
scripts/
  02_run_perturbation_agent.py       show the concrete sites found on a real task
  03_variance_demo.py                one score can't separate fixers; the distribution can
  04_rsi_training_demo.py            train with vs without perturbations, measure the gap
  05_metamorphic_and_adaptive.py     the value-hardcoding exploit + metamorphic channel + adaptive gate
  06_rsi_loops.py                    the RSI loop: naive Yes/No vs. perturbation evaluator, over rounds, + plot
  07_daytona_smoke_test.py           REAL eval pipeline check (gold resolves, empty doesn't) — no Anthropic cost
  08_daytona_real_round.py           REAL adaptive skeptical eval on one instance (agent + real grading)
out/
  rsi_accuracy.png                   accuracy-per-round plot written by script 06
web/
  skeptical-evaluator.html       the pitch site (problem, approach, variance chart, plant demo)
```

Open `web/skeptical-evaluator.html` directly in a browser, or serve it:
`npx http-server web -p 8877` then visit `http://127.0.0.1:8877/skeptical-evaluator.html`.

## Run it

```bash
pip install -r requirements.txt

# 0. The offline test suite — 8 tests, ~0.1s, no network/keys/cost
pytest tests/ -v          # add -s to see the channel coverage table

# 1. The perturbation agent on a real task (astropy__astropy-12907 by default)
python scripts/02_run_perturbation_agent.py
python scripts/02_run_perturbation_agent.py astropy__astropy-14182   # any Lite instance

# 2. Mean+variance separates genuine fixes from reward hacks
python scripts/03_variance_demo.py

# 3. Does training on perturbations actually help the model learn?
python scripts/04_rsi_training_demo.py

# 4. The value-hardcoding exploit the old channels miss, the metamorphic channel
#    that catches it, and the adaptive gate's sample count
python scripts/05_metamorphic_and_adaptive.py

# 5. The full RSI loop: naive Yes/No baseline (reward-hacks, plateaus) vs. the
#    perturbation evaluator (keeps improving); writes out/rsi_accuracy.png
python scripts/06_rsi_loops.py
```

## The headline result (script 06)

A greedy code-fix agent iterates for 6 rounds under two evaluators. **Reward hacking
is not coded into the agent — it emerges from what the evaluator rewards.** Under
the naive Yes/No check the cheapest way to flip fail→pass is to hard-code to the one
configuration checked, so the agent hacks and the loop freezes on it; under the
perturbation evaluator hacking is penalized, so the same agent generalizes.

| round | naive: believed | naive: truly resolved | skeptical: truly resolved |
|--:|--:|--:|--:|
| 0 | 0% | 0% | 0% |
| 3 | 95% | 30% | 88% |
| 6 | 100% | 35% | 100% |

The naive loop's own evaluator reports 100%, but only 35% of those fixes actually
generalize. `se/rsi_loop.py` holds the loop.

## Real evaluation via Daytona (ground-truth numbers)

`DaytonaFixer` replaces the modeled scores with **real SWE-bench pass/fail**. No
local Docker required — each evaluation creates a Daytona sandbox *directly from
the official SWE-bench instance image* (repo already at the base commit with exact
deps), applies the patch, runs the official eval script, and grades with SWE-bench's
own grader. All of that runs **inside the Linux sandbox**, because `swebench`
imports the Unix-only `resource` module and cannot run on Windows.

```
DaytonaFixer.score_under(inst, channel)
  -> Perturbation Agent builds the perturbed input for that channel
  -> SimpleSweAgent explores/edits in a sandbox   -> unified diff
  -> DaytonaRunner grades that diff on a FRESH sandbox (clean base + hidden tests)
  -> real % of F2P+P2P passing
```

Because it satisfies the same interface as the modeled `Fixer`, it drops straight
into `evaluate_skeptically` / `evaluate_adaptive` with no other changes.

**Setup:** `pip install daytona`, then `cp .env.example .env` and paste your keys
into `.env` (it is gitignored):

```
DAYTONA_API_KEY=...     # https://app.daytona.io/dashboard/keys — needed by 07 and 08
ANTHROPIC_API_KEY=...   # only script 08 (the agent) and paraphrase perturbations
```

`se/secrets.py` reads these at call time and hands them straight to the SDK
clients — keys are never printed or logged (`describe_key` shows only a masked
fingerprint). An OS environment variable of the same name takes precedence if you
prefer that. Do **not** `pip install swebench` locally — the runner installs it
inside each Linux sandbox.

```bash
# FREE (no Anthropic cost): prove the real eval path works
python scripts/07_daytona_smoke_test.py     # gold patch must resolve; empty must not

# PAID: one real adaptive skeptical eval (agent + real grading, per axis)
python scripts/08_daytona_real_round.py
```

**Constraints.** A Daytona sandbox caps at 4 vCPU / 8GB RAM / **10GB disk**. The
instance image is the sandbox rootfs (no nested Docker), so light repos — `flask`,
`requests`, `pytest`, `marshmallow` — fit comfortably; heavy scientific repos
(astropy, sympy, scikit-learn) may exceed it. Script 07 picks a light repo
automatically. Cost per channel ≈ 2 sandboxes + 1 agent loop, so prefer
`evaluate_adaptive` (one probe per axis, early stop) over the fixed gate, and pass
`SimpleSweAgent(model="claude-sonnet-5")` as a cheaper agent.

On Windows PowerShell, prefix with `$env:PYTHONIOENCODING="utf-8";` so the μ/σ
glyphs print. Data is cached under `data/` after the first fetch.

## Wiring in a real agent (next step)

`se/fixers.py:RealAgentFixer` documents the drop-in: materialize the perturbed
repo at `base_commit`, run an open-source SWE-bench agent (Aider / SWE-agent /
Agentless) on the perturbed problem statement to get a patch, then in a Docker
container apply that patch + the hidden `test_patch` and run the F2P/P2P tests.
The `AppliedPerturbation` objects the Perturbation Agent already produces carry
everything those steps need (perturbed problem statement, file overrides, new
files, and the evaluator's test-order hints).
