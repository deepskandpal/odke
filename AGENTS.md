# AGENTS.md

openodke is the verification layer between any extractor and a graph store
(DECISIONS #24). The **Validator** checks each extracted fact against its
source text, resolves entities, and merges facts across sources with their
support. The **Evaluator** measures a pipeline against labels or public
benchmarks, and says what to change. openodke sells numbers people trust, so
its checks are the product. This file is the map; [CONTRIBUTING.md](CONTRIBUTING.md)
and [DECISIONS.md](DECISIONS.md) are the detail.

## Check your work

- **The whole check:** `scripts/verify.sh`, ten steps, about seven minutes.
  CI runs the same script on Python 3.11 to 3.14, plus a live Neo4j job and
  a bench job (`bench/test_competitors.py`, in `bench/requirements.txt`).
- **The rules alone:** `uv run pytest tests/rules`, about 15 seconds. They
  also run after every edit under `src/` ([.claude/settings.json](.claude/settings.json)).
- **A quick loop:** the test files your change touches. Tests never reach the
  network: model calls are answered by `src/openodke/llm/testing.py`.

## Where things are

| Path | What it holds |
|---|---|
| `src/openodke/validator.py` | the Validator: ground, resolve, corroborate, gate, write |
| `src/openodke/ground/`, `src/openodke/corroborate/` | the stages it runs |
| `src/openodke/sinks/`, `src/openodke/tenants.py` | the stores: JSONL, Neo4j, Cypher file, CSV, RDF |
| `src/openodke/eval/` | the Evaluator: scorers, statistics, labelling sheets, benchmarks |
| `src/openodke/llm/` | every model call: clients, cache, budget, providers |
| `src/openodke/run/`, `src/openodke/cli/main.py` | `odke run` and the rest of the `odke` command |
| `tests/`, `tests/rules/` | the suite, and the six rules |
| `bench/` | the scripts behind the published benchmark tables |
| `docs/`, `examples/` | the site, and examples that run with no key |
| `scripts/` | the check (`scripts/verify.sh`), recordings (`scripts/record.py`), the evidence bundle (`scripts/evidence.py`), the edit hook (`scripts/edit_hook.py`) |

**Decisions.** [DECISIONS.md](DECISIONS.md) is append-only: each call, its
reason and what it cost. [docs/decisions.md](docs/decisions.md) is its index,
one line each. Read the entry before you change what it decided. A new call
gets the next number and an index row, in the pull request that acts on it.

## The six rules

Each rule is one test file that checks every place in the codebase where its
kind of bug can happen. A failure message says what to do.

| Rule | Test |
|---|---|
| R1. A failure never becomes a number | `tests/rules/test_r1_failures.py` |
| R2. The tenant goes everywhere | `tests/rules/test_r2_tenants.py` |
| R3. A metric is computed in one place | `tests/rules/test_r3_one_metric.py` |
| R4. A recorded answer never passes for a real one | `tests/rules/test_r4_recorded_answers.py` |
| R5. Secrets stay home | `tests/rules/test_r5_secrets.py` |
| R6. Intervals behave | `tests/rules/test_r6_intervals.py` |

Where the code still breaks a rule, the test is `xfail(strict=True)`, and its
reason names the finding and the fix PR: findings 1–20 and S1–S30 of the 1.0
release audit (11 Oct 2026) or of the 9 Oct audit, and fix PRs 1–5, the pull
requests into the 1.0 release branch that follow it. A fix removes the marker: a rule
that passes with its marker still on fails CI. A new entry point (a command
that scores, a sink, a read path, a cached client) goes into its rule's list.

## The pull request loop

1. **Check the base:** `scripts/verify.sh` on the base commit. Note the result.
2. **Reproduce:** a test that fails on the base, a rule's or the finding's own.
3. **Record the failure:** `scripts/evidence.py record --side base --commit <base>
   --run "<title>::<command>"`: the failing test (a rule's with `--runxfail`, since
   an xfail passes), and the command a user would type.
4. **Fix** on a branch off the release branch, and remove the rule's marker.
5. **Drive it:** run the CLI end to end on fixtures with recorded answers.
6. **Record the fix:** the same commands, `--side head --here`.
7. **Open the pull request** into the release branch, with the evidence summary.
8. **Review:** fresh-context reviewers. A finding needs a failing case and a
   separate check; each is recorded as fixed, or dropped as unproven.
9. **Watch CI** and fix what is red. A flaky test is a finding: find out why.
10. **Escalate** what needs judgment: a published number, the API pin, a
    DECISIONS entry, a label question, spend. It goes under "Needs you".
11. **Merge** is the maintainer's, unless the maintainer has approved merges
    for a phase of work.

**The evidence** is one `evidence.json` per pull request, in `--out`, else
`$ODKE_DESK/<branch>`, else `$TMPDIR/desk/<branch>`, beside its `.cast`
recordings. It holds `branch`, `pr`, `title`, a `summary` of five sentences or
fewer, the `findings`, `rules` and `decisions`, and `risk_tags` from numbers,
tenants, secrets, statistics, api and docs. `base` and `head` each hold
`commit`, `verify` and `recordings` (title, command, file, exit). Then
`changes` (file, what, tags), `review` (finding, outcome, commit),
`build_failures` (what, fix), `needs_you`, `ci` and `merged`. Build it with
`scripts/evidence.py init`, `record`, `set` and `ci`; `check` says what the
desk could not read.

## What needs a human

Anything that needs the maintainer's judgment, login or admin rights is
listed in [HUMAN.md](HUMAN.md). Write it there, say so in the pull request,
and stop. Never post on GitHub for the maintainer, and never push a tag.
