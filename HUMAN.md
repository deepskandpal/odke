# HUMAN.md

What only the maintainer can do: it needs the maintainer's own judgment, login
or admin rights. An agent that reaches one of these lists it here, says so in
its pull request under "Needs the maintainer", and stops. It does not work
around it.

## Open

- [ ] **The tagline (#100).** The maintainer writes it. An agent may collect
      options in the issue, but does not choose one or put one in the docs.
- [ ] **Merging `release/1.0.0` into `main` (#202), and tagging `v1.0.0rc1`.**
      The maintainer's call, after the fix PRs and the re-check. An agent may
      prepare the merge PR and the release notes; it does not merge it, and it
      does not push a tag (a `v*` tag publishes to PyPI through `release.yml`).
- [ ] **Anything that needs the maintainer's PyPI or GitHub login.** For
      example: a pending or trusted publisher, an environment's protection
      rules, a ruleset, repository settings, secrets, or a release that has to
      be approved by hand. An agent writes down exactly what to click or run,
      here, and stops.

## Standing rules

- **Merging is the maintainer's.** A pull request stops at "ready to merge",
  unless the maintainer has approved merges for a phase of work, in writing.
- **Nothing is posted for the maintainer.** Review comments, replies,
  announcements and issue labels are drafted for the maintainer, never posted.

## Done

- [x] **#45: PyPI Trusted Publishing.** Done 14 Sep 2026. The environments
      `pypi` (deploys only from `main` or a `v*` tag) and `testpypi` exist, and
      `release.yml` uses both. Pending publishers were added on pypi.org and
      test.pypi.org for project `openodke`, owner `deepskandpal`, repository
      `odke`, workflow `release.yml`, environments `pypi` and `testpypi`.
- [x] **#46: Branch protection on `main`.** Done 14 Sep 2026: ruleset
      `main-protections` blocks deletion and force-push, and requires a pull
      request (0 approvals, so the maintainer can merge their own) plus passing
      `verify` 3.11/3.12/3.13 and the Neo4j live tests.
- [x] **#41 and #42: the first release.** Done 14 Sep 2026: openodke 0.1.0 on
      PyPI and TestPyPI, and the GitHub release v0.1.0.
      https://pypi.org/project/openodke/0.1.0/ ·
      https://github.com/deepskandpal/odke/releases/tag/v0.1.0
- [x] **The board is linked to the repo:** project 6. New issues are added by
      hand (`gh project item-add 6 --owner deepskandpal --url <issue-url>`); the
      token-based auto-add was dropped, see DECISIONS #13.
