# End to end: text and tables to a queryable graph

A small invented corpus, run through all thirteen stages into JSON Lines or
Neo4j, ending with the Cypher that shows where every fact came from. Every
company and person in it is fictional.

It runs on **recorded model responses**, so the first run needs no key, no
network and no database. Those responses were written by hand to exercise every
path of the pipeline — see [what the recorded responses are](#4-what-the-recorded-responses-are)
before reading anything into the numbers.

| Path | What it is |
|---|---|
| `corpus/register.csv` | A company-register extract: name, number, incorporation date, head office. Tier `curated` |
| `corpus/staff.csv` | A staff directory. Tier `authoritative` |
| `corpus/factsheet.md` | A factsheet of `Key: value` lines, loaded as semi-structured |
| `corpus/notes/*.md` | Two trade-press notes, prose. Tier `community` |
| `ontology.json` | Two types, seven predicates; `headquarters` and `chief_executive` are single-valued |
| `e2e_stages.py` | One stage of the example's own: registration numbers become external ids |
| `recorded/` | The hand-authored model responses |
| `odke.yaml`, `odke.neo4j.yaml` | The run, into JSON Lines or into Neo4j |
| `queries.cypher` | The queries in section 3 |
| `gold.jsonl` | 36 labelled facts, for the ablation in section 5 |
| `docker-compose.yml` | A throwaway Neo4j 5.26 Community |

## 1. Run it

```bash
pip install "openodke[yaml]"
odke run examples/e2e/odke.yaml
```

```text
odke run
documents     8 (8 chunks; 0 skipped, 0 deferred)
extractor     paths (paths llm+pattern, chunks 8, pattern_facts 20, llm_facts 18, merged 2, model_calls 3), rejections (quote not in the passage 1)
grounder      facts 36, calls 36, prompt_tokens 4764, completion_tokens 216, supported 20, contradicted 1, not_found 15, span (facts 36, located 36)
corroborator  conflicts (lost 1, won 2)
validator     accepted 25, refused (contradicted 1)
refused       1
graph         25 facts (8 edges, 17 properties), 7 entities, 4 links (different 1, similar 3)
cost          39 model calls, 4980 tokens, USD unknown
wrote         jsonl → …/examples/e2e/out: entities.jsonl 7, facts.jsonl 25, links.jsonl 4, manifest.json
```

Line by line:

- **documents.** One per register row and per staff row — each record is
  rendered to text, so a span can point into it — plus the factsheet and the two
  notes. Their ids are their paths: `corpus/register.csv#L2` is the register's
  second line. A labelled set and a Cypher query can both name them.
- **extractor.** The pattern path read 20 facts from the CSV rows and the
  factsheet's `Key: value` lines, exactly and without a model call. The model
  read the three chunks with prose in them, once each; two of its facts matched
  the factsheet's pattern facts and merged into them. One quote the model gave,
  *"headquartered in Leeds"*, is not in its passage, and was dropped before
  anything else looked at it.
- **grounder.** One call per fact, each shown the fact and its own cited span.
  The one `contradicted` is the model's claim that Halden Robotics GmbH has its
  head office in Leeds, citing *"Halden Robotics GmbH, a Munich engineering
  firm"*. Most of the fifteen `not_found` are register and staff cells: a cell
  is grounded against itself, `Leeds`, which cannot say whose head office it is.
  One is a real miss: the model claimed Maya Okafor works for Halden Robotics
  GmbH on the strength of a sentence that does not mention her.
- **corroborator.** Normalisation made the note's *"10 March 2014"* and the
  register's `2014-03-10` one claim, with `support: 2`. The register (curated)
  and a note (community) disagree on Halden Robotics Ltd's head office; Leeds
  won, and Sheffield is kept at reduced confidence with the reason attached.
- **validator.** The default gate refuses `contradicted` and keeps
  `not_found` — refusing `not_found` here would throw away most of the register.
- **graph.** 25 facts on 7 entities, and 4 links between entities: one
  `DIFFERENT`, three `SIMILAR`. Nothing was merged away.

The output is in `out/`: `facts.jsonl` has every fact with its evidence,
verdict, confidence and support, and `manifest.json` has the counts above.

## 2. Into Neo4j

```bash
pip install "openodke[neo4j,yaml]"
export NEO4J_PASSWORD=...            # eight characters or more
docker compose -f examples/e2e/docker-compose.yml up -d
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j

odke run examples/e2e/odke.neo4j.yaml --dry-run
odke run examples/e2e/odke.neo4j.yaml
```

The dry run connects to nothing. It prints the 15 statements `bootstrap: true`
will apply and the 16 `UNWIND … MERGE` statements the sink will send:

```text
bootstrap     would apply 15 statements
  CREATE INDEX odke_entity_key IF NOT EXISTS FOR (n:`Entity`) ON (n.key)
  CREATE CONSTRAINT odke_claim_signature IF NOT EXISTS FOR (c:`Claim`) REQUIRE c.signature IS UNIQUE
  CREATE CONSTRAINT odke_key_Company IF NOT EXISTS FOR (n:`Company`) REQUIRE n.key IS UNIQUE
  …
  CREATE CONSTRAINT odke_signature_headquarters IF NOT EXISTS FOR ()-[r:`headquarters`]-() REQUIRE r.signature IS UNIQUE
would write   neo4j → $NEO4J_URI: 16 statements, 52 rows
       5 × MERGE (n:`Company` {key: row.key})
       2 × MERGE (n:`Person` {key: row.key})
       6 × MERGE (s)-[r:`employer` {signature: row.signature}]->(o)
  …
       1 × MERGE (a)-[l:`DIFFERENT`]->(b)
```

Every constraint is `IF NOT EXISTS` and every write is a `MERGE` on an entity key
or a fact signature, so running it again changes nothing in the store. The
uniqueness constraints on relationships need **Neo4j 5.7 or later**.

Unset `NEO4J_URI` and `NEO4J_PASSWORD` before running `./scripts/verify.sh`: it
refuses to start with a database credential in the environment.

## 3. Ask it why

Open the browser at http://localhost:7474 and run these. They are also in
`queries.cypher`, and the test suite runs every one against a live Neo4j.

### Every fact from one source

```cypher
MATCH (s)-[r]->(o)
WHERE 'corpus/notes/halden-robotics.md' IN r.evidence_doc_ids
RETURN s.label AS subject, type(r) AS predicate, coalesce(o.label, o.value) AS object,
       r.verdict AS verdict, r.evidence_starts AS starts, r.evidence_ends AS ends
ORDER BY predicate, object
```

Eight rows. Each relationship carries its evidence as parallel lists — document
ids, character offsets, tiers — so `starts[i]` and `ends[i]` are where in
document `i` the fact was read. A fact two sources agree on lists both. The
GmbH's invented head office is not among the rows: the gate refused it.

### Why is this value here?

```cypher
MATCH (c:Company {key: 'Company:halden robotics ltd'})-[r:headquarters]->(v:Claim)
RETURN v.value AS headquarters, r.evidence_doc_ids AS sources, r.evidence_tiers AS tiers,
       r.confidence AS confidence, r.`odke.conflict` AS conflict
ORDER BY confidence DESC
```

Two rows, Leeds first. A literal fact is a relationship to a `:Claim` node so
that two values can both be held, each with its own receipts. Sheffield's
`conflict` says why it lost:

> Kept 'Leeds' over 'Sheffield': 'Leeds' is backed by 1 independent source
> (curated, retrieved …) and 'Sheffield' by 1 independent source (community,
> retrieved …). Ranked on source trust, freshness, and agreement discounted by
> how much each source asserts: 1.35 to 0.64.

### Two companies with one name

```cypher
MATCH (a:Company)-[l:DIFFERENT]->(b:Company)
RETURN a.label AS company, b.label AS other, l.score AS name_similarity, l.reason AS reason
```

One row: Halden Robotics GmbH and Halden Robotics Ltd, name similarity 1.0,
reason `external_id mismatch: DE-551902 vs HR-104233`. Their names normalise to
the same key, and their registration numbers disagree, so resolution recorded
that they are two companies instead of proposing to merge them (DECISIONS #16).
The three `SIMILAR` links are the ones it could not settle — *"Halden Robotics"*
in a note, with no number, is similar to both. Nothing was merged, so a
threshold can be changed and the resolver re-run.

The numbers came from `e2e_stages.py`, a stage of this example's own named in
the config as `e2e_stages:RegistryResolver`. It is twenty lines: this corpus
has an identifier, the package cannot know that, and a stage is how it says so.

### A company with two head offices

```cypher
MATCH (s)-[r:`headquarters`]->(o)
WHERE r.polarity = 'asserted' AND r.valid_to IS NULL
WITH s, collect(DISTINCT coalesce(o.key, o.value)) AS objects
WHERE size(objects) > 1
RETURN labels(s) AS labels, s.key AS subject, objects
```

One row: `Company:halden robotics ltd`, `["Leeds", "Sheffield"]`. This is the
check openodke compiles from the ontology for every single-valued predicate — Neo4j
cannot enforce relationship cardinality itself. From Python it is
`Neo4jSink(...).check(ontology)`. It reports and never repairs: which head office
is right is a decision for a person, and both are still there to decide on.

### What changed this week

```cypher
MATCH (s)-[r]->(o)
WHERE r.signature IS NOT NULL AND r.extracted_at >= datetime() - duration('P7D')
RETURN type(r) AS predicate, count(*) AS facts
ORDER BY predicate
```

`extracted_at` is when the run wrote the fact; `retrieved_at` is when its
freshest source was read; `valid_from` and `valid_to`, when set, are when the
claim was true in the world. Three clocks, three different questions
(DECISIONS #17).

## 4. What the recorded responses are

They were **written by hand**, not recorded from a provider.

- `recorded/extract.json` answers the model's three extraction calls. Most facts
  are right; one quote is not in its passage; two facts cite a real span that
  does not support them; two dates are copied as written, not normalised.
- `recorded/ground.json` answers the 29 distinct grounding questions by one
  rule: `supported` when the passage states the relation and the value (a name
  as written counts), `not_found` when the passage is only the value or does not
  mention the claim, `contradicted` when it states an incompatible value.

Token counts on recorded grounding responses are estimates, and no response
carries a price, so cost is reported as unknown. To run against real models,
delete the two `replay` lines from the config; the provider reads its own key
from the environment. To record real responses for a test, wrap a client in
`openodke.llm.testing.RecordingClient`.

## 5. The ablation, on this example

> **A demonstration on hand-authored recorded responses, not a benchmark.** The
> responses and the labels were both written for this example, so the numbers
> below measure how this fixture was written and nothing about any model. They
> show what the command reports and how to read it. Run it on your own labels.

`gold.jsonl` labels the 36 facts the eight documents state, each under the
document id `odke run` gives it. The same config, run three ways:

```bash
odke eval ablation --config examples/e2e/odke.yaml --labels examples/e2e/gold.jsonl
```

| Configuration | Precision | Recall | F1 | TP | FP | FN | Facts written | Model calls |
|---|---|---|---|---|---|---|---|---|
| extraction alone | 0.889 | 0.889 | 0.889 | 32 | 4 | 4 | 36 | 3 |
| + grounding (gate refuses `contradicted`) | 0.914 | 0.889 | 0.901 | 32 | 3 | 4 | 35 | 39 |
| + corroboration (normalise, resolve, corroborate, score) | 0.971 | 0.944 | 0.958 | 34 | 1 | 2 | 25 | 39 |

Cost is unknown for all three: the recorded responses carry no price.

What it says about this fixture:

- **Grounding moved precision from 0.889 to 0.914, and not recall.** Of the four
  wrong candidates it caught one, the head office the model invented for the
  GmbH. The second invented fact — Maya Okafor working for the GmbH — came back
  `not_found`, which the default gate keeps.
- **Refusing `not_found` too would have been worse here.** It would keep 19 of
  the 32 true extracted facts and 1 of the 4 false ones (precision 0.950): a
  register cell is grounded against itself, which cannot say whose value it is,
  so a careful grounder answers `not_found` for most structured facts.
- **Most of the gain came after grounding, from normalisation.** Two dates the
  model copied as written ("10 March 2014", "1 June 2019") became the ISO dates
  the labels use. Corroboration merged the 36 candidates into 26 facts, and the
  gate refused one of those; a merged fact is scored once in each document it
  cites, so merging alone moves neither number. Resolution proposed links and
  re-keyed nothing.
