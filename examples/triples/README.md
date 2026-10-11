# Five triples someone else wrote

The input format from [docs/inputs.md](../../docs/inputs.md), run once. Five
hand-written triples about one short passage, one of each kind an extractor
might hand over:

| Row | What it carries | What happens |
|---|---|---|
| founder, Mara Quist | offsets and the quote at them | the span check passes; the model says `supported` |
| office in, Lyon | a quote and no offsets | found in the text, then `supported` |
| chief executive, Tomas Ferrand | nothing | asked about against the whole text: `supported` |
| office in, Berlin | nothing, and wrong | asked about against the whole text: `not_found` |
| founded, 2012 | a quote that is not in the text | refused by the free span check; no model call |

```bash
odke run examples/triples/odke.yaml
odke eval spans --facts examples/triples/out
```

It runs on recorded responses (`recorded/ground.json`), so it needs no key and
no network. Delete the `replay` lines in `odke.yaml` to ask a real model.

`odke run` reports the extractor's rows by how their evidence was made: 1 cited,
1 quoted, 1 quote not found, 2 context. The grounder makes four calls for five
facts. `odke eval spans` counts three of the five as having no span of their
own: the two uncited facts, whose whole passage is not a citation, and the
quote that is not in the text.

## Scoring it as a pipeline

`gold.jsonl` labels the four facts the passage states, and `pipeline.py` is a
stand-in pipeline that hands back the five rows. `odke eval pipeline` runs it
and scores what comes back:

```bash
odke eval pipeline --labels examples/triples/gold.jsonl \
    --documents examples/triples/texts --ontology examples/triples/ontology.json \
    --cmd "python examples/triples/pipeline.py {in} {out}"
```

`--run examples/triples/pipeline.py:extract` calls it instead, and
`--predictions examples/triples/triples.jsonl` reads the rows as written; all
three give the same report. Precision is 0.600 and recall 0.750: the founder,
the Lyon office and the chief executive are right, Berlin is spurious, and 2012
is a wrong value (one fact written wrong, the gold 2014 not written).
`--validator --config examples/triples/odke.yaml` adds the Validator's row, on
the recorded responses. It scores the same here, because this config's gate
keeps `not_found`; with `refuse_not_found: true` it refuses Berlin and 2012, and
precision is 1.000.
