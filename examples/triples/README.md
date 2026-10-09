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
