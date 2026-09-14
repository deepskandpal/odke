# Loaders & extraction

!!! note "Lands with M2"
    This page lands with M2: loaders, the span-preserving chunker, and the
    pattern, LLM and hybrid extractors (`odke.loaders`, `odke.extract`).

Until then:

- The `Loader`, `Chunker`, `Router` and `Extractor` Protocols and their
  pass-through defaults are described in [Concepts](concepts.md#the-thirteen-protocols).
- Any class with an `extract(chunk, ontology)` method is an extractor. The
  [Home](index.md#five-minutes-no-keys) page has a working one.
- What an extractor is prompted with is an ontology snippet, which you can
  inspect today: see [Ontology](ontology.md#snippets-what-the-model-is-shown).
