# Decisions

The design calls, one line each. Each links to its full entry, with the
reasoning and what it cost, in [DECISIONS.md](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md).

## What openodke is

| # | Decision |
|---|---|
| 24 {#24} | [openodke is the verification layer between any extractor and the store](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#24-openodke-is-the-verification-layer-between-any-extractor-and-the-store). It checks any extractor's output before the store, and evaluates the result; openodke's own extractor stays as a reference. |
| 26 {#26} | [The Validator is the layer; the gate is a stage](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#26-the-validator-is-the-layer-the-gate-is-a-stage). `Validator` names the layer, and the gate stage is `Gate`. |

## The rest of the log

| # | Decision |
|---|---|
| 1 {#1} | [The base install talks to nothing](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#1-the-base-install-talks-to-nothing). No model provider, database driver or HTTP client. |
| 2 {#2} | [One `Fact` class for edges and properties](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#2-one-fact-class-for-edges-and-properties). |
| 3 {#3} | [Spans are character offsets, not quoted strings](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#3-spans-are-character-offsets-not-quoted-strings). A span either resolves to its text or it does not, and checking is free. |
| 4 {#4} | [Everything is frozen](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#4-everything-is-frozen). Stages return new objects. |
| 5 {#5} | [Stages are Protocols, not base classes](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#5-stages-are-protocols-not-base-classes). Write one method to supply your own stage. |
| 6 {#6} | [Snippets are data, rendered two ways](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#6-snippets-are-data-rendered-two-ways). One object gives the prompt's prose and the output schema. |
| 7 {#7} | [Two model adapters, not one, and not a hundred](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#7-two-model-adapters-not-one-and-not-a-hundred). A standard-library OpenAI-compatible client, and litellm behind `[llm]`. |
| 7a {#7a} | [Model roles are part of the configuration](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#7a-model-roles-are-part-of-the-configuration). Grounding defaults to a smaller model than extraction. |
| 7b {#7b} | [A registry, so an internal gateway is not a fork](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#7b-a-registry-so-an-internal-gateway-is-not-a-fork). |
| 8 {#8} | [Ontology inference is a bootstrap, not a mode](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#8-ontology-inference-is-a-bootstrap-not-a-mode). An inferred ontology is reviewed and frozen before use. |
| 9 {#9} | [The Initiator and Retriever are optional](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#9-the-initiator-and-retriever-are-optional). |
| 10 {#10} | [Trust tiers are an enum with weights, not a free float](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#10-trust-tiers-are-an-enum-with-weights-not-a-free-float). Four named tiers with fixed weights. |
| 11 {#11} | [`Fact.signature` excludes reconcilable qualifiers](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#11-factsignature-excludes-reconcilable-qualifiers). |
| 12 {#12} | [Verification is one script](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#12-verification-is-one-script). `scripts/verify.sh` is what CI and contributors run. |
| 13 {#13} | [The board is linked to the repo, not auto-populated](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#13-the-board-is-linked-to-the-repo-not-auto-populated). New issues are added to the account-owned board by hand. |
| 14 {#14} | [Polarity is a field, and it is part of the signature](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#14-polarity-is-a-field-and-it-is-part-of-the-signature). A denial and its assertion never merge. |
| 15 {#15} | [The ontology says which qualifiers bear identity; the fact carries the answer](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#15-the-ontology-says-which-qualifiers-bear-identity-the-fact-carries-the-answer). `Qualifier(identity=True)` puts a key in `Fact.signature`. |
| 16 {#16} | [Resolution proposes links; it never replaces nodes](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#16-resolution-proposes-links-it-never-replaces-nodes). `SAME_AS`, `SIMILAR` and `DIFFERENT` links; nodes are never merged. |
| 17 {#17} | [Two clocks: valid time on the fact, transaction time on the evidence](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#17-two-clocks-valid-time-on-the-fact-transaction-time-on-the-evidence). |
| 18 {#18} | [An entity records how its key was decided](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#18-an-entity-records-how-its-key-was-decided). `Entity.resolution` says whether the caller, an external id or a linker decided. |
| 19 {#19} | [The chunk is the unit of the pipeline, and the router sees chunks](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#19-the-chunk-is-the-unit-of-the-pipeline-and-the-router-sees-chunks). |
| 20 {#20} | [Thirteen Protocols, each with a pass-through default](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#20-thirteen-protocols-each-with-a-pass-through-default). |
| 21 {#21} | [A stage the platform also does is warned about, never forbidden](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#21-a-stage-the-platform-also-does-is-warned-about-never-forbidden). |
| 22 {#22} | [The package is `openodke`; the tool and its data keep `odke`](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#22-the-package-is-openodke-the-tool-and-its-data-keep-odke). |
| 23 {#23} | [Evidence cites the claim-bearing clause, and carries the mention beside it](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#23-evidence-cites-the-claim-bearing-clause-and-carries-the-mention-beside-it). |
| 25 {#25} | [A span says who chose it](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#25-a-span-says-who-chose-it). `Evidence.span_origin` is `cited`, `located` or `context`. |
| 27 {#27} | [A prompt is a registered, versioned object](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#27-a-prompt-is-a-registered-versioned-object). Every prompt is `id@version`, locked by its hash. |
| 28 {#28} | [An inverse comes from the schema, is marked, and is never grounded twice](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#28-an-inverse-comes-from-the-schema-is-marked-and-is-never-grounded-twice). |
| 29 {#29} | [A comparison has three verdicts, and inconclusive is not a pass](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#29-a-comparison-has-three-verdicts-and-inconclusive-is-not-a-pass). Better, worse or inconclusive with its detection limit; the CI gate fails worse. |
| 31 {#31} | [The store is looked up, never loaded, and a lookup never changes it](https://github.com/deepskandpal/odke/blob/main/DECISIONS.md#31-the-store-is-looked-up-never-loaded-and-a-lookup-never-changes-it). A proof re-keys the incoming facts onto the stored entity as stored; anything weaker is a link. |
