# Licences of the data in bench/labels

The label sets quote passages and facts from public datasets. That quoted text
and those facts keep their sources' licences. openodke's Apache-2.0 covers the
scripts here and the fields this project adds (ids, splits, strata, verdicts),
not the quoted data.

| Source | In | Licence | Stated in |
|---|---|---|---|
| [Text2KGBench](https://github.com/cenguix/Text2KGBench), Wikidata-TekGen part (Mihindukulasooriya et al., ISWC 2023) | G | Data: Creative Commons Attribution-ShareAlike 4.0, as the README's text names it (its badge says CC BY 4.0). Code: Apache-2.0. | the repository's README |
| TekGen corpus, part of KELM (Agarwal et al., NAACL 2021), which Text2KGBench's Wikidata-TekGen sentences come from | G | CC BY-SA 2.0 | Text2KGBench's README |
| [Re-DocRED](https://github.com/tonytan48/Re-DocRED) (Tan et al., EMNLP 2022) | G, R | MIT, Copyright (c) 2023 tonytan48 | the repository's `LICENSE` |
| [DocRED](https://github.com/thunlp/DocRED) (Yao et al., ACL 2019), which Re-DocRED revises | G, R | MIT | the repository's `LICENSE` |
| Wikipedia, where both datasets' text comes from | G, R | CC BY-SA 4.0 (3.0 for text contributed before June 2023) | Wikipedia's terms of use |

So the Text2KGBench rows in `G/` are shared under CC BY-SA 4.0, the share-alike
terms of the dataset and of TekGen. The Re-DocRED rows in `G/` and every row
in `R/` are shared under MIT for the annotation, and the Wikipedia passages
they quote under CC BY-SA. A copy of any of these rows carries the same terms
and the attribution above.

The extracted triples in `G/` are model output from the published comparison
(PR #105, `anthropic/claude-sonnet-5-5`). The three extractors are openodke,
LangChain's LLMGraphTransformer and neo4j-graphrag. No library's code is copied
here.
