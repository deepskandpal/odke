"""Every prompt openodke sends, as a registered, versioned, hashed object.

A calibration card has to say what it measured: which model, which prompt. "The
grounder's prompt" names nothing, because it is whatever the source said on the
day of the run. So each prompt is registered here as `Prompt(id, version, text,
source)` under the key `id@version`, and every model call records the key it
sent (DECISIONS #27).

A registered text is never edited. `prompts.lock.json`, beside this file, holds
the SHA-256 of every key, and the suite fails when a text no longer matches its
hash. To change a prompt, register the new text as the next version and add its
hash to the lock. The old version stays, so a card that names it still names a
real prompt. `get` without a version gives the latest, and the latest is what
the stages send.

Only the fixed instruction text is registered. How a claim, a passage or an
ontology snippet is rendered into a message is code, and the package version
names it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

LOCK = Path(__file__).with_name("prompts.lock.json")


@dataclass(frozen=True, slots=True)
class Prompt:
    """One prompt text, exactly as it is sent, and where it came from."""

    id: str
    version: int
    text: str
    # Provenance in a word or two: "openodke", or the paper section it is copied from.
    source: str

    @property
    def key(self) -> str:
        """`id@version`: what a model call, a run report and a calibration card record."""
        return f"{self.id}@{self.version}"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


_REGISTRY: dict[str, dict[int, Prompt]] = {}


def _register(id: str, version: int, text: str, *, source: str) -> Prompt:
    if version < 1:
        raise ValueError(f"a prompt version starts at 1, got {id}@{version}")
    versions = _REGISTRY.setdefault(id, {})
    if version in versions:
        raise ValueError(f"{id}@{version} is already registered; a new text is a new version")
    versions[version] = Prompt(id, version, text, source)
    return versions[version]


def get(id: str, version: int | None = None) -> Prompt:
    """The prompt `id` at `version`, or its latest version when none is given.

    A key reads too: `get("ground.span@1")` is `get("ground.span", 1)`, which is
    how a recorded key is turned back into the text it names.
    """
    if version is None and "@" in id:
        id, _, number = id.rpartition("@")
        version = int(number)
    versions = _REGISTRY.get(id)
    if not versions:
        raise LookupError(f"no prompt {id!r}; registered: {', '.join(sorted(_REGISTRY))}")
    if version is None:
        return versions[max(versions)]
    if version not in versions:
        known = ", ".join(str(v) for v in sorted(versions))
        raise LookupError(f"no prompt {id}@{version}; {id} has version {known}")
    return versions[version]


def registered() -> list[Prompt]:
    """Every registered prompt, every version, by id and then version."""
    return [_REGISTRY[id][v] for id in sorted(_REGISTRY) for v in sorted(_REGISTRY[id])]


def read_lock(path: Path = LOCK) -> dict[str, str]:
    """The lock: each key's SHA-256, as recorded when it was registered."""
    found = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in found.items()}


def check(
    prompts: Iterable[Prompt] | None = None, lock: Mapping[str, str] | None = None
) -> list[str]:
    """What is wrong between the registry and the lock; empty when they agree.

    Three things can be: a text edited in place, which must become the next
    version instead; a key with no hash yet; and a hash with no key, which is a
    removed version that an old calibration card may still name.
    """
    found = registered() if prompts is None else list(prompts)
    recorded = read_lock() if lock is None else lock
    problems: list[str] = []
    for prompt in found:
        want = recorded.get(prompt.key)
        if want is None:
            problems.append(
                f'{prompt.key} is not in the lock: add "{prompt.key}": "{prompt.sha256}" '
                f"to {LOCK.name}"
            )
        elif want != prompt.sha256:
            problems.append(
                f"bump the version: {prompt.id} (the text of {prompt.key} no longer matches "
                f"its hash in {LOCK.name}; restore it, and register the new text as the "
                "next version)"
            )
    for key in sorted(set(recorded) - {p.key for p in found}):
        problems.append(
            f"{key} is in the lock but no longer registered: restore it, because an old "
            "calibration card may name it"
        )
    return problems


# --------------------------------------------------------------------------- #
# The prompts. Never edit a text here; register the next version instead.
# --------------------------------------------------------------------------- #

# The default grounder's question (`openodke.ground.LLMGrounder`). Short on
# purpose: it is sent once per fact, and the cost story of the whole stage rests
# on it being a fraction of the extraction prompt.
_register(
    "ground.span",
    1,
    (
        "You check one knowledge-graph claim against one passage from its source. "
        "Judge from the passage alone; ignore anything you know from elsewhere.\n"
        'Answer "supported" if the passage states the claim or clearly entails it, '
        '"contradicted" if the passage states the opposite or an incompatible value, '
        'and "not_found" if the passage does not settle it either way.\n'
        'Reply with JSON only: {"verdict": "supported" | "contradicted" | "not_found"}'
    ),
    source="openodke",
)

# The paper's grounder prompt, sent by `LLMGrounder(verdicts="binary")`: ODKE+
# App. B word for word, with its worked example left out.
_register(
    "ground.paper",
    1,
    (
        "Given a context about a subject and a triple in the format of "
        "<subject, predicate(qualifier: optional), object>, your task is to verify if the "
        "given triple can be found from or grounded in the context and only respond with "
        "True or False. For some object, they may have been listed there in form of a list "
        "of object, not single."
    ),
    source="ODKE+ App. B, verbatim",
)

# The extractor's instructions (`openodke.extract.LLMExtractor`). The ontology
# snippets are appended after it, and the passage is the whole user message.
_register(
    "extract",
    1,
    """\
Extract facts from the passage in the user message, using only the entity types \
and properties listed below. Anything else is ignored.

Every fact needs "quote": the words of the passage that state it, copied exactly, \
with the same spelling, capitals, punctuation and spacing. Quote the whole clause \
that supports the fact, not just the word naming the value; where one clause \
states several facts, quote it in full for each and put the words that tell this \
one apart in "mention". A fact whose quote is not in the passage is discarded. \
Give "start", the character offset where the quote begins (the passage's first \
character is 0); your best count is fine. "mention" is "" when there is nothing \
to tell the fact apart, and a qualifier the passage does not state is "".

"polarity" is "denied" when the passage says the fact is not so, "partial" when \
it holds only with a limitation the passage states, and "asserted" otherwise. \
For a property whose range is an entity type, "value" is that entity's name as \
the passage writes it.

Reply with one JSON object and nothing else, in this shape:
{"entities": [{"type": "...", "name": "...", "facts": [{"predicate": "...", \
"value": "...", "quote": "...", "start": 0, "mention": "...", \
"polarity": "asserted", "qualifiers": {}}]}]}
Reply {"entities": []} when the passage states none of these properties.""",
    source="openodke",
)

# Sent as a user turn after a reply that was not the extraction contract.
_register(
    "extract.repair",
    1,
    (
        'That reply was not a JSON object with an "entities" array. Reply again with '
        "only that object, following the same rules."
    ),
    source="openodke",
)

# The re-extract hook's instructions (`LLMExtractor.reextract`, #102): one gap
# window the coverage report found, the properties that could fill it, and the
# facts already taken from it. The idea is GraphRAG's "gleaning" pass (Edge et
# al. 2024, arXiv 2404.16130), scoped to one window, and the grounder checks
# what comes back like any other fact. After its first paragraph it is extract@1
# from "Every fact needs" to the end, unchanged, so the reply has the same shape
# and passes the same span checks. The flagged properties' snippets follow it.
_EXTRACT_RULES = _REGISTRY["extract"][1].text
_EXTRACT_RULES = _EXTRACT_RULES[_EXTRACT_RULES.index('Every fact needs "quote"') :]
_register(
    "reextract",
    1,
    (
        "You find facts that an earlier extraction missed in one passage. Use only the "
        "properties listed below. The facts already extracted from this passage are listed "
        "in the user message: do not return any of them again, even in other words. Return "
        "only facts the passage itself states.\n\n"
    )
    + _EXTRACT_RULES,
    source=(
        "openodke, after GraphRAG's gleaning pass (Edge et al. 2024, arXiv 2404.16130), "
        "scoped to one window and grounded after"
    ),
)

# The ontology proposer's instructions (`openodke.infer.LLMProposer`). A template:
# `{literals}` and `{max_types}` are filled in per run, and the hash is of the
# template, since the values are the run's settings rather than the prompt.
_register(
    "infer",
    1,
    """\
You are drafting a small ontology for a corpus. Deterministic tools have read a \
sample of it and proposed the candidate entity types and properties in the user \
message, each with the number of documents that support it and the tool that \
found it. Your job is to name, merge and rank those candidates, not to invent a \
schema of your own.

- Name each type in singular PascalCase and each property in snake_case, choosing \
the name a reader of these documents would expect.
- Merge candidates that mean the same thing into one entry, and list every \
candidate it covers, by the name shown, in "from". A candidate you rename goes in \
"from" too.
- A property's "domain" lists the types it describes. Its "range" is one of the \
types you return, or one of: {literals}.
- Return at most {max_types} types. List types and properties most important first.
- Propose something no candidate covers only when a passage states it, and give \
"quote": words of that passage, copied exactly. An entry with neither "from" nor \
a quote that is in the passages is discarded.
- Candidates you leave out are kept as found, for a person to review.

Reply with one JSON object and nothing else, in this shape:
{"types": [{"name": "...", "description": "...", "parents": [], "from": []}], \
"predicates": [{"name": "...", "description": "...", "domain": ["..."], \
"range": "...", "cardinality": "single", "from": []}]}""",
    source="openodke",
)

# Sent as a user turn after a reply that was not the proposal contract.
_register(
    "infer.repair",
    1,
    (
        'That reply was not a JSON object with "types" and "predicates" arrays. Reply '
        "again with only that object, following the same rules."
    ),
    source="openodke",
)

# The entity-pair judge's question (`openodke.corroborate.PairJudge`, #150): two
# mentions the resolver's rules left open, each with its type and the text
# around it. Asked once in each order, (A, B) and (B, A).
_PAIR_SOURCE = (
    "openodke, after LLM entity matching (Peeters, Steiner & Bizer 2023, arXiv 2310.11244), "
    "asked in both orders for position bias (Zheng et al. 2023, arXiv 2306.05685 §3.4)"
)
_register(
    "pair",
    1,
    (
        "You decide whether two mentions name the same real-world entity. Each mention comes "
        "with its entity type and the text around it.\n"
        "You may use general knowledge of names: abbreviations, short forms, former names, "
        "transliterations. You may not decide two mentions are the same only because a "
        "well-known entity has that name; the two contexts must describe one entity. Mentions "
        "with the same name can be different entities, and mentions with different names can "
        "be the same entity.\n"
        'Answer "same" if the contexts describe one entity, "different" if they describe two, '
        'and "unsure" if the text shown does not settle it.\n'
        'Reply with JSON only: {"because": "<at most 20 words>", "decision": "same" | '
        '"different" | "unsure"}'
    ),
    source=_PAIR_SOURCE,
)

# Its user message, a template: each `{field}` is filled in per pair, once, so
# a context that contains braces is never read as a field. `{a_aliases}` is
# "; also known as: …" for a stored entity and empty otherwise, on both sides,
# since the swap puts either entity first.
_register(
    "pair.user",
    1,
    (
        'Mention A: "{a_surface}" (type: {a_type}){a_aliases}\n'
        "Context A: {a_context}\n"
        "\n"
        'Mention B: "{b_surface}" (type: {b_type}){b_aliases}\n'
        "Context B: {b_context}"
    ),
    source=_PAIR_SOURCE,
)

# The fact-equivalence judge's question (`openodke.eval.equivalence.FactJudge`,
# #143): a gold fact the scorer counted missed, a prediction the pre-filter
# paired with it, and the gold fact's evidence. The gold fact is the reference
# the prediction is judged against, as in reference-guided judging, and a
# quantity must match, as EnterpriseRAG-Bench's correctness rule has it. Asked
# once in each order: gold first, then the prediction first.
_FACT_EQUIV_SOURCE = (
    "openodke, after reference-guided judging (Zheng et al. 2023, arXiv 2306.05685) and "
    'EnterpriseRAG-Bench\'s rule that "quantities must match" (Onyx, arXiv 2605.05253, MIT), '
    "asked in both orders"
)
_register(
    "fact_equiv",
    1,
    (
        "You compare two knowledge-graph facts taken from the same passage and decide whether "
        "they state the same fact. Wording may differ: a short form of a name, another date "
        "format, a synonym for the value. They are the same only if all three parts match in "
        "meaning: the same subject, the same relation and the same object, in the same "
        "direction. If both state a quantity, date or number, it must be the same value.\n"
        'Answer "same", "different", or "unsure" if the passage does not settle it.\n'
        'Reply with JSON only: {"decision": "same" | "different" | "unsure"}'
    ),
    source=_FACT_EQUIV_SOURCE,
)

# Its user message, a template: each `{field}` is filled in once, so a passage
# that contains braces is never read as a field. Both facts are rendered as the
# grounder renders a claim (`render_claim`), and the swap puts either first.
_register(
    "fact_equiv.user",
    1,
    (
        "Relation: {predicate}: {description}\n"
        "\n"
        "Fact 1: {fact_1}\n"
        "Fact 2: {fact_2}\n"
        "\n"
        "Passage:\n"
        "{passage}"
    ),
    source=_FACT_EQUIV_SOURCE,
)


__all__ = ["LOCK", "Prompt", "check", "get", "read_lock", "registered"]
