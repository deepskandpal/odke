"""The `odke` command.

Thin on purpose: every command is a few lines over the library, so that anything
reachable from the CLI is reachable from Python and neither grows a behaviour the
other lacks.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import typer

from openodke import __version__
from openodke.ontology import Ontology, OntologyLoadError

if TYPE_CHECKING:
    from openodke.eval.compare import Comparison
    from openodke.run import RunConfig

app = typer.Typer(
    name="odke",
    help="Ontology-guided knowledge extraction: text in, a grounded knowledge graph out.",
    no_args_is_help=True,
    add_completion=False,
)
ontology_app = typer.Typer(help="Inspect and compile ontologies.", no_args_is_help=True)
app.add_typer(ontology_app, name="ontology")


def _version(value: bool) -> None:
    if value:
        typer.echo(f"openodke {__version__}")
        raise typer.Exit()


def _load(path: Path) -> Ontology:
    # Never strict: these commands exist to show what is wrong with a schema,
    # and refusing to load it would hide exactly that.
    if path.suffix.lower() in {".yaml", ".yml"}:
        return Ontology.from_yaml(path, strict=False)
    return Ontology.from_json(path, strict=False)


def _load_or_exit(path: Path) -> Ontology:
    try:
        return _load(path)
    except (OntologyLoadError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None


def _load_run_config(path: Path) -> RunConfig:
    """`load_config`, with what it warned about printed as `warning:` lines.

    Python hides a `DeprecationWarning` raised inside a library, and the reader
    of a config file is a person at a terminal: a key that still works under an
    old name (`validator:`, now `gate:`) has to be said where they will see it.
    """
    from openodke.run import load_config

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        config = load_config(path)
    for warning in caught:
        typer.echo(f"warning: {warning.message}", err=True)
    return config


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _qualified(model: str | None, provider: str | None) -> str | None:
    """`--model` and `--model-provider` as one provider-qualified string, or None.

    Exits 2 rather than raising: choosing a model wrongly is a usage error, and
    it is worth saying so before a config is read or a document is loaded.
    """
    if model is None:
        if provider is None:
            return None
        typer.echo("error: --model-provider needs --model", err=True)
        raise typer.Exit(2)
    from openodke.llm.providers import qualify

    try:
        return qualify(model, provider)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """openodke — ontology-guided knowledge extraction."""


@ontology_app.command("snippet")
def ontology_snippet(
    path: Path = typer.Argument(..., help="Ontology JSON or YAML file."),
    entity_type: str = typer.Argument(..., help="Entity type to render a snippet for."),
    limit: int = typer.Option(25, help="Maximum predicates in the snippet."),
    as_json_schema: bool = typer.Option(False, "--json-schema", help="Emit JSON Schema instead."),
) -> None:
    """Print the exact schema fragment the extractor would be prompted with.

    Being able to see this without spending a token is most of what makes an
    ontology debuggable: a bad extraction is usually a bad snippet.
    """
    snippet = _load_or_exit(path).snippet(entity_type, limit=limit)
    typer.echo(json.dumps(snippet.json_schema(), indent=2) if as_json_schema else snippet.render())


@ontology_app.command("types")
def ontology_types(
    path: Path = typer.Argument(..., help="Ontology JSON or YAML file."),
) -> None:
    """List entity types and how many predicates each one can carry."""
    ontology = _load_or_exit(path)
    for name in sorted(ontology.types):
        typer.echo(f"{name}\t{len(ontology.predicates_for(name))} predicates")


@ontology_app.command("validate")
def ontology_validate(
    paths: list[Path] = typer.Argument(..., help="Ontology JSON or YAML files."),
) -> None:
    """Print every diagnostic, and exit 1 if any file has an error.

    Warnings print and pass, so this can gate a pre-commit hook or a CI step
    without teaching anyone to ignore a red build. It takes several files
    because that is how pre-commit calls it.
    """
    errors = warned = 0
    for path in paths:
        try:
            diagnostics = _load(path).validate()
        except (OntologyLoadError, ImportError) as exc:
            typer.echo(f"error: {exc}")
            errors += 1
            continue
        for diagnostic in diagnostics:
            typer.echo(f"{path}: {diagnostic}")
        found = sum(d.severity == "error" for d in diagnostics)
        errors += found
        warned += len(diagnostics) - found
    typer.echo(f"{_count(errors, 'error')}, {_count(warned, 'warning')}")
    if errors:
        raise typer.Exit(1)


@ontology_app.command("diff")
def ontology_diff(
    old: Path = typer.Argument(..., help="The ontology as it was."),
    new: Path = typer.Argument(..., help="The ontology as it is now."),
    fail_on_breaking: bool = typer.Option(
        False, "--fail-on-breaking", help="Exit 1 if any change is breaking."
    ),
) -> None:
    """Added, removed and changed types, predicates, qualifiers and cardinality.

    Breaking changes print first, because "does this break the live graph?" is
    the question being asked; `--fail-on-breaking` makes the answer an exit code.
    """
    changes = _load_or_exit(old).diff(_load_or_exit(new))
    for change in sorted(changes, key=lambda c: not c.breaking):
        typer.echo(str(change))
    breaking = sum(c.breaking for c in changes)
    typer.echo(f"{_count(len(changes), 'change')}, {breaking} breaking")
    if breaking and fail_on_breaking:
        raise typer.Exit(1)


MODEL_HELP = "Provider-qualified model for every role, e.g. openai/gpt-5.5. Overrides `models`."
MODEL_PROVIDER_HELP = "Provider for --model when the string does not name one, e.g. openai."
CACHE_HELP = (
    "A directory of model answers: a call asked before is answered from it for nothing, "
    "and every new answer is kept there. Overrides models.cache."
)
BUDGET_USD_HELP = (
    "Stop the run cleanly before it spends more than this many US dollars, keeping what is "
    "done. Overrides models.budget.usd. Exit 3 when it stops."
)
BUDGET_CALLS_HELP = (
    "Stop the run cleanly before it makes more than this many model calls, keeping what is "
    "done. Overrides models.budget.calls. Exit 3 when it stops."
)
# The exit status of a run a budget stopped: what it kept was written.
EXIT_BUDGET = 3


@app.command("models")
def models_command() -> None:
    """List every provider openodke can address, and whether its key is set.

    Which provider serves a model string, which environment variable that
    provider reads, whether it is set here, and which client makes the call —
    so "use OpenAI" is something to check rather than to infer. No value is
    printed and no key is written anywhere: openodke reads the environment and
    nothing else.
    """
    # Imported here so `odke --version` stays light. Nothing on this path imports
    # a provider: the [llm] extra is looked for, never loaded.
    from openodke.llm.providers import describe
    from openodke.llm.registry import registered_providers

    typer.echo(describe(os.environ, registered=registered_providers()))


@app.command("run")
def run_command(
    config: Path = typer.Argument(..., help="Run config, YAML or JSON. See examples/run.yaml."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Load, extract and ground, then print what would be written instead of writing it.",
    ),
    model: str | None = typer.Option(None, "--model", help=MODEL_HELP),
    model_provider: str | None = typer.Option(None, "--model-provider", help=MODEL_PROVIDER_HELP),
    widen: bool = typer.Option(
        False,
        "--widen",
        help="Give a not_found whose cited span is narrower than its sentence one more "
        "grounding call, against the sentence. Needs stages.grounder: llm.",
    ),
    cache: Path | None = typer.Option(None, "--cache", help=CACHE_HELP),
    budget_usd: float | None = typer.Option(None, "--budget-usd", min=0, help=BUDGET_USD_HELP),
    budget_calls: int | None = typer.Option(None, "--budget-calls", min=0, help=BUDGET_CALLS_HELP),
) -> None:
    """Run the whole pipeline from a config file.

    The config names the inputs, the ontology, the models and which
    implementation fills each of the thirteen stages. A dry run still calls
    the models; it is the store it spares. Exit 2 is a config that cannot run,
    exit 1 a run that failed, exit 3 a run its budget stopped, which still
    wrote what it kept.

    `--model` puts every role on one model and prints which, so a run states
    what it called instead of leaving it to be read out of a config; per-role
    models stay a config decision. `odke models` lists what can be named.
    """
    # Imported here so `odke --version` and the ontology commands stay light.
    from openodke.llm.base import ProviderError
    from openodke.llm.budget import BudgetExceeded
    from openodke.run import ConfigError, execute

    chosen = _qualified(model, model_provider)
    try:
        loaded = _load_run_config(config)
        if chosen is not None:
            loaded = loaded.with_model(chosen)
            typer.echo(f"models: every role on {chosen}")
        if widen:
            loaded = loaded.with_widen()
        if cache is not None:
            loaded = loaded.with_cache(cache)
        loaded = loaded.with_budget(usd=budget_usd, calls=budget_calls)
        result = execute(loaded, dry_run=dry_run)
    except (ConfigError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except BudgetExceeded as exc:
        # Raised by a stage outside the pipeline's reach, so nothing was kept.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_BUDGET) from exc
    except (ProviderError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for warning in result.warnings:
        typer.echo(f"warning: {warning}", err=True)
    typer.echo(result.render())
    if result.stats.get("stopped"):
        raise typer.Exit(EXIT_BUDGET)
    _every_document_failed(result.stats.get("failed"), int(result.stats.get("documents", 0)))


# --------------------------------------------------------------------------- #
# odke ground — a graph from somewhere else
# --------------------------------------------------------------------------- #

ADAPTERS = ("triples", "langchain", "langextract", "graphrag", "neo4j")
FACTS_HELP = (
    "The facts: a triples JSONL file, an adapter's output file, or a Neo4j URI "
    "(bolt://, neo4j://) for neo4j and graphrag."
)
TEXTS_HELP = (
    "The texts the facts cite: a file or a directory. Triples need them; LangChain, "
    "LangExtract and neo4j-graphrag output carries its own."
)
ADAPTER_HELP = "How to read --facts: triples, langchain, langextract, graphrag or neo4j."
ONTOLOGY_HELP = (
    "Ontology JSON or YAML: adds the free checks that the relation is in it and the types fit."
)
CONFIG_HELP = (
    "A file with a `models` block (a run config will do): the model and recorded responses."
)
TEXT_PROPERTY_HELP = (
    "neo4j: the relationship property holding its source text, for a graph openodke did not write."
)
DATABASE_HELP = "Neo4j: the database to read."
USER_HELP = "Neo4j: the user."
PASSWORD_ENV_HELP = "Neo4j: the environment variable holding the password. Never a flag or a file."


class _Store:
    """How to reach a Neo4j source: the sink's settings, the password from the environment."""

    def __init__(
        self, *, text_property: str | None, database: str | None, user: str, password_env: str
    ) -> None:
        self.text_property = text_property
        self.database = database
        self.user = user
        self.password_env = password_env

    def driver(self, uri: str) -> Any:
        password = os.environ.get(self.password_env)
        if password is None:
            raise ValueError(
                f"{self.password_env} is not set: the Neo4j password is read from the "
                "environment variable --password-env names, never from a flag or a file"
            )
        from openodke.sinks.neo4j import _connect

        return _connect(uri, (self.user, password))


def _read_facts(
    adapter: str, facts: str, texts: Path | None, store: _Store
) -> tuple[list[Any], list[Any], Any]:
    """The rows `--facts` holds, the texts they cite, and the driver if a store was read.

    Raises ValueError for input that does not fit the adapter: a usage error.
    """
    from openodke import interop
    from openodke.loaders import DirectoryLoader
    from openodke.run import with_path_ids

    if adapter not in ADAPTERS:
        raise ValueError(f"unknown adapter {adapter!r}; one of {', '.join(ADAPTERS)}")
    uri = "://" in facts
    if uri and adapter not in ("neo4j", "graphrag"):
        raise ValueError(f"{adapter} reads a file; only neo4j and graphrag read a Neo4j URI")
    if adapter == "neo4j" and not uri:
        raise ValueError("neo4j reads a store: give its URI as --facts, e.g. bolt://localhost:7687")
    if adapter in ("langchain", "langextract") and texts is not None:
        raise ValueError(f"{adapter} output carries its own texts; leave --texts out")
    if adapter == "graphrag" and uri and texts is not None:
        raise ValueError("a neo4j-graphrag store holds its own chunks; leave --texts out")
    if store.text_property is not None and adapter != "neo4j":
        raise ValueError("--text-property is for the neo4j adapter")
    docs = with_path_ids(list(DirectoryLoader().load(texts)), Path.cwd()) if texts else []
    if adapter == "triples":
        if texts is None:
            raise ValueError("triples cite texts by name: give them as --texts")
        return interop.read_triples(facts), docs, None
    if adapter == "langchain":
        rows, docs = interop.from_graph_documents(facts)
        return rows, docs, None
    if adapter == "langextract":
        rows, docs = interop.from_langextract(facts)
        return rows, docs, None
    if adapter == "graphrag" and not uri:
        if len(docs) > 1:
            raise ValueError("graphrag takes one text with --texts: the one the graph came from")
        rows, docs = interop.from_graphrag(facts, document=docs[0] if docs else None)
        return rows, docs, None
    if adapter == "neo4j" and texts is None and store.text_property is None:
        raise ValueError(
            "neo4j needs --texts (the documents a graph openodke wrote was built from) "
            "or --text-property (the relationship property holding each one's text)"
        )
    driver = store.driver(facts)
    try:
        if adapter == "graphrag":
            rows, docs = interop.read_graphrag(driver, database=store.database)
        else:
            rows, read = interop.read_neo4j(
                driver,
                documents=docs,
                text_property=store.text_property,
                database=store.database,
            )
            # Every text given, not only those matched by id or URI: a row that
            # names a file by its name still finds it in the extractor.
            docs = list({doc.id: doc for doc in [*docs, *read]}.values())
    except BaseException:
        driver.close()
        raise
    return rows, docs, driver


def _grounder(
    config: Path | None,
    chosen: str | None,
    *,
    locate: bool,
    paper: bool,
    cache: Path | None = None,
    budget_usd: float | None = None,
    budget_calls: int | None = None,
) -> Any:
    """The model grounder `odke run` would build from these models, or the config's replay.

    Every call holds a slot of its provider's limit, from the `models` block's
    `limits`, process-wide. A budget, from the flags over the block's, counts
    every call that goes out. The response cache is `--cache` when given, else
    the block's, and answers in front of both, for nothing.
    """
    from openodke.ground import LLMGrounder
    from openodke.llm.base import ProviderNotInstalled
    from openodke.llm.budget import Ledger
    from openodke.llm.cache import CachedClient
    from openodke.llm.limits import PROVIDER_LIMITS, LimitedClient
    from openodke.llm.registry import resolve as resolve_client
    from openodke.run import ModelsConfig, load_models
    from openodke.run.build import OnFirstCall, _replay_client, response_cache

    models, base = load_models(config) if config is not None else (ModelsConfig(), Path.cwd())
    if chosen is not None:
        models = models.with_model(chosen)
        typer.echo(f"models: grounding on {chosen}")
    roles = models.roles()
    models = models.with_budget(usd=budget_usd, calls=budget_calls)
    PROVIDER_LIMITS.update(models.limits)
    if cache is not None:
        store = response_cache(cache.expanduser(), "--cache")
    elif models.cache is not None:
        store = response_cache(base / Path(models.cache).expanduser())
    else:
        store = None
    replay = models.replay.get("ground")
    client: Any
    if replay is not None:
        client = _replay_client(base / replay, "ground")
    else:
        try:
            client = resolve_client(roles.ground)
        except ProviderNotInstalled:
            # Behind a cache, a rerun answered from it needs no adapter at all.
            if store is None:
                raise
            client = OnFirstCall(roles.ground)
    client = LimitedClient(client)
    if models.budget is not None:
        client = Ledger(models.budget).client(client)
    if store is not None:
        client = CachedClient(client, store)
    mode: dict[str, Any] = {"context": "document", "verdicts": "binary"} if paper else {}
    return LLMGrounder(roles, client=client, locate=locate, **mode)


def _strict_ontology(path: Path) -> Ontology:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return Ontology.from_yaml(path)
    return Ontology.from_json(path)


def _echo_warnings(caught: list[warnings.WarningMessage]) -> None:
    for warning in caught:
        typer.echo(f"warning: {warning.message}", err=True)


@app.command("ground")
def ground_command(
    facts: str = typer.Option(..., "--facts", help=FACTS_HELP),
    out: Path = typer.Option(
        ..., "--out", "-o", help="Directory to write facts.jsonl and summary.json into."
    ),
    texts: Path | None = typer.Option(None, "--texts", help=TEXTS_HELP),
    adapter: str = typer.Option("triples", "--adapter", help=ADAPTER_HELP),
    ontology: Path | None = typer.Option(None, "--ontology", help=ONTOLOGY_HELP),
    config: Path | None = typer.Option(None, "--config", help=CONFIG_HELP),
    model: str | None = typer.Option(None, "--model", help=MODEL_HELP),
    model_provider: str | None = typer.Option(None, "--model-provider", help=MODEL_PROVIDER_HELP),
    locate: bool = typer.Option(
        False, "--locate", help="Find the sentence naming both ends of a fact that cited nothing."
    ),
    paper: bool = typer.Option(
        False, "--paper", help="ODKE+'s own grounder: the whole document, True or False."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="The free checks and the locator only: no model, no cost."
    ),
    text_property: str | None = typer.Option(None, "--text-property", help=TEXT_PROPERTY_HELP),
    database: str | None = typer.Option(None, "--database", help=DATABASE_HELP),
    user: str = typer.Option("neo4j", "--user", envvar="NEO4J_USER", help=USER_HELP),
    password_env: str = typer.Option("NEO4J_PASSWORD", "--password-env", help=PASSWORD_ENV_HELP),
    write_back: bool = typer.Option(
        False,
        "--write-verdicts",
        help="neo4j: set odke_verdict on each relationship read. Off unless asked.",
    ),
    cache: Path | None = typer.Option(None, "--cache", help=CACHE_HELP),
    budget_usd: float | None = typer.Option(None, "--budget-usd", min=0, help=BUDGET_USD_HELP),
    budget_calls: int | None = typer.Option(None, "--budget-calls", min=0, help=BUDGET_CALLS_HELP),
) -> None:
    """Ground a graph openodke did not build, and say what is wrong with it.

    Reads the facts through an adapter and checks each in three steps, cheapest
    first: the free checks (the mention is in the text; with --ontology, the
    relation is in it and the types fit its domain and range), the span
    locator with --locate, then the model. A fact the free checks refuse is
    never sent to the model, and a dry run sends none.

    Writes the same facts back with their verdicts to OUT/facts.jsonl, and a
    summary to OUT/summary.json and here: counts per verdict, the facts with no
    span, the `odke eval spans` width split, and the two ways a fact fails:
    evidence that does not support it, and a citation too narrow for its claim.
    Models come from --config's `models` block, as in `odke run`.

    Exit 2 is input that cannot be read, exit 1 a run that failed, exit 3 a run
    its budget stopped, which still wrote what it grounded.
    """
    # Imported here so `odke --version` and the ontology commands stay light.
    from openodke.interop import ground_graph, write_verdicts
    from openodke.llm.base import ProviderError
    from openodke.run import ConfigError

    chosen = _qualified(model, model_provider)
    try:
        if write_back and adapter != "neo4j":
            raise ValueError("--write-verdicts writes to the Neo4j graph the neo4j adapter read")
        if write_back and dry_run:
            raise ValueError("a dry run asks no model, so it has no verdicts to write back")
        schema = _strict_ontology(ontology) if ontology is not None else None
        grounder = (
            None
            if dry_run
            else _grounder(
                config,
                chosen,
                locate=locate,
                paper=paper,
                cache=cache,
                budget_usd=budget_usd,
                budget_calls=budget_calls,
            )
        )
        store = _Store(
            text_property=text_property, database=database, user=user, password_env=password_env
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows, docs, driver = _read_facts(adapter, facts, texts, store)
    except (ValueError, ConfigError, OntologyLoadError, ImportError, OSError) as exc:
        _data_error(exc)
    except ProviderError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    _echo_warnings(caught)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # With a model, the grounder locates; a dry run's free checks do it themselves.
            grounded = ground_graph(
                rows, docs, ontology=schema, grounder=grounder, locate=locate and dry_run
            )
        _echo_warnings(caught)
        if write_back:
            written = write_verdicts(driver, grounded.facts, database=database)
        paths = grounded.write(out)
    except (ProviderError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        if driver is not None:
            driver.close()
    typer.echo(grounded.summary.render())
    typer.echo("")
    typer.echo(f"wrote {', '.join(str(p) for p in paths)}")
    if write_back:
        typer.echo(f"wrote odke_verdict on {_count(written, 'relationship')}")
    if grounded.summary.stopped:
        raise typer.Exit(EXIT_BUDGET)
    if grounded.summary.failed and not grounded.summary.facts:
        _every_document_failed(grounded.summary.failed, len(grounded.summary.failed))


# --------------------------------------------------------------------------- #
# odke validate — the whole layer
# --------------------------------------------------------------------------- #

VALIDATE_CONFIG_HELP = (
    "A run config whose extractor is triples: its inputs, ontology, models, stages and "
    "sinks. With --facts, only its models block is read, as odke ground reads it."
)


@app.command("validate")
def validate_command(
    config: Path | None = typer.Option(None, "--config", help=VALIDATE_CONFIG_HELP),
    facts: str | None = typer.Option(None, "--facts", help=FACTS_HELP),
    texts: Path | None = typer.Option(None, "--texts", help=TEXTS_HELP),
    adapter: str = typer.Option("triples", "--adapter", help=ADAPTER_HELP),
    ontology: Path | None = typer.Option(None, "--ontology", help=ONTOLOGY_HELP),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Write the graph here as JSONL, beside any sinks the config names.",
    ),
    merge: bool = typer.Option(
        False,
        "--merge",
        help="Merge into the JSONL already at -o: a fact it holds gains this batch's sources.",
    ),
    update: bool = typer.Option(
        False,
        "--update",
        help="The texts are new versions of ones the store cites: retract each first, so "
        "what a new version no longer states loses it. Implies --merge.",
    ),
    model: str | None = typer.Option(None, "--model", help=MODEL_HELP),
    model_provider: str | None = typer.Option(None, "--model-provider", help=MODEL_PROVIDER_HELP),
    locate: bool = typer.Option(
        False, "--locate", help="Find the sentence naming both ends of a fact that cited nothing."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="The free checks, the locator and every deterministic stage: no model, no write.",
    ),
    text_property: str | None = typer.Option(None, "--text-property", help=TEXT_PROPERTY_HELP),
    database: str | None = typer.Option(None, "--database", help=DATABASE_HELP),
    user: str = typer.Option("neo4j", "--user", envvar="NEO4J_USER", help=USER_HELP),
    password_env: str = typer.Option("NEO4J_PASSWORD", "--password-env", help=PASSWORD_ENV_HELP),
    cache: Path | None = typer.Option(None, "--cache", help=CACHE_HELP),
    budget_usd: float | None = typer.Option(None, "--budget-usd", min=0, help=BUDGET_USD_HELP),
    budget_calls: int | None = typer.Option(None, "--budget-calls", min=0, help=BUDGET_CALLS_HELP),
) -> None:
    """Validate another extractor's facts: ground, resolve, corroborate, gate and write.

    The whole verification layer over one batch, as `openodke.Validator` runs
    it. The facts come as `odke ground` takes them (--facts, --texts,
    --adapter, and --config's models block), or from a run config alone whose
    extractor is `triples`. A stage the config leaves out is the Validator's
    default, not the pass-through: the free checks and the model grounder, the
    value normaliser, the native resolver, the signature corroborator, the
    evidence scorer, and the verdict gate with the schema checks.

    Writes the graph to -o as JSONL and to the config's sinks, and prints the
    report: facts in, refused, merged, linked and derived, the cost, the
    prompts sent and the coverage report. A dry run calls no model and writes
    nothing.

    A store merges: a fact a Neo4j sink already holds gains this batch's
    sources rather than a second edge, and so does one in the JSONL at -o with
    --merge, or in a config's jsonl sink with `merge: true`. With --update the
    texts replace the versions the store cites: each is retracted from the
    sinks first, so a fact the new version no longer states loses that source,
    and is retired when it has none. To retract a text that is gone, use
    `odke reconcile --delete`.

    Exit 2 is input or a config that cannot be read, exit 1 a run that failed,
    exit 3 a run its budget stopped, which still wrote what it kept.
    """
    # Imported here so `odke --version` and the ontology commands stay light.
    from openodke import Retractable, SignatureCorroborator, Validator
    from openodke.interop import TriplesExtractor
    from openodke.llm.base import ProviderError
    from openodke.run import ConfigError, build
    from openodke.run.execute import _bootstrap
    from openodke.sinks.jsonl import JsonlSink
    from openodke.validator import stores_of

    chosen = _qualified(model, model_provider)
    driver: Any = None
    applied: list[str] = []
    plans: list[Any] = []
    sinks: list[Any] = []
    # The store lookup's plan, which closes a connection it opened itself.
    lookups: list[Any] = []
    caught: list[warnings.WarningMessage] = []
    try:
        if facts is None and config is None:
            raise ValueError("give the facts with --facts, or a run config with --config")
        if merge and out is None:
            raise ValueError("--merge merges into the JSONL at -o: give -o")
        merge = merge or update
        if facts is not None:
            schema = _strict_ontology(ontology) if ontology is not None else None
            grounder = (
                None
                if dry_run
                else _grounder(
                    config,
                    chosen,
                    locate=locate,
                    paper=False,
                    cache=cache,
                    budget_usd=budget_usd,
                    budget_calls=budget_calls,
                )
            )
            store = _Store(
                text_property=text_property,
                database=database,
                user=user,
                password_env=password_env,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rows, docs, driver = _read_facts(adapter, facts, texts, store)
            validator = Validator(schema, grounder=grounder, locate=locate and grounder is None)
            named: dict[str, Any] = {}
        else:
            assert config is not None  # one of the two is given
            given = (texts, ontology, text_property, database)
            if any(v is not None for v in given) or adapter != "triples":
                raise ValueError(
                    "a run config names its own inputs, ontology and extractor: leave out "
                    "--texts, --ontology, --adapter and the Neo4j source options"
                )
            loaded = _load_run_config(config)
            if chosen is not None:
                loaded = loaded.with_model(chosen)
                typer.echo(f"models: every role on {chosen}")
            if cache is not None:
                loaded = loaded.with_cache(cache)
            loaded = loaded.with_budget(usd=budget_usd, calls=budget_calls)
            if locate:
                raise ValueError("with a run config, the locator is the grounder's: locate: true")
            built = build(loaded)
            extractor = built.stages["extractor"]
            if not isinstance(extractor, TriplesExtractor):
                raise ConfigError(
                    "stages.extractor: odke validate checks the triples another extractor "
                    "wrote, so it takes {use: triples, path: ...}; to extract, use odke run"
                )
            docs = built.documents()
            rows = [row for group in extractor.rows.values() for row in group]
            named = {"extractor": extractor.extractor, "confidence": extractor.confidence}
            stage = built.stages
            plans = built.sinks
            lookups = [built.lookup] if built.lookup is not None else []
            if not dry_run:
                sinks = [plan.open() for plan in plans]
                # Before any model is called, as `odke run` applies it.
                if loaded.bootstrap:
                    applied = _bootstrap(built, sinks)
                built.open_lookup(sinks)
            elif built.lookup is not None:
                typer.echo(f"warning: {built.lookup.dry_run_note}", err=True)
            if dry_run and loaded.bootstrap:
                constrainer = built.stages["constrainer"]
                applied = [
                    s for p in plans if p.can_bootstrap for s in p.ddl(built.ontology, constrainer)
                ]
            validator = Validator(
                built.ontology,
                grounder=stage["grounder"],
                roles=built.context.roles,
                client=built.context.client("ground"),
                normalizer=stage["normalizer"],
                resolver=stage["resolver"],
                corroborator=stage["corroborator"],
                scorer=stage["scorer"],
                gate=stage["gate"],
                inverses=loaded.inverses,
                coverage=loaded.coverage,
            )
        if out is not None and not dry_run:
            sinks.append(JsonlSink(out, merge=merge))
        validator.sinks = tuple(sinks)
        if update and not dry_run and not any(isinstance(s, Retractable) for s in sinks):
            raise ValueError(
                "--update retracts the old versions from the store: give -o, or a config "
                "with a jsonl or neo4j sink"
            )
        # A corroborator the config names merges with the sinks, as the default does.
        configured = validator.corroborator
        if isinstance(configured, SignatureCorroborator) and not configured.store:
            configured.store = stores_of(sinks)
    except (ValueError, ConfigError, OntologyLoadError, ImportError, OSError) as exc:
        _close(driver, [*lookups, *sinks])
        _data_error(exc)
    except ProviderError as exc:
        _close(driver, [*lookups, *sinks])
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    _echo_warnings(caught)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kg, report = validator.validate(rows, docs, dry_run=dry_run, update=update, **named)
    except (ProviderError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        _close(driver, [*lookups, *sinks])
    _echo_warnings(caught)
    typer.echo(report.render())
    verb = "would write" if dry_run else "wrote"
    if applied:
        done = "would apply" if dry_run else "applied"
        typer.echo(f"{'bootstrap':<13} {done} {_count(len(applied), 'statement')}")
    written = [line for plan in plans for line in plan.describe(kg)]
    if out is not None:
        written.append(
            f"jsonl → {out}: entities.jsonl {len(kg.entities)}, facts.jsonl {len(kg.facts)}, "
            f"links.jsonl {len(kg.links)}, manifest.json"
        )
    for line in written or ["nothing: give -o, or name a sink in the config"]:
        typer.echo(f"{verb:<13} {line}")
    if report.stopped:
        raise typer.Exit(EXIT_BUDGET)
    _every_document_failed(report.failed, report.documents)


def _every_document_failed(failed: Any, documents: int) -> None:
    """Exit 1 when every document failed on its own: the run did nothing.

    One document's failure is left out of the graph and named in the report,
    and the run goes on (#162). When that is every document, the cause is
    almost certainly not in the documents, and a zero exit would hide it.
    """
    if documents and isinstance(failed, dict) and len(failed) >= documents:
        typer.echo(
            f"error: every document failed; the first: {next(iter(failed.values()))}", err=True
        )
        raise typer.Exit(1)


def _close(driver: Any, sinks: list[Any]) -> None:
    """The Neo4j source and every sink that holds a connection, closed."""
    for held in [driver, *sinks]:
        close = getattr(held, "close", None)
        if callable(close):
            close()


# --------------------------------------------------------------------------- #
# odke reconcile — a source that is gone
# --------------------------------------------------------------------------- #


@app.command("reconcile")
def reconcile_command(
    sink: str = typer.Option(
        ...,
        "--sink",
        help="The store: a directory odke wrote as JSONL, or a Neo4j URI (bolt://, neo4j://).",
    ),
    delete: list[str] = typer.Option(
        ..., "--delete", help="The id of a document that is gone. Repeat for several."
    ),
    hard_delete: bool = typer.Option(
        False, "--hard-delete", help="Delete a fact left with no source, rather than retire it."
    ),
    ontology: Path | None = typer.Option(
        None,
        "--ontology",
        help="Neo4j: decides whether a value projected again is one value or a list.",
    ),
    database: str | None = typer.Option(None, "--database", help="Neo4j: the database."),
    user: str = typer.Option("neo4j", "--user", envvar="NEO4J_USER", help=USER_HELP),
    password_env: str = typer.Option("NEO4J_PASSWORD", "--password-env", help=PASSWORD_ENV_HELP),
) -> None:
    """Retract documents that are gone from the store, and retire facts left with no source.

    Every fact a document backed loses that document: its evidence, and its
    place in the support list. A fact with another source keeps it; a fact
    with none is retired, marked with the time it lost its last source and
    kept, or deleted with --hard-delete. Its valid clock is left alone. Run
    the same command twice and the second changes nothing.

    A document that changed is an update, not a delete: give the new version
    to `odke validate --update`, which retracts the old one first, so the
    facts the new version still states regain their support.

    Exit 2 is a store that cannot be read.
    """
    from openodke.reconcile import Reconciler
    from openodke.sinks.jsonl import JsonlSink

    store: Any = None
    try:
        if "://" in sink:
            schema = _strict_ontology(ontology) if ontology is not None else None
            password = os.environ.get(password_env)
            if password is None:
                raise ValueError(
                    f"{password_env} is not set: the Neo4j password is read from the "
                    "environment variable --password-env names, never from a flag or a file"
                )
            from openodke.sinks.neo4j import Neo4jSink

            store = Neo4jSink(sink, (user, password), database=database, ontology=schema)
        else:
            if ontology is not None or database is not None:
                raise ValueError("--ontology and --database are for a Neo4j store")
            directory = Path(sink)
            if not (directory / "facts.jsonl").is_file():
                raise ValueError(f"{directory} holds no facts.jsonl: not a store odke wrote")
            store = JsonlSink(directory, merge=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = Reconciler([store], hard_delete=hard_delete).delete(delete)
    except (ValueError, OntologyLoadError, ImportError, OSError) as exc:
        _close(None, [store])
        _data_error(exc)
    _close(None, [store])
    _echo_warnings(caught)
    typer.echo(report.render())


@ontology_app.command("infer")
def ontology_infer(
    paths: list[Path] = typer.Argument(..., help="Files or directories to infer a schema from."),
    out: Path = typer.Option(
        ..., "--out", "-o", help="Where to write the draft: .yaml, .yml or .json."
    ),
    max_types: int = typer.Option(10, "--max-types", help="Keep at most this many entity types."),
    max_predicates: int = typer.Option(
        25, "--max-predicates", help="Keep at most this many predicates."
    ),
    sample_words: int = typer.Option(
        8_000, "--sample-words", help="How many words of the corpus to sample."
    ),
    seed: int = typer.Option(
        0, "--seed", help="Sampling seed: the same files and seed give the same draft."
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Deterministic proposers only: no model, no key, no cost."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Model for the infer role, e.g. ollama/llama3.1."
    ),
    model_provider: str | None = typer.Option(None, "--model-provider", help=MODEL_PROVIDER_HELP),
    name: str = typer.Option("inferred", "--name", help="Name of the drafted ontology."),
) -> None:
    """Draft an ontology for a corpus that has none: for review, never to use as is.

    Record shapes, Hearst patterns and co-occurrence propose types and predicates,
    each with the spans that produced it; unless --no-llm, a model (the infer role)
    then names and merges what they found. Writes the draft marked inferred, with
    its evidence as comments, and the full evidence beside it as
    <out>.evidence.json, and prints what backs each proposal. Then validate, edit,
    and `odke ontology freeze` it.
    """
    # Imported here so `odke --version` and the other commands stay light.
    from openodke.infer.build import infer_ontology
    from openodke.infer.review import (
        evidence_path,
        format_for,
        inferred_header,
        notes,
        render,
        summary,
    )
    from openodke.llm import ModelSpec, ProviderError
    from openodke.loaders import DirectoryLoader

    chosen = _qualified(model, model_provider)
    try:
        fmt = format_for(out)
        loader = DirectoryLoader()
        docs = [doc for path in paths for doc in loader.load(path)]
    except (OSError, ValueError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    if not docs:
        # From the loader's own table, so the hint cannot fall behind what it reads.
        suffixes = ", ".join(loader.loaders)
        typer.echo(
            f"error: no documents: no file under those paths has a loader ({suffixes})",
            err=True,
        )
        raise typer.Exit(2)
    try:
        inference = infer_ontology(
            docs,
            llm=not no_llm,
            spec=ModelSpec(model=chosen) if chosen else None,
            name=name,
            seed=seed,
            sample_words=sample_words,
            max_types=max_types,
            max_predicates=max_predicates,
        )
    except ProviderError as exc:
        typer.echo(f"error: {exc}\n--no-llm runs the deterministic proposers alone.", err=True)
        raise typer.Exit(1) from None
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None

    evidence = evidence_path(out)
    try:
        text = render(
            inference.ontology,
            fmt=fmt,
            header=inferred_header(inference, evidence_file=evidence.name),
            notes=notes(inference),
        )
    except ImportError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    evidence.write_text(
        inference.model_dump_json(indent=2, exclude={"ontology"}) + "\n", encoding="utf-8"
    )
    typer.echo(summary(inference))
    typer.echo(f"wrote {out} — inferred: review it, then `odke ontology freeze {out}`")
    typer.echo(f"wrote {evidence}")


@ontology_app.command("freeze")
def ontology_freeze(
    path: Path = typer.Argument(..., help="The reviewed ontology, JSON or YAML."),
    by: str | None = typer.Option(
        None, "--by", help="Who reviewed it. Defaults to the current user."
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Write the frozen ontology here instead of in place."
    ),
) -> None:
    """Mark a reviewed ontology frozen: inferred cleared, reviewer and time recorded.

    Refuses while `validate` finds an error, printing each one and exiting 1, and
    leaves the file untouched. The frozen file is written without the review
    comments; the evidence file beside a draft is left where it is.
    """
    from openodke.infer.review import format_for, frozen_header, render
    from openodke.ontology import OntologyFreezeError

    ontology = _load_or_exit(path)
    try:
        frozen = ontology.freeze(by=by if by is not None else _current_user())
    except OntologyFreezeError as exc:
        for diagnostic in exc.diagnostics:
            typer.echo(f"{path}: {diagnostic}")
        typer.echo(f"not frozen: {_count(len(exc.diagnostics), 'error')}")
        raise typer.Exit(1) from None
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    target = out if out is not None else path
    try:
        text = render(frozen, fmt=format_for(target), header=frozen_header(frozen))
    except (ValueError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    target.write_text(text, encoding="utf-8")
    when = frozen.frozen_at.isoformat() if frozen.frozen_at else "now"
    typer.echo(f"frozen: {target} (by {frozen.frozen_by} at {when})")


def _current_user() -> str:
    import getpass

    try:
        return getpass.getuser()
    except (OSError, KeyError, ImportError):
        return "unknown"


@app.command("eval")
def eval_stage(
    stage: str = typer.Argument(
        ...,
        help="route, extract, ground, resolve, score, validate, ablation (with --config), "
        "spans (with --facts, and no labels at all), compare (two runs' --items files), "
        "pipeline (your own, run with --cmd, --run or --predictions), or precision (--facts "
        "a judge graded, corrected by --labels on a random sample; no gold).",
    ),
    runs: list[Path] | None = typer.Argument(
        None, help="compare only: run A's --items file, then run B's.", show_default=False
    ),
    labels: Path | None = typer.Option(None, "--labels", help="Your labelled rows, as JSONL."),
    predictions: Path | None = typer.Option(
        None,
        "--predictions",
        help="What the stage produced, as JSONL. For pipeline, what it already wrote.",
    ),
    run: str | None = typer.Option(
        None,
        "--run",
        help="package.module:Name — run that stage over the labels instead. For pipeline: "
        "module:function or path/file.py:function, called with the documents.",
    ),
    ontology: Path | None = typer.Option(
        None,
        "--ontology",
        help="Ontology JSON: for --run with extract or validate, and extract's conformance.",
    ),
    documents: Path | None = typer.Option(
        None,
        "--documents",
        help="Document JSONL, for --run with extract. For pipeline, also a folder of texts.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Run config: for ablation, the pipeline to run three ways; for pipeline "
        "--validator, the Validator's stages and models.",
    ),
    facts: Path | None = typer.Option(
        None,
        "--facts",
        help="For spans and precision: a run's facts.jsonl, or the directory a sink wrote. "
        "For precision, after a judge (the grounder) set each verdict.",
    ),
    describe: bool = typer.Option(
        False, "--describe", help="Print what a label row and a prediction row are, and exit."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the stage's report as JSON; for pipeline, the eval report."
    ),
    items: Path | None = typer.Option(
        None,
        "--items",
        help="Also write each labelled item's outcome here, as JSONL, for `compare`: "
        "extract, ground, validate and route.",
    ),
    metric: str | None = typer.Option(
        None,
        "--metric",
        help="compare: the primary metric, which decides the verdict: f1, precision, recall "
        "or accuracy. Default f1 for documents, accuracy for items; the rest are guardrails.",
    ),
    resamples: int = typer.Option(
        2000, "--resamples", min=100, help="compare: bootstrap resamples."
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="compare: the resampling seed; the same seed gives the same interval. "
        "precision: the seed --make-sheet draws the sample with.",
    ),
    fail_under: float | None = typer.Option(
        None,
        "--fail-under",
        min=0.0,
        max=1.0,
        help="compare, as a CI gate: also exit 1 when the low end of B's 95% range for the "
        "primary metric is under this, e.g. 0.80. A worse verdict always exits 1; an "
        "inconclusive one exits 0 unless --fail-on-inconclusive.",
    ),
    fail_on_inconclusive: bool = typer.Option(
        False,
        "--fail-on-inconclusive",
        help="compare: exit 1 on inconclusive as well, a change this set cannot tell from noise.",
    ),
    report_to: Path | None = typer.Option(
        None, "--report", help="Also write the versioned eval report, as JSON, here."
    ),
    cmd: str | None = typer.Option(
        None,
        "--cmd",
        help="pipeline: a command that reads {in}, a folder of texts, and writes triples to "
        '{out}, e.g. "python my_extract.py {in} {out}". Run without a shell.',
    ),
    bench: Path | None = typer.Option(
        None, "--bench", help="pipeline: a set `odke bench prepare` wrote, to score against."
    ),
    adapter: str = typer.Option(
        "triples",
        "--adapter",
        help="pipeline: the output's format: triples, langchain, langextract or graphrag.",
    ),
    validator: bool = typer.Option(
        False,
        "--validator",
        help="pipeline: also run the Validator over the same output, with --config's stages "
        "and models (a bench set's own odke.json by default), and report both rows.",
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", min=0.0, help="pipeline: seconds --cmd may run before it is stopped."
    ),
    make_sheet: Path | None = typer.Option(
        None,
        "--make-sheet",
        help="precision: draw a random sample of --facts and write it here as `odke label` "
        "grounding sheets, with the texts from --documents.",
    ),
    sample_size: int = typer.Option(
        150, "--n", min=1, help="precision --make-sheet: how many facts to draw."
    ),
    by_predicate: bool = typer.Option(
        False,
        "--by-predicate",
        help="precision --make-sheet: give each predicate its share of the sample.",
    ),
    adjudicate: Path | None = typer.Option(
        None,
        "--adjudicate",
        help="pipeline, with --labels: ground each prediction the gold lacks three times; one "
        "supported in two is possibly missing from gold. Prints an adjudicated precision "
        "beside the strict one and writes the list here, as JSONL.",
    ),
) -> None:
    """Score one stage against your own labelled data.

    Bring your own labelled dataset: openodke ships the formats and the arithmetic,
    never a corpus. The fixtures in its test suite are examples of the formats
    and are not a benchmark. Run with --describe to see what to label.

    `ablation` runs a whole config three ways — extraction alone, + grounding,
    + corroboration — over your labelled extraction set.

    `spans` is the exception: it needs no labelled data. It reads a run's facts
    and reports span width split by the grounder's verdict, which scores the
    too-narrow-citation error with no gold set at all.

    `compare A B` asks whether the change between two runs was real. Write each
    run's outcomes with `--items`, then compare them: a paired bootstrap over the
    items both scored, with a verdict of better, worse or inconclusive and the
    smallest change the set can detect. It exits 1 when B is worse, which makes
    it a CI gate; `--fail-under` adds a floor and `--fail-on-inconclusive` a
    stricter bar.

    `pipeline` runs your whole extraction pipeline, a command (--cmd), a
    function (--run) or output already written (--predictions), and scores
    its triples against your labels (--labels, --documents) or a prepared
    benchmark (--bench). --validator adds the row with the Validator's check.
    --adjudicate LIST grounds each prediction your gold lacks three times and
    prints an adjudicated precision beside the strict one.

    `precision` needs no gold. A judge, the grounder, graded every fact in
    --facts; `--make-sheet DIR` draws a random sample of them to label by hand,
    and the labels read back (--labels) correct the judge's precision by
    prediction-powered inference. It prints the judge's number, the corrected
    one with its 95% interval and the labels' own. Recall is never claimed.

    Extraction, the ablation and pipeline print precision, recall and F1 with
    95% ranges over your documents. `--report` writes the versioned eval
    report; `--json` prints the stage's own report, as 0.x did, and for
    pipeline the eval report itself.
    """
    # Imported here so `odke --version` and the ontology commands stay light.
    from openodke.eval.compare import gate
    from openodke.eval.eval_report import Dataset, Run, from_stage
    from openodke.eval.formats import describe as describe_formats
    from openodke.eval.harness import PipelineError
    from openodke.eval.runner import load_inputs, report_inputs
    from openodke.llm.base import ProviderError

    inputs = (labels, predictions, run, ontology, documents, config, facts, items)
    compare_flags = [
        flag
        for flag, used in (
            ("--metric", metric is not None),
            ("--resamples", resamples != 2000),
            ("--seed", seed != 0),
            ("--fail-under", fail_under is not None),
            ("--fail-on-inconclusive", fail_on_inconclusive),
        )
        if used
    ]
    piped = {
        "--cmd": cmd is not None,
        "--bench": bench is not None,
        "--adapter": adapter != "triples",
        "--validator": validator,
        "--timeout": timeout != 600.0,
        "--adjudicate": adjudicate is not None,
    }
    sampled = {
        "--make-sheet": make_sheet is not None,
        "--n": sample_size != 150,
        "--by-predicate": by_predicate,
    }
    # The flags compare shares with another stage, which takes them as its own.
    shared = {"precision": ("--seed",)}
    compare_flags = [flag for flag in compare_flags if flag not in shared.get(stage, ())]
    try:
        if stage != "pipeline" and any(piped.values()):
            named = ", ".join(flag for flag, given in piped.items() if given)
            raise ValueError(f"{named}: these are for pipeline")
        if stage != "precision" and any(sampled.values()):
            named = ", ".join(flag for flag, given in sampled.items() if given)
            raise ValueError(f"{named}: these are for precision")
        if stage == "compare":
            if any(v is not None for v in inputs):
                raise ValueError("compare reads two --items files and takes no other inputs")
            comparison = _eval_compare(runs or [], describe, as_json, metric, resamples, seed)
            if comparison is None:
                return
            if report_to is not None:
                from openodke.eval.eval_report import EvalReport

                items_compared = comparison.primary.items
                EvalReport(title="compare", n=items_compared, comparison=comparison).write(
                    report_to
                )
                if not as_json:
                    typer.echo(f"wrote {report_to}")
            reasons = gate(
                comparison, fail_under=fail_under, fail_on_inconclusive=fail_on_inconclusive
            )
            for reason in reasons:
                typer.echo(f"gate: fail: {reason}", err=True)
            if reasons:
                raise typer.Exit(1)
            return
        if runs:
            raise ValueError(f"unexpected argument {str(runs[0])!r}: only compare takes runs")
        if compare_flags:
            owners = {"--seed": "compare and precision"}
            raise ValueError(
                "; ".join(
                    f"{flag}: for {owners.get(flag, 'compare')} only" for flag in compare_flags
                )
            )
        if stage == "pipeline":
            from openodke.eval.harness import DESCRIPTION, evaluate_pipeline

            if describe:
                typer.echo(DESCRIPTION)
                return
            if facts is not None:
                raise ValueError("--facts is for spans")
            if items is not None:
                raise ValueError("--items is written for extract, ground, validate and route")
            if config is not None and not (validator or adjudicate is not None):
                raise ValueError(
                    "--config is for ablation, and for pipeline with --validator or --adjudicate"
                )
            report = evaluate_pipeline(
                command=cmd,
                function=run,
                predictions=predictions,
                adapter=adapter,
                labels=labels,
                documents=documents,
                bench=bench,
                ontology=ontology,
                validator=validator,
                config=config,
                timeout=timeout,
                adjudicate=adjudicate,
            )
        elif stage == "precision":
            report = _eval_precision(
                facts,
                labels,
                documents,
                ontology,
                describe=describe,
                others=(predictions, run, config, items),
                make_sheet=make_sheet,
                sample_size=sample_size,
                seed=seed,
                by_predicate=by_predicate,
            )
            if report is None:
                return
        elif stage == "spans":
            from openodke.eval.spans import DESCRIPTION, evaluate_spans, load_facts

            if describe:
                typer.echo(DESCRIPTION)
                return
            if labels is not None:
                raise ValueError(
                    "spans needs no labelled data; pass the run's facts as --facts instead"
                )
            # A facts file is what --predictions already means for the fact
            # stages, so it is taken as --facts rather than refused.
            source = facts if facts is not None else predictions
            if any(v is not None for v in (run, ontology, documents, config, items)):
                raise ValueError("spans reads a run's facts; it takes --facts and nothing else")
            if source is None:
                raise ValueError(
                    "spans needs --facts: a run's facts.jsonl, or the directory a sink wrote"
                )
            dataset = Dataset(name=Path(source).name, path=str(source))
            report = from_stage(evaluate_spans(load_facts(source)), run=Run(dataset=dataset))
        elif stage == "ablation":
            from openodke.eval.ablation import DESCRIPTION, report_ablation
            from openodke.eval.formats import GoldFact, load_jsonl

            if describe:
                typer.echo(
                    f"{DESCRIPTION}\n\n{describe_formats('extract').split('--predictions')[0]}"
                )
                return
            if any(v is not None for v in (predictions, run, ontology, documents, facts, items)):
                raise ValueError("ablation runs the config itself; it takes --config and --labels")
            if config is None or labels is None:
                raise ValueError("ablation needs --config and --labels (or --describe)")
            report = report_ablation(
                _load_run_config(config), load_jsonl(labels, GoldFact), labels=labels
            )
        else:
            if describe:
                typer.echo(describe_formats(stage))
                return
            if labels is None:
                raise ValueError("--labels is required (or --describe to see the format)")
            if config is not None:
                raise ValueError("--config is for ablation")
            if facts is not None:
                raise ValueError("--facts is for spans")
            rows, predicted = load_inputs(
                stage, labels, predictions, run=run, ontology=ontology, documents=documents
            )
            report = report_inputs(stage, rows, predicted, labels=labels, ontology=ontology)
            if items is not None:
                _write_items(items, stage, rows, predicted)
        if report_to is not None:
            report.write(report_to)
    except (ValueError, OSError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except (ProviderError, PipelineError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if as_json:
        # The 0.x stages print their own report, as they did; the newer ones have no 0.x form.
        shown = report if stage in ("pipeline", "precision") else report.stages[0]
        typer.echo(shown.model_dump_json(indent=2))
        return
    typer.echo(report.render())
    if report_to is not None:
        typer.echo(f"wrote {report_to}")


def _eval_precision(
    facts: Path | None,
    labels: Path | None,
    documents: Path | None,
    ontology: Path | None,
    *,
    describe: bool,
    others: tuple[Any, ...],
    make_sheet: Path | None,
    sample_size: int,
    seed: int,
    by_predicate: bool,
) -> Any:
    """`odke eval precision`: its report, after drawing the sample if asked; None to describe."""
    from openodke.eval.eval_report import Dataset
    from openodke.eval.formats import GroundingLabel, load_jsonl
    from openodke.eval.harness import load_documents
    from openodke.eval.ppi import DESCRIPTION, SAMPLE_FILE, report_precision, write_sample
    from openodke.eval.spans import load_facts

    if describe:
        typer.echo(DESCRIPTION)
        return None
    if any(v is not None for v in others):
        raise ValueError(
            "precision reads --facts, with --labels, --documents and --ontology; nothing else"
        )
    if facts is None:
        raise ValueError(
            "precision needs --facts: a run's facts after a judge (the grounder) graded them"
        )
    if make_sheet is not None and labels is not None:
        raise ValueError(
            "--make-sheet draws the sample to label and --labels reads it back: one step at a time"
        )
    judged = load_facts(facts)
    docs = load_documents(documents) if documents is not None else None
    if make_sheet is not None:
        if docs is None:
            raise ValueError("--make-sheet needs --documents: a sheet shows the text a fact cites")
        drawn, made = write_sample(
            judged, docs, make_sheet, n=sample_size, seed=seed, by_predicate=by_predicate
        )
        how = f"seed {seed}" + (", by predicate" if by_predicate else "")
        typer.echo(
            f"drew {len(drawn)} of {_count(len(judged), 'judged fact')} ({how}) into "
            f"{make_sheet / SAMPLE_FILE}",
            err=True,
        )
        for warning in made.warnings:
            typer.echo(f"warning: {warning}", err=True)
        typer.echo(
            f"wrote {_count(len(made.sheets), 'sheet')} to {make_sheet}: {made.first} to "
            f"{made.last}; tick them, read them back with `odke label read {make_sheet} -o "
            "labels.jsonl`, and pass that file as --labels",
            err=True,
        )
    rows = load_jsonl(labels, GroundingLabel) if labels is not None else []
    return report_precision(
        judged,
        rows,
        documents=docs,
        ontology=_load(ontology) if ontology is not None else None,
        dataset=Dataset(
            name=Path(facts).name,
            path=str(facts),
            documents=len(docs) if docs is not None else None,
            labels=len(rows) if labels is not None else None,
        ),
    )


def _write_items(path: Path, stage: str, rows: Any, predicted: Any) -> None:
    from openodke.eval.compare import item_rows, write_items

    outcomes, warnings = item_rows(stage, rows, predicted)
    write_items(path, outcomes)
    for warning in warnings:
        typer.echo(f"warning: {warning}", err=True)
    unit = "document" if stage == "extract" else "item"
    typer.echo(f"wrote {path}: {_count(len(outcomes), unit)}", err=True)


def _eval_compare(
    runs: list[Path], describe: bool, as_json: bool, metric: str | None, resamples: int, seed: int
) -> Comparison | None:
    from openodke.eval.compare import DESCRIPTION, compare_files

    if describe:
        typer.echo(DESCRIPTION)
        return None
    if len(runs) != 2:
        raise ValueError(f"compare takes two runs' --items files, A then B; got {len(runs)}")
    comparison = compare_files(runs[0], runs[1], metric=metric, resamples=resamples, seed=seed)
    typer.echo(comparison.model_dump_json(indent=2) if as_json else comparison.render())
    return comparison


# --------------------------------------------------------------------------- #
# odke label — labelling by hand
# --------------------------------------------------------------------------- #

label_app = typer.Typer(
    help="Write sheets to label by hand, and read the ticks back as labels.",
    no_args_is_help=True,
)
app.add_typer(label_app, name="label")


def _data_error(exc: Exception) -> NoReturn:
    # One `error:` line per problem: a sheet can have several, each with its line.
    for line in str(exc).splitlines():
        typer.echo(f"error: {line}", err=True)
    raise typer.Exit(2) from exc


@label_app.command("make")
def label_make(
    kind: str = typer.Argument(..., help="grounding or pair."),
    items: Path = typer.Argument(..., help="The rows to label, as JSONL."),
    out: Path = typer.Option(..., "--out", "-o", help="A directory with no sheets in it yet."),
    per_sheet: int = typer.Option(
        50, "--per-sheet", min=1, help="Items per sheet: one sheet is one sitting."
    ),
) -> None:
    """Write numbered markdown sheets to tick by hand, with the rows kept beside them.

    A grounding row is a GroundingLabel without its verdict: the fact and the
    text it cites. A pair row is two mentions, `a` and `b`, each a key and a
    type with an optional label and context. Each item gets a heading, what to
    judge and one box per answer; sheet-NN.items.jsonl keeps its rows. The same
    rows write the same sheets, byte for byte, and a directory that already
    holds sheets is refused. Exit 2 is a row or a directory it cannot use.
    """
    # Imported here so `odke --version` and the ontology commands stay light.
    from openodke.eval.sheets import make_sheets

    try:
        made = make_sheets(kind, items, out, per_sheet=per_sheet)
    except (ValueError, OSError) as exc:
        _data_error(exc)
    for warning in made.warnings:
        typer.echo(f"warning: {warning}", err=True)
    typer.echo(f"wrote {_count(len(made.sheets), 'sheet')} to {out}: {made.first} to {made.last}")


@label_app.command("read")
def label_read(
    path: Path = typer.Argument(..., help="A sheet, or the directory `odke label make` wrote."),
    out: Path = typer.Option(..., "--out", "-o", help="Where to write the labels, as JSONL."),
) -> None:
    """Read ticked sheets back as GroundingLabel or PairLabel rows.

    One tick is a label. No tick leaves the item unlabelled, and it is counted.
    Two ticks, or a heading the sidecar does not know, exits 2 with the file and
    the line of the item's heading, and nothing is written. Notes go beside the
    labels as <out>.notes.jsonl; pairs answered unsure as <out>.unsure.jsonl.
    """
    from openodke.eval.sheets import read_sheets

    try:
        reading = read_sheets(path)
        written = reading.write(out)
    except (ValueError, OSError) as exc:
        _data_error(exc)
    typer.echo(reading.summary())
    for target, n in written:
        typer.echo(f"wrote {target}: {_count(n, 'row')}")


# --------------------------------------------------------------------------- #
# odke bench — public benchmarks
# --------------------------------------------------------------------------- #

bench_app = typer.Typer(
    help="Public benchmarks: fetch a dataset, prepare a run, run it three ways and score it.",
    no_args_is_help=True,
)
app.add_typer(bench_app, name="bench")


def _dataset(name: str) -> Any:
    from openodke.eval.datasets import DATASETS

    if name not in DATASETS:
        typer.echo(f"error: unknown dataset {name!r}; one of {', '.join(DATASETS)}", err=True)
        raise typer.Exit(2)
    return DATASETS[name]


@bench_app.command("fetch")
def bench_fetch(
    dataset: str = typer.Argument(..., help="text2kgbench or redocred."),
    dest: Path = typer.Argument(..., help="Directory to download into."),
    source: str = typer.Option(
        "wikidata_tekgen", help="text2kgbench only: wikidata_tekgen or dbpedia_webnlg."
    ),
    ontology: list[str] | None = typer.Option(
        None, "--ontology", help="text2kgbench only: an ontology id, repeatable (default: all)."
    ),
    split: list[str] | None = typer.Option(
        None, "--split", help="redocred only: dev, test or train, repeatable (default: dev, test)."
    ),
) -> None:
    """Download a dataset's official files. Nothing is redistributed by openodke."""
    module = _dataset(dataset)
    try:
        if dataset == "text2kgbench":
            module.fetch(dest, source=source, ontologies=ontology or None)
        else:
            module.fetch(dest, splits=tuple(split or ("dev", "test")))
    except (ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"fetched {dataset} into {dest}")


@bench_app.command("prepare")
def bench_prepare(
    dataset: str = typer.Argument(..., help="text2kgbench or redocred."),
    root: Path = typer.Argument(..., help="Where `odke bench fetch` downloaded it."),
    out: Path = typer.Option(..., "--out", help="Directory to write the runnable set into."),
    ontology: str | None = typer.Option(
        None, "--ontology", help="text2kgbench only: the ontology id, e.g. ont_1_movie."
    ),
    source: str = typer.Option(
        "wikidata_tekgen", help="text2kgbench only: wikidata_tekgen or dbpedia_webnlg."
    ),
    split: str = typer.Option("test", help="redocred only: dev, test or train."),
    limit: int | None = typer.Option(None, help="Keep the first N documents (a cheap pilot)."),
    extract_model: str | None = typer.Option(
        None,
        "--extract-model",
        help="Any LiteLLM model string: openai/gpt-5, anthropic/..., ollama/llama3.1.",
    ),
    ground_model: str | None = typer.Option(
        None, "--ground-model", help="Any LiteLLM model string; a smaller one than the extractor."
    ),
    max_tokens: int = typer.Option(
        16000, "--max-tokens", help="The extractor's output room; lower it for a small model."
    ),
    paper: bool = typer.Option(
        False,
        "--paper",
        help="ODKE+'s own grounder and gate: the whole context, True/False, affirmed facts only.",
    ),
) -> None:
    """Write documents, ontology, gold and an `odke.json` run config. Calls no model."""
    module = _dataset(dataset)
    models = {
        "extract_model": extract_model,
        "ground_model": ground_model,
        "paper": paper,
        "max_tokens": max_tokens,
    }
    try:
        if dataset == "text2kgbench":
            if ontology is None:
                raise ValueError("text2kgbench needs --ontology, e.g. ont_1_movie")
            module.prepare(root, ontology, out, source=source, limit=limit, **models)
        else:
            module.prepare(root, out, split=split, limit=limit, **models)
    except (ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"prepared {out}; run it with: odke bench run {dataset} {out}")


@bench_app.command("run")
def bench_run(
    dataset: str = typer.Argument(..., help="text2kgbench or redocred."),
    prepared: Path = typer.Argument(..., help="A directory `odke bench prepare` wrote."),
    as_json: bool = typer.Option(False, "--json", help="The dataset's own report as JSON."),
    report_to: Path | None = typer.Option(
        None, "--report", help="Where to write the eval report. Default: <prepared>/report.json."
    ),
) -> None:
    """Run the prepared config three ways and score each row. Calls the models it names.

    Extraction and grounding are each called once; + grounding and + corroboration
    replay them, so the bill is one run's, not three. Each row's precision,
    recall and F1 carry a 95% range over the documents, and the versioned eval
    report is written beside the predictions.
    """
    from openodke.llm.base import ProviderError

    module = _dataset(dataset)
    target = report_to if report_to is not None else prepared / "report.json"
    try:
        report = module.evaluate(prepared)
        report.write(target)
    except (ValueError, OSError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except ProviderError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(report.stages[0].model_dump_json(indent=2))
        return
    typer.echo(report.render())
    typer.echo(f"wrote {target}")


if __name__ == "__main__":  # pragma: no cover
    app()
