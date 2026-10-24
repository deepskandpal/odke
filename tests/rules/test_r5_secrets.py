"""R5. Secrets stay home.

A key lives in the environment and goes only to the endpoint it was named for.
A password in a model URL is part of the config, which the manifest redacts,
but an error message that quotes the URL carries it into the manifest's
`failed` list, the `document.failed` event, the report and the warning on
stderr (release audit, finding 9). Sending the provider's key to whatever host
`base_url` names was the 9 Oct audit's finding 6.

The check: run with a fake key and a URL password against a gateway that
cannot be reached, then search everything the run printed or wrote for either.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import yaml
from _support import copy_example
from typer.testing import CliRunner

from openodke.cli.main import app

runner = CliRunner()

KEY = "sk-rule5-fake-0f9e8d7c6b5a"
PASSWORD = "Hunter2Rule5"
GATEWAY = f"https://svc:{PASSWORD}@gateway.invalid/v1"
NO_RETRY = {"attempts": 1}


class _Unreachable:
    """Stands in for `urllib.request.urlopen`: notes each request, then cannot connect."""

    def __init__(self) -> None:
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: Any = None) -> Any:
        self.requests.append(request)
        raise urllib.error.URLError("connection refused")


def _project(tmp_path: Path, model: dict[str, Any], *, extract: bool) -> Path:
    """`examples/triples`, its models sent to `model`; with `extract`, an LLM extractor too."""
    project = copy_example("triples", tmp_path)
    config = yaml.safe_load((project / "odke.yaml").read_text(encoding="utf-8"))
    config["models"] = {"ground": model, "extract": model}
    config["stages"]["grounder"] = {"use": "llm", "retry": NO_RETRY}
    if extract:
        config["stages"]["extractor"] = {"use": "llm", "retry": NO_RETRY}
    (project / "odke.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return project


def _channels(project: Path, result: Any, logs: str) -> dict[str, str]:
    """Everything the run printed, logged or wrote, by where it went. The config is the input."""
    channels = {"stdout": result.stdout, "stderr": result.stderr, "logs": logs}
    for path in sorted(project.rglob("*")):
        if path.is_file() and path.name != "odke.yaml":
            channels[path.relative_to(project).as_posix()] = path.read_text("utf-8", "replace")
    return channels


GATEWAY_MODEL = {"model": "openai/gateway-large", "base_url": GATEWAY, "api_key_env": "GATEWAY_KEY"}


# Which stages call the gateway (`extract`: the extractor and the grounder; `ground`: the
# grounder alone, after a model-free extractor), and how the run logs. JSON logs carry no message
# text unless `--log-text` asks for it (#41), so a grounding failure leaks nothing there.
SCENARIOS = [
    pytest.param("extract", ["--log-format", "text"], id="extract-text"),
    pytest.param("extract", ["--log-format", "json"], id="extract-json"),
    pytest.param("extract", ["--log-format", "json", "--log-text"], id="extract-json-text"),
    pytest.param("ground", ["--log-format", "text"], id="ground-text"),
    pytest.param("ground", ["--log-format", "json"], id="ground-json"),
    pytest.param("ground", ["--log-format", "json", "--log-text"], id="ground-json-text"),
]


@pytest.mark.parametrize(("stages", "logs"), SCENARIOS)
def test_no_key_or_url_password_reaches_the_manifest_the_logs_or_stderr(
    stages: str,
    logs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    project = _project(tmp_path, GATEWAY_MODEL, extract=stages == "extract")
    monkeypatch.setenv("GATEWAY_KEY", KEY)
    unreachable = _Unreachable()
    monkeypatch.setattr(urllib.request, "urlopen", unreachable)

    with caplog.at_level(logging.DEBUG):
        result = runner.invoke(app, ["run", str(project / "odke.yaml"), *logs])
    # The gateway was asked, and the key went to it: the endpoint it was named for.
    assert unreachable.requests, result.output
    assert {r.get_header("Authorization") for r in unreachable.requests} == {f"Bearer {KEY}"}

    channels = _channels(project, result, caplog.text)
    leaks = {
        name: [where for where, text in channels.items() if secret in text]
        for name, secret in (("the key", KEY), ("the URL password", PASSWORD))
    }
    leaks = {name: where for name, where in leaks.items() if where}
    assert not leaks, (
        "; ".join(f"{name} reached {', '.join(where)}" for name, where in leaks.items())
        + ". A secret stays in the environment: remove a URL's password where an error "
        "message is made (`could not reach ... at <url>`), and in `_batch.reason()` as the "
        "backstop for messages from third-party code."
    )


def test_a_providers_key_is_never_sent_to_a_host_base_url_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `base_url` and no `api_key_env`, the provider's own key stays home."""
    project = _project(
        tmp_path, {"model": "openai/llama3.1", "base_url": "http://gpu-01.invalid:8000/v1"},
        extract=False,
    )  # fmt: skip
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    unreachable = _Unreachable()
    monkeypatch.setattr(urllib.request, "urlopen", unreachable)

    result = runner.invoke(app, ["run", str(project / "odke.yaml")])
    assert unreachable.requests, result.output
    sent = [json.dumps(dict(r.header_items())) for r in unreachable.requests]
    assert not any(KEY in headers for headers in sent), (
        "OPENAI_API_KEY was sent to the host `base_url` names. A `base_url` sends the call "
        "somewhere the provider's key was never meant to go: read a key for it only from "
        "the variable `api_key_env` names (`openodke.llm.providers.key_env_for`)."
    )
