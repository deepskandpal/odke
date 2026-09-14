"""Value normalisation: one canonical spelling per value, the original kept.

Two sources that agree still disagree textually. "10 December 1815",
"1815-12-10" and "Dec 10, 1815" are one fact, and `Fact.signature` compares
`repr(object_value)` — so unless they are rewritten to one form the corroborator
sees three claims with `support = 1` each, where there is one with `support = 3`.

Deterministic and dependency-free on purpose. `dateparser` and `pint` cover more
forms, but they are large, locale-dependent and occasionally surprising, and a
normaliser that guesses is worse than one that leaves a value alone: a value
left alone costs a missed merge, a wrong rewrite costs a wrong fact. So every
rule here refuses when unsure — `03/04/2020` is not rewritten unless the caller
says which way round dates go.

Nothing is destructive. The original spelling of every rewritten value is kept
under `qualifiers["odke.source_form"]`, and entity labels are never touched: a
name's comparison key goes to `Entity.attributes["odke.name_key"]` for the
resolver, and the label stays what the source said.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from odke.corroborate.provenance import NAME_KEY, SOURCE_FORM, source_forms
from odke.ontology import Ontology
from odke.types import Entity, Fact

# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}  # fmt: skip

_ORD = r"(?:st|nd|rd|th)?"
_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}.*")
_YMD = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
_D_MONTH_Y = re.compile(rf"(\d{{1,2}}){_ORD}\s+(?:of\s+)?([a-z]+)\.?,?\s+(\d{{4}})", re.I)
_MONTH_D_Y = re.compile(rf"([a-z]+)\.?\s+(\d{{1,2}}){_ORD},?\s+(\d{{4}})", re.I)
_NUMERIC = re.compile(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})")
_MONTH_Y = re.compile(r"([a-z]+)\.?,?\s+(\d{4})", re.I)
_YM = re.compile(r"(\d{4})-(\d{1,2})")


def _ymd(year: int, month: int | None, day: int) -> str | None:
    if month is None:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_date(text: str, *, day_first: bool | None = None) -> str | None:
    """An ISO 8601 date (`1815-12-10`), month (`1815-12`) or datetime, or `None`.

    `None` means "not a date I am sure of", and the caller keeps the value as it
    was. All-numeric day/month forms are read only when unambiguous — one part
    above 12, or both equal — unless `day_first` settles it: `03/04/2020` is the
    third of April in London and the fourth of March in New York, and guessing
    would silently corrupt half of them.
    """
    s = " ".join(text.split())
    if _DATETIME.fullmatch(s):
        try:
            return datetime.fromisoformat(s).isoformat()
        except ValueError:
            return None
    if m := _YMD.fullmatch(s):
        return _ymd(int(m[1]), int(m[2]), int(m[3]))
    if m := _D_MONTH_Y.fullmatch(s):
        return _ymd(int(m[3]), _MONTHS.get(m[2].lower()), int(m[1]))
    if m := _MONTH_D_Y.fullmatch(s):
        return _ymd(int(m[3]), _MONTHS.get(m[1].lower()), int(m[2]))
    if m := _NUMERIC.fullmatch(s):
        a, b, year = int(m[1]), int(m[2]), int(m[3])
        if a == b or (a > 12 >= b):
            day, month = a, b
        elif b > 12 >= a:
            day, month = b, a
        elif day_first is None:
            return None
        else:
            day, month = (a, b) if day_first else (b, a)
        return _ymd(year, month, day)
    if m := _MONTH_Y.fullmatch(s):
        named = _MONTHS.get(m[1].lower())
        return f"{int(m[2]):04d}-{named:02d}" if named else None
    if m := _YM.fullmatch(s):
        numbered = int(m[2])
        return f"{m[1]}-{numbered:02d}" if 1 <= numbered <= 12 else None
    return None


# --------------------------------------------------------------------------- #
# Numbers and units
# --------------------------------------------------------------------------- #

# Thousands separators only in groups of three, so "1,2" is not read as 12.
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+")
# Full words scale anything; abbreviations only money — "5 m" is five metres.
_SCALE_WORDS = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12}
_SCALE_ABBREV = {"k": 3, "m": 6, "mn": 6, "bn": 9, "tn": 12}
_SCALE = re.compile(r"(thousand|million|billion|trillion|mn|bn|tn|k|m)\b\s*", re.I)
_CURRENCY_SYMBOLS = (("US$", "USD"), ("Rs.", "INR"), ("Rs", "INR"), ("$", "USD"),
                     ("€", "EUR"), ("£", "GBP"), ("¥", "JPY"), ("₹", "INR"))  # fmt: skip
_CURRENCY_CODES = frozenset({
    "USD", "EUR", "GBP", "JPY", "INR", "CNY", "CHF", "CAD", "AUD", "NZD",
    "SGD", "HKD", "SEK", "NOK", "DKK", "ZAR", "BRL", "MXN", "KRW",
})  # fmt: skip
_PERCENT = frozenset({"%", "percent", "per cent", "pct"})
# SI multiples are powers of 1000 and IEC ones powers of 1024, as the standards
# say. Case-sensitive: "Mb" is megabits, and guessing it is megabytes is wrong.
_BYTES = {
    "B": 1, "byte": 1, "bytes": 1,
    "kB": 10**3, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12, "PB": 10**15,
    "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40, "PiB": 2**50,
}  # fmt: skip


def _plain(d: Decimal) -> str:
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def normalize_quantity(text: str) -> int | float | str | None:
    """A bare number, or a canonical `"<number> <unit>"` string, or `None`.

    `"1,234"` → `1234`; `"12.5 percent"` → `"12.5 %"`; `"$1.2bn"` →
    `"1200000000 USD"`; `"1.5 GiB"` → `"1610612736 B"`. Units are carried as a
    string rather than as a converted float so that the canonical form is exact,
    readable in the graph, and has one `repr` for the signature to compare.
    A leading zero (`"0123"`) is an identifier, not a number, and is refused.
    """
    s = " ".join(text.split())
    currency: str | None = None
    for symbol, code in _CURRENCY_SYMBOLS:
        if s.startswith(symbol):
            currency, s = code, s[len(symbol) :].lstrip()
            break
    else:
        if len(s) > 3 and s[:3] in _CURRENCY_CODES and not s[3].isalpha():
            currency, s = s[:3], s[3:].lstrip()

    m = _NUMBER.match(s)
    if m is None or re.match(r"[-+]?0\d", m[0]):
        return None
    try:
        value = Decimal(m[0].replace(",", ""))
    except InvalidOperation:
        return None
    rest = s[m.end() :].lstrip()

    if scale := _SCALE.match(rest):
        word = scale[1].lower()
        if word in _SCALE_WORDS:
            value *= Decimal(10) ** _SCALE_WORDS[word]
        elif currency is not None or rest[scale.end() :] in _CURRENCY_CODES:
            value *= Decimal(10) ** _SCALE_ABBREV[word]
        else:
            return None
        rest = rest[scale.end() :]

    if currency is None and rest in _CURRENCY_CODES:
        currency, rest = rest, ""
    if currency is not None:
        return f"{_plain(value)} {currency}" if not rest else None
    if rest in _PERCENT:
        return f"{_plain(value)} %"
    if rest in _BYTES:
        return f"{_plain(value * _BYTES[rest])} B"
    if rest:
        return None
    return int(value) if value == value.to_integral_value() else float(value)


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #

_LEGAL_SUFFIXES = frozenset({
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "llc", "llp", "lp", "plc", "gmbh", "ag", "kg", "sa", "sas", "sarl", "srl", "spa",
    "bv", "nv", "oy", "oyj", "ab", "as", "asa", "pty", "pvt", "private", "kk", "se", "ulc",
})  # fmt: skip
_HONORIFICS = frozenset({"mr", "mrs", "ms", "miss", "mx", "dr", "prof", "sir", "dame"})
_GENERATIONAL = frozenset({"jr", "sr", "ii", "iii", "iv", "phd", "md"})
_DOTTED = re.compile(r"\b((?:\w\.){2,})")


def name_key(text: str, *, person: bool = False) -> str:
    """The form two spellings of one name share. For comparison, never display.

    Casefolded, accents and punctuation dropped, dotted initialisms closed up
    (`S.A.` → `sa`), `&` read as `and`. Organisation-style names also lose a
    leading "the" and trailing legal forms — `"Acme, Inc."`, `"ACME Inc"` and
    `"Acme Corporation"` all key to `acme` — but never their last word, so a
    company called "Limited" keeps a key. `person=True` instead reads
    `"Lovelace, Ada"` as `"ada lovelace"` and drops honorifics and generational
    suffixes, and keeps initials: `"J. Smith"` is not `"John Smith"`, and making
    it so is the resolver's call, with a score, not the normaliser's.
    """
    s = unicodedata.normalize("NFKD", text)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = _DOTTED.sub(lambda m: m[1].replace(".", ""), s)
    if person and s.count(",") == 1:
        last, _, first = s.partition(",")
        s = f"{first} {last}"
    tokens = re.findall(r"[^\W_]+", s.replace("&", " and "))
    if person:
        while len(tokens) > 1 and tokens[0] in _HONORIFICS:
            tokens.pop(0)
        while len(tokens) > 1 and tokens[-1] in _GENERATIONAL:
            tokens.pop()
    else:
        if len(tokens) > 1 and tokens[0] == "the":
            tokens.pop(0)
        while len(tokens) > 1 and tokens[-1] in _LEGAL_SUFFIXES:
            tokens.pop()
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #

_DATE_RANGES = frozenset({"date", "datetime"})
_NUMBER_RANGES = frozenset({"integer", "number", "float"})
_VERBATIM_RANGES = frozenset({"string", "boolean"})


def normalize_value(value: Any, *, range: str | None = None, day_first: bool | None = None) -> Any:
    """One value in its canonical form, or unchanged when no rule is sure.

    `range` is the predicate's range from the ontology when there is one. A
    `string` range is left verbatim apart from whitespace — a registration number
    `"114322"` must not become the integer 114322 — `date` ranges are only read
    as dates and numeric ranges only as quantities. With no range, the shape of
    the value decides.
    """
    if isinstance(value, datetime | date):
        return value.isoformat()
    if not isinstance(value, str):
        return value
    s = " ".join(value.split())
    if not s or range in _VERBATIM_RANGES:
        return s
    if range not in _NUMBER_RANGES and (as_date := normalize_date(s, day_first=day_first)):
        return as_date
    if range not in _DATE_RANGES and (quantity := normalize_quantity(s)) is not None:
        return quantity
    return s


def _changed(old: Any, new: Any) -> bool:
    return type(old) is not type(new) or old != new


class ValueNormalizer:
    """Dates, numbers, units and name forms to one canonical shape each.

    Rewrites `object_value` and every qualifier value, records the original
    spelling of each rewritten one under `qualifiers["odke.source_form"]`, and
    stamps each entity's name comparison key into `attributes["odke.name_key"]`.
    Idempotent: normalising a normalised fact changes nothing.

    `ontology` supplies predicate ranges; `person_types` names the entity types
    whose names are people's — which types those are is the caller's schema,
    not something this package assumes.
    """

    def __init__(
        self,
        ontology: Ontology | None = None,
        *,
        day_first: bool | None = None,
        person_types: Collection[str] = (),
    ) -> None:
        self.ontology = ontology
        self.day_first = day_first
        self.person_types = frozenset(person_types)

    def normalize(self, fact: Fact) -> Fact:
        forms = source_forms(fact.qualifiers.get(SOURCE_FORM))
        predicate = self.ontology.predicates.get(fact.predicate) if self.ontology else None
        # An edge's range is an entity type, not a literal; it says nothing here.
        literal = predicate.range if predicate and not fact.is_edge else None

        obj = normalize_value(fact.object_value, range=literal, day_first=self.day_first)
        if _changed(fact.object_value, obj):
            _record(forms, "object_value", fact.object_value)

        qualifiers: dict[str, Any] = {}
        for key, value in fact.qualifiers.items():
            if key == SOURCE_FORM:
                continue
            qualifiers[key] = normalize_value(value, day_first=self.day_first)
            if _changed(value, qualifiers[key]):
                _record(forms, key, value)
        if forms:
            qualifiers[SOURCE_FORM] = forms

        subject = self._entity(fact.subject)
        object_entity = self._entity(fact.object_entity) if fact.object_entity else None
        update: dict[str, Any] = {}
        if _changed(fact.object_value, obj):
            update["object_value"] = obj
        if qualifiers != fact.qualifiers:
            update["qualifiers"] = qualifiers
        if subject is not fact.subject:
            update["subject"] = subject
        if object_entity is not fact.object_entity:
            update["object_entity"] = object_entity
        return fact.model_copy(update=update) if update else fact

    def _entity(self, entity: Entity) -> Entity:
        key = name_key(entity.label or entity.key, person=entity.type in self.person_types)
        if entity.attributes.get(NAME_KEY) == key:
            return entity
        return entity.model_copy(update={"attributes": {**entity.attributes, NAME_KEY: key}})


def _record(forms: dict[str, tuple[str, ...]], field: str, original: Any) -> None:
    # Only text has a spelling worth keeping; a datetime's ISO form loses nothing.
    if isinstance(original, str) and original not in forms.get(field, ()):
        forms[field] = (*forms.get(field, ()), original)


__all__ = [
    "ValueNormalizer",
    "name_key",
    "normalize_date",
    "normalize_quantity",
    "normalize_value",
]
