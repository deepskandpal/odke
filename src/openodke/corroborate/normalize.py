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

from openodke.corroborate.merge import DEFAULT_INTERVALS
from openodke.corroborate.provenance import NAME_KEY, SOURCE_FORM, source_forms
from openodke.ontology import Ontology
from openodke.types import Entity, Fact

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

# English notation: a dot decimal, and thousands separated by commas in groups
# of three, so "1,2" is not read as 12 and "1,5 km" (a decimal comma) is refused.
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+")
# Full words scale anything; abbreviations only money — "5 m" is five metres.
_SCALE_WORDS = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12}
_SCALE_ABBREV = {"k": 3, "m": 6, "mn": 6, "bn": 9, "tn": 12}
_SCALE = re.compile(r"(thousand|million|billion|trillion|mn|bn|tn|k|m)\b\s*", re.I)
# Currency is never converted: rates change, and a graph is read later. A symbol
# becomes an ISO code only when one currency owns it. "$" is the dollar of a
# dozen countries, "¥" both the yen and the yuan, "Rs" the rupee of four, so
# each is kept as written, after the amount: "$1.2bn" and "$1,200 million" are
# one value, and neither is "1.2bn USD".
_CURRENCY_SYMBOLS = (
    ("US$", "USD"), ("A$", "AUD"), ("CA$", "CAD"), ("NZ$", "NZD"), ("HK$", "HKD"),
    ("S$", "SGD"), ("€", "EUR"), ("£", "GBP"), ("₹", "INR"),
    ("Rs.", "Rs"), ("Rs", "Rs"), ("$", "$"), ("¥", "¥"),
)  # fmt: skip
_CURRENCY_SUFFIXES = dict(_CURRENCY_SYMBOLS)
_CURRENCY_CODES = frozenset({
    "USD", "EUR", "GBP", "JPY", "INR", "CNY", "CHF", "CAD", "AUD", "NZD",
    "SGD", "HKD", "SEK", "NOK", "DKK", "ZAR", "BRL", "MXN", "KRW",
})  # fmt: skip

# One canonical unit per dimension: the SI base unit where there is one (metre,
# kilogram, second), the byte for data, and % for a percentage. Every factor is
# exact by definition (an inch is 0.0254 m, a pound 0.45359237 kg) and held as a
# Decimal, so "5 km" is "5000 m" and never "5000.000000000001 m". Symbols are
# matched as SI writes them, case and all; words in any case.
#
# Data sizes follow SI and IEC: k, M, G, T and P are powers of 1000 whoever
# writes them, "KB" included, and only KiB, MiB, GiB, TiB and PiB are powers of
# 1024. "Mb" is megabits and is refused, not guessed to be megabytes.
#
# A year is not a fixed number of seconds (365 days, 365.25, or a leap year?),
# so it is a unit of its own: "5 yrs" and "5 years" agree, "1 year" and "365
# days" do not. Left out, so their values stay as written: "pound" (also money),
# "ton" (short, long or metric), "month" (no fixed length), "d" (also a penny),
# fluid and troy ounces, and temperature, whose scales differ by an offset as
# well as a factor.
_UNIT_TABLE: tuple[tuple[str, str | int, tuple[str, ...], tuple[str, ...]], ...] = (
    # canonical, factor, symbols, words
    ("m", "0.001", ("mm",), ("millimetre", "millimetres", "millimeter", "millimeters")),
    ("m", "0.01", ("cm",), ("centimetre", "centimetres", "centimeter", "centimeters")),
    ("m", 1, ("m",), ("metre", "metres", "meter", "meters")),
    ("m", 1000, ("km",), ("kilometre", "kilometres", "kilometer", "kilometers")),
    ("m", "0.0254", ("in",), ("inch", "inches")),
    ("m", "0.3048", ("ft",), ("foot", "feet")),
    ("m", "1609.344", ("mi",), ("mile", "miles")),
    ("kg", "0.000001", ("mg",), ("milligram", "milligrams", "milligramme", "milligrammes")),
    ("kg", "0.001", ("g",), ("gram", "grams", "gramme", "grammes")),
    ("kg", 1, ("kg",), ("kilogram", "kilograms", "kilogramme", "kilogrammes")),
    ("kg", 1000, ("t",), ("tonne", "tonnes")),
    ("kg", "0.45359237", ("lb", "lbs"), ()),
    ("kg", "0.028349523125", ("oz",), ("ounce", "ounces")),
    ("s", "0.001", ("ms",), ("millisecond", "milliseconds")),
    ("s", 1, ("s", "sec", "secs"), ("second", "seconds")),
    ("s", 60, ("min", "mins"), ("minute", "minutes")),
    ("s", 3600, ("h", "hr", "hrs"), ("hour", "hours")),
    ("s", 86400, (), ("day", "days")),
    ("s", 604800, ("wk", "wks"), ("week", "weeks")),
    ("year", 1, ("yr", "yrs"), ("year", "years")),
    ("B", 1, ("B",), ("byte", "bytes")),
    ("B", 10**3, ("kB", "KB"), ("kilobyte", "kilobytes")),
    ("B", 10**6, ("MB",), ("megabyte", "megabytes")),
    ("B", 10**9, ("GB",), ("gigabyte", "gigabytes")),
    ("B", 10**12, ("TB",), ("terabyte", "terabytes")),
    ("B", 10**15, ("PB",), ("petabyte", "petabytes")),
    ("B", 2**10, ("KiB",), ("kibibyte", "kibibytes")),
    ("B", 2**20, ("MiB",), ("mebibyte", "mebibytes")),
    ("B", 2**30, ("GiB",), ("gibibyte", "gibibytes")),
    ("B", 2**40, ("TiB",), ("tebibyte", "tebibytes")),
    ("B", 2**50, ("PiB",), ("pebibyte", "pebibytes")),
    ("%", 1, ("%",), ("percent", "per cent", "pct")),
)
_UNIT_SYMBOLS = {s: (Decimal(f), unit) for unit, f, symbols, _ in _UNIT_TABLE for s in symbols}
_UNIT_WORDS = {w: (Decimal(f), unit) for unit, f, _, words in _UNIT_TABLE for w in words}


def _unit(text: str) -> tuple[Decimal, str] | None:
    """`(factor, canonical unit)` for a unit in the table, or `None`."""
    return _UNIT_SYMBOLS.get(text) or _UNIT_WORDS.get(text.casefold())


def _plain(d: Decimal) -> str:
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def normalize_quantity(text: str) -> int | float | str | None:
    """A bare number, or a canonical `"<number> <unit>"` string, or `None`.

    `"1,234"` → `1234`; `"5 km"` → `"5000 m"`; `"12.5 percent"` → `"12.5 %"`;
    `"€1.2bn"` → `"1200000000 EUR"`; `"$1.2bn"` → `"1200000000 $"`;
    `"1.5 GiB"` → `"1610612736 B"`. Units are carried as a string rather than as
    a converted float so that the canonical form is exact, readable in the
    graph, and has one `repr` for the signature to compare: "5 km" and "5 kg"
    can never meet. A unit not in the table, a leading zero (`"0123"` is an
    identifier) and a decimal comma are refused.
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

    # A unit before a scale: "5 m" is five metres, and only money has millions.
    if currency is None and (unit := _unit(rest)):
        return f"{_plain(value * unit[0])} {unit[1]}"

    if scale := _SCALE.match(rest):
        word = scale[1].lower()
        after = rest[scale.end() :]
        if word in _SCALE_WORDS:
            value *= Decimal(10) ** _SCALE_WORDS[word]
        elif currency is not None or after in _CURRENCY_CODES or after in _CURRENCY_SUFFIXES:
            value *= Decimal(10) ** _SCALE_ABBREV[word]
        else:
            return None
        rest = after

    if currency is None and rest in _CURRENCY_CODES:
        currency, rest = rest, ""
    if currency is None and rest in _CURRENCY_SUFFIXES:
        currency, rest = _CURRENCY_SUFFIXES[rest], ""
    if currency is not None:
        return f"{_plain(value)} {currency}" if not rest else None
    if unit := _unit(rest):
        return f"{_plain(value * unit[0])} {unit[1]}"
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
# `quantity` is a number with its unit, held as text so the unit survives a
# structured-output call that would hold a `number` range to a bare number.
_NUMBER_RANGES = frozenset({"integer", "number", "float", "quantity"})
_VERBATIM_RANGES = frozenset({"string", "boolean"})


def normalize_value(value: Any, *, range: str | None = None, day_first: bool | None = None) -> Any:
    """One value in its canonical form, or unchanged when no rule is sure.

    `range` is the predicate's range from the ontology when there is one. A
    `string` range is left verbatim apart from whitespace — a registration number
    `"114322"` must not become the integer 114322 — `date` ranges are only read
    as dates, and numeric and `quantity` ranges only as quantities. With no
    range, the shape of the value decides.
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
    The interval bounds the corroborator reconciles (`DEFAULT_INTERVALS`:
    `start_time`, `end_time`, …) are read as dates, so they stay comparable.
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
            # An interval bound is a date. Read by shape, "2019" would become the
            # integer 2019, which the corroborator cannot order against "2018-06".
            bound = "date" if key in DEFAULT_INTERVALS else None
            qualifiers[key] = normalize_value(value, range=bound, day_first=self.day_first)
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
