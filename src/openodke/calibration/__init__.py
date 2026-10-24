"""Calibration cards: how far each judge agrees with a person, and what follows (#135).

Provisional (DECISIONS #48): the card format and the bars settle with the first
published cards (roadmap §9.4), and may change in a minor release, with a
CHANGELOG line.

openodke asks a model four kinds of question: does this passage support this
fact (the grounder, P1, and its challengers), do these two mentions name one
entity (the pair judge, P4), and is this prediction the gold fact in other
words (the fact-equivalence judge, P5). A model's answer is a measurement, and
a measurement is only as good as its agreement with a person. A **card** is that
agreement for one registered prompt on one dated model, against one version of
one hand-labelled set (`Card`, `load_card`), with the bar the judge must clear
to be a default and whether it does.

Every run report prints each judge's card line, or **UNCALIBRATED** where there
is no card (`status`, `CardStatus.line`). Below its bar, or with no card, a
judge still runs, but nothing is dropped on its word (DECISIONS #54,
`openodke.calibration.policy`): the gate marks a refused fact for review
instead of dropping it, and the pair judge sends its pairs to review.

`odke calibrate` makes a card from your own labels; `ODKE_CARDS` names where
your cards are, and a run reads them as it reads the package's.
"""

from openodke.calibration.card import (
    BAR_TEXT,
    BARS,
    CARD_VERSION,
    UNCALIBRATED,
    Band,
    Bar,
    Card,
    Equivalence,
    Grounding,
    Kappa,
    LabelSet,
    Pairs,
    Share,
    label_hash,
    load_card,
    uncalibrated_line,
)
from openodke.calibration.lookup import (
    ACTIONS,
    CARDS_DIR,
    ENV,
    CardStatus,
    card_for,
    cards,
    judge_of,
    standing,
    status,
    status_of,
)
from openodke.calibration.measure import (
    BANDS,
    FLIP_LIMIT,
    JudgedFact,
    JudgedPair,
    Made,
    equivalence_card,
    grounding_card,
    kappa,
    pair_card,
    share,
    swap_rule,
    voted,
)

# The switch itself, `policy.DEFAULT_POLICY`, is read where it lives, so it is not re-exported.
from openodke.calibration.policy import Policy, Resolved, resolve

__all__ = [
    "ACTIONS",
    "BANDS",
    "BARS",
    "BAR_TEXT",
    "CARDS_DIR",
    "CARD_VERSION",
    "ENV",
    "FLIP_LIMIT",
    "UNCALIBRATED",
    "Band",
    "Bar",
    "Card",
    "CardStatus",
    "Equivalence",
    "Grounding",
    "JudgedFact",
    "JudgedPair",
    "Kappa",
    "LabelSet",
    "Made",
    "Pairs",
    "Policy",
    "Resolved",
    "Share",
    "card_for",
    "cards",
    "equivalence_card",
    "grounding_card",
    "judge_of",
    "kappa",
    "label_hash",
    "load_card",
    "pair_card",
    "resolve",
    "share",
    "standing",
    "status",
    "status_of",
    "swap_rule",
    "uncalibrated_line",
    "voted",
]
