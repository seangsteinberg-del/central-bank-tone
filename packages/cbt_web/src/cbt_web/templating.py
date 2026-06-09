"""Jinja2 template configuration and presentation helpers for the web UI.

Presentation-only logic (mapping a tone to a colour class, a score to an axis position, a
datetime to a date string) lives here as Jinja filters so templates stay declarative and the
mapping is defined once. The templates and static assets are loaded from this package by
filesystem path, so they work from a source checkout without a build step.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi.templating import Jinja2Templates

from cbt_core import CentralBank, ToneLabel

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

# Tone -> CSS modifier class, for colouring tone badges and timeline points consistently.
_TONE_CLASS = {
    ToneLabel.HAWKISH: "hawkish",
    ToneLabel.DOVISH: "dovish",
    ToneLabel.NEUTRAL: "neutral",
    ToneLabel.MIXED: "mixed",
}


def _tone_class(tone: ToneLabel) -> str:
    """Return the CSS modifier class for a tone label."""
    return _TONE_CLASS[tone]


def _tone_label(tone: ToneLabel) -> str:
    """Return a human-readable label for a tone (for example ``Hawkish``)."""
    return tone.value.replace("_", " ").title()


def _bank_label(bank: CentralBank) -> str:
    """Return a human-readable label for a central bank (for example ``Federal Reserve``)."""
    return bank.value.replace("_", " ").title()


def _format_date(value: datetime) -> str:
    """Format a datetime as an ISO date (``YYYY-MM-DD``)."""
    return value.date().isoformat()


def _clean_title(title: str, speaker_name: str) -> str:
    """Strip a leading ``"<speaker name>: "`` that BIS Review titles prepend, to avoid redundancy.

    On a speaker or speech page the speaker's name is already shown, so the prefix is noise. Only an
    exact leading name match is removed, so a real title that merely contains a colon is left intact.

    Args:
        title: The raw speech title.
        speaker_name: The speaker whose name may prefix the title.

    Returns:
        The title without the redundant leading speaker-name prefix.
    """
    prefix = f"{speaker_name}:"
    if title.lower().startswith(prefix.lower()):
        return title[len(prefix) :].strip() or title
    return title


def _year(value: datetime) -> str:
    """Format a datetime as its four-digit year."""
    return f"{value.year}"


def _score_chip(score: float) -> str:
    """Return the CSS modifier for a score chip: positive (hawkish), negative (dovish), or zero."""
    if score > 0.02:
        return "pos"
    if score < -0.02:
        return "neg"
    return "zero"


def _tone_side(score: float) -> str:
    """Return the diverging-bar side for a tone score: ``hawk`` (>0), ``dove`` (<0), or ``flat``."""
    if score > 0.02:
        return "hawk"
    if score < -0.02:
        return "dove"
    return "flat"


def _bar_width(score: float) -> str:
    """Return a diverging tone bar's width percent for a score on ``[-1, 1]`` (half-track is 1.0)."""
    return f"{min(abs(score), 1.0) * 50.0:.1f}"


def _delta(value: float | None) -> str:
    """Format a signed stance change with a direction arrow, or an em dash when there is none."""
    if value is None:
        return "—"  # em dash: no reading that far back to compare against, not a zero
    if value > 0.02:
        return f"{value:+.2f} ▲"
    if value < -0.02:
        return f"{value:+.2f} ▼"
    return f"{value:+.2f}"


def _delta_class(value: float | None) -> str:
    """Return the CSS modifier for a stance change: ``up`` (hawkish), ``down`` (dovish), or flat."""
    if value is None or -0.02 <= value <= 0.02:
        return "flat"
    return "up" if value > 0.02 else "down"


def build_templates() -> Jinja2Templates:
    """Build the Jinja2 templates object with the UI's presentation filters registered.

    Returns:
        A configured :class:`~fastapi.templating.Jinja2Templates` instance.
    """
    instance = Jinja2Templates(directory=str(TEMPLATES_DIR))
    instance.env.filters["tone_class"] = _tone_class
    instance.env.filters["tone_label"] = _tone_label
    instance.env.filters["bank_label"] = _bank_label
    instance.env.filters["format_date"] = _format_date
    instance.env.filters["clean_title"] = _clean_title
    instance.env.filters["year"] = _year
    instance.env.filters["score_chip"] = _score_chip
    instance.env.filters["tone_side"] = _tone_side
    instance.env.filters["bar_width"] = _bar_width
    instance.env.filters["delta"] = _delta
    instance.env.filters["delta_class"] = _delta_class
    return instance


# A single templates instance shared by the views and the error handlers. Constructing it only
# builds a filesystem loader (no IO, no configuration), so a module-level instance is safe.
templates = build_templates()
