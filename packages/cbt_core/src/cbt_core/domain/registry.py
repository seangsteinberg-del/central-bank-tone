"""The schema spine: the registry of central banks the system tracks (CLAUDE.md section 2).

This enum is the single source of truth for the institutions the system knows about. Adding a
member here makes domain validation, the mapped persistence column, and every adapter schema
recognize the institution. Do not hardcode this knowledge anywhere else; the persistence layer
references this enum rather than re-listing its values.
"""

from __future__ import annotations

from enum import StrEnum


class CentralBank(StrEnum):
    """A central bank whose speeches the system ingests and scores.

    The values are stable, lower-snake identifiers used as the storage representation. To add
    an institution, add a member here and ship a migration that extends the database enum
    (CLAUDE.md section 6); validation and adapters pick it up automatically.
    """

    FEDERAL_RESERVE = "federal_reserve"
    ECB = "ecb"
    BANK_OF_ENGLAND = "bank_of_england"
    BANK_OF_JAPAN = "bank_of_japan"
    BANK_OF_CANADA = "bank_of_canada"
    RESERVE_BANK_OF_AUSTRALIA = "reserve_bank_of_australia"
    SWISS_NATIONAL_BANK = "swiss_national_bank"
    PEOPLES_BANK_OF_CHINA = "peoples_bank_of_china"
