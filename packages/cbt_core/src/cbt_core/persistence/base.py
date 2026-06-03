"""SQLAlchemy declarative base and the shared constraint naming convention.

A stable naming convention keeps constraint names deterministic across databases and
migrations, so alembic autogenerate and hand-written migrations agree (CLAUDE.md section 6).
"""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for every ORM row. Rows never leave the persistence layer."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
