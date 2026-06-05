"""Backfill the structured stance decomposition for already-ingested speeches (ADR 0021).

Re-scores the existing corpus through the structured pipeline using each speech's stored holistic
headline (no re-summarizing, and the immutable tone record is untouched), and upserts a derived
``speech_stance`` row per speech. Idempotent and resumable: a speech that already has a decomposition
is skipped, so the run can be stopped and restarted. Work is parallel over the I/O-bound Gemini
classify calls, one call per speech.

Run: ``uv run python scripts/rescore_stance.py [--bank FEDERAL_RESERVE] [--limit N] [--workers N]``
(needs the live database and a Gemini key).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from cbt_core.domain.registry import CentralBank
from cbt_core.llm.gemini import build_gemini_client
from cbt_core.persistence.engine import create_engine_from_settings, make_session_factory
from cbt_core.persistence.repositories import SpeechStanceRepository
from cbt_core.persistence.rows import SpeechRow, SpeechStanceRow
from cbt_core.services import StanceService, build_stance
from cbt_core.settings import Settings

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

_DEFAULT_WORKERS = 8
_MAX_WORKERS = 12


def _pending(session_factory: sessionmaker[Session], bank: CentralBank | None) -> list[Any]:
    """The speeches that still need a decomposition (optionally limited to one institution)."""
    with session_factory() as session:
        done = set(session.scalars(select(SpeechStanceRow.speech_id)).all())
        statement = select(
            SpeechRow.id,
            SpeechRow.central_bank,
            SpeechRow.score,
            SpeechRow.tone,
            SpeechRow.body,
        )
        if bank is not None:
            statement = statement.where(SpeechRow.central_bank == bank)
        rows = session.execute(statement).all()
    return [row for row in rows if row.id not in done]


def _process(
    row: Any,
    *,
    stance_service: StanceService,
    session_factory: sessionmaker[Session],
    model_id: str,
) -> None:
    """Score one speech through the structured pipeline and upsert its decomposition."""
    assessment = stance_service.assess(
        row.body,
        headline_score=row.score,
        headline_tone=row.tone,
        central_bank=row.central_bank,
    )
    stance = build_stance(row.id, assessment, model_id=model_id)
    with session_factory() as session:
        SpeechStanceRepository(session).upsert(stance)
        session.commit()


def main() -> int:
    """Backfill the structured stance decomposition over the corpus."""
    parser = argparse.ArgumentParser(description="Backfill speech_stance for ingested speeches.")
    parser.add_argument("--bank", default=None, help="restrict to one CentralBank name")
    parser.add_argument("--limit", type=int, default=None, help="cap the number scored")
    parser.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    args = parser.parse_args()
    workers = max(1, min(_MAX_WORKERS, args.workers))
    bank = CentralBank[args.bank] if args.bank else None

    settings = Settings()
    engine = create_engine_from_settings(settings)
    session_factory = make_session_factory(engine)
    stance_service = StanceService(build_gemini_client(settings))

    pending = _pending(session_factory, bank)
    if args.limit is not None:
        pending = pending[: args.limit]
    total = len(pending)
    print(f"re-scoring {total} speeches ({bank.value if bank else 'all banks'}), {workers} workers")
    if total == 0:
        return 0

    started = time.monotonic()
    done = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process,
                row,
                stance_service=stance_service,
                session_factory=session_factory,
                model_id=settings.gemini_model,
            ): row.id
            for row in pending
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001  (isolate one speech's failure, keep going)
                failures += 1
                print(f"  failed {futures[future]}: {type(exc).__name__}: {exc}")
            done += 1
            if done % 50 == 0 or done == total:
                rate = done / max(time.monotonic() - started, 1e-9) * 60
                print(f"  {done}/{total} done, {failures} failed, {rate:.0f}/min")
    print(f"finished: {done - failures} scored, {failures} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
