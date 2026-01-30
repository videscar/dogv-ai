"""
Rebuild BM25 tsvector in batches for a given language.

Usage:
    python scripts/rebuild_tsv.py --language va_va --ts-config catalan --batch-size 5000
"""

import argparse

from sqlalchemy import text as sa_text

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal


def rebuild_tsv(language: str, ts_config: str, batch_size: int) -> None:
    db = SessionLocal()
    try:
        total = db.execute(
            sa_text("SELECT COUNT(*) FROM rag_chunk WHERE language = :language"),
            {"language": language},
        ).scalar_one()
        print(f"Rebuilding tsv for language={language} rows={total}")

        last_id = 0
        updated = 0
        while True:
            ids = db.execute(
                sa_text(
                    """
                    SELECT id
                    FROM rag_chunk
                    WHERE language = :language AND id > :last_id
                    ORDER BY id
                    LIMIT :batch_size
                    """
                ),
                {"language": language, "last_id": last_id, "batch_size": batch_size},
            ).scalars().all()
            if not ids:
                break

            db.execute(
                sa_text(
                    """
                    UPDATE rag_chunk
                    SET tsv = (
                        setweight(
                            to_tsvector(
                                CAST(:ts_config AS regconfig),
                                COALESCE(metadata->>'title', '')
                            ),
                            'A'
                        )
                        || setweight(
                            to_tsvector(CAST(:ts_config AS regconfig), text),
                            'B'
                        )
                    )
                    WHERE id = ANY(:ids)
                    """
                ),
                {"ts_config": ts_config, "ids": ids},
            )
            db.commit()
            updated += len(ids)
            last_id = ids[-1]
            print(f"[progress] updated {updated}/{total}")
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="va_va")
    parser.add_argument("--ts-config", default="catalan")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    rebuild_tsv(args.language, args.ts_config, args.batch_size)


if __name__ == "__main__":
    main()
