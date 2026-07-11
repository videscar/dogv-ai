from pathlib import Path

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from sqlalchemy import text

from api import models  # noqa: F401  # ensure models are imported
from api.db import Base, engine

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def apply_sql_migrations():
    """Apply the sql/*.sql migrations (indexes, pgvector tables not covered by
    the ORM models). Idempotent: each file uses IF NOT EXISTS / IF EXISTS."""
    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        with engine.begin() as conn:
            conn.execute(text(sql_file.read_text()))
        print(f"Applied {sql_file.name}")


def init_db():
    Base.metadata.create_all(bind=engine)
    apply_sql_migrations()
    print("Database initialized")


if __name__ == "__main__":
    init_db()
