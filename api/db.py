from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
)

try:
    from pgvector.psycopg2 import register_vector
except Exception:  # pragma: no cover - optional dependency until runtime
    register_vector = None

if register_vector:
    @event.listens_for(engine, "connect")
    def _register_vector(dbapi_connection, connection_record):
        register_vector(dbapi_connection)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
