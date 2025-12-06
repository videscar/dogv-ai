try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import Base, engine
from api import models  # noqa: F401  # ensure models are imported

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database initialized")

if __name__ == "__main__":
    init_db()
