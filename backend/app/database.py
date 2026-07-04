from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

# SQLite single-writer constraint: WAL mode allows concurrent readers, but
# only ONE writer at a time. API requests, scrape threads, training threads,
# and APScheduler jobs all write to this file — busy_timeout (below) makes
# concurrent writers queue instead of failing instantly with
# "database is locked". Keep background writers coarse-grained (commit in
# batches) and do not add worker processes without moving to Postgres.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # SQLite-specific
    echo=False,
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable WAL mode, foreign keys, and performance pragmas for SQLite."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=30000")           # queue writers for up to 30s
    cursor.execute("PRAGMA cache_size=-64000")            # 64 MB page cache
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.execute("PRAGMA mmap_size=268435456")          # 256 MB memory-mapped I/O
    cursor.execute("PRAGMA journal_size_limit=67108864")  # 64 MB WAL size limit
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
