from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
import os
from typing import Optional

from .models import Base

class DatabaseManager:
    def __init__(self, database_url: Optional[str] = None):
        if database_url is None:
            # Default to SQLite in database directory
            import os
            from pathlib import Path

            # Get project root directory (parent of src)
            project_root = Path(__file__).parent.parent.parent
            database_dir = project_root / "database"
            database_dir.mkdir(exist_ok=True)

            database_path = database_dir / "nba_dfs.db"
            database_url = f"sqlite:///{database_path}"

        # SQLite optimizations
        self.engine = create_engine(
            database_url,
            echo=False,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 20
            }
        )

        # Enable WAL mode and other optimizations for SQLite
        if database_url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                # Enable WAL mode for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL")
                # Increase cache size (10MB)
                cursor.execute("PRAGMA cache_size=10000")
                # Faster synchronization
                cursor.execute("PRAGMA synchronous=NORMAL")
                # Memory-mapped I/O
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                # Faster temp storage
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session_sync(self) -> Session:
        """Get a synchronous session (caller responsible for cleanup)."""
        return self.SessionLocal()

# Global database manager instance
db_manager = DatabaseManager()

def init_database(database_url: Optional[str] = None):
    """Initialize the database with optional custom URL."""
    global db_manager
    db_manager = DatabaseManager(database_url)
    db_manager.create_tables()

def get_db_session():
    """Dependency for getting database session."""
    return db_manager.get_session()

def get_db_session_sync():
    """Get synchronous database session."""
    return db_manager.get_session_sync()