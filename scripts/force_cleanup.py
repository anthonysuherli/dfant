#!/usr/bin/env python3
"""
Force cleanup of database files with multiple strategies.
"""

import os
import shutil
import time
from pathlib import Path
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def force_cleanup():
    """Force cleanup with multiple strategies."""
    logger = logging.getLogger(__name__)
    project_root = Path(__file__).parent.parent

    db_files = ["nba_dfs.db", "nba_dfs.db-shm", "nba_dfs.db-wal"]
    database_dir = project_root / "database"
    database_dir.mkdir(exist_ok=True)

    logger.info("Attempting forced cleanup of database files...")

    # Strategy 1: Copy files to new location, then remove originals
    for db_file in db_files:
        source_path = project_root / db_file
        dest_path = database_dir / db_file

        if source_path.exists():
            try:
                # Copy first
                shutil.copy2(str(source_path), str(dest_path))
                logger.info(f"Copied {db_file} to database directory")

                # Try to remove original
                try:
                    source_path.unlink()
                    logger.info(f"Removed original {db_file} from root")
                except Exception as e:
                    logger.warning(f"Could not remove original {db_file}: {e}")
                    logger.info(f"File copied but original remains. Manual removal needed.")

            except Exception as e:
                logger.error(f"Could not copy {db_file}: {e}")

    # Test new database connection
    logger.info("Testing new database connection...")
    try:
        import sys
        sys.path.append(str(project_root / "src"))
        from database.connection import get_db_session

        with get_db_session() as session:
            session.execute("SELECT 1")
            logger.info("New database connection successful")

    except Exception as e:
        logger.error(f"Database connection test failed: {e}")

    # Show final status
    logger.info("\nFinal status:")
    for db_file in db_files:
        root_file = project_root / db_file
        db_file_new = database_dir / db_file

        if root_file.exists():
            logger.warning(f"❌ {db_file} still in root directory")
        else:
            logger.info(f"✅ {db_file} removed from root")

        if db_file_new.exists():
            logger.info(f"✅ {db_file} exists in database directory")

if __name__ == "__main__":
    setup_logging()
    force_cleanup()