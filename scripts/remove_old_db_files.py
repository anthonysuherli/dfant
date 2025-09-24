#!/usr/bin/env python3
"""
Remove old database files from root directory (run when no processes are using them).
"""

import os
from pathlib import Path
import logging

def main():
    """Remove old database files from root."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    project_root = Path(__file__).parent.parent
    db_files = ["nba_dfs.db", "nba_dfs.db-shm", "nba_dfs.db-wal"]

    logger.info("Attempting to remove old database files from root directory...")

    removed_count = 0
    for db_file in db_files:
        file_path = project_root / db_file
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"✅ Removed {db_file}")
                removed_count += 1
            except Exception as e:
                logger.error(f"❌ Could not remove {db_file}: {e}")
                logger.info("   → Close any IDE connections, terminal sessions, or scripts using the database")
        else:
            logger.info(f"ℹ️  {db_file} not found in root (already removed)")

    if removed_count == len([f for f in db_files if (project_root / f).exists()]):
        logger.info("🎉 Root directory cleanup completed!")
    else:
        logger.info("⚠️  Some files remain. Close all database connections and try again.")

if __name__ == "__main__":
    main()