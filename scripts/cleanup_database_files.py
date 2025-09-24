#!/usr/bin/env python3
"""
Clean up database files from root directory and move to proper location.

This script handles moving existing database files to the correct directory
and provides options for organizing data storage.
"""

import shutil
import argparse
from pathlib import Path
import logging


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def move_database_files(destination: str = "database") -> None:
    """Move database files to specified destination directory."""
    logger = logging.getLogger(__name__)

    # Get project root
    project_root = Path(__file__).parent.parent

    # Database files to move
    db_files = [
        "nba_dfs.db",
        "nba_dfs.db-shm",
        "nba_dfs.db-wal"
    ]

    # Determine destination directory
    if destination == "database":
        dest_dir = project_root / "database"
    elif destination == "data_parquet":
        dest_dir = project_root / "data_parquet" / "database"
    else:
        raise ValueError("Destination must be 'database' or 'data_parquet'")

    dest_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0

    for db_file in db_files:
        source_path = project_root / db_file
        dest_path = dest_dir / db_file

        if source_path.exists():
            try:
                # Try to move the file
                shutil.move(str(source_path), str(dest_path))
                logger.info(f"Moved {db_file} to {dest_dir}")
                moved_count += 1
            except Exception as e:
                if "being used by another process" in str(e) or "Device or resource busy" in str(e):
                    logger.warning(f"Cannot move {db_file} - file is in use. Stop any running processes and try again.")
                else:
                    logger.error(f"Error moving {db_file}: {e}")
        else:
            logger.info(f"File {db_file} not found in root directory")

    if moved_count == 0:
        logger.info("No database files moved. They may already be in the correct location or in use.")
    else:
        logger.info(f"Successfully moved {moved_count} database files to {dest_dir}")


def update_gitignore():
    """Update .gitignore to exclude database files in proper locations."""
    logger = logging.getLogger(__name__)
    project_root = Path(__file__).parent.parent
    gitignore_path = project_root / ".gitignore"

    gitignore_entries = [
        "# Database files",
        "database/*.db*",
        "data_parquet/database/*.db*",
        "",
        "# SQLite temporary files",
        "*.db-shm",
        "*.db-wal",
        "*.sqlite*",
        ""
    ]

    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            existing_content = f.read()
    else:
        existing_content = ""

    # Add entries if not already present
    new_entries = []
    for entry in gitignore_entries:
        if entry.strip() and entry not in existing_content:
            new_entries.append(entry)

    if new_entries:
        with open(gitignore_path, 'a') as f:
            f.write("\n" + "\n".join(new_entries))
        logger.info(f"Updated .gitignore with database file exclusions")
    else:
        logger.info(".gitignore already contains database file exclusions")


def create_directory_structure():
    """Create clean directory structure for project data."""
    logger = logging.getLogger(__name__)
    project_root = Path(__file__).parent.parent

    directories = [
        "database",           # SQLite database files
        "data_parquet/core",  # Core NBA data (player logs, team logs)
        "data_parquet/external",  # External data (betting, injuries)
        "data_parquet/features",  # ML features and engineered data
        "models",            # Trained models and artifacts
        "logs",              # Application logs
    ]

    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Clean up and organize database files"
    )
    parser.add_argument(
        "--destination",
        type=str,
        choices=["database", "data_parquet"],
        default="database",
        help="Where to move database files (default: database)"
    )
    parser.add_argument(
        "--setup-structure",
        action="store_true",
        help="Create complete project directory structure"
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting database cleanup and organization...")

    try:
        # Move database files
        move_database_files(args.destination)

        # Update gitignore
        update_gitignore()

        # Create directory structure if requested
        if args.setup_structure:
            create_directory_structure()

        logger.info("[COMPLETED] Database cleanup finished successfully!")
        logger.info(f"Database files are now in: {args.destination} directory")

    except Exception as e:
        logger.error(f"[ERROR] Cleanup failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())