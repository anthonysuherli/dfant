#!/usr/bin/env python3
"""
Remove collection timestamps from existing parquet filenames.

This script renames files from:
  games_20211019_20250924_021536.parquet
To:
  games_20211019.parquet

And from:
  player_logs_20211019_20250924_024839_game_20211019_BKN@MIL.parquet
To:
  player_logs_20211019_game_20211019_BKN@MIL.parquet
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import logging

class ParquetFileRenamer:
    """Remove collection timestamps from parquet filenames."""

    def __init__(self, data_dir: str = "data", dry_run: bool = True):
        self.data_dir = Path(data_dir)
        self.dry_run = dry_run
        self.logger = self._setup_logger()

        # Pattern to match collection timestamp: _20250924_HHMMSS (collection date/time)
        # Match specifically the 2025 collection dates
        self.timestamp_pattern = r'_2025\d{4}_\d{6}'

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _extract_clean_filename(self, original_filename: str) -> str:
        """Extract clean filename using simple naming convention."""
        # Remove collection timestamp first
        clean_name = re.sub(self.timestamp_pattern, '', original_filename)

        # Apply new naming convention: [folder_name]_[date].parquet
        if clean_name.startswith('games_'):
            # games_20211019.parquet
            match = re.match(r'games_(\d{8})\.parquet', clean_name)
            if match:
                return clean_name

        elif clean_name.startswith('player_logs_'):
            # player_logs_20211019_game_20211019_BKN@MIL.parquet -> player_logs_20211019_BKN@MIL.parquet
            match = re.match(r'player_logs_(\d{8})_game_\d{8}_(.+)\.parquet', clean_name)
            if match:
                date, teams = match.groups()
                return f'player_logs_{date}_{teams}.parquet'
            else:
                # Handle other player_logs formats
                match = re.match(r'player_logs_(\d{8})_(.+)\.parquet', clean_name)
                if match:
                    date, rest = match.groups()
                    if rest.startswith('game_'):
                        # Extract team matchup from game_ID format
                        teams = rest.split('_')[-1] if '_' in rest else rest
                        return f'player_logs_{date}_{teams}.parquet'
                    return f'player_logs_{date}_{rest}.parquet'

        elif clean_name.startswith('injuries_'):
            # injuries_20230405_to_20230430.parquet -> injuries_20230405_to_20230430.parquet
            return clean_name

        elif clean_name.startswith('dfs_salaries_'):
            # dfs_salaries_20211019.parquet
            match = re.match(r'dfs_salaries_(\d{8})\.parquet', clean_name)
            if match:
                return clean_name

        elif clean_name.startswith('teams_') or clean_name.startswith('players_'):
            # teams_current.parquet, players_current.parquet
            if 'current' in clean_name:
                folder_name = clean_name.split('_')[0]
                return f'{folder_name}_current.parquet'

        return clean_name

    def _find_duplicate_targets(self, directory: Path) -> dict:
        """Find files that would have the same target name after renaming."""
        target_mapping = defaultdict(list)

        for file_path in directory.glob("*.parquet"):
            clean_name = self._extract_clean_filename(file_path.name)
            target_mapping[clean_name].append(file_path)

        # Return only duplicates
        duplicates = {target: files for target, files in target_mapping.items()
                     if len(files) > 1}
        return duplicates

    def _handle_duplicates(self, duplicates: dict, directory: Path) -> None:
        """Handle files that would have duplicate target names."""
        self.logger.info(f"Found {len(duplicates)} sets of duplicate targets in {directory.name}")

        for target_name, file_list in duplicates.items():
            self.logger.info(f"\nDuplicate target: {target_name}")

            # Sort by modification time (keep newest)
            file_list.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            newest_file = file_list[0]
            older_files = file_list[1:]

            self.logger.info(f"  KEEP: {newest_file.name} (newest)")

            for old_file in older_files:
                self.logger.info(f"  DELETE: {old_file.name} (older)")
                if not self.dry_run:
                    old_file.unlink()
                    self.logger.info(f"    Deleted: {old_file.name}")

            # Rename the newest file
            new_path = directory / target_name
            if not self.dry_run:
                newest_file.rename(new_path)
                self.logger.info(f"    Renamed: {newest_file.name} -> {target_name}")

    def rename_directory(self, directory_name: str) -> None:
        """Rename all parquet files in a specific directory."""
        directory = self.data_dir / directory_name

        if not directory.exists():
            self.logger.warning(f"Directory {directory} does not exist")
            return

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Processing directory: {directory_name}")
        self.logger.info(f"{'='*50}")

        parquet_files = list(directory.glob("*.parquet"))
        if not parquet_files:
            self.logger.info("No parquet files found")
            return

        self.logger.info(f"Found {len(parquet_files)} parquet files")

        # Check for potential duplicates
        duplicates = self._find_duplicate_targets(directory)

        if duplicates:
            self._handle_duplicates(duplicates, directory)
        else:
            # No duplicates - simple rename
            renamed_count = 0
            for file_path in parquet_files:
                clean_name = self._extract_clean_filename(file_path.name)

                if clean_name != file_path.name:  # Only rename if different
                    new_path = directory / clean_name

                    if self.dry_run:
                        self.logger.info(f"WOULD RENAME: {file_path.name} -> {clean_name}")
                    else:
                        file_path.rename(new_path)
                        self.logger.info(f"RENAMED: {file_path.name} -> {clean_name}")
                    renamed_count += 1
                else:
                    self.logger.debug(f"SKIP: {file_path.name} (already clean)")

            self.logger.info(f"Processed {renamed_count} files in {directory_name}")

    def rename_all_directories(self) -> None:
        """Rename files in all data directories."""
        directories = [
            "games",
            "player_logs",
            "injuries",
            "dfs_salaries",
            "teams",
            "players"
        ]

        mode = "DRY RUN" if self.dry_run else "LIVE MODE"
        self.logger.info(f"\nStarting parquet file renaming - {mode}")
        self.logger.info(f"Data directory: {self.data_dir.absolute()}")

        for directory_name in directories:
            self.rename_directory(directory_name)

        self.logger.info(f"\nCompleted parquet file renaming - {mode}")

        if self.dry_run:
            self.logger.info("\n⚠️  This was a DRY RUN - no files were actually renamed")
            self.logger.info("Run with dry_run=False to perform actual renaming")

def main():
    """Main function to run the renaming."""

    # First run in dry-run mode to see what would happen
    print("=== DRY RUN MODE ===")
    renamer = ParquetFileRenamer(dry_run=True)
    renamer.rename_all_directories()

    # Ask for confirmation before actual renaming
    print("\n" + "="*60)
    response = input("Do you want to proceed with actual renaming? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        print("\n=== LIVE MODE - RENAMING FILES ===")
        renamer = ParquetFileRenamer(dry_run=False)
        renamer.rename_all_directories()
    else:
        print("Renaming cancelled by user")

if __name__ == "__main__":
    main()