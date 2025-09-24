"""
CSV Data Collector for NBA DFS Historical Data

This module provides functionality to collect NBA data and save it to CSV files
for machine learning training and analysis.
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from tank01_api import NBAFantasyAPI


@dataclass
class CollectionConfig:
    """Configuration for CSV data collection."""
    start_date: str  # YYYYMMDD format
    end_date: str    # YYYYMMDD format
    data_dir: str = "data"
    rate_limit: float = 1.0  # Requests per second
    max_retries: int = 3
    backoff_factor: float = 2.0
    chunk_size: int = 30  # Days per chunk
    save_frequency: int = 10  # Save every N successful requests


class CSVDataCollector:
    """Collects historical NBA data and saves to CSV files."""

    def __init__(self, api: NBAFantasyAPI, config: CollectionConfig):
        self.api = api
        self.config = config
        self.logger = self._setup_logger()
        self.request_count = 0
        self.last_request_time = 0.0

        # Ensure data directories exist
        self._setup_directories()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('csv_collector')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_directories(self):
        """Create necessary directories for data storage."""
        base_dir = Path(self.config.data_dir)
        directories = [
            'games',
            'player_logs',
            'injuries',
            'dfs_salaries',
            'betting_odds'
        ]

        for directory in directories:
            (base_dir / directory).mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        """Enforce rate limiting between API requests."""
        if self.config.rate_limit > 0:
            time_since_last = time.time() - self.last_request_time
            min_interval = 1.0 / self.config.rate_limit

            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)

            # Add additional buffer for 429 prevention
            time.sleep(0.5)

        self.last_request_time = time.time()
        self.request_count += 1

    def _make_request_with_retry(self, func, *args, **kwargs) -> Optional[pd.DataFrame]:
        """Make API request with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                wait_time = self.config.backoff_factor ** attempt
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}. "
                    f"Retrying in {wait_time:.1f} seconds."
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Request failed after {self.config.max_retries} attempts: {e}")
                    return None

        return None

    def _date_range_generator(self, start_date: str, end_date: str):
        """Generate date range for iteration."""
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')

        current = start
        while current <= end:
            yield current.strftime('%Y%m%d')
            current += timedelta(days=1)

    def collect_games_for_date_range(self):
        """Collect game data for the specified date range."""
        self.logger.info(f"Starting game collection from {self.config.start_date} to {self.config.end_date}")

        dates = list(self._date_range_generator(self.config.start_date, self.config.end_date))
        games_collected = 0

        for date_str in tqdm(dates, desc="Collecting games"):
            try:
                games_df = self._make_request_with_retry(
                    self.api.get_games_for_date, date_str
                )

                if games_df is not None and not games_df.empty:
                    # Save individual date file
                    date_file = Path(self.config.data_dir) / 'games' / f'games_{date_str}.csv'
                    games_df.to_csv(date_file, index=False)

                    games_collected += len(games_df)
                    self.logger.info(f"Saved {len(games_df)} games for {date_str}")
                else:
                    self.logger.info(f"No games found for {date_str}")

            except Exception as e:
                self.logger.error(f"Failed to collect games for {date_str}: {e}")

        # Create consolidated file
        self._consolidate_csv_files('games', 'all_games.csv')
        self.logger.info(f"Game collection completed. Total games: {games_collected}")

    def collect_player_game_logs_for_date_range(self):
        """Collect player game logs for the specified date range."""
        self.logger.info(f"Starting player game log collection from {self.config.start_date} to {self.config.end_date}")

        # First get all games to know which games exist
        games_data = []
        dates = list(self._date_range_generator(self.config.start_date, self.config.end_date))

        # Collect games to get game IDs
        for date_str in dates:
            try:
                games_df = self._make_request_with_retry(
                    self.api.get_games_for_date, date_str
                )
                if games_df is not None and not games_df.empty:
                    games_data.append(games_df)
            except Exception as e:
                self.logger.error(f"Failed to get games for {date_str}: {e}")

        if not games_data:
            self.logger.error("No games found in date range")
            return

        all_games_df = pd.concat(games_data, ignore_index=True)
        self.logger.info(f"Found {len(all_games_df)} games in date range")

        # Collect player logs for each game
        logs_collected = 0
        for _, game_row in tqdm(all_games_df.iterrows(), desc="Collecting player logs", total=len(all_games_df)):
            try:
                game_id = game_row.get('gameID', game_row.get('game_id'))
                if not game_id:
                    continue

                logs_df = self._make_request_with_retry(
                    self.api.get_player_game_logs, game_id
                )

                if logs_df is not None and not logs_df.empty:
                    # Add collection metadata
                    logs_df['collection_date'] = datetime.now().isoformat()

                    # Save to CSV
                    date_str = game_row.get('gameDate', game_row.get('game_date', ''))
                    if date_str:
                        log_file = Path(self.config.data_dir) / 'player_logs' / f'logs_{date_str}_{game_id}.csv'
                        logs_df.to_csv(log_file, index=False)
                        logs_collected += len(logs_df)

            except Exception as e:
                self.logger.error(f"Failed to collect logs for game {game_id}: {e}")

        # Create consolidated file
        self._consolidate_csv_files('player_logs', 'all_player_logs.csv')
        self.logger.info(f"Player log collection completed. Total logs: {logs_collected}")

    def collect_injury_data_for_date_range(self):
        """Collect injury data for the specified date range."""
        self.logger.info(f"Starting injury data collection from {self.config.start_date} to {self.config.end_date}")

        dates = list(self._date_range_generator(self.config.start_date, self.config.end_date))
        injuries_collected = 0

        for date_str in tqdm(dates, desc="Collecting injuries"):
            try:
                injuries_df = self._make_request_with_retry(
                    self.api.get_injuries_for_date, date_str
                )

                if injuries_df is not None and not injuries_df.empty:
                    # Save individual date file
                    date_file = Path(self.config.data_dir) / 'injuries' / f'injuries_{date_str}.csv'
                    injuries_df.to_csv(date_file, index=False)

                    injuries_collected += len(injuries_df)
                    self.logger.info(f"Saved {len(injuries_df)} injuries for {date_str}")

            except Exception as e:
                self.logger.error(f"Failed to collect injuries for {date_str}: {e}")

        # Create consolidated file
        self._consolidate_csv_files('injuries', 'all_injuries.csv')
        self.logger.info(f"Injury data collection completed. Total injuries: {injuries_collected}")

    def collect_dfs_salaries_for_date_range(self):
        """Collect DFS salary data for the specified date range."""
        self.logger.info(f"Starting DFS salary collection from {self.config.start_date} to {self.config.end_date}")

        dates = list(self._date_range_generator(self.config.start_date, self.config.end_date))
        salaries_collected = 0

        for date_str in tqdm(dates, desc="Collecting DFS salaries"):
            try:
                salaries_df = self._make_request_with_retry(
                    self.api.get_dfs_salaries_for_date, date_str
                )

                if salaries_df is not None and not salaries_df.empty:
                    # Save individual date file
                    date_file = Path(self.config.data_dir) / 'dfs_salaries' / f'salaries_{date_str}.csv'
                    salaries_df.to_csv(date_file, index=False)

                    salaries_collected += len(salaries_df)
                    self.logger.info(f"Saved {len(salaries_df)} salary entries for {date_str}")

            except Exception as e:
                self.logger.error(f"Failed to collect DFS salaries for {date_str}: {e}")

        # Create consolidated file
        self._consolidate_csv_files('dfs_salaries', 'all_dfs_salaries.csv')
        self.logger.info(f"DFS salary collection completed. Total entries: {salaries_collected}")

    def collect_betting_odds_for_date_range(self):
        """Collect betting odds for the specified date range."""
        self.logger.info(f"Starting betting odds collection from {self.config.start_date} to {self.config.end_date}")

        dates = list(self._date_range_generator(self.config.start_date, self.config.end_date))
        odds_collected = 0

        for date_str in tqdm(dates, desc="Collecting betting odds"):
            try:
                odds_df = self._make_request_with_retry(
                    self.api.get_betting_odds_for_date, date_str
                )

                if odds_df is not None and not odds_df.empty:
                    # Save individual date file
                    date_file = Path(self.config.data_dir) / 'betting_odds' / f'odds_{date_str}.csv'
                    odds_df.to_csv(date_file, index=False)

                    odds_collected += len(odds_df)
                    self.logger.info(f"Saved {len(odds_df)} odds entries for {date_str}")

            except Exception as e:
                self.logger.error(f"Failed to collect betting odds for {date_str}: {e}")

        # Create consolidated file
        self._consolidate_csv_files('betting_odds', 'all_betting_odds.csv')
        self.logger.info(f"Betting odds collection completed. Total odds records: {odds_collected}")

    def _consolidate_csv_files(self, subdirectory: str, output_filename: str):
        """Consolidate individual CSV files into a single file."""
        data_dir = Path(self.config.data_dir) / subdirectory
        csv_files = list(data_dir.glob('*.csv'))

        # Filter out the consolidated file itself
        csv_files = [f for f in csv_files if f.name != output_filename]

        if not csv_files:
            self.logger.warning(f"No CSV files found in {data_dir}")
            return

        # Read and concatenate all files
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
            except Exception as e:
                self.logger.error(f"Failed to read {csv_file}: {e}")

        if all_data:
            consolidated_df = pd.concat(all_data, ignore_index=True)
            output_path = data_dir / output_filename
            consolidated_df.to_csv(output_path, index=False)
            self.logger.info(f"Consolidated {len(csv_files)} files into {output_path}")

    def run_full_collection(self):
        """Run complete data collection for all data types."""
        self.logger.info("Starting full historical data collection")

        # Collect in order of importance for DFS modeling
        self.collect_games_for_date_range()
        self.collect_player_game_logs_for_date_range()
        self.collect_injury_data_for_date_range()
        self.collect_dfs_salaries_for_date_range()
        self.collect_betting_odds_for_date_range()

        self.logger.info("Full historical data collection completed successfully")

    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of collected data."""
        summary = {}

        data_types = ['games', 'player_logs', 'injuries', 'dfs_salaries', 'betting_odds']

        for data_type in data_types:
            data_dir = Path(self.config.data_dir) / data_type
            consolidated_file = data_dir / f'all_{data_type}.csv'

            if consolidated_file.exists():
                try:
                    df = pd.read_csv(consolidated_file)
                    file_size_mb = round(consolidated_file.stat().st_size / (1024 * 1024), 2)
                    summary[data_type] = {
                        'total_records': len(df),
                        'file_size_mb': file_size_mb
                    }
                except Exception as e:
                    summary[data_type] = {'status': 'error', 'message': str(e)}
            else:
                summary[data_type] = {'status': 'not_collected'}

        return summary