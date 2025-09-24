#!/usr/bin/env python3
"""
Convert existing database tables to optimized parquet format.

This script extracts data from your SQLite database and converts it to
partitioned parquet files with optimized data types for NBA DFS ML pipeline.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from sqlalchemy import text
from sqlalchemy.orm import Session

# Import your existing database components
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from database.connection import get_db_session
from database.models import (
    PlayerGameLog, Game, Player, Team, Injury,
    BettingOdds, DFSSalary, TeamGameLog, FeatureStore, Projection
)
from sqlalchemy.exc import OperationalError

class DatabaseToParquetConverter:
    """Convert database tables to optimized parquet format."""

    def __init__(self, output_dir: str = "data_parquet"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()

        # Create subdirectories for organized storage
        (self.output_dir / "core").mkdir(exist_ok=True)
        (self.output_dir / "external").mkdir(exist_ok=True)
        (self.output_dir / "features").mkdir(exist_ok=True)

        # Check database connectivity
        self._check_database_connection()

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

    def _check_database_connection(self) -> None:
        """Check database connectivity and table existence."""
        try:
            with get_db_session() as session:
                # Simple connectivity test
                session.execute(text("SELECT 1"))
                self.logger.info("Database connection successful")
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise

    def _table_exists(self, session: Session, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            session.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1"))
            return True
        except OperationalError:
            return False

    def _safe_query_count(self, session: Session, model_class) -> int:
        """Safely get row count for a table."""
        try:
            return session.query(model_class).count()
        except OperationalError:
            return 0

    def convert_player_game_logs(self) -> None:
        """Convert PlayerGameLog table to optimized parquet with partitioning."""
        self.logger.info("Converting PlayerGameLog table...")

        with get_db_session() as session:
            # Check if table exists
            if not self._table_exists(session, 'player_game_logs'):
                self.logger.warning("player_game_logs table does not exist. Skipping conversion.")
                return

            # Check row count
            row_count = self._safe_query_count(session, PlayerGameLog)
            if row_count == 0:
                self.logger.warning("player_game_logs table is empty. Creating empty parquet structure.")
                self._create_empty_player_logs_parquet()
                return

            self.logger.info(f"Found {row_count} records in player_game_logs table")

            # Query all player game logs with related data
            try:
                query = session.query(PlayerGameLog).join(Game, isouter=True).join(Player, isouter=True)
            except Exception as e:
                self.logger.warning(f"Join query failed, using simple query: {e}")
                query = session.query(PlayerGameLog)

            # Convert to pandas first (SQLAlchemy compatibility)
            data = []
            for log in query:
                record = {
                    # Identifiers
                    'player_id': log.player_id,
                    'game_id': log.game_id,
                    'game_date': log.game_date,

                    # Team context - handle missing relationships
                    'team_id': getattr(log.player, 'team_id', None) if hasattr(log, 'player') and log.player else None,
                    'opponent_team_id': log.opponent_team_id,
                    'is_home': log.is_home,

                    # Basic NBA statistics
                    'minutes': log.minutes or 0.0,
                    'points': log.points or 0,
                    'field_goals_made': log.field_goals_made or 0,
                    'field_goals_attempted': log.field_goals_attempted or 0,
                    'three_pointers_made': log.three_pointers_made or 0,
                    'three_pointers_attempted': log.three_pointers_attempted or 0,
                    'free_throws_made': log.free_throws_made or 0,
                    'free_throws_attempted': log.free_throws_attempted or 0,
                    'rebounds_offensive': log.rebounds_offensive or 0,
                    'rebounds_defensive': log.rebounds_defensive or 0,
                    'rebounds_total': log.rebounds_total or 0,
                    'assists': log.assists or 0,
                    'steals': log.steals or 0,
                    'blocks': log.blocks or 0,
                    'turnovers': log.turnovers or 0,
                    'personal_fouls': log.personal_fouls or 0,
                    'plus_minus': log.plus_minus or 0,

                    # Fantasy scoring
                    'dfs_points_dk': log.dfs_points_dk or 0.0,
                    'dfs_points_fd': log.dfs_points_fd or 0.0,
                    'dfs_salary_dk': log.dfs_salary_dk or 0,
                    'dfs_salary_fd': log.dfs_salary_fd or 0,

                    # Advanced metrics
                    'usage_rate': log.usage_rate or 0.0,
                    'true_shooting_percentage': log.true_shooting_percentage or 0.0,
                    'player_efficiency_rating': log.player_efficiency_rating or 0.0,
                    'effective_field_goal_percentage': log.effective_field_goal_percentage or 0.0,

                    # Game context
                    'rest_days': log.rest_days or 0,
                    'is_starter': log.is_starter or False,
                }
                data.append(record)

        if not data:
            self.logger.warning("No player game logs found in database")
            return

        # Convert to Polars DataFrame with optimized data types
        df = pl.DataFrame(data)
        df_optimized = self._optimize_player_game_logs_schema(df)

        # Write partitioned parquet
        output_path = self.output_dir / "core" / "player_game_logs"
        self._write_partitioned_parquet(df_optimized, output_path, ['season', 'month'])

        self.logger.info(f"Converted {len(data)} player game logs to parquet")

    def _create_empty_player_logs_parquet(self) -> None:
        """Create empty parquet structure for player game logs."""
        # Create empty DataFrame with correct schema
        empty_data = {
            'player_id': [], 'game_id': [], 'game_date': [], 'team_id': [],
            'opponent_team_id': [], 'is_home': [], 'minutes': [], 'points': [],
            'field_goals_made': [], 'field_goals_attempted': [], 'three_pointers_made': [],
            'three_pointers_attempted': [], 'free_throws_made': [], 'free_throws_attempted': [],
            'rebounds_offensive': [], 'rebounds_defensive': [], 'rebounds_total': [],
            'assists': [], 'steals': [], 'blocks': [], 'turnovers': [], 'personal_fouls': [],
            'plus_minus': [], 'dfs_points_dk': [], 'dfs_points_fd': [], 'dfs_salary_dk': [],
            'dfs_salary_fd': [], 'usage_rate': [], 'true_shooting_percentage': [],
            'player_efficiency_rating': [], 'effective_field_goal_percentage': [],
            'rest_days': [], 'is_starter': []
        }

        df = pl.DataFrame(empty_data)
        df_optimized = self._optimize_player_game_logs_schema(df)

        output_path = self.output_dir / "core" / "player_game_logs"
        output_path.mkdir(parents=True, exist_ok=True)

        # Create a single empty parquet file
        (output_path / "empty.parquet").touch()
        self.logger.info("Created empty player game logs parquet structure")

    def _optimize_player_game_logs_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply optimized data types and add partitioning columns."""
        return df.with_columns([
            # Optimize data types for NBA statistics
            pl.col('player_id').cast(pl.UInt16),
            pl.col('team_id').cast(pl.UInt8),
            pl.col('opponent_team_id').cast(pl.UInt8),
            pl.col('points').cast(pl.UInt8),
            pl.col('field_goals_made').cast(pl.UInt8),
            pl.col('field_goals_attempted').cast(pl.UInt8),
            pl.col('three_pointers_made').cast(pl.UInt8),
            pl.col('three_pointers_attempted').cast(pl.UInt8),
            pl.col('free_throws_made').cast(pl.UInt8),
            pl.col('free_throws_attempted').cast(pl.UInt8),
            pl.col('rebounds_offensive').cast(pl.UInt8),
            pl.col('rebounds_defensive').cast(pl.UInt8),
            pl.col('rebounds_total').cast(pl.UInt8),
            pl.col('assists').cast(pl.UInt8),
            pl.col('steals').cast(pl.UInt8),
            pl.col('blocks').cast(pl.UInt8),
            pl.col('turnovers').cast(pl.UInt8),
            pl.col('personal_fouls').cast(pl.UInt8),
            pl.col('plus_minus').cast(pl.Int8),
            pl.col('rest_days').cast(pl.UInt8),

            # Fantasy and salary data
            pl.col('dfs_points_dk').cast(pl.Float32),
            pl.col('dfs_points_fd').cast(pl.Float32),
            pl.col('dfs_salary_dk').cast(pl.UInt16),
            pl.col('dfs_salary_fd').cast(pl.UInt16),

            # Advanced metrics
            pl.col('minutes').cast(pl.Float32),
            pl.col('usage_rate').cast(pl.Float32),
            pl.col('true_shooting_percentage').cast(pl.Float32),
            pl.col('player_efficiency_rating').cast(pl.Float32),
            pl.col('effective_field_goal_percentage').cast(pl.Float32),

            # Add partitioning columns for efficient querying
            pl.col('game_date').dt.month().alias('month'),
            pl.when(pl.col('game_date').dt.month() >= 10)
              .then(
                  pl.col('game_date').dt.year().cast(pl.Utf8) + '-' +
                  (pl.col('game_date').dt.year() + 1).cast(pl.Utf8).str.slice(2, 2)
              )
              .otherwise(
                  (pl.col('game_date').dt.year() - 1).cast(pl.Utf8).str.slice(2, 2) + '-' +
                  pl.col('game_date').dt.year().cast(pl.Utf8).str.slice(2, 2)
              )
              .alias('season'),
        ])

    def convert_betting_odds(self) -> None:
        """Convert BettingOdds table to daily partitioned parquet."""
        self.logger.info("Converting BettingOdds table...")

        with get_db_session() as session:
            # Check if table exists
            if not self._table_exists(session, 'betting_odds'):
                self.logger.warning("betting_odds table does not exist. Skipping conversion.")
                return

            row_count = self._safe_query_count(session, BettingOdds)
            if row_count == 0:
                self.logger.warning("betting_odds table is empty. Skipping conversion.")
                return

            self.logger.info(f"Found {row_count} records in betting_odds table")

            try:
                query = session.query(BettingOdds).join(Game, isouter=True)
            except Exception as e:
                self.logger.warning(f"Join query failed, using simple query: {e}")
                query = session.query(BettingOdds)

            data = []
            for odds in query:
                record = {
                    'game_id': odds.game_id,
                    'game_date': odds.game.game_date if odds.game else None,
                    'sportsbook': odds.sportsbook,
                    'timestamp': odds.timestamp,

                    # Betting lines
                    'home_spread': odds.home_spread or 0.0,
                    'away_spread': odds.away_spread or 0.0,
                    'home_spread_odds': odds.home_spread_odds or 0,
                    'away_spread_odds': odds.away_spread_odds or 0,
                    'home_moneyline': odds.home_moneyline or 0,
                    'away_moneyline': odds.away_moneyline or 0,
                    'total_points': odds.total_points or 0.0,
                    'over_odds': odds.over_odds or 0,
                    'under_odds': odds.under_odds or 0,
                }
                data.append(record)

        if not data:
            self.logger.warning("No betting odds found in database")
            return

        df = pl.DataFrame(data)
        df_optimized = df.with_columns([
            pl.col('home_spread').cast(pl.Float32),
            pl.col('away_spread').cast(pl.Float32),
            pl.col('home_spread_odds').cast(pl.Int16),
            pl.col('away_spread_odds').cast(pl.Int16),
            pl.col('home_moneyline').cast(pl.Int16),
            pl.col('away_moneyline').cast(pl.Int16),
            pl.col('total_points').cast(pl.Float32),
            pl.col('over_odds').cast(pl.Int16),
            pl.col('under_odds').cast(pl.Int16),
        ])

        output_path = self.output_dir / "external" / "betting_odds"
        self._write_partitioned_parquet(df_optimized, output_path, ['game_date'])

        self.logger.info(f"Converted {len(data)} betting odds records to parquet")

    def convert_injury_reports(self) -> None:
        """Convert Injury table to daily partitioned parquet."""
        self.logger.info("Converting Injury table...")

        with get_db_session() as session:
            query = session.query(Injury)

            data = []
            for injury in query:
                record = {
                    'player_id': injury.player_id,
                    'injury_date': injury.injury_date,
                    'status': injury.status or 'Unknown',
                    'description': injury.description or '',
                    'body_part': injury.body_part or '',
                    'return_date': injury.return_date,
                    'games_missed': injury.games_missed or 0,
                }
                data.append(record)

        if not data:
            self.logger.warning("No injury reports found in database")
            return

        df = pl.DataFrame(data)
        df_optimized = df.with_columns([
            pl.col('player_id').cast(pl.UInt16),
            pl.col('games_missed').cast(pl.UInt8),
        ])

        output_path = self.output_dir / "external" / "injury_reports"
        self._write_partitioned_parquet(df_optimized, output_path, ['injury_date'])

        self.logger.info(f"Converted {len(data)} injury records to parquet")

    def convert_dfs_salaries(self) -> None:
        """Convert DFSSalary table to daily partitioned parquet."""
        self.logger.info("Converting DFSSalary table...")

        with get_db_session() as session:
            query = session.query(DFSSalary)

            data = []
            for salary in query:
                record = {
                    'player_id': salary.player_id,
                    'game_date': salary.game_date,
                    'platform': salary.platform,
                    'salary': salary.salary,
                    'position': salary.position or '',
                    'is_available': salary.is_available or True,
                }
                data.append(record)

        if not data:
            self.logger.warning("No DFS salaries found in database")
            return

        df = pl.DataFrame(data)
        df_optimized = df.with_columns([
            pl.col('player_id').cast(pl.UInt16),
            pl.col('salary').cast(pl.UInt16),
        ])

        output_path = self.output_dir / "external" / "dfs_salaries"
        self._write_partitioned_parquet(df_optimized, output_path, ['game_date', 'platform'])

        self.logger.info(f"Converted {len(data)} DFS salary records to parquet")

    def convert_feature_store(self) -> None:
        """Convert FeatureStore table to versioned parquet."""
        self.logger.info("Converting FeatureStore table...")

        with get_db_session() as session:
            query = session.query(FeatureStore)

            data = []
            for feature in query:
                record = {
                    'player_id': feature.player_id,
                    'game_date': feature.game_date,
                    'feature_set_version': feature.feature_set_version,
                    'features': feature.features,  # JSON data
                    'created_at': feature.created_at,
                }
                data.append(record)

        if not data:
            self.logger.warning("No feature store records found in database")
            return

        df = pl.DataFrame(data)
        df_optimized = df.with_columns([
            pl.col('player_id').cast(pl.UInt16),
        ])

        output_path = self.output_dir / "features" / "engineered_features"
        self._write_partitioned_parquet(df_optimized, output_path, ['feature_set_version', 'game_date'])

        self.logger.info(f"Converted {len(data)} feature store records to parquet")

    def convert_team_game_logs(self) -> None:
        """Convert TeamGameLog table to parquet."""
        self.logger.info("Converting TeamGameLog table...")

        with get_db_session() as session:
            query = session.query(TeamGameLog)

            data = []
            for log in query:
                record = {
                    'team_id': log.team_id,
                    'game_id': log.game_id,
                    'game_date': log.game_date,
                    'opponent_team_id': log.opponent_team_id,
                    'is_home': log.is_home,
                    'rest_days': log.rest_days or 0,

                    # Team statistics
                    'points': log.points or 0,
                    'field_goals_made': log.field_goals_made or 0,
                    'field_goals_attempted': log.field_goals_attempted or 0,
                    'field_goal_percentage': log.field_goal_percentage or 0.0,
                    'three_pointers_made': log.three_pointers_made or 0,
                    'three_pointers_attempted': log.three_pointers_attempted or 0,
                    'three_point_percentage': log.three_point_percentage or 0.0,
                    'free_throws_made': log.free_throws_made or 0,
                    'free_throws_attempted': log.free_throws_attempted or 0,
                    'free_throw_percentage': log.free_throw_percentage or 0.0,
                    'rebounds_total': log.rebounds_total or 0,
                    'assists': log.assists or 0,
                    'steals': log.steals or 0,
                    'blocks': log.blocks or 0,
                    'turnovers': log.turnovers or 0,
                    'personal_fouls': log.personal_fouls or 0,

                    # Advanced team metrics
                    'pace': log.pace or 0.0,
                    'offensive_rating': log.offensive_rating or 0.0,
                    'defensive_rating': log.defensive_rating or 0.0,
                    'net_rating': log.net_rating or 0.0,
                    'true_shooting_percentage': log.true_shooting_percentage or 0.0,
                    'effective_field_goal_percentage': log.effective_field_goal_percentage or 0.0,
                    'turnover_rate': log.turnover_rate or 0.0,
                    'offensive_rebound_rate': log.offensive_rebound_rate or 0.0,
                    'defensive_rebound_rate': log.defensive_rebound_rate or 0.0,
                }
                data.append(record)

        if not data:
            self.logger.warning("No team game logs found in database")
            return

        df = pl.DataFrame(data)
        df_optimized = df.with_columns([
            # Team IDs
            pl.col('team_id').cast(pl.UInt8),
            pl.col('opponent_team_id').cast(pl.UInt8),
            pl.col('rest_days').cast(pl.UInt8),

            # Basic stats
            pl.col('points').cast(pl.UInt8),
            pl.col('field_goals_made').cast(pl.UInt8),
            pl.col('field_goals_attempted').cast(pl.UInt8),
            pl.col('three_pointers_made').cast(pl.UInt8),
            pl.col('three_pointers_attempted').cast(pl.UInt8),
            pl.col('free_throws_made').cast(pl.UInt8),
            pl.col('free_throws_attempted').cast(pl.UInt8),
            pl.col('rebounds_total').cast(pl.UInt8),
            pl.col('assists').cast(pl.UInt8),
            pl.col('steals').cast(pl.UInt8),
            pl.col('blocks').cast(pl.UInt8),
            pl.col('turnovers').cast(pl.UInt8),
            pl.col('personal_fouls').cast(pl.UInt8),

            # Percentages and rates
            pl.col('field_goal_percentage').cast(pl.Float32),
            pl.col('three_point_percentage').cast(pl.Float32),
            pl.col('free_throw_percentage').cast(pl.Float32),
            pl.col('pace').cast(pl.Float32),
            pl.col('offensive_rating').cast(pl.Float32),
            pl.col('defensive_rating').cast(pl.Float32),
            pl.col('net_rating').cast(pl.Float32),
            pl.col('true_shooting_percentage').cast(pl.Float32),
            pl.col('effective_field_goal_percentage').cast(pl.Float32),
            pl.col('turnover_rate').cast(pl.Float32),
            pl.col('offensive_rebound_rate').cast(pl.Float32),
            pl.col('defensive_rebound_rate').cast(pl.Float32),

            # Add season partitioning
            pl.col('game_date').dt.month().alias('month'),
            pl.when(pl.col('game_date').dt.month() >= 10)
              .then(
                  pl.col('game_date').dt.year().cast(pl.Utf8) + '-' +
                  (pl.col('game_date').dt.year() + 1).cast(pl.Utf8).str.slice(2, 2)
              )
              .otherwise(
                  (pl.col('game_date').dt.year() - 1).cast(pl.Utf8).str.slice(2, 2) + '-' +
                  pl.col('game_date').dt.year().cast(pl.Utf8).str.slice(2, 2)
              )
              .alias('season'),
        ])

        output_path = self.output_dir / "core" / "team_game_logs"
        self._write_partitioned_parquet(df_optimized, output_path, ['season', 'month'])

        self.logger.info(f"Converted {len(data)} team game logs to parquet")

    def _write_partitioned_parquet(self, df: pl.DataFrame, output_path: Path,
                                 partition_cols: List[str]) -> None:
        """Write DataFrame to partitioned parquet with optimal settings."""
        try:
            df.write_parquet(
                str(output_path),
                partition_by=partition_cols,
                compression='snappy',          # Fast compression/decompression
                statistics=True,               # Enable predicate pushdown
                row_group_size=5000,          # ~1 day of NBA games
                use_pyarrow=True,             # Better optimization
            )
            self.logger.info(f"Written partitioned parquet to {output_path}")
        except Exception as e:
            self.logger.error(f"Error writing parquet to {output_path}: {e}")
            raise

    def convert_all_tables(self) -> None:
        """Convert all database tables to parquet format."""
        self.logger.info("Starting conversion of all database tables to parquet...")

        try:
            # Core NBA data tables
            self.convert_player_game_logs()
            self.convert_team_game_logs()

            # External data sources
            self.convert_betting_odds()
            self.convert_injury_reports()
            self.convert_dfs_salaries()

            # ML features
            self.convert_feature_store()

            self.logger.info("Successfully converted all database tables to parquet format")
            self._print_conversion_summary()

        except Exception as e:
            self.logger.error(f"Error during conversion: {e}")
            raise

    def _print_conversion_summary(self) -> None:
        """Print summary of converted files and their sizes."""
        self.logger.info("\n" + "="*50)
        self.logger.info("CONVERSION SUMMARY")
        self.logger.info("="*50)

        total_size = 0
        for parquet_file in self.output_dir.rglob("*.parquet"):
            size_mb = parquet_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            relative_path = parquet_file.relative_to(self.output_dir)
            self.logger.info(f"{relative_path}: {size_mb:.2f} MB")

        self.logger.info(f"\nTotal parquet size: {total_size:.2f} MB")
        self.logger.info(f"Files saved to: {self.output_dir.absolute()}")
        self.logger.info("="*50)


def main():
    """Main function to run the conversion."""
    converter = DatabaseToParquetConverter()
    converter.convert_all_tables()


if __name__ == "__main__":
    main()