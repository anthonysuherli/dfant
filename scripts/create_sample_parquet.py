#!/usr/bin/env python3
"""
Create sample parquet files with NBA DFS data structure for testing and development.

This script creates realistic sample data when the database is empty,
allowing you to test the parquet schema and query patterns.
"""

import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import logging


class SampleParquetCreator:
    """Create sample NBA DFS parquet data for testing."""

    def __init__(self, output_dir: str = "data_parquet"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()

        # Create subdirectories
        (self.output_dir / "core").mkdir(exist_ok=True)
        (self.output_dir / "external").mkdir(exist_ok=True)
        (self.output_dir / "features").mkdir(exist_ok=True)

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

    def create_sample_player_game_logs(self, num_players: int = 50, num_games: int = 10) -> None:
        """Create sample player game log data."""
        self.logger.info(f"Creating sample player game logs: {num_players} players, {num_games} games each")

        # Generate sample data
        data = []
        base_date = datetime(2024, 1, 1)

        for player_id in range(1, num_players + 1):
            for game_num in range(num_games):
                game_date = base_date + timedelta(days=game_num * 2)

                # Realistic NBA statistics with some randomness
                points = max(0, int(np.random.normal(15, 8)))
                minutes = max(0, min(48, np.random.normal(28, 10)))

                record = {
                    # Identifiers
                    'player_id': player_id,
                    'game_id': f"2024010{game_num:02d}",
                    'game_date': game_date.date(),

                    # Team context
                    'team_id': (player_id % 30) + 1,  # 30 NBA teams
                    'opponent_team_id': ((player_id + game_num) % 30) + 1,
                    'is_home': bool(game_num % 2),

                    # Basic NBA statistics
                    'minutes': round(minutes, 1),
                    'points': points,
                    'field_goals_made': max(0, int(points * 0.4)),
                    'field_goals_attempted': max(1, int(points * 0.8)),
                    'three_pointers_made': max(0, int(np.random.poisson(1.5))),
                    'three_pointers_attempted': max(0, int(np.random.poisson(3.0))),
                    'free_throws_made': max(0, int(np.random.poisson(2.0))),
                    'free_throws_attempted': max(0, int(np.random.poisson(2.5))),
                    'rebounds_offensive': max(0, int(np.random.poisson(1.5))),
                    'rebounds_defensive': max(0, int(np.random.poisson(4.0))),
                    'rebounds_total': max(0, int(np.random.poisson(5.5))),
                    'assists': max(0, int(np.random.poisson(3.0))),
                    'steals': max(0, int(np.random.poisson(1.0))),
                    'blocks': max(0, int(np.random.poisson(0.7))),
                    'turnovers': max(0, int(np.random.poisson(2.5))),
                    'personal_fouls': max(0, int(np.random.poisson(2.0))),
                    'plus_minus': int(np.random.normal(0, 12)),

                    # Fantasy scoring (DraftKings formula)
                    'dfs_points_dk': round(
                        points * 1.0 +
                        max(0, int(np.random.poisson(5.5))) * 1.25 +  # rebounds
                        max(0, int(np.random.poisson(3.0))) * 1.5 +   # assists
                        max(0, int(np.random.poisson(1.0))) * 2.0 +   # steals
                        max(0, int(np.random.poisson(0.7))) * 2.0 +   # blocks
                        max(0, int(np.random.poisson(2.5))) * (-0.5), # turnovers
                        1
                    ),
                    'dfs_points_fd': round(
                        points * 1.0 +
                        max(0, int(np.random.poisson(5.5))) * 1.2 +   # rebounds
                        max(0, int(np.random.poisson(3.0))) * 1.5 +   # assists
                        max(0, int(np.random.poisson(1.0))) * 3.0 +   # steals
                        max(0, int(np.random.poisson(0.7))) * 3.0 +   # blocks
                        max(0, int(np.random.poisson(2.5))) * (-1.0), # turnovers
                        1
                    ),
                    'dfs_salary_dk': int(np.random.uniform(3000, 11500)),
                    'dfs_salary_fd': int(np.random.uniform(3500, 12000)),

                    # Advanced metrics
                    'usage_rate': round(np.random.uniform(15.0, 35.0), 2),
                    'true_shooting_percentage': round(np.random.uniform(0.45, 0.65), 3),
                    'player_efficiency_rating': round(np.random.uniform(8.0, 30.0), 2),
                    'effective_field_goal_percentage': round(np.random.uniform(0.40, 0.60), 3),

                    # Game context
                    'rest_days': int(np.random.choice([0, 1, 2, 3], p=[0.1, 0.6, 0.25, 0.05])),
                    'is_starter': bool(np.random.choice([True, False], p=[0.6, 0.4])),
                }
                data.append(record)

        # Convert to Polars DataFrame with optimized types
        df = pl.DataFrame(data)
        df_optimized = self._optimize_player_logs_schema(df)

        # Write partitioned parquet
        output_path = self.output_dir / "core" / "player_game_logs"
        df_optimized.write_parquet(
            str(output_path),
            partition_by=['season', 'month'],
            compression='snappy',
            statistics=True,
            use_pyarrow=True,
        )

        self.logger.info(f"Created sample player game logs: {len(data)} records")

    def _optimize_player_logs_schema(self, df: pl.DataFrame) -> pl.DataFrame:
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

            # Add partitioning columns
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

    def create_sample_betting_odds(self, num_games: int = 20) -> None:
        """Create sample betting odds data."""
        self.logger.info(f"Creating sample betting odds for {num_games} games")

        data = []
        base_date = datetime(2024, 1, 1)

        for game_num in range(num_games):
            game_date = base_date + timedelta(days=game_num)

            record = {
                'game_id': f"2024010{game_num:02d}",
                'game_date': game_date.date(),
                'sportsbook': np.random.choice(['DraftKings', 'FanDuel', 'BetMGM']),
                'timestamp': game_date,

                # Betting lines
                'home_spread': round(np.random.uniform(-12.0, 12.0), 1),
                'away_spread': round(np.random.uniform(-12.0, 12.0), 1),
                'home_spread_odds': int(np.random.uniform(-120, -100)),
                'away_spread_odds': int(np.random.uniform(-120, -100)),
                'home_moneyline': int(np.random.uniform(-300, 300)),
                'away_moneyline': int(np.random.uniform(-300, 300)),
                'total_points': round(np.random.uniform(200.0, 250.0), 1),
                'over_odds': int(np.random.uniform(-120, -100)),
                'under_odds': int(np.random.uniform(-120, -100)),
            }
            data.append(record)

        df = pl.DataFrame(data)
        df_optimized = df.with_columns([
            pl.col('home_spread').cast(pl.Float32),
            pl.col('away_spread').cast(pl.Float32),
            pl.col('total_points').cast(pl.Float32),
            pl.col('home_spread_odds').cast(pl.Int16),
            pl.col('away_spread_odds').cast(pl.Int16),
            pl.col('home_moneyline').cast(pl.Int16),
            pl.col('away_moneyline').cast(pl.Int16),
            pl.col('over_odds').cast(pl.Int16),
            pl.col('under_odds').cast(pl.Int16),
        ])

        output_path = self.output_dir / "external" / "betting_odds"
        df_optimized.write_parquet(
            str(output_path),
            partition_by=['game_date'],
            compression='snappy',
            statistics=True,
            use_pyarrow=True,
        )

        self.logger.info(f"Created sample betting odds: {len(data)} records")

    def create_all_sample_data(self) -> None:
        """Create all sample parquet files."""
        self.logger.info("Creating comprehensive sample NBA DFS parquet data...")

        try:
            self.create_sample_player_game_logs(num_players=100, num_games=20)
            self.create_sample_betting_odds(num_games=50)

            self._print_sample_summary()
            self.logger.info("[COMPLETED] Sample parquet data creation finished successfully!")

        except Exception as e:
            self.logger.error(f"[ERROR] Sample data creation failed: {e}")
            raise

    def _print_sample_summary(self) -> None:
        """Print summary of created sample files."""
        self.logger.info("\n" + "="*50)
        self.logger.info("SAMPLE DATA SUMMARY")
        self.logger.info("="*50)

        total_size = 0
        for parquet_file in self.output_dir.rglob("*.parquet"):
            size_mb = parquet_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            relative_path = parquet_file.relative_to(self.output_dir)
            self.logger.info(f"{relative_path}: {size_mb:.2f} MB")

        self.logger.info(f"\nTotal sample data size: {total_size:.2f} MB")
        self.logger.info(f"Files saved to: {self.output_dir.absolute()}")
        self.logger.info("="*50)


def main():
    """Create sample parquet data for testing."""
    creator = SampleParquetCreator()
    creator.create_all_sample_data()


if __name__ == "__main__":
    main()