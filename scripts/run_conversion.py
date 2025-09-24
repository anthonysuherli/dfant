#!/usr/bin/env python3
"""
Runner script for database to parquet conversion with validation and error handling.

Usage:
    python run_conversion.py --output-dir data_parquet --validate
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional
import polars as pl
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from convert_db_to_parquet import DatabaseToParquetConverter


class ConversionValidator:
    """Validate converted parquet files against original database."""

    def __init__(self, parquet_dir: Path):
        self.parquet_dir = parquet_dir
        self.logger = logging.getLogger(__name__)

    def validate_player_game_logs(self) -> bool:
        """Validate player game logs conversion."""
        try:
            parquet_path = self.parquet_dir / "core" / "player_game_logs"
            if not parquet_path.exists():
                self.logger.error("Player game logs parquet not found")
                return False

            # Read parquet data
            df = pl.scan_parquet(str(parquet_path / "**" / "*.parquet")).collect()

            # Basic validation checks
            if df.is_empty():
                self.logger.error("Player game logs parquet is empty")
                return False

            # Check required columns
            required_cols = ['player_id', 'game_date', 'points', 'dfs_points_dk']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return False

            # Check data types
            if df['player_id'].dtype != pl.UInt16:
                self.logger.warning("player_id not optimized to UInt16")

            if df['points'].dtype != pl.UInt8:
                self.logger.warning("points not optimized to UInt8")

            # Check for null values in critical columns
            null_counts = df.select([
                pl.col('player_id').null_count(),
                pl.col('game_date').null_count()
            ]).row(0)

            if any(count > 0 for count in null_counts):
                self.logger.warning("Found null values in critical columns")

            self.logger.info(f"Player game logs validation passed: {len(df)} records")
            return True

        except Exception as e:
            self.logger.error(f"Player game logs validation failed: {e}")
            return False

    def validate_all(self) -> bool:
        """Run all validation checks."""
        validations = [
            self.validate_player_game_logs(),
        ]

        return all(validations)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration with Windows console compatibility."""
    # Set UTF-8 encoding for console output on Windows
    import os
    if os.name == 'nt':  # Windows
        import sys
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('conversion.log', encoding='utf-8')
        ]
    )


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Convert database tables to optimized parquet format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_parquet",
        help="Output directory for parquet files (default: data_parquet)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks after conversion"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--tables",
        type=str,
        nargs="+",
        choices=["player_logs", "team_logs", "betting", "injuries", "salaries", "features"],
        help="Specific tables to convert (default: all)"
    )
    parser.add_argument(
        "--db-location",
        type=str,
        choices=["database", "data_parquet"],
        default="database",
        help="Database file location: 'database' folder or 'data_parquet' folder (default: database)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Configure database location if specified
        if args.db_location == "data_parquet":
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent / "src"))
            from database.connection import init_database

            # Initialize database in data_parquet directory
            db_path = Path(args.output_dir) / "database" / "nba_dfs.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            init_database(f"sqlite:///{db_path}")
            logger.info(f"Using database location: {db_path}")

        # Initialize converter
        converter = DatabaseToParquetConverter(output_dir=args.output_dir)

        logger.info(f"Starting conversion to {args.output_dir}")
        logger.info(f"Converting tables: {args.tables if args.tables else 'all'}")

        # Run specific table conversions or all
        if args.tables:
            table_map = {
                "player_logs": converter.convert_player_game_logs,
                "team_logs": converter.convert_team_game_logs,
                "betting": converter.convert_betting_odds,
                "injuries": converter.convert_injury_reports,
                "salaries": converter.convert_dfs_salaries,
                "features": converter.convert_feature_store,
            }

            for table in args.tables:
                if table in table_map:
                    logger.info(f"Converting {table}...")
                    table_map[table]()
                else:
                    logger.warning(f"Unknown table: {table}")
        else:
            # Convert all tables
            converter.convert_all_tables()

        # Run validation if requested
        if args.validate:
            logger.info("Running validation checks...")
            validator = ConversionValidator(Path(args.output_dir))

            if validator.validate_all():
                logger.info("[SUCCESS] All validation checks passed")
            else:
                logger.error("[FAILED] Some validation checks failed")
                sys.exit(1)

        logger.info("[COMPLETED] Conversion completed successfully!")

    except Exception as e:
        logger.error(f"[ERROR] Conversion failed: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()