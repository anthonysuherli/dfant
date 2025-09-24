#!/usr/bin/env python3
"""
ETL Pipeline Script

This script runs the ETL pipeline to clean, validate, and process collected NBA data.

Usage:
    python run_etl_pipeline.py --start-date 20240101 --end-date 20240131
    python run_etl_pipeline.py --all  # Process all data
    python run_etl_pipeline.py --recent-days 30  # Process last 30 days
"""

import argparse
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from database.connection import init_database
from processing.etl_pipeline import ETLPipeline

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('etl.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Run ETL pipeline on NBA data")

    # Date range options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--start-date", help="Start date (YYYYMMDD)")
    date_group.add_argument("--all", action="store_true", help="Process all data")
    date_group.add_argument("--recent-days", type=int, help="Process last N days")

    parser.add_argument("--end-date", help="End date (YYYYMMDD), required with --start-date")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without doing it")

    # Processing options
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip data cleaning steps")
    parser.add_argument("--skip-advanced-stats", action="store_true", help="Skip advanced stats calculation")
    parser.add_argument("--skip-dfs-scoring", action="store_true", help="Skip DFS scoring calculation")
    parser.add_argument("--skip-rest-days", action="store_true", help="Skip rest days calculation")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Determine date range
    if args.start_date:
        if not args.end_date:
            parser.error("--end-date is required when using --start-date")
        start_date = args.start_date
        end_date = args.end_date
    elif args.all:
        # Process all data (use very early date)
        start_date = "20200101"
        end_date = datetime.now().strftime('%Y%m%d')
        logger.info(f"Processing all data: {start_date} to {end_date}")
    elif args.recent_days:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=args.recent_days)).strftime('%Y%m%d')
        logger.info(f"Processing recent {args.recent_days} days: {start_date} to {end_date}")

    # Validate dates
    try:
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        if start_dt > end_dt:
            raise ValueError("Start date must be before end date")

    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    logger.info(f"ETL processing period: {start_date} to {end_date}")

    if args.dry_run:
        logger.info("DRY RUN MODE - No data will be processed")
        days = (end_dt - start_dt).days + 1
        logger.info(f"Would process {days} days of data")
        return

    try:
        # Initialize database
        logger.info("Initializing database...")
        init_database()

        # Create ETL pipeline
        logger.info("Initializing ETL pipeline...")
        etl = ETLPipeline()

        logger.info("Starting ETL processing...")

        # Data cleaning (unless skipped)
        if not args.skip_cleaning:
            logger.info("Running data cleaning...")
            etl.clean_player_data()
            etl.normalize_team_data()
            etl.validate_game_logs()
        else:
            logger.info("Skipping data cleaning")

        # Advanced stats calculation (unless skipped)
        if not args.skip_advanced_stats:
            logger.info("Calculating advanced statistics...")
            etl.calculate_advanced_stats(start_date, end_date)
        else:
            logger.info("Skipping advanced stats calculation")

        # DFS scoring calculation (unless skipped)
        if not args.skip_dfs_scoring:
            logger.info("Calculating DFS scoring...")
            etl.calculate_dfs_scoring('DraftKings')
            etl.calculate_dfs_scoring('FanDuel')
        else:
            logger.info("Skipping DFS scoring calculation")

        # Rest days calculation (unless skipped)
        if not args.skip_rest_days:
            logger.info("Calculating rest days...")
            etl.calculate_rest_days()
        else:
            logger.info("Skipping rest days calculation")

        logger.info("ETL pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.info("ETL processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"ETL processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()