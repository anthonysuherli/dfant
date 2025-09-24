#!/usr/bin/env python3
"""
Historical Data Collection Script

This script collects historical NBA data using the Tank01 API and stores it in the database.
Run with different date ranges to collect data for specific periods.

Usage:
    python collect_historical_data.py --start-date 20240101 --end-date 20240131
    python collect_historical_data.py --season 2024  # Collect full season
    python collect_historical_data.py --recent-days 30  # Last 30 days
"""

import argparse
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tank01_api import NBAFantasyAPI
from database.connection import init_database
from collectors.historical_collector import HistoricalDataCollector, CollectionConfig

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('collection.log')
        ]
    )

def get_season_date_range(season_year: int) -> tuple:
    """Get start and end dates for NBA season."""
    # NBA season typically starts in October and ends in April
    start_date = f"{season_year - 1}1001"  # October 1st
    end_date = f"{season_year}0430"        # April 30th
    return start_date, end_date

def main():
    parser = argparse.ArgumentParser(description="Collect historical NBA data")

    # Date range options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--start-date", help="Start date (YYYYMMDD)")
    date_group.add_argument("--season", type=int, help="Season year (e.g., 2024 for 2023-24 season)")
    date_group.add_argument("--recent-days", type=int, help="Collect last N days")

    parser.add_argument("--end-date", help="End date (YYYYMMDD), required with --start-date")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="Requests per second (default: 1.0)")
    parser.add_argument("--chunk-size", type=int, default=30, help="Days per chunk (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be collected without doing it")

    # Collection options
    parser.add_argument("--games-only", action="store_true", help="Only collect game data")
    parser.add_argument("--logs-only", action="store_true", help="Only collect player game logs")
    parser.add_argument("--injuries-only", action="store_true", help="Only collect injury data")
    parser.add_argument("--salaries-only", action="store_true", help="Only collect DFS salary data")

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
    elif args.season:
        start_date, end_date = get_season_date_range(args.season)
        logger.info(f"Season {args.season}: {start_date} to {end_date}")
    elif args.recent_days:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=args.recent_days)).strftime('%Y%m%d')
        logger.info(f"Recent {args.recent_days} days: {start_date} to {end_date}")

    # Validate dates
    try:
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        if start_dt > end_dt:
            raise ValueError("Start date must be before end date")

        if end_dt > datetime.now():
            logger.warning("End date is in the future, adjusting to today")
            end_date = datetime.now().strftime('%Y%m%d')

    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    logger.info(f"Collection period: {start_date} to {end_date}")

    if args.dry_run:
        logger.info("DRY RUN MODE - No data will be collected")
        days = (end_dt - start_dt).days + 1
        logger.info(f"Would collect {days} days of data")
        return

    try:
        # Initialize database
        logger.info("Initializing database...")
        init_database()

        # Initialize API
        logger.info("Initializing NBA API...")
        api = NBAFantasyAPI()

        # Create collection configuration
        config = CollectionConfig(
            start_date=start_date,
            end_date=end_date,
            rate_limit=args.rate_limit,
            chunk_size=args.chunk_size
        )

        # Create collector
        collector = HistoricalDataCollector(api, config)

        logger.info("Starting data collection...")

        # Run specific collections based on flags
        if args.games_only:
            logger.info("Collecting games only...")
            collector.collect_games_for_date_range()
        elif args.logs_only:
            logger.info("Collecting player game logs only...")
            collector.collect_player_game_logs_for_date_range()
        elif args.injuries_only:
            logger.info("Collecting injury data only...")
            collector.collect_injury_data_for_date_range()
        elif args.salaries_only:
            logger.info("Collecting DFS salary data only...")
            collector.collect_dfs_salaries_for_date_range()
        else:
            # Run full collection
            logger.info("Running full collection...")
            collector.run_full_collection()

        logger.info("Data collection completed successfully!")

    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()