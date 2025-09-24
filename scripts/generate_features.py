#!/usr/bin/env python3
"""
Feature Generation Script

This script generates machine learning features for NBA DFS optimization.

Usage:
    python generate_features.py --date 20250301  # Single date
    python generate_features.py --start-date 20240101 --end-date 20240131  # Date range
    python generate_features.py --recent-days 30  # Last 30 days
"""

import argparse
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from database.connection import init_database
from processing.feature_engineering import FeatureEngineer, FeatureConfig

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('feature_generation.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Generate ML features for NBA DFS")

    # Date options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--date", help="Single date (YYYYMMDD)")
    date_group.add_argument("--start-date", help="Start date (YYYYMMDD)")
    date_group.add_argument("--recent-days", type=int, help="Generate features for last N days")

    parser.add_argument("--end-date", help="End date (YYYYMMDD), required with --start-date")
    parser.add_argument("--feature-version", default="v1", help="Feature set version (default: v1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--output", help="Output CSV file path (optional)")

    # Feature configuration
    parser.add_argument("--lookback-windows", nargs="+", type=int, default=[5, 10, 15, 30],
                       help="Lookback windows for rolling features (default: 5 10 15 30)")
    parser.add_argument("--min-games", type=int, default=5,
                       help="Minimum games threshold for features (default: 5)")
    parser.add_argument("--no-opponent", action="store_true", help="Exclude opponent features")
    parser.add_argument("--no-injury", action="store_true", help="Exclude injury features")
    parser.add_argument("--no-rest", action="store_true", help="Exclude rest features")
    parser.add_argument("--no-home-away", action="store_true", help="Exclude home/away features")
    parser.add_argument("--no-salary", action="store_true", help="Exclude salary features")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Determine date range
    if args.date:
        start_date = end_date = args.date
        logger.info(f"Generating features for single date: {args.date}")
    elif args.start_date:
        if not args.end_date:
            parser.error("--end-date is required when using --start-date")
        start_date = args.start_date
        end_date = args.end_date
        logger.info(f"Generating features for date range: {start_date} to {end_date}")
    elif args.recent_days:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=args.recent_days)).strftime('%Y%m%d')
        logger.info(f"Generating features for recent {args.recent_days} days: {start_date} to {end_date}")

    # Validate dates
    try:
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        if start_dt > end_dt:
            raise ValueError("Start date must be before end date")

    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    try:
        # Initialize database
        logger.info("Initializing database...")
        init_database()

        # Create feature configuration
        config = FeatureConfig(
            lookback_windows=args.lookback_windows,
            include_opponent_features=not args.no_opponent,
            include_injury_features=not args.no_injury,
            include_rest_features=not args.no_rest,
            include_home_away_features=not args.no_home_away,
            include_salary_features=not args.no_salary,
            min_games_threshold=args.min_games
        )

        logger.info(f"Feature configuration: {config}")

        # Create feature engineer
        logger.info("Initializing feature engineer...")
        engineer = FeatureEngineer(config)

        logger.info("Starting feature generation...")

        if start_date == end_date:
            # Single date
            features_df = engineer.generate_features_for_date(start_date, args.feature_version)
            logger.info(f"Generated features for {len(features_df)} players")
        else:
            # Date range
            features_df = engineer.generate_features_batch(start_date, end_date, args.feature_version)
            logger.info(f"Generated features for {len(features_df)} player-date combinations")

        # Save to CSV if requested
        if args.output and not features_df.empty:
            features_df.to_csv(args.output, index=False)
            logger.info(f"Features saved to {args.output}")

        # Display summary
        if not features_df.empty:
            logger.info(f"Feature summary:")
            logger.info(f"  Total records: {len(features_df)}")
            logger.info(f"  Feature columns: {len(features_df.columns)}")
            logger.info(f"  Date range: {features_df['target_date'].min()} to {features_df['target_date'].max()}")

            # Show sample of feature names
            feature_cols = [col for col in features_df.columns if col not in ['player_id', 'target_date']]
            logger.info(f"  Sample features: {feature_cols[:10]}")

        logger.info("Feature generation completed successfully!")

    except KeyboardInterrupt:
        logger.info("Feature generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()