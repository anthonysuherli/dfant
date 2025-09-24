#!/usr/bin/env python3
"""
Training Data Export Script

This script exports training data from collected CSV files for ML model training.

Usage:
    python export_training_data.py --start-date 20240101 --end-date 20240131
    python export_training_data.py --recent-days 90  # Last 90 days
    python export_training_data.py --season 2024  # Full season
"""

import argparse
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from csv_data_loader import CSVDataLoader

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training_export.log')
        ]
    )

def get_season_date_range(season_year: int) -> tuple:
    """Get start and end dates for NBA season."""
    start_date = f"{season_year - 1}1001"  # October 1st
    end_date = f"{season_year}0430"        # April 30th
    return start_date, end_date

def main():
    parser = argparse.ArgumentParser(description="Export training data from CSV files")

    # Date range options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--start-date", help="Start date (YYYYMMDD)")
    date_group.add_argument("--season", type=int, help="Season year (e.g., 2024 for 2023-24 season)")
    date_group.add_argument("--recent-days", type=int, help="Export last N days")

    parser.add_argument("--end-date", help="End date (YYYYMMDD), required with --start-date")
    parser.add_argument("--data-dir", default="data", help="Directory with CSV files (default: data)")
    parser.add_argument("--output", default="training_data.csv", help="Output training data file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    # Filter options
    parser.add_argument("--min-salary", type=int, default=3000, help="Minimum salary threshold (default: 3000)")
    parser.add_argument("--min-minutes", type=float, default=5.0, help="Minimum minutes threshold (default: 5.0)")

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

    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    try:
        # Initialize data loader
        logger.info(f"Initializing data loader for directory: {args.data_dir}")
        loader = CSVDataLoader(data_dir=args.data_dir)

        # Check available data
        logger.info("Checking available data...")
        data_summary = loader.get_data_summary()

        logger.info("Data Summary:")
        for data_type, stats in data_summary.items():
            if 'total_records' in stats:
                logger.info(f"  {data_type}: {stats['total_records']} records ({stats['date_range']})")
            else:
                logger.info(f"  {data_type}: {stats}")

        # Export training data
        logger.info(f"Exporting training data from {start_date} to {end_date}...")

        training_df = loader.export_training_data(
            start_date=start_date,
            end_date=end_date,
            output_file=args.output
        )

        if training_df.empty:
            logger.error("No training data could be exported. Check if required CSV files exist.")
            sys.exit(1)

        # Apply filters
        initial_count = len(training_df)

        # Filter by minimum salary
        if args.min_salary > 0:
            training_df = training_df[
                (training_df['salary_dk'] >= args.min_salary) |
                (training_df['salary_fd'] >= args.min_salary)
            ]
            logger.info(f"Filtered by minimum salary ({args.min_salary}): {len(training_df)} records remaining")

        # Filter by minimum minutes
        if args.min_minutes > 0:
            training_df = training_df[training_df['actual_minutes'] >= args.min_minutes]
            logger.info(f"Filtered by minimum minutes ({args.min_minutes}): {len(training_df)} records remaining")

        # Final statistics
        logger.info("Training Data Export Summary:")
        logger.info(f"  Total records: {len(training_df)}")
        logger.info(f"  Filtered out: {initial_count - len(training_df)} records")
        logger.info(f"  Date range: {training_df['target_date'].min()} to {training_df['target_date'].max()}")
        logger.info(f"  Unique players: {training_df['player_id'].nunique()}")

        # Feature summary
        feature_cols = [col for col in training_df.columns
                       if col not in ['player_id', 'target_date', 'actual_fantasy_points',
                                     'actual_points', 'actual_rebounds', 'actual_assists',
                                     'actual_minutes', 'salary_dk', 'salary_fd', 'value_dk']]
        logger.info(f"  Feature columns: {len(feature_cols)}")

        # Show sample features
        if len(feature_cols) > 0:
            logger.info(f"  Sample features: {feature_cols[:10]}")

        # Target variable stats
        if 'actual_fantasy_points' in training_df.columns:
            fp_stats = training_df['actual_fantasy_points'].describe()
            logger.info(f"  Fantasy points - Mean: {fp_stats['mean']:.1f}, Std: {fp_stats['std']:.1f}")

        # Save final filtered data
        if args.output != "training_data.csv":
            final_output = Path(args.data_dir) / args.output
            training_df.to_csv(final_output, index=False)
            logger.info(f"Final training data saved to: {final_output}")

        logger.info("Training data export completed successfully!")

    except KeyboardInterrupt:
        logger.info("Export interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()