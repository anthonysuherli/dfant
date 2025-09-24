#!/usr/bin/env python3
"""
Test script for the updated historical data collector.

This script demonstrates the new file-saving functionality with detailed logging.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tank01_api import NBAFantasyAPI
from collectors.historical_collector import HistoricalDataCollector, CollectionConfig


def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('collection.log', encoding='utf-8')
        ]
    )


def test_file_only_collection():
    """Test collection with file saving only (no database)."""
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("TESTING FILE-ONLY COLLECTION")
    logger.info("="*60)

    # Configure for file-only collection
    config = CollectionConfig(
        start_date="20240101",
        end_date="20240103",  # Small range for testing
        rate_limit=0.5,  # Slow rate to be respectful
        data_dir="data",
        save_to_database=False,  # File only
        file_format="parquet"  # Efficient format
    )

    # Initialize API and collector
    api = NBAFantasyAPI()
    collector = HistoricalDataCollector(api, config)

    try:
        # Test individual collections
        logger.info("Testing teams data collection...")
        collector.collect_teams_data()

        logger.info("Testing players data collection...")
        collector.collect_players_data()

        logger.info("Testing games collection for date range...")
        collector.collect_games_for_date_range()

        logger.info("File-only collection test completed!")

    except Exception as e:
        logger.error(f"Error during file-only collection test: {e}")
        raise


def test_hybrid_collection():
    """Test collection with both file and database saving."""
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("TESTING HYBRID COLLECTION (FILES + DATABASE)")
    logger.info("="*60)

    # Configure for hybrid collection
    config = CollectionConfig(
        start_date="20240101",
        end_date="20240102",  # Very small range
        rate_limit=0.5,
        data_dir="data",
        save_to_database=True,  # Both files and database
        file_format="parquet"
    )

    # Initialize API and collector
    api = NBAFantasyAPI()
    collector = HistoricalDataCollector(api, config)

    try:
        # Run limited collection
        logger.info("Testing hybrid collection...")
        collector.collect_teams_data()
        collector.collect_games_for_date_range()

        logger.info("Hybrid collection test completed!")

    except Exception as e:
        logger.error(f"Error during hybrid collection test: {e}")
        raise


def show_file_structure():
    """Display the created file structure."""
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("CREATED FILE STRUCTURE")
    logger.info("="*60)

    data_dir = Path("data")
    if data_dir.exists():
        for item in data_dir.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                logger.info(f"📁 {item}: {size_mb:.2f} MB")
    else:
        logger.warning("No data directory found")


def main():
    """Run the test suite."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting Historical Data Collector Tests")
    logger.info(f"Test started at: {datetime.now()}")

    try:
        # Test file-only collection first
        test_file_only_collection()

        # Test hybrid collection
        # test_hybrid_collection()  # Uncomment to test database saving too

        # Show results
        show_file_structure()

        logger.info("✅ All tests completed successfully!")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())