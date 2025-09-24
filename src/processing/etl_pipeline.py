import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func
import logging

from ..database.connection import get_db_session
from ..database.models import (
    PlayerGameLog, Game, Player, Team, Injury,
    TeamGameLog, DFSSalary
)

class ETLPipeline:
    """ETL Pipeline for cleaning and normalizing NBA data."""

    def __init__(self):
        self.logger = self._setup_logger()

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

    def calculate_advanced_stats(self, start_date: str, end_date: str):
        """Calculate advanced basketball statistics for player game logs."""
        self.logger.info(f"Calculating advanced stats from {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        with get_db_session() as session:
            # Get player game logs in date range
            game_logs = session.query(PlayerGameLog).filter(
                PlayerGameLog.game_date >= start_dt,
                PlayerGameLog.game_date <= end_dt
            ).all()

            self.logger.info(f"Processing {len(game_logs)} game logs")

            for log in game_logs:
                self._calculate_player_advanced_stats(log)
                session.merge(log)

            session.commit()

        self.logger.info("Advanced stats calculation completed")

    def _calculate_player_advanced_stats(self, log: PlayerGameLog):
        """Calculate advanced statistics for a single player game log."""
        # True Shooting Percentage
        if log.field_goals_attempted and log.free_throws_attempted:
            tsa = 2 * (log.field_goals_attempted + 0.44 * log.free_throws_attempted)
            if tsa > 0:
                log.true_shooting_percentage = log.points / tsa
            else:
                log.true_shooting_percentage = 0.0

        # Effective Field Goal Percentage
        if log.field_goals_attempted and log.field_goals_attempted > 0:
            log.effective_field_goal_percentage = (
                log.field_goals_made + 0.5 * log.three_pointers_made
            ) / log.field_goals_attempted
        else:
            log.effective_field_goal_percentage = 0.0

        # Usage Rate (simplified calculation - needs team data for accuracy)
        if log.minutes and log.minutes > 0:
            # Simplified usage rate calculation
            possessions_used = (
                log.field_goals_attempted +
                0.44 * (log.free_throws_attempted or 0) +
                (log.turnovers or 0)
            )
            # This is a simplified version - actual calculation requires team stats
            log.usage_rate = possessions_used / (log.minutes / 48) if log.minutes > 0 else 0.0

    def clean_player_data(self):
        """Clean and validate player data."""
        self.logger.info("Starting player data cleaning")

        with get_db_session() as session:
            # Remove duplicate players
            duplicates = session.query(Player.name, func.count(Player.id).label('count')).group_by(
                Player.name
            ).having(func.count(Player.id) > 1).all()

            for name, count in duplicates:
                self.logger.warning(f"Found {count} duplicate entries for player: {name}")
                # Keep the most recent entry
                players = session.query(Player).filter(Player.name == name).order_by(Player.id.desc()).all()
                for player in players[1:]:  # Remove all but the first (most recent)
                    session.delete(player)

            session.commit()

        self.logger.info("Player data cleaning completed")

    def validate_game_logs(self):
        """Validate and clean player game logs."""
        self.logger.info("Starting game log validation")

        with get_db_session() as session:
            # Find logs with impossible values
            invalid_logs = session.query(PlayerGameLog).filter(
                (PlayerGameLog.minutes < 0) |
                (PlayerGameLog.minutes > 48) |
                (PlayerGameLog.points < 0) |
                (PlayerGameLog.field_goals_made > PlayerGameLog.field_goals_attempted) |
                (PlayerGameLog.three_pointers_made > PlayerGameLog.three_pointers_attempted) |
                (PlayerGameLog.free_throws_made > PlayerGameLog.free_throws_attempted)
            ).all()

            self.logger.info(f"Found {len(invalid_logs)} logs with validation issues")

            for log in invalid_logs:
                self._fix_game_log_issues(log)
                session.merge(log)

            session.commit()

        self.logger.info("Game log validation completed")

    def _fix_game_log_issues(self, log: PlayerGameLog):
        """Fix common issues in game logs."""
        # Cap minutes at 48
        if log.minutes and log.minutes > 48:
            log.minutes = 48.0

        # Ensure made <= attempted for shooting stats
        if log.field_goals_made and log.field_goals_attempted:
            if log.field_goals_made > log.field_goals_attempted:
                log.field_goals_made = log.field_goals_attempted

        if log.three_pointers_made and log.three_pointers_attempted:
            if log.three_pointers_made > log.three_pointers_attempted:
                log.three_pointers_made = log.three_pointers_attempted

        if log.free_throws_made and log.free_throws_attempted:
            if log.free_throws_made > log.free_throws_attempted:
                log.free_throws_made = log.free_throws_attempted

        # Set negative values to 0
        for attr in ['points', 'rebounds_total', 'assists', 'steals', 'blocks']:
            value = getattr(log, attr)
            if value and value < 0:
                setattr(log, attr, 0)

    def calculate_dfs_scoring(self, platform: str = 'DraftKings'):
        """Calculate DFS points using standard scoring."""
        self.logger.info(f"Calculating {platform} DFS scoring")

        # DraftKings scoring
        if platform == 'DraftKings':
            scoring_rules = {
                'points': 1.0,
                'rebounds_total': 1.25,
                'assists': 1.5,
                'steals': 3.0,
                'blocks': 3.0,
                'turnovers': -1.0,
                'double_double': 1.5,
                'triple_double': 3.0
            }
        # FanDuel scoring
        elif platform == 'FanDuel':
            scoring_rules = {
                'points': 1.0,
                'rebounds_total': 1.2,
                'assists': 1.5,
                'steals': 3.0,
                'blocks': 3.0,
                'turnovers': -1.0
            }
        else:
            raise ValueError(f"Unknown platform: {platform}")

        with get_db_session() as session:
            game_logs = session.query(PlayerGameLog).filter(
                PlayerGameLog.dfs_points_dk.is_(None) if platform == 'DraftKings'
                else PlayerGameLog.dfs_points_fd.is_(None)
            ).all()

            self.logger.info(f"Calculating DFS points for {len(game_logs)} game logs")

            for log in game_logs:
                dfs_points = self._calculate_single_dfs_score(log, scoring_rules)

                if platform == 'DraftKings':
                    log.dfs_points_dk = dfs_points
                else:
                    log.dfs_points_fd = dfs_points

                session.merge(log)

            session.commit()

        self.logger.info(f"{platform} DFS scoring calculation completed")

    def _calculate_single_dfs_score(self, log: PlayerGameLog, scoring_rules: Dict[str, float]) -> float:
        """Calculate DFS score for a single game log."""
        score = 0.0

        # Basic stats
        score += (log.points or 0) * scoring_rules.get('points', 0)
        score += (log.rebounds_total or 0) * scoring_rules.get('rebounds_total', 0)
        score += (log.assists or 0) * scoring_rules.get('assists', 0)
        score += (log.steals or 0) * scoring_rules.get('steals', 0)
        score += (log.blocks or 0) * scoring_rules.get('blocks', 0)
        score += (log.turnovers or 0) * scoring_rules.get('turnovers', 0)

        # Bonus scoring
        if 'double_double' in scoring_rules:
            if self._is_double_double(log):
                score += scoring_rules['double_double']

        if 'triple_double' in scoring_rules:
            if self._is_triple_double(log):
                score += scoring_rules['triple_double']

        return round(score, 2)

    def _is_double_double(self, log: PlayerGameLog) -> bool:
        """Check if game log represents a double-double."""
        stats = [
            log.points or 0,
            log.rebounds_total or 0,
            log.assists or 0,
            log.steals or 0,
            log.blocks or 0
        ]
        return sum(1 for stat in stats if stat >= 10) >= 2

    def _is_triple_double(self, log: PlayerGameLog) -> bool:
        """Check if game log represents a triple-double."""
        stats = [
            log.points or 0,
            log.rebounds_total or 0,
            log.assists or 0,
            log.steals or 0,
            log.blocks or 0
        ]
        return sum(1 for stat in stats if stat >= 10) >= 3

    def calculate_rest_days(self):
        """Calculate rest days for all player game logs."""
        self.logger.info("Calculating rest days for game logs")

        with get_db_session() as session:
            # Get all players
            players = session.query(Player).filter(Player.is_active == True).all()

            for player in players:
                self._calculate_player_rest_days(session, player.player_id)

            session.commit()

        self.logger.info("Rest days calculation completed")

    def _calculate_player_rest_days(self, session: Session, player_id: int):
        """Calculate rest days for a specific player."""
        # Get all game logs for player, ordered by date
        game_logs = session.query(PlayerGameLog).filter(
            PlayerGameLog.player_id == player_id
        ).order_by(PlayerGameLog.game_date).all()

        for i, log in enumerate(game_logs):
            if i == 0:
                log.rest_days = 0  # First game has no rest days
            else:
                prev_game_date = game_logs[i-1].game_date
                current_game_date = log.game_date
                rest_days = (current_game_date.date() - prev_game_date.date()).days - 1
                log.rest_days = max(0, rest_days)  # Ensure non-negative

            session.merge(log)

    def normalize_team_data(self):
        """Normalize and standardize team data."""
        self.logger.info("Normalizing team data")

        with get_db_session() as session:
            teams = session.query(Team).all()

            for team in teams:
                # Standardize abbreviations
                team.abbreviation = team.abbreviation.upper()

                # Add conference/division info if missing
                if not team.conference:
                    team.conference, team.division = self._get_team_conference_division(team.abbreviation)

                session.merge(team)

            session.commit()

        self.logger.info("Team data normalization completed")

    def _get_team_conference_division(self, abbreviation: str) -> Tuple[str, str]:
        """Get conference and division for team abbreviation."""
        team_info = {
            'ATL': ('Eastern', 'Southeast'),
            'BOS': ('Eastern', 'Atlantic'),
            'BKN': ('Eastern', 'Atlantic'),
            'CHA': ('Eastern', 'Southeast'),
            'CHI': ('Eastern', 'Central'),
            'CLE': ('Eastern', 'Central'),
            'DAL': ('Western', 'Southwest'),
            'DEN': ('Western', 'Northwest'),
            'DET': ('Eastern', 'Central'),
            'GSW': ('Western', 'Pacific'),
            'HOU': ('Western', 'Southwest'),
            'IND': ('Eastern', 'Central'),
            'LAC': ('Western', 'Pacific'),
            'LAL': ('Western', 'Pacific'),
            'MEM': ('Western', 'Southwest'),
            'MIA': ('Eastern', 'Southeast'),
            'MIL': ('Eastern', 'Central'),
            'MIN': ('Western', 'Northwest'),
            'NOP': ('Western', 'Southwest'),
            'NYK': ('Eastern', 'Atlantic'),
            'OKC': ('Western', 'Northwest'),
            'ORL': ('Eastern', 'Southeast'),
            'PHI': ('Eastern', 'Atlantic'),
            'PHX': ('Western', 'Pacific'),
            'POR': ('Western', 'Northwest'),
            'SAC': ('Western', 'Pacific'),
            'SAS': ('Western', 'Southwest'),
            'TOR': ('Eastern', 'Atlantic'),
            'UTA': ('Western', 'Northwest'),
            'WAS': ('Eastern', 'Southeast'),
        }

        return team_info.get(abbreviation, ('Unknown', 'Unknown'))

    def run_full_etl(self, start_date: str, end_date: str):
        """Run complete ETL pipeline."""
        self.logger.info("Starting full ETL pipeline")

        try:
            # Data cleaning
            self.clean_player_data()
            self.normalize_team_data()
            self.validate_game_logs()

            # Calculate derived metrics
            self.calculate_advanced_stats(start_date, end_date)
            self.calculate_dfs_scoring('DraftKings')
            self.calculate_dfs_scoring('FanDuel')
            self.calculate_rest_days()

            self.logger.info("Full ETL pipeline completed successfully")

        except Exception as e:
            self.logger.error(f"Error during ETL pipeline: {e}")
            raise