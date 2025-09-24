import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc, asc
from functools import lru_cache
import logging
import pickle
import os
from pathlib import Path

from ..database.connection import get_db_session
from ..database.models import (
    PlayerGameLog, Game, Player, Team, Injury,
    DFSSalary, FeatureStore, BettingOdds, TeamGameLog
)

class DataRepository:
    """Optimized data access layer with caching for NBA DFS data."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
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

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def _save_to_cache(self, data: Any, cache_key: str):
        """Save data to cache."""
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_key}: {e}")

    def _load_from_cache(self, cache_key: str, max_age_hours: int = 24) -> Optional[Any]:
        """Load data from cache if it exists and is not too old."""
        try:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                # Check age
                cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
                if cache_age < max_age_hours * 3600:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                else:
                    # Remove stale cache
                    cache_path.unlink()
        except Exception as e:
            self.logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None

    def get_player_game_logs(self, player_id: int, start_date: str = None,
                           end_date: str = None, limit: int = None) -> pd.DataFrame:
        """Get player game logs with optional date filtering."""
        cache_key = f"player_logs_{player_id}_{start_date}_{end_date}_{limit}"
        cached_data = self._load_from_cache(cache_key, max_age_hours=6)

        if cached_data is not None:
            return cached_data

        with get_db_session() as session:
            query = session.query(PlayerGameLog).filter(
                PlayerGameLog.player_id == player_id
            )

            if start_date:
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                query = query.filter(PlayerGameLog.game_date >= start_dt)

            if end_date:
                end_dt = datetime.strptime(end_date, '%Y%m%d')
                query = query.filter(PlayerGameLog.game_date <= end_dt)

            query = query.order_by(desc(PlayerGameLog.game_date))

            if limit:
                query = query.limit(limit)

            logs = query.all()

            # Convert to DataFrame
            df = pd.DataFrame([{
                'player_id': log.player_id,
                'game_id': log.game_id,
                'game_date': log.game_date,
                'minutes': log.minutes,
                'points': log.points,
                'rebounds': log.rebounds_total,
                'assists': log.assists,
                'steals': log.steals,
                'blocks': log.blocks,
                'turnovers': log.turnovers,
                'field_goals_made': log.field_goals_made,
                'field_goals_attempted': log.field_goals_attempted,
                'three_pointers_made': log.three_pointers_made,
                'three_pointers_attempted': log.three_pointers_attempted,
                'free_throws_made': log.free_throws_made,
                'free_throws_attempted': log.free_throws_attempted,
                'dfs_points_dk': log.dfs_points_dk,
                'dfs_points_fd': log.dfs_points_fd,
                'true_shooting_percentage': log.true_shooting_percentage,
                'usage_rate': log.usage_rate,
                'is_home': log.is_home,
                'opponent_team_id': log.opponent_team_id,
                'rest_days': log.rest_days,
                'is_starter': log.is_starter
            } for log in logs])

            self._save_to_cache(df, cache_key)
            return df

    def get_games_for_date(self, game_date: str) -> pd.DataFrame:
        """Get all games for a specific date."""
        cache_key = f"games_{game_date}"
        cached_data = self._load_from_cache(cache_key, max_age_hours=12)

        if cached_data is not None:
            return cached_data

        with get_db_session() as session:
            target_date = datetime.strptime(game_date, '%Y%m%d').date()

            games = session.query(Game).options(
                joinedload(Game.home_team),
                joinedload(Game.away_team)
            ).filter(
                func.date(Game.game_date) == target_date
            ).all()

            df = pd.DataFrame([{
                'game_id': game.game_id,
                'game_date': game.game_date,
                'home_team_id': game.home_team_id,
                'away_team_id': game.away_team_id,
                'home_team_abbr': game.home_team.abbreviation if game.home_team else None,
                'away_team_abbr': game.away_team.abbreviation if game.away_team else None,
                'home_score': game.home_score,
                'away_score': game.away_score,
                'game_status': game.game_status,
                'total_line': game.total_line,
                'home_spread': game.home_spread,
                'away_spread': game.away_spread
            } for game in games])

            self._save_to_cache(df, cache_key)
            return df

    def get_player_features(self, player_id: int, target_date: str,
                          feature_version: str = "v1") -> Optional[Dict[str, Any]]:
        """Get pre-computed features for a player on a specific date."""
        with get_db_session() as session:
            target_dt = datetime.strptime(target_date, '%Y%m%d')

            feature_record = session.query(FeatureStore).filter(
                and_(
                    FeatureStore.player_id == player_id,
                    FeatureStore.game_date == target_dt,
                    FeatureStore.feature_set_version == feature_version
                )
            ).first()

            if feature_record:
                return feature_record.features
            else:
                return None

    def get_dfs_slate_data(self, game_date: str, platform: str = 'DraftKings') -> pd.DataFrame:
        """Get complete DFS slate data for a specific date."""
        cache_key = f"dfs_slate_{game_date}_{platform}"
        cached_data = self._load_from_cache(cache_key, max_age_hours=2)

        if cached_data is not None:
            return cached_data

        with get_db_session() as session:
            target_date = datetime.strptime(game_date, '%Y%m%d').date()

            # Get DFS salaries with player and game info
            query = session.query(
                DFSSalary.player_id,
                DFSSalary.salary,
                DFSSalary.position,
                Player.name.label('player_name'),
                Player.team_id,
                Team.abbreviation.label('team_abbr'),
                Game.game_id,
                Game.home_team_id,
                Game.away_team_id,
                Game.total_line,
                Game.home_spread,
                Game.away_spread
            ).join(Player, DFSSalary.player_id == Player.player_id)\
             .join(Team, Player.team_id == Team.team_id)\
             .join(Game, or_(
                 and_(Game.home_team_id == Player.team_id, func.date(Game.game_date) == target_date),
                 and_(Game.away_team_id == Player.team_id, func.date(Game.game_date) == target_date)
             )).filter(
                and_(
                    DFSSalary.game_date == target_date,
                    DFSSalary.platform == platform,
                    DFSSalary.is_available == True
                )
             )

            results = query.all()

            df = pd.DataFrame([{
                'player_id': result.player_id,
                'player_name': result.player_name,
                'team_id': result.team_id,
                'team_abbr': result.team_abbr,
                'position': result.position,
                'salary': result.salary,
                'game_id': result.game_id,
                'total_line': result.total_line,
                'spread': result.home_spread if result.team_id == result.home_team_id else result.away_spread,
                'is_home': result.team_id == result.home_team_id
            } for result in results])

            self._save_to_cache(df, cache_key)
            return df

    def get_training_data(self, start_date: str, end_date: str,
                         feature_version: str = "v1", min_salary: int = 3000) -> pd.DataFrame:
        """Get training data with features and targets for ML models."""
        cache_key = f"training_data_{start_date}_{end_date}_{feature_version}_{min_salary}"
        cached_data = self._load_from_cache(cache_key, max_age_hours=24)

        if cached_data is not None:
            return cached_data

        with get_db_session() as session:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            # Get features with corresponding game outcomes
            query = session.query(
                FeatureStore.player_id,
                FeatureStore.game_date,
                FeatureStore.features,
                PlayerGameLog.dfs_points_dk.label('actual_dfs_points'),
                PlayerGameLog.minutes,
                PlayerGameLog.points,
                PlayerGameLog.rebounds_total,
                PlayerGameLog.assists,
                DFSSalary.salary,
                DFSSalary.position
            ).join(
                PlayerGameLog,
                and_(
                    FeatureStore.player_id == PlayerGameLog.player_id,
                    FeatureStore.game_date == PlayerGameLog.game_date
                )
            ).join(
                DFSSalary,
                and_(
                    FeatureStore.player_id == DFSSalary.player_id,
                    FeatureStore.game_date == DFSSalary.game_date,
                    DFSSalary.platform == 'DraftKings'
                )
            ).filter(
                and_(
                    FeatureStore.game_date >= start_dt,
                    FeatureStore.game_date <= end_dt,
                    FeatureStore.feature_set_version == feature_version,
                    DFSSalary.salary >= min_salary
                )
            )

            results = query.all()

            # Convert to DataFrame
            training_data = []
            for result in results:
                row = result.features.copy()
                row.update({
                    'player_id': result.player_id,
                    'game_date': result.game_date,
                    'actual_dfs_points': result.actual_dfs_points,
                    'actual_minutes': result.minutes,
                    'actual_points': result.points,
                    'actual_rebounds': result.rebounds_total,
                    'actual_assists': result.assists,
                    'salary': result.salary,
                    'position': result.position,
                    'value': result.actual_dfs_points / (result.salary / 1000) if result.salary > 0 else 0
                })
                training_data.append(row)

            df = pd.DataFrame(training_data)
            self._save_to_cache(df, cache_key)
            return df

    @lru_cache(maxsize=128)
    def get_player_info(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Get cached player information."""
        with get_db_session() as session:
            player = session.query(Player).options(
                joinedload(Player.team)
            ).filter(Player.player_id == player_id).first()

            if player:
                return {
                    'player_id': player.player_id,
                    'name': player.name,
                    'position': player.position,
                    'team_id': player.team_id,
                    'team_abbr': player.team.abbreviation if player.team else None,
                    'is_active': player.is_active
                }
            else:
                return None

    def get_team_performance_vs_position(self, team_id: int, position: str,
                                       start_date: str, end_date: str) -> Dict[str, float]:
        """Get team's defensive performance against specific positions."""
        cache_key = f"team_vs_pos_{team_id}_{position}_{start_date}_{end_date}"
        cached_data = self._load_from_cache(cache_key, max_age_hours=12)

        if cached_data is not None:
            return cached_data

        with get_db_session() as session:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            # Get opponent performance against this team
            opponent_stats = session.query(
                func.avg(PlayerGameLog.points).label('avg_points_allowed'),
                func.avg(PlayerGameLog.rebounds_total).label('avg_rebounds_allowed'),
                func.avg(PlayerGameLog.assists).label('avg_assists_allowed'),
                func.avg(PlayerGameLog.dfs_points_dk).label('avg_dfs_allowed'),
                func.count(PlayerGameLog.id).label('games_count')
            ).join(Player, PlayerGameLog.player_id == Player.player_id)\
             .filter(
                and_(
                    PlayerGameLog.opponent_team_id == team_id,
                    Player.position == position,
                    PlayerGameLog.game_date >= start_dt,
                    PlayerGameLog.game_date <= end_dt
                )
             ).first()

            result = {
                'avg_points_allowed': float(opponent_stats.avg_points_allowed or 0),
                'avg_rebounds_allowed': float(opponent_stats.avg_rebounds_allowed or 0),
                'avg_assists_allowed': float(opponent_stats.avg_assists_allowed or 0),
                'avg_dfs_allowed': float(opponent_stats.avg_dfs_allowed or 0),
                'games_sample_size': int(opponent_stats.games_count or 0)
            }

            self._save_to_cache(result, cache_key)
            return result

    def get_player_recent_trends(self, player_id: int, days: int = 30) -> Dict[str, Any]:
        """Get recent performance trends for a player."""
        cache_key = f"player_trends_{player_id}_{days}"
        cached_data = self._load_from_cache(cache_key, max_age_hours=6)

        if cached_data is not None:
            return cached_data

        with get_db_session() as session:
            cutoff_date = datetime.now() - timedelta(days=days)

            recent_logs = session.query(PlayerGameLog).filter(
                and_(
                    PlayerGameLog.player_id == player_id,
                    PlayerGameLog.game_date >= cutoff_date
                )
            ).order_by(desc(PlayerGameLog.game_date)).all()

            if not recent_logs:
                return {}

            # Calculate trends
            dfs_points = [log.dfs_points_dk or 0 for log in recent_logs]
            minutes = [log.minutes or 0 for log in recent_logs]
            points = [log.points or 0 for log in recent_logs]

            # Linear trend calculation
            if len(dfs_points) >= 3:
                x = np.arange(len(dfs_points))
                dfs_trend = np.polyfit(x, dfs_points, 1)[0]
                minutes_trend = np.polyfit(x, minutes, 1)[0]
                points_trend = np.polyfit(x, points, 1)[0]
            else:
                dfs_trend = minutes_trend = points_trend = 0

            result = {
                'games_played': len(recent_logs),
                'avg_dfs_points': np.mean(dfs_points),
                'avg_minutes': np.mean(minutes),
                'avg_points': np.mean(points),
                'dfs_trend': dfs_trend,
                'minutes_trend': minutes_trend,
                'points_trend': points_trend,
                'consistency': 1.0 / (1.0 + np.std(dfs_points)) if len(dfs_points) > 1 else 1.0
            }

            self._save_to_cache(result, cache_key)
            return result

    def get_injury_report(self, game_date: str) -> pd.DataFrame:
        """Get injury report for a specific date."""
        with get_db_session() as session:
            target_date = datetime.strptime(game_date, '%Y%m%d').date()

            # Get recent injuries (within last 7 days)
            injuries = session.query(Injury).options(
                joinedload(Injury.player)
            ).filter(
                and_(
                    Injury.injury_date <= target_date,
                    Injury.injury_date >= target_date - timedelta(days=7)
                )
            ).order_by(desc(Injury.injury_date)).all()

            df = pd.DataFrame([{
                'player_id': injury.player_id,
                'player_name': injury.player.name if injury.player else None,
                'injury_date': injury.injury_date,
                'status': injury.status,
                'description': injury.description,
                'body_part': injury.body_part,
                'games_missed': injury.games_missed
            } for injury in injuries])

            return df

    def get_ownership_projections(self, game_date: str, platform: str = 'DraftKings') -> pd.DataFrame:
        """Get ownership projections based on historical patterns."""
        # This would typically come from a separate ownership model
        # For now, return basic salary-based projections
        slate_data = self.get_dfs_slate_data(game_date, platform)

        if slate_data.empty:
            return pd.DataFrame()

        # Simple ownership model based on salary and position
        slate_data['ownership_projection'] = self._calculate_basic_ownership(slate_data)

        return slate_data[['player_id', 'player_name', 'salary', 'ownership_projection']]

    def _calculate_basic_ownership(self, slate_data: pd.DataFrame) -> pd.Series:
        """Calculate basic ownership projections."""
        # Simple model: higher salary players get higher ownership within position
        ownership = pd.Series(index=slate_data.index, dtype=float)

        for position in slate_data['position'].unique():
            pos_mask = slate_data['position'] == position
            pos_salaries = slate_data.loc[pos_mask, 'salary']

            # Normalize salaries to 0-1 range within position
            min_sal = pos_salaries.min()
            max_sal = pos_salaries.max()

            if max_sal > min_sal:
                normalized = (pos_salaries - min_sal) / (max_sal - min_sal)
                # Convert to ownership percentage (5% to 25% range)
                pos_ownership = 5 + (normalized * 20)
            else:
                pos_ownership = pd.Series(15.0, index=pos_salaries.index)

            ownership.loc[pos_mask] = pos_ownership

        return ownership

    def clear_cache(self):
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def get_correlation_matrix(self, players: List[int], start_date: str,
                             end_date: str) -> pd.DataFrame:
        """Get correlation matrix between players' DFS performances."""
        cache_key = f"correlation_{hash(tuple(players))}_{start_date}_{end_date}"
        cached_data = self._load_from_cache(cache_key, max_age_hours=24)

        if cached_data is not None:
            return cached_data

        # Get game logs for all players
        all_logs = []
        for player_id in players:
            logs = self.get_player_game_logs(player_id, start_date, end_date)
            if not logs.empty:
                logs = logs[['game_date', 'dfs_points_dk']].copy()
                logs.columns = ['game_date', f'player_{player_id}']
                all_logs.append(logs)

        if not all_logs:
            return pd.DataFrame()

        # Merge all player data on game_date
        merged_data = all_logs[0]
        for logs in all_logs[1:]:
            merged_data = pd.merge(merged_data, logs, on='game_date', how='outer')

        # Calculate correlation matrix
        correlation_matrix = merged_data.drop('game_date', axis=1).corr()

        self._save_to_cache(correlation_matrix, cache_key)
        return correlation_matrix