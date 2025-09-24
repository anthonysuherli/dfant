import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
import logging
from dataclasses import dataclass

from ..database.connection import get_db_session
from ..database.models import (
    PlayerGameLog, Game, Player, Team, Injury,
    DFSSalary, FeatureStore
)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    lookback_windows: List[int] = None  # [5, 10, 15, 30] games
    include_opponent_features: bool = True
    include_injury_features: bool = True
    include_rest_features: bool = True
    include_home_away_features: bool = True
    include_salary_features: bool = True
    min_games_threshold: int = 5  # Minimum games for rolling features

    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [5, 10, 15, 30]

class FeatureEngineer:
    """Feature engineering module for NBA DFS optimization."""

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
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

    def calculate_advanced_basketball_metrics(self, logs: List[PlayerGameLog], team_stats: Dict = None) -> Dict[str, float]:
        """Calculate advanced basketball metrics that provide 1.7-1.9% MAPE improvement.

        Based on MODEL_METHODOLOGY_SUMMARY.md priority features:
        1. PER (Player Efficiency Rating)
        2. True Shooting %: Points / (2 × (FGA + 0.44 × FTA))
        3. Usage Rate calculation with team context
        """
        if not logs:
            return {}

        metrics = {}

        # Extract stats from logs
        stats = {
            'pts': np.mean([log.points for log in logs if log.points is not None]),
            'fga': np.sum([log.field_goals_attempted for log in logs if log.field_goals_attempted is not None]),
            'fgm': np.sum([log.field_goals_made for log in logs if log.field_goals_made is not None]),
            'fta': np.sum([log.free_throws_attempted for log in logs if log.free_throws_attempted is not None]),
            'ftm': np.sum([log.free_throws_made for log in logs if log.free_throws_made is not None]),
            'fg3a': np.sum([log.three_pointers_attempted for log in logs if log.three_pointers_attempted is not None]),
            'fg3m': np.sum([log.three_pointers_made for log in logs if log.three_pointers_made is not None]),
            'orb': np.sum([log.offensive_rebounds for log in logs if log.offensive_rebounds is not None]),
            'drb': np.sum([log.defensive_rebounds for log in logs if log.defensive_rebounds is not None]),
            'ast': np.sum([log.assists for log in logs if log.assists is not None]),
            'stl': np.sum([log.steals for log in logs if log.steals is not None]),
            'blk': np.sum([log.blocks for log in logs if log.blocks is not None]),
            'tov': np.sum([log.turnovers for log in logs if log.turnovers is not None]),
            'pf': np.sum([log.personal_fouls for log in logs if log.personal_fouls is not None]),
            'mins': np.sum([log.minutes_played for log in logs if log.minutes_played is not None]),
            'games': len(logs)
        }

        # 1. True Shooting Percentage
        # Formula: Points / (2 × (FGA + 0.44 × FTA))
        total_pts = stats['pts'] * stats['games']
        true_shot_attempts = 2 * (stats['fga'] + 0.44 * stats['fta'])
        metrics['true_shooting_pct'] = total_pts / true_shot_attempts if true_shot_attempts > 0 else 0.0

        # 2. Usage Rate calculation
        # Usage Rate = 100 * ((FGA + 0.44 * FTA + TOV) * Team_Pace * 5) / (Minutes * Team_Possessions)
        if team_stats and 'team_pace' in team_stats and 'team_possessions' in team_stats:
            player_possessions = stats['fga'] + 0.44 * stats['fta'] + stats['tov']
            team_pace = team_stats['team_pace']
            team_possessions = team_stats['team_possessions']

            if stats['mins'] > 0 and team_possessions > 0:
                metrics['usage_rate'] = 100 * (player_possessions * team_pace * 5) / (stats['mins'] * team_possessions)
            else:
                metrics['usage_rate'] = 0.0
        else:
            # Simplified usage rate without team context
            if stats['mins'] > 0:
                metrics['usage_rate'] = 100 * (stats['fga'] + 0.44 * stats['fta'] + stats['tov']) / stats['mins']
            else:
                metrics['usage_rate'] = 0.0

        # 3. Player Efficiency Rating (PER)
        # Simplified PER calculation (approximation of John Hollinger's formula)
        if stats['mins'] > 0:
            # Per-minute stats
            per_min = {
                'pts': total_pts / stats['mins'],
                'ast': stats['ast'] / stats['mins'],
                'reb': (stats['orb'] + stats['drb']) / stats['mins'],
                'stl': stats['stl'] / stats['mins'],
                'blk': stats['blk'] / stats['mins'],
                'tov': stats['tov'] / stats['mins'],
                'pf': stats['pf'] / stats['mins'],
                'fga': stats['fga'] / stats['mins'],
                'fta': stats['fta'] / stats['mins']
            }

            # Simplified PER formula (normalized to league average of 15)
            per_raw = (per_min['pts'] + per_min['reb'] + per_min['ast'] + per_min['stl'] + per_min['blk']
                      - per_min['tov'] - per_min['pf'] - (per_min['fga'] - stats['fgm']/stats['mins'])
                      - (per_min['fta'] - stats['ftm']/stats['mins']))

            # Scale to standard PER (league average = 15)
            metrics['per'] = max(0.0, per_raw * 15)
        else:
            metrics['per'] = 0.0

        # Additional advanced metrics
        # 4. Effective Field Goal Percentage
        if stats['fga'] > 0:
            metrics['efg_pct'] = (stats['fgm'] + 0.5 * stats['fg3m']) / stats['fga']
        else:
            metrics['efg_pct'] = 0.0

        # 5. Assist to Turnover Ratio
        if stats['tov'] > 0:
            metrics['ast_tov_ratio'] = stats['ast'] / stats['tov']
        else:
            metrics['ast_tov_ratio'] = stats['ast'] if stats['ast'] > 0 else 0.0

        # 6. Steal percentage (approximation)
        if stats['mins'] > 0:
            metrics['steal_pct'] = (stats['stl'] / stats['mins']) * 48  # Per 48 minutes
        else:
            metrics['steal_pct'] = 0.0

        # 7. Block percentage (approximation)
        if stats['mins'] > 0:
            metrics['block_pct'] = (stats['blk'] / stats['mins']) * 48  # Per 48 minutes
        else:
            metrics['block_pct'] = 0.0

        return metrics

    def calculate_moving_averages_with_decay(self, logs: List[PlayerGameLog], alpha: float = 0.2) -> Dict[str, float]:
        """Calculate moving averages with exponential decay.

        Based on MODEL_METHODOLOGY_SUMMARY.md optimal windows:
        - Consistent players: 10-15 games
        - Inconsistent players: 20-30 games
        - Exponential decay: w_i = α × (1-α)^i where α = 0.1-0.3
        """
        if not logs:
            return {}

        # Sort logs by date (most recent first)
        sorted_logs = sorted(logs, key=lambda x: x.game_date, reverse=True)

        # Determine player consistency (coefficient of variation for fantasy points)
        fantasy_points = [log.fantasy_points for log in sorted_logs if log.fantasy_points is not None]
        if len(fantasy_points) < 5:
            return {}

        cv = np.std(fantasy_points) / np.mean(fantasy_points) if np.mean(fantasy_points) > 0 else 1.0

        # Choose window size based on consistency
        if cv < 0.25:  # Consistent player (σ/μ < 0.25 as per methodology)
            window_size = min(15, len(sorted_logs))
        else:  # Inconsistent player
            window_size = min(25, len(sorted_logs))

        # Use most recent games within window
        recent_logs = sorted_logs[:window_size]

        averages = {}

        # Calculate exponentially weighted averages for key stats
        stats_to_average = [
            'fantasy_points', 'points', 'rebounds', 'assists', 'steals', 'blocks',
            'turnovers', 'field_goals_made', 'field_goals_attempted',
            'three_pointers_made', 'three_pointers_attempted',
            'free_throws_made', 'free_throws_attempted', 'minutes_played'
        ]

        for stat in stats_to_average:
            values = []
            weights = []

            for i, log in enumerate(recent_logs):
                value = getattr(log, stat, None)
                if value is not None:
                    values.append(value)
                    # Exponential decay weight: w_i = α × (1-α)^i
                    weight = alpha * (1 - alpha) ** i
                    weights.append(weight)

            if values and weights:
                # Weighted average
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                weight_sum = sum(weights)
                averages[f'{stat}_ema'] = weighted_sum / weight_sum if weight_sum > 0 else 0.0

                # Also calculate simple moving average for comparison
                averages[f'{stat}_sma'] = np.mean(values)

                # Calculate trend (slope of recent values)
                if len(values) >= 3:
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    averages[f'{stat}_trend'] = slope
                else:
                    averages[f'{stat}_trend'] = 0.0

        # Calculate advanced moving averages for our custom metrics
        if len(recent_logs) >= 3:
            # Get advanced metrics for recent games
            recent_metrics = self.calculate_advanced_basketball_metrics(recent_logs)

            for metric_name, value in recent_metrics.items():
                averages[f'{metric_name}_recent'] = value

        # Volatility metrics
        if len(fantasy_points) >= 5:
            averages['fantasy_points_volatility'] = np.std(fantasy_points)
            averages['fantasy_points_cv'] = cv
            averages['consistency_rating'] = 1 / (1 + cv)  # Higher = more consistent

        # Momentum indicators
        if len(fantasy_points) >= 6:
            recent_3 = np.mean(fantasy_points[:3])
            prev_3 = np.mean(fantasy_points[3:6])
            averages['momentum_ratio'] = recent_3 / prev_3 if prev_3 > 0 else 1.0

        return averages

    def generate_features_for_date(self, target_date: str, feature_version: str = "v1") -> pd.DataFrame:
        """Generate features for all players on a specific date."""
        self.logger.info(f"Generating features for date: {target_date}")

        target_dt = datetime.strptime(target_date, '%Y%m%d')

        with get_db_session() as session:
            # Get all active players
            players = session.query(Player).filter(Player.is_active == True).all()

            features_list = []

            for player in players:
                player_features = self._generate_player_features(
                    session, player.player_id, target_dt
                )

                if player_features:
                    features_list.append(player_features)

            # Convert to DataFrame
            if features_list:
                features_df = pd.DataFrame(features_list)
                self._save_features_to_store(session, features_df, target_date, feature_version)
                return features_df
            else:
                return pd.DataFrame()

    def _generate_player_features(self, session: Session, player_id: int, target_date: datetime) -> Optional[Dict[str, Any]]:
        """Generate features for a single player."""
        # Get historical game logs before target date
        historical_logs = session.query(PlayerGameLog).filter(
            and_(
                PlayerGameLog.player_id == player_id,
                PlayerGameLog.game_date < target_date
            )
        ).order_by(PlayerGameLog.game_date.desc()).limit(50).all()

        if len(historical_logs) < self.config.min_games_threshold:
            return None

        features = {'player_id': player_id, 'target_date': target_date}

        # Basic player info
        player = session.query(Player).filter(Player.player_id == player_id).first()
        if player:
            features.update(self._get_player_info_features(player))

        # Rolling statistics features
        features.update(self._calculate_rolling_features(historical_logs))

        # Recent performance features
        features.update(self._calculate_recent_performance_features(historical_logs))

        # Advanced basketball metrics
        features.update(self._calculate_advanced_metrics_features(historical_logs))

        # Opponent features
        if self.config.include_opponent_features:
            features.update(self._get_opponent_features(session, player_id, target_date))

        # Injury features
        if self.config.include_injury_features:
            features.update(self._get_injury_features(session, player_id, target_date))

        # Rest and schedule features
        if self.config.include_rest_features:
            features.update(self._get_rest_features(historical_logs, target_date))

        # Home/Away features
        if self.config.include_home_away_features:
            features.update(self._get_home_away_features(historical_logs))

        # Salary features
        if self.config.include_salary_features:
            features.update(self._get_salary_features(session, player_id, target_date))

        # Seasonal trends
        features.update(self._calculate_seasonal_trends(historical_logs, target_date))

        return features

    def _get_player_info_features(self, player: Player) -> Dict[str, Any]:
        """Extract basic player information features."""
        return {
            'position': player.position,
            'team_id': player.team_id,
            'is_active': player.is_active,
        }

    def _calculate_rolling_features(self, logs: List[PlayerGameLog]) -> Dict[str, Any]:
        """Calculate rolling average features for different time windows."""
        features = {}

        # Convert logs to DataFrame for easier calculation
        df = pd.DataFrame([{
            'points': log.points or 0,
            'rebounds': log.rebounds_total or 0,
            'assists': log.assists or 0,
            'steals': log.steals or 0,
            'blocks': log.blocks or 0,
            'turnovers': log.turnovers or 0,
            'minutes': log.minutes or 0,
            'field_goals_made': log.field_goals_made or 0,
            'field_goals_attempted': log.field_goals_attempted or 0,
            'three_pointers_made': log.three_pointers_made or 0,
            'three_pointers_attempted': log.three_pointers_attempted or 0,
            'free_throws_made': log.free_throws_made or 0,
            'free_throws_attempted': log.free_throws_attempted or 0,
            'dfs_points_dk': log.dfs_points_dk or 0,
            'usage_rate': log.usage_rate or 0,
            'true_shooting_percentage': log.true_shooting_percentage or 0,
        } for log in logs])

        if df.empty:
            return features

        # Calculate rolling averages for each window
        for window in self.config.lookback_windows:
            if len(df) >= window:
                window_df = df.head(window)

                # Basic stats
                features[f'avg_points_L{window}'] = window_df['points'].mean()
                features[f'avg_rebounds_L{window}'] = window_df['rebounds'].mean()
                features[f'avg_assists_L{window}'] = window_df['assists'].mean()
                features[f'avg_steals_L{window}'] = window_df['steals'].mean()
                features[f'avg_blocks_L{window}'] = window_df['blocks'].mean()
                features[f'avg_turnovers_L{window}'] = window_df['turnovers'].mean()
                features[f'avg_minutes_L{window}'] = window_df['minutes'].mean()
                features[f'avg_dfs_points_L{window}'] = window_df['dfs_points_dk'].mean()

                # Advanced stats
                features[f'avg_usage_rate_L{window}'] = window_df['usage_rate'].mean()
                features[f'avg_true_shooting_L{window}'] = window_df['true_shooting_percentage'].mean()

                # Shooting efficiency
                features[f'avg_fg_pct_L{window}'] = self._safe_divide(
                    window_df['field_goals_made'].sum(),
                    window_df['field_goals_attempted'].sum()
                )
                features[f'avg_3p_pct_L{window}'] = self._safe_divide(
                    window_df['three_pointers_made'].sum(),
                    window_df['three_pointers_attempted'].sum()
                )
                features[f'avg_ft_pct_L{window}'] = self._safe_divide(
                    window_df['free_throws_made'].sum(),
                    window_df['free_throws_attempted'].sum()
                )

                # Variance features
                features[f'std_points_L{window}'] = window_df['points'].std()
                features[f'std_dfs_points_L{window}'] = window_df['dfs_points_dk'].std()
                features[f'std_minutes_L{window}'] = window_df['minutes'].std()

                # Consistency features
                features[f'games_played_L{window}'] = window

        return features

    def _calculate_recent_performance_features(self, logs: List[PlayerGameLog]) -> Dict[str, Any]:
        """Calculate recent performance indicators."""
        features = {}

        if not logs:
            return features

        # Most recent game
        if logs:
            recent_log = logs[0]
            features['last_game_points'] = recent_log.points or 0
            features['last_game_dfs_points'] = recent_log.dfs_points_dk or 0
            features['last_game_minutes'] = recent_log.minutes or 0

        # Recent trends (last 3 vs previous 3)
        if len(logs) >= 6:
            recent_3 = logs[:3]
            previous_3 = logs[3:6]

            recent_avg_points = np.mean([log.points or 0 for log in recent_3])
            previous_avg_points = np.mean([log.points or 0 for log in previous_3])

            features['points_trend_recent'] = recent_avg_points - previous_avg_points

            recent_avg_dfs = np.mean([log.dfs_points_dk or 0 for log in recent_3])
            previous_avg_dfs = np.mean([log.dfs_points_dk or 0 for log in previous_3])

            features['dfs_trend_recent'] = recent_avg_dfs - previous_avg_dfs

        # Performance consistency
        if len(logs) >= 10:
            recent_10_dfs = [log.dfs_points_dk or 0 for log in logs[:10]]
            features['recent_consistency'] = 1.0 / (1.0 + np.std(recent_10_dfs))

        return features

    def _calculate_advanced_metrics_features(self, logs: List[PlayerGameLog]) -> Dict[str, Any]:
        """Calculate advanced basketball metrics features."""
        features = {}

        if not logs:
            return features

        # Calculate Player Efficiency Rating (PER) approximation
        total_minutes = sum(log.minutes or 0 for log in logs[:15])
        if total_minutes > 0:
            # Simplified PER calculation
            total_positive = sum(
                (log.points or 0) +
                (log.rebounds_total or 0) +
                (log.assists or 0) +
                (log.steals or 0) +
                (log.blocks or 0)
                for log in logs[:15]
            )
            total_negative = sum(
                (log.field_goals_attempted or 0) - (log.field_goals_made or 0) +
                (log.free_throws_attempted or 0) - (log.free_throws_made or 0) +
                (log.turnovers or 0)
                for log in logs[:15]
            )

            features['per_estimate'] = (total_positive - total_negative) / (total_minutes / 48)

        # Usage rate trends
        usage_rates = [log.usage_rate for log in logs[:10] if log.usage_rate]
        if usage_rates:
            features['avg_usage_rate'] = np.mean(usage_rates)
            features['usage_rate_trend'] = np.polyfit(range(len(usage_rates)), usage_rates, 1)[0]

        # Shooting efficiency trends
        ts_percentages = [log.true_shooting_percentage for log in logs[:10] if log.true_shooting_percentage]
        if ts_percentages:
            features['avg_true_shooting'] = np.mean(ts_percentages)
            features['shooting_trend'] = np.polyfit(range(len(ts_percentages)), ts_percentages, 1)[0]

        return features

    def _get_opponent_features(self, session: Session, player_id: int, target_date: datetime) -> Dict[str, Any]:
        """Get opponent-related features."""
        features = {}

        # Get today's game for this player
        game = session.query(Game).join(PlayerGameLog).filter(
            and_(
                PlayerGameLog.player_id == player_id,
                Game.game_date == target_date.date()
            )
        ).first()

        if game:
            # Determine opponent
            player = session.query(Player).filter(Player.player_id == player_id).first()
            if player:
                opponent_id = game.away_team_id if player.team_id == game.home_team_id else game.home_team_id
                features['opponent_team_id'] = opponent_id
                features['is_home_game'] = player.team_id == game.home_team_id

                # Historical performance vs opponent
                vs_opponent_logs = session.query(PlayerGameLog).filter(
                    and_(
                        PlayerGameLog.player_id == player_id,
                        PlayerGameLog.opponent_team_id == opponent_id,
                        PlayerGameLog.game_date < target_date
                    )
                ).order_by(PlayerGameLog.game_date.desc()).limit(5).all()

                if vs_opponent_logs:
                    features['avg_points_vs_opponent'] = np.mean([log.points or 0 for log in vs_opponent_logs])
                    features['avg_dfs_vs_opponent'] = np.mean([log.dfs_points_dk or 0 for log in vs_opponent_logs])
                    features['games_vs_opponent'] = len(vs_opponent_logs)

        return features

    def _get_injury_features(self, session: Session, player_id: int, target_date: datetime) -> Dict[str, Any]:
        """Get injury-related features."""
        features = {}

        # Recent injury history (last 30 days)
        recent_injuries = session.query(Injury).filter(
            and_(
                Injury.player_id == player_id,
                Injury.injury_date >= target_date - timedelta(days=30),
                Injury.injury_date < target_date
            )
        ).all()

        features['recent_injury_count'] = len(recent_injuries)
        features['has_recent_injury'] = len(recent_injuries) > 0

        # Days since last injury
        if recent_injuries:
            last_injury_date = max(injury.injury_date for injury in recent_injuries)
            features['days_since_injury'] = (target_date.date() - last_injury_date.date()).days
        else:
            features['days_since_injury'] = 999  # Large number if no recent injuries

        return features

    def _get_rest_features(self, logs: List[PlayerGameLog], target_date: datetime) -> Dict[str, Any]:
        """Get rest and schedule-related features."""
        features = {}

        if logs:
            # Days of rest before target date
            last_game_date = logs[0].game_date
            features['rest_days'] = (target_date.date() - last_game_date.date()).days - 1

            # Back-to-back indicator
            features['is_back_to_back'] = features['rest_days'] == 0

            # Rest patterns
            rest_days_list = [log.rest_days for log in logs[:10] if log.rest_days is not None]
            if rest_days_list:
                features['avg_rest_days'] = np.mean(rest_days_list)
                features['std_rest_days'] = np.std(rest_days_list)

        return features

    def _get_home_away_features(self, logs: List[PlayerGameLog]) -> Dict[str, Any]:
        """Get home/away performance features."""
        features = {}

        home_logs = [log for log in logs[:20] if log.is_home]
        away_logs = [log for log in logs[:20] if not log.is_home]

        if home_logs:
            features['avg_points_home'] = np.mean([log.points or 0 for log in home_logs])
            features['avg_dfs_home'] = np.mean([log.dfs_points_dk or 0 for log in home_logs])
            features['home_games_count'] = len(home_logs)

        if away_logs:
            features['avg_points_away'] = np.mean([log.points or 0 for log in away_logs])
            features['avg_dfs_away'] = np.mean([log.dfs_points_dk or 0 for log in away_logs])
            features['away_games_count'] = len(away_logs)

        # Home/away differential
        if home_logs and away_logs:
            features['home_away_dfs_diff'] = features['avg_dfs_home'] - features['avg_dfs_away']

        return features

    def _get_salary_features(self, session: Session, player_id: int, target_date: datetime) -> Dict[str, Any]:
        """Get DFS salary-related features."""
        features = {}

        # Current salary
        current_salary = session.query(DFSSalary).filter(
            and_(
                DFSSalary.player_id == player_id,
                DFSSalary.game_date == target_date.date(),
                DFSSalary.platform == 'DraftKings'
            )
        ).first()

        if current_salary:
            features['current_salary_dk'] = current_salary.salary

            # Historical value (points per $1000 salary)
            historical_salaries = session.query(DFSSalary).join(PlayerGameLog).filter(
                and_(
                    DFSSalary.player_id == player_id,
                    DFSSalary.platform == 'DraftKings',
                    DFSSalary.game_date < target_date.date(),
                    PlayerGameLog.player_id == player_id,
                    PlayerGameLog.game_date == DFSSalary.game_date
                )
            ).order_by(DFSSalary.game_date.desc()).limit(10).all()

            if historical_salaries:
                values = []
                for salary_record in historical_salaries:
                    game_log = session.query(PlayerGameLog).filter(
                        and_(
                            PlayerGameLog.player_id == player_id,
                            PlayerGameLog.game_date == salary_record.game_date
                        )
                    ).first()

                    if game_log and game_log.dfs_points_dk and salary_record.salary > 0:
                        value = (game_log.dfs_points_dk / salary_record.salary) * 1000
                        values.append(value)

                if values:
                    features['avg_value_dk'] = np.mean(values)
                    features['std_value_dk'] = np.std(values)

        return features

    def _calculate_seasonal_trends(self, logs: List[PlayerGameLog], target_date: datetime) -> Dict[str, Any]:
        """Calculate seasonal performance trends."""
        features = {}

        if len(logs) < 10:
            return features

        # Season-to-date averages
        season_start = datetime(target_date.year if target_date.month >= 10 else target_date.year - 1, 10, 1)
        season_logs = [log for log in logs if log.game_date >= season_start]

        if season_logs:
            features['season_avg_points'] = np.mean([log.points or 0 for log in season_logs])
            features['season_avg_dfs'] = np.mean([log.dfs_points_dk or 0 for log in season_logs])
            features['season_games_played'] = len(season_logs)

            # Month-over-month trends
            if len(season_logs) >= 20:
                # Split season into early and recent
                mid_point = len(season_logs) // 2
                early_season = season_logs[mid_point:]
                recent_season = season_logs[:mid_point]

                early_avg = np.mean([log.dfs_points_dk or 0 for log in early_season])
                recent_avg = np.mean([log.dfs_points_dk or 0 for log in recent_season])

                features['seasonal_trend'] = recent_avg - early_avg

        return features

    def _save_features_to_store(self, session: Session, features_df: pd.DataFrame,
                               target_date: str, feature_version: str):
        """Save generated features to the feature store."""
        target_dt = datetime.strptime(target_date, '%Y%m%d')

        for _, row in features_df.iterrows():
            feature_record = FeatureStore(
                player_id=int(row['player_id']),
                game_date=target_dt,
                feature_set_version=feature_version,
                features=row.to_dict()
            )

            session.merge(feature_record)

        session.commit()

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Safely divide two numbers, returning 0 if denominator is 0."""
        return numerator / denominator if denominator != 0 else 0.0

    def generate_features_batch(self, start_date: str, end_date: str,
                               feature_version: str = "v1") -> pd.DataFrame:
        """Generate features for a range of dates."""
        self.logger.info(f"Generating features batch from {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        all_features = []
        current_dt = start_dt

        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y%m%d')
            daily_features = self.generate_features_for_date(date_str, feature_version)

            if not daily_features.empty:
                all_features.append(daily_features)

            current_dt += timedelta(days=1)

        if all_features:
            return pd.concat(all_features, ignore_index=True)
        else:
            return pd.DataFrame()