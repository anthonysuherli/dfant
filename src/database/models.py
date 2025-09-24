from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text,
    ForeignKey, UniqueConstraint, Index, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Team(Base):
    __tablename__ = 'teams'

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, unique=True, nullable=False)
    abbreviation = Column(String(5), nullable=False)
    name = Column(String(100), nullable=False)
    conference = Column(String(10))
    division = Column(String(20))

    # Relationships
    games_home = relationship("Game", foreign_keys="Game.home_team_id", back_populates="home_team")
    games_away = relationship("Game", foreign_keys="Game.away_team_id", back_populates="away_team")
    players = relationship("Player", back_populates="team")

class Player(Base):
    __tablename__ = 'players'

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=True)
    position = Column(String(10))
    height = Column(String(10))
    weight = Column(Integer)
    birth_date = Column(DateTime)
    jersey_number = Column(String(5))
    is_active = Column(Boolean, default=True)

    # Relationships
    team = relationship("Team", back_populates="players")
    game_logs = relationship("PlayerGameLog", back_populates="player")
    injuries = relationship("Injury", back_populates="player")
    projections = relationship("Projection", back_populates="player")

    __table_args__ = (
        Index('idx_player_team', 'team_id'),
        Index('idx_player_active', 'is_active'),
    )

class Game(Base):
    __tablename__ = 'games'

    id = Column(Integer, primary_key=True)
    game_id = Column(String(20), unique=True, nullable=False)
    game_date = Column(DateTime, nullable=False)
    season = Column(String(10), nullable=False)
    season_type = Column(String(20))  # Regular Season, Playoffs
    home_team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
    away_team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    game_status = Column(String(20))  # Scheduled, In Progress, Final
    game_time = Column(String(10))

    # Betting data
    total_line = Column(Float)
    home_spread = Column(Float)
    away_spread = Column(Float)
    home_moneyline = Column(Integer)
    away_moneyline = Column(Integer)

    # Game context
    home_rest_days = Column(Integer)
    away_rest_days = Column(Integer)
    is_back_to_back_home = Column(Boolean, default=False)
    is_back_to_back_away = Column(Boolean, default=False)

    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="games_home")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="games_away")
    player_game_logs = relationship("PlayerGameLog", back_populates="game")
    betting_odds = relationship("BettingOdds", back_populates="game")

    __table_args__ = (
        Index('idx_game_date', 'game_date'),
        Index('idx_game_season', 'season'),
        Index('idx_game_teams', 'home_team_id', 'away_team_id'),
    )

class PlayerGameLog(Base):
    __tablename__ = 'player_game_logs'

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.player_id'), nullable=False)
    game_id = Column(String(20), ForeignKey('games.game_id'), nullable=False)
    game_date = Column(DateTime, nullable=False)

    # Basic stats
    minutes = Column(Float)
    points = Column(Integer)
    field_goals_made = Column(Integer)
    field_goals_attempted = Column(Integer)
    field_goal_percentage = Column(Float)
    three_pointers_made = Column(Integer)
    three_pointers_attempted = Column(Integer)
    three_point_percentage = Column(Float)
    free_throws_made = Column(Integer)
    free_throws_attempted = Column(Integer)
    free_throw_percentage = Column(Float)
    rebounds_offensive = Column(Integer)
    rebounds_defensive = Column(Integer)
    rebounds_total = Column(Integer)
    assists = Column(Integer)
    steals = Column(Integer)
    blocks = Column(Integer)
    turnovers = Column(Integer)
    personal_fouls = Column(Integer)
    plus_minus = Column(Integer)

    # Advanced metrics (calculated)
    true_shooting_percentage = Column(Float)
    effective_field_goal_percentage = Column(Float)
    usage_rate = Column(Float)
    player_efficiency_rating = Column(Float)

    # Fantasy scoring
    dfs_points_dk = Column(Float)  # DraftKings scoring
    dfs_points_fd = Column(Float)  # FanDuel scoring
    dfs_salary_dk = Column(Integer)
    dfs_salary_fd = Column(Integer)

    # Game context
    is_starter = Column(Boolean)
    is_home = Column(Boolean)
    opponent_team_id = Column(Integer, ForeignKey('teams.team_id'))
    rest_days = Column(Integer)

    # Relationships
    player = relationship("Player", back_populates="game_logs")
    game = relationship("Game", back_populates="player_game_logs")
    opponent_team = relationship("Team", foreign_keys=[opponent_team_id])

    __table_args__ = (
        UniqueConstraint('player_id', 'game_id'),
        Index('idx_game_log_date', 'game_date'),
        Index('idx_game_log_player', 'player_id'),
        Index('idx_game_log_fantasy', 'dfs_points_dk'),
    )

class Injury(Base):
    __tablename__ = 'injuries'

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.player_id'), nullable=False)
    injury_date = Column(DateTime, nullable=False)
    status = Column(String(50))  # Out, Questionable, Probable, Day-to-Day
    description = Column(Text)
    body_part = Column(String(50))
    return_date = Column(DateTime)
    games_missed = Column(Integer)

    # Relationships
    player = relationship("Player", back_populates="injuries")

    __table_args__ = (
        Index('idx_injury_date', 'injury_date'),
        Index('idx_injury_player', 'player_id'),
        Index('idx_injury_status', 'status'),
    )

class BettingOdds(Base):
    __tablename__ = 'betting_odds'

    id = Column(Integer, primary_key=True)
    game_id = Column(String(20), ForeignKey('games.game_id'), nullable=False)
    sportsbook = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)

    # Spread betting
    home_spread = Column(Float)
    away_spread = Column(Float)
    home_spread_odds = Column(Integer)
    away_spread_odds = Column(Integer)

    # Moneyline
    home_moneyline = Column(Integer)
    away_moneyline = Column(Integer)

    # Total (Over/Under)
    total_points = Column(Float)
    over_odds = Column(Integer)
    under_odds = Column(Integer)

    # Relationships
    game = relationship("Game", back_populates="betting_odds")

    __table_args__ = (
        Index('idx_odds_game', 'game_id'),
        Index('idx_odds_timestamp', 'timestamp'),
    )

class Projection(Base):
    __tablename__ = 'projections'

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.player_id'), nullable=False)
    game_date = Column(DateTime, nullable=False)
    projection_date = Column(DateTime, nullable=False)  # When projection was made
    source = Column(String(50), nullable=False)  # API, Model name, etc.

    # Projected stats
    projected_minutes = Column(Float)
    projected_points = Column(Float)
    projected_rebounds = Column(Float)
    projected_assists = Column(Float)
    projected_steals = Column(Float)
    projected_blocks = Column(Float)
    projected_turnovers = Column(Float)
    projected_field_goals_made = Column(Float)
    projected_field_goals_attempted = Column(Float)
    projected_three_pointers_made = Column(Float)
    projected_three_pointers_attempted = Column(Float)
    projected_free_throws_made = Column(Float)
    projected_free_throws_attempted = Column(Float)

    # Fantasy projections
    projected_dfs_points_dk = Column(Float)
    projected_dfs_points_fd = Column(Float)

    # Confidence metrics
    projection_confidence = Column(Float)  # 0-1 scale
    variance_estimate = Column(Float)

    # Relationships
    player = relationship("Player", back_populates="projections")

    __table_args__ = (
        Index('idx_projection_game_date', 'game_date'),
        Index('idx_projection_player', 'player_id'),
        Index('idx_projection_source', 'source'),
    )

class TeamGameLog(Base):
    __tablename__ = 'team_game_logs'

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
    game_id = Column(String(20), ForeignKey('games.game_id'), nullable=False)
    game_date = Column(DateTime, nullable=False)

    # Team stats
    points = Column(Integer)
    field_goals_made = Column(Integer)
    field_goals_attempted = Column(Integer)
    field_goal_percentage = Column(Float)
    three_pointers_made = Column(Integer)
    three_pointers_attempted = Column(Integer)
    three_point_percentage = Column(Float)
    free_throws_made = Column(Integer)
    free_throws_attempted = Column(Integer)
    free_throw_percentage = Column(Float)
    rebounds_offensive = Column(Integer)
    rebounds_defensive = Column(Integer)
    rebounds_total = Column(Integer)
    assists = Column(Integer)
    steals = Column(Integer)
    blocks = Column(Integer)
    turnovers = Column(Integer)
    personal_fouls = Column(Integer)

    # Advanced team metrics
    pace = Column(Float)
    offensive_rating = Column(Float)
    defensive_rating = Column(Float)
    net_rating = Column(Float)
    true_shooting_percentage = Column(Float)
    effective_field_goal_percentage = Column(Float)
    turnover_rate = Column(Float)
    offensive_rebound_rate = Column(Float)
    defensive_rebound_rate = Column(Float)

    # Game context
    is_home = Column(Boolean)
    opponent_team_id = Column(Integer, ForeignKey('teams.team_id'))
    rest_days = Column(Integer)

    __table_args__ = (
        UniqueConstraint('team_id', 'game_id'),
        Index('idx_team_game_log_date', 'game_date'),
        Index('idx_team_game_log_team', 'team_id'),
    )

class DFSSalary(Base):
    __tablename__ = 'dfs_salaries'

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.player_id'), nullable=False)
    game_date = Column(DateTime, nullable=False)
    platform = Column(String(20), nullable=False)  # DraftKings, FanDuel
    salary = Column(Integer, nullable=False)
    position = Column(String(10))
    is_available = Column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint('player_id', 'game_date', 'platform'),
        Index('idx_dfs_salary_date', 'game_date'),
        Index('idx_dfs_salary_platform', 'platform'),
    )

class FeatureStore(Base):
    __tablename__ = 'feature_store'

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.player_id'), nullable=False)
    game_date = Column(DateTime, nullable=False)
    feature_set_version = Column(String(20), nullable=False)

    # Store features as JSON for flexibility
    features = Column(JSON, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('player_id', 'game_date', 'feature_set_version'),
        Index('idx_feature_store_date', 'game_date'),
        Index('idx_feature_store_version', 'feature_set_version'),
    )