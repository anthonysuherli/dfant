import time
import logging
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from tank01_api import NBAFantasyAPI
from database.connection import get_db_session
from database.models import (
    Game, Player, Team, PlayerGameLog, Injury,
    BettingOdds, DFSSalary, TeamGameLog
)

@dataclass
class CollectionConfig:
    """Configuration for historical data collection."""
    start_date: str  # YYYYMMDD format
    end_date: str    # YYYYMMDD format
    rate_limit: float = 1.0  # Requests per second
    max_retries: int = 3
    backoff_factor: float = 2.0
    chunk_size: int = 30  # Days per chunk
    save_frequency: int = 10  # Save every N successful requests
    data_dir: str = "data"  # Directory to save raw data files
    save_to_database: bool = True  # Whether to also save to database
    file_format: str = "parquet"  # "parquet", "csv", or "json"

class HistoricalDataCollector:
    """Collects historical NBA data and stores in files and/or database."""

    def __init__(self, api: NBAFantasyAPI, config: CollectionConfig):
        self.api = api
        self.config = config
        self.logger = self._setup_logger()
        self.request_count = 0
        self.last_request_time = 0.0

        # Setup data directory structure
        self.data_dir = Path(self.config.data_dir)
        self._setup_data_directories()

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

    def _setup_data_directories(self):
        """Create necessary data directories."""
        directories = [
            self.data_dir / "games",
            self.data_dir / "player_logs",
            self.data_dir / "injuries",
            self.data_dir / "dfs_salaries",
            self.data_dir / "betting_odds",
            self.data_dir / "teams",
            self.data_dir / "players"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created/verified directory: {directory}")

    def _save_to_file(self, data: pd.DataFrame, data_type: str, date_str: str = None, additional_info: str = "") -> str:
        """Save DataFrame to file and return file path."""
        if data.empty:
            self.logger.warning(f"No data to save for {data_type}")
            return ""

        # Create simple filename: [folder_name]_[date].parquet
        if date_str:
            filename = f"{data_type}_{date_str}"
        else:
            filename = f"{data_type}"

        # Special case for player_logs: player_logs_[date]_[TeamA@TeamB].parquet
        if data_type == "player_logs" and additional_info:
            # Extract game info from additional_info (e.g., "game_20211019_BKN@MIL")
            if additional_info.startswith("game_"):
                game_info = additional_info.replace("game_", "").replace(date_str + "_", "")
                filename = f"{data_type}_{date_str}_{game_info}"
            else:
                filename = f"{data_type}_{date_str}_{additional_info}"
        elif additional_info and data_type != "player_logs":
            # For non-player_logs, only add additional_info for special cases
            if additional_info in ["current"]:
                filename = f"{data_type}_{additional_info}"

        # Select file extension and save method
        if self.config.file_format == "parquet":
            filename += ".parquet"
            file_path = self.data_dir / data_type / filename
            data.to_parquet(file_path, index=False)
        elif self.config.file_format == "csv":
            filename += ".csv"
            file_path = self.data_dir / data_type / filename
            data.to_csv(file_path, index=False)
        elif self.config.file_format == "json":
            filename += ".json"
            file_path = self.data_dir / data_type / filename
            data.to_json(file_path, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported file format: {self.config.file_format}")

        self.logger.info(f"SAVED FILE: {file_path} ({len(data)} records)")
        return str(file_path)

    def _rate_limit(self):
        """Enforce rate limiting between API requests."""
        if self.config.rate_limit > 0:
            time_since_last = time.time() - self.last_request_time
            min_interval = 1.0 / self.config.rate_limit

            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1

    def _make_request_with_retry(self, func, *args, **kwargs) -> Optional[pd.DataFrame]:
        """Make API request with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                wait_time = self.config.backoff_factor ** attempt
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}. "
                    f"Retrying in {wait_time:.1f} seconds."
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Request failed after {self.config.max_retries} attempts: {e}")
                    return None

        return None

    def _date_range_generator(self, start_date: str, end_date: str):
        """Generate date range for iteration."""
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')

        current = start
        while current <= end:
            yield current.strftime('%Y%m%d')
            current += timedelta(days=1)

    def collect_games_for_date_range(self):
        """Collect game data for the specified date range."""
        self.logger.info(f"Starting game collection from {self.config.start_date} to {self.config.end_date}")

        dates = list(self._date_range_generator(self.config.start_date, self.config.end_date))
        progress_bar = tqdm(dates, desc="Collecting games")

        games_collected = 0

        with get_db_session() as session:
            for date_str in progress_bar:
                progress_bar.set_description(f"Collecting games for {date_str}")

                # Check if data already exists
                existing_games = session.query(Game).filter(
                    Game.game_date >= datetime.strptime(date_str, '%Y%m%d').date(),
                    Game.game_date < datetime.strptime(date_str, '%Y%m%d').date() + timedelta(days=1)
                ).count()

                if existing_games > 0:
                    self.logger.debug(f"Games for {date_str} already exist, skipping")
                    continue

                games_df = self._make_request_with_retry(
                    self.api.get_games_for_date,
                    game_date=date_str
                )

                if games_df is not None and not games_df.empty:
                    # Save to file first
                    file_path = self._save_to_file(games_df, "games", date_str)

                    # Save to database if configured
                    if self.config.save_to_database:
                        games_saved = self._save_games_data(session, games_df, date_str)
                        games_collected += games_saved

                        if games_collected % self.config.save_frequency == 0:
                            session.commit()
                            self.logger.info(f"Database: Saved {games_collected} games so far")
                    else:
                        games_collected += len(games_df)

                progress_bar.set_postfix({"Games": games_collected})

        self.logger.info(f"Game collection completed. Total games: {games_collected}")

    def _save_games_data(self, session, games_df: pd.DataFrame, date_str: str) -> int:
        """Save games data to database."""
        games_saved = 0

        for _, row in games_df.iterrows():
            try:
                # Extract game data
                game_date = datetime.strptime(date_str, '%Y%m%d')

                game = Game(
                    game_id=str(row.get('gameID', '')),
                    game_date=game_date,
                    season=self._determine_season(game_date),
                    season_type=row.get('seasonType', 'Regular Season'),
                    home_team_id=int(row.get('homeTeamID', 0)),
                    away_team_id=int(row.get('awayTeamID', 0)),
                    home_score=row.get('homeScore'),
                    away_score=row.get('awayScore'),
                    game_status=row.get('gameStatus', 'Scheduled'),
                    game_time=row.get('gameTime', ''),
                    # Additional fields can be added as available in API
                )

                session.merge(game)
                games_saved += 1

            except Exception as e:
                self.logger.error(f"Error saving game data: {e}")
                continue

        return games_saved

    def collect_player_game_logs_for_date_range(self):
        """Collect player game logs for the specified date range."""
        self.logger.info(f"Starting player game log collection from {self.config.start_date} to {self.config.end_date}")

        # First get all players
        with get_db_session() as session:
            # Get all games in date range
            start_date = datetime.strptime(self.config.start_date, '%Y%m%d')
            end_date = datetime.strptime(self.config.end_date, '%Y%m%d')

            games = session.query(Game).filter(
                Game.game_date >= start_date,
                Game.game_date <= end_date
            ).all()

            self.logger.info(f"Found {len(games)} games in date range")

            # Get all active players
            players_df = self._make_request_with_retry(self.api.get_player_list)
            if players_df is None or players_df.empty:
                self.logger.error("Failed to get player list")
                return

            # Process each game
            for game in tqdm(games, desc="Processing games for player logs"):
                self._collect_box_score_for_game(session, game)

    def _collect_box_score_for_game(self, session, game: Game):
        """Collect box score data for a specific game."""
        # Check if box score already exists
        existing_logs = session.query(PlayerGameLog).filter(
            PlayerGameLog.game_id == game.game_id
        ).count()

        if existing_logs > 0:
            self.logger.debug(f"Box score for game {game.game_id} already exists, skipping")
            return

        box_score_df = self._make_request_with_retry(
            self.api.get_box_score,
            game_id=game.game_id,
            fantasy_points=True
        )

        if box_score_df is not None and not box_score_df.empty:
            # Save to file first
            date_str = game.game_date.strftime('%Y%m%d')
            file_path = self._save_to_file(box_score_df, "player_logs", date_str, f"game_{game.game_id}")

            # Save to database if configured
            if self.config.save_to_database:
                self._save_player_game_logs(session, box_score_df, game)

    def _save_player_game_logs(self, session, box_score_df: pd.DataFrame, game: Game):
        """Save player game logs to database."""
        for _, row in box_score_df.iterrows():
            try:
                player_id = int(row.get('playerID', 0))

                # Skip invalid player records
                if player_id <= 0:
                    self.logger.debug(f"Skipping invalid player_id: {player_id}")
                    continue

                # Ensure player exists before creating game log
                existing_player = session.query(Player).filter(Player.player_id == player_id).first()
                if not existing_player:
                    # Create minimal player record from box score data
                    new_player = Player(
                        player_id=player_id,
                        name=row.get('longName', row.get('playerName', '')),
                        team_id=int(row.get('teamID', 0)) if row.get('teamID') else None,
                        position=row.get('pos', ''),
                        height='',
                        weight=None,
                        birth_date=None,
                        jersey_number='',
                        is_active=True
                    )
                    session.add(new_player)
                    session.flush()  # Ensure player is created before game log

                # Check if game log already exists (upsert logic)
                existing_log = session.query(PlayerGameLog).filter(
                    PlayerGameLog.player_id == player_id,
                    PlayerGameLog.game_id == game.game_id
                ).first()

                # Determine if player is on home team
                is_home = int(row.get('teamID', 0)) == game.home_team_id
                opponent_team_id = game.away_team_id if is_home else game.home_team_id

                if existing_log:
                    # Update existing game log
                    existing_log.minutes = self._safe_float(row.get('min'))
                    existing_log.points = self._safe_int(row.get('pts'))
                    existing_log.field_goals_made = self._safe_int(row.get('fgm'))
                    existing_log.field_goals_attempted = self._safe_int(row.get('fga'))
                    existing_log.field_goal_percentage = self._safe_float(row.get('fg_pct'))
                    existing_log.three_pointers_made = self._safe_int(row.get('tpm'))
                    existing_log.three_pointers_attempted = self._safe_int(row.get('tpa'))
                    existing_log.three_point_percentage = self._safe_float(row.get('tp_pct'))
                    existing_log.free_throws_made = self._safe_int(row.get('ftm'))
                    existing_log.free_throws_attempted = self._safe_int(row.get('fta'))
                    existing_log.free_throw_percentage = self._safe_float(row.get('ft_pct'))
                    existing_log.rebounds_offensive = self._safe_int(row.get('oreb'))
                    existing_log.rebounds_defensive = self._safe_int(row.get('dreb'))
                    existing_log.rebounds_total = self._safe_int(row.get('reb'))
                    existing_log.assists = self._safe_int(row.get('ast'))
                    existing_log.steals = self._safe_int(row.get('stl'))
                    existing_log.blocks = self._safe_int(row.get('blk'))
                    existing_log.turnovers = self._safe_int(row.get('tov'))
                    existing_log.personal_fouls = self._safe_int(row.get('pf'))
                    existing_log.plus_minus = self._safe_int(row.get('plusMinus'))
                    existing_log.dfs_points_dk = self._safe_float(row.get('fantasyPoints'))
                    existing_log.is_home = is_home
                    existing_log.opponent_team_id = opponent_team_id
                    existing_log.is_starter = row.get('starter', '').lower() == 'true'
                else:
                    # Create new game log
                    new_log = PlayerGameLog(
                        player_id=player_id,
                        game_id=game.game_id,
                        game_date=game.game_date,
                        minutes=self._safe_float(row.get('min')),
                        points=self._safe_int(row.get('pts')),
                        field_goals_made=self._safe_int(row.get('fgm')),
                        field_goals_attempted=self._safe_int(row.get('fga')),
                        field_goal_percentage=self._safe_float(row.get('fg_pct')),
                        three_pointers_made=self._safe_int(row.get('tpm')),
                        three_pointers_attempted=self._safe_int(row.get('tpa')),
                        three_point_percentage=self._safe_float(row.get('tp_pct')),
                        free_throws_made=self._safe_int(row.get('ftm')),
                        free_throws_attempted=self._safe_int(row.get('fta')),
                        free_throw_percentage=self._safe_float(row.get('ft_pct')),
                        rebounds_offensive=self._safe_int(row.get('oreb')),
                        rebounds_defensive=self._safe_int(row.get('dreb')),
                        rebounds_total=self._safe_int(row.get('reb')),
                        assists=self._safe_int(row.get('ast')),
                        steals=self._safe_int(row.get('stl')),
                        blocks=self._safe_int(row.get('blk')),
                        turnovers=self._safe_int(row.get('tov')),
                        personal_fouls=self._safe_int(row.get('pf')),
                        plus_minus=self._safe_int(row.get('plusMinus')),
                        dfs_points_dk=self._safe_float(row.get('fantasyPoints')),
                        is_home=is_home,
                        opponent_team_id=opponent_team_id,
                        is_starter=row.get('starter', '').lower() == 'true'
                    )
                    session.add(new_log)

            except Exception as e:
                self.logger.error(f"Error saving player game log: {e}")
                continue

    def collect_injury_data_for_date_range(self):
        """Collect injury data for the specified date range."""
        self.logger.info(f"Starting injury data collection from {self.config.start_date} to {self.config.end_date}")

        # Collect injury data in chunks to avoid overwhelming the API
        start_date = datetime.strptime(self.config.start_date, '%Y%m%d')
        end_date = datetime.strptime(self.config.end_date, '%Y%m%d')

        current_date = start_date
        with get_db_session() as session:
            while current_date <= end_date:
                chunk_end = min(current_date + timedelta(days=self.config.chunk_size), end_date)

                self.logger.info(f"Collecting injuries from {current_date.strftime('%Y%m%d')} to {chunk_end.strftime('%Y%m%d')}")

                injury_df = self._make_request_with_retry(
                    self.api.get_injury_list,
                    beginning_inj_date=current_date.strftime('%Y%m%d'),
                    end_inj_date=chunk_end.strftime('%Y%m%d')
                )

                if injury_df is not None and not injury_df.empty:
                    # Save to file first
                    chunk_start_str = current_date.strftime('%Y%m%d')
                    chunk_end_str = chunk_end.strftime('%Y%m%d')
                    file_path = self._save_to_file(injury_df, "injuries", f"{chunk_start_str}_to_{chunk_end_str}")

                    # Save to database if configured
                    if self.config.save_to_database:
                        self._save_injury_data(session, injury_df)

                current_date = chunk_end + timedelta(days=1)

    def _save_injury_data(self, session, injury_df: pd.DataFrame):
        """Save injury data to database."""
        for _, row in injury_df.iterrows():
            try:
                injury_date_str = row.get('injDate', '')
                if not injury_date_str:
                    continue

                injury_date = datetime.strptime(injury_date_str, '%Y%m%d')

                injury = Injury(
                    player_id=int(row.get('playerID', 0)),
                    injury_date=injury_date,
                    status=row.get('status', ''),
                    description=row.get('description', ''),
                    body_part=row.get('bodyPart', ''),
                )

                session.merge(injury)

            except Exception as e:
                self.logger.error(f"Error saving injury data: {e}")
                continue

    def collect_dfs_salaries_for_date_range(self):
        """Collect DFS salary data for the specified date range."""
        self.logger.info(f"Starting DFS salary collection from {self.config.start_date} to {self.config.end_date}")

        dates = list(self._date_range_generator(self.config.start_date, self.config.end_date))

        with get_db_session() as session:
            for date_str in tqdm(dates, desc="Collecting DFS salaries"):
                # Check if data already exists
                existing_salaries = session.query(DFSSalary).filter(
                    DFSSalary.game_date >= datetime.strptime(date_str, '%Y%m%d').date(),
                    DFSSalary.game_date < datetime.strptime(date_str, '%Y%m%d').date() + timedelta(days=1)
                ).count()

                if existing_salaries > 0:
                    continue

                dfs_df = self._make_request_with_retry(
                    self.api.get_dfs_salaries,
                    date=date_str
                )

                if dfs_df is not None and not dfs_df.empty:
                    # Save to file first
                    file_path = self._save_to_file(dfs_df, "dfs_salaries", date_str)

                    # Save to database if configured
                    if self.config.save_to_database:
                        self._save_dfs_salary_data(session, dfs_df, date_str)

    def _save_dfs_salary_data(self, session, dfs_df: pd.DataFrame, date_str: str):
        """Save DFS salary data to database."""
        game_date = datetime.strptime(date_str, '%Y%m%d')

        for _, row in dfs_df.iterrows():
            try:
                # New format: each row has platform and salary columns
                if pd.notna(row.get('salary')) and pd.notna(row.get('platform')):
                    dfs_salary = DFSSalary(
                        player_id=int(row.get('playerID', 0)),
                        game_date=game_date,
                        platform=row.get('platform'),
                        salary=int(row.get('salary', 0)),
                        position=row.get('pos', ''),
                        is_available=True
                    )
                    session.merge(dfs_salary)

            except Exception as e:
                self.logger.error(f"Error saving DFS salary data: {e}")
                continue

    def collect_betting_odds_for_date_range(self):
        """Collect betting odds data for the specified date range."""
        self.logger.info(f"Starting betting odds collection from {self.config.start_date} to {self.config.end_date}")

        dates = list(self._date_range_generator(self.config.start_date, self.config.end_date))

        with get_db_session() as session:
            for date_str in tqdm(dates, desc="Collecting betting odds"):
                # Check if data already exists by joining with Game table
                target_date = datetime.strptime(date_str, '%Y%m%d').date()
                existing_odds = session.query(BettingOdds).join(Game).filter(
                    Game.game_date >= target_date,
                    Game.game_date < target_date + timedelta(days=1)
                ).count()

                if existing_odds > 0:
                    continue

                betting_df = self._make_request_with_retry(
                    self.api.get_betting_odds,
                    game_date=date_str
                )

                if betting_df is not None and not betting_df.empty:
                    # Save to file first
                    file_path = self._save_to_file(betting_df, "betting_odds", date_str)

                    # Save to database if configured
                    if self.config.save_to_database:
                        self._save_betting_odds_data(session, betting_df, date_str)

    def _save_betting_odds_data(self, session, betting_df: pd.DataFrame, date_str: str):
        """Save betting odds data to database."""
        current_timestamp = datetime.utcnow()

        for _, row in betting_df.iterrows():
            try:
                # Each row represents odds from one sportsbook for one game
                betting_odds = BettingOdds(
                    game_id=row.get('gameID'),
                    sportsbook=row.get('sportsbook'),
                    timestamp=current_timestamp,
                    home_spread=self._safe_float(row.get('homeTeamSpread')),
                    away_spread=self._safe_float(row.get('awayTeamSpread')),
                    home_spread_odds=self._safe_int(row.get('homeTeamSpreadOdds')),
                    away_spread_odds=self._safe_int(row.get('awayTeamSpreadOdds')),
                    home_moneyline=self._safe_int(row.get('homeTeamMLOdds')),
                    away_moneyline=self._safe_int(row.get('awayTeamMLOdds')),
                    total_points=self._safe_float(row.get('totalOver')),
                    over_odds=self._safe_int(row.get('overOdds')),
                    under_odds=self._safe_int(row.get('underOdds'))
                )
                session.merge(betting_odds)

            except Exception as e:
                self.logger.error(f"Error saving betting odds data: {e}")
                continue

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely convert value to int."""
        try:
            if pd.isna(value) or value == '':
                return None
            return int(float(value))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        try:
            if pd.isna(value) or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _determine_season(game_date: datetime) -> str:
        """Determine NBA season from game date."""
        if game_date.month >= 10:  # October to December
            return f"{game_date.year}-{str(game_date.year + 1)[2:]}"
        else:  # January to September
            return f"{game_date.year - 1}-{str(game_date.year)[2:]}"

    def collect_players_data(self):
        """Collect current players list."""
        self.logger.info("Starting players data collection")

        players_df = self._make_request_with_retry(self.api.get_player_list)

        if players_df is not None and not players_df.empty:
            # Save to file
            file_path = self._save_to_file(players_df, "players", additional_info="current")

            # Save to database if configured
            if self.config.save_to_database:
                self._save_players_data(players_df)
        else:
            self.logger.warning("No players data retrieved")

    def collect_teams_data(self):
        """Collect current teams list."""
        self.logger.info("Starting teams data collection")

        teams_df = self._make_request_with_retry(self.api.get_teams)

        if teams_df is not None and not teams_df.empty:
            # Save to file
            file_path = self._save_to_file(teams_df, "teams", additional_info="current")

            # Save to database if configured
            if self.config.save_to_database:
                self._save_teams_data(teams_df)
        else:
            self.logger.warning("No teams data retrieved")

    def _save_players_data(self, players_df: pd.DataFrame):
        """Save players data to database using upsert logic."""
        with get_db_session() as session:
            for _, row in players_df.iterrows():
                try:
                    player_id = int(row.get('playerID', 0))

                    # Check if player already exists
                    existing_player = session.query(Player).filter(Player.player_id == player_id).first()

                    if existing_player:
                        # Update existing player
                        existing_player.name = row.get('playerName', '')
                        existing_player.team_id = int(row.get('teamID', 0)) if row.get('teamID') else None
                        existing_player.position = row.get('pos', '')
                        existing_player.height = row.get('height', '')
                        existing_player.weight = self._safe_int(row.get('weight'))
                        existing_player.jersey_number = row.get('jersey', '')
                        existing_player.is_active = True
                    else:
                        # Create new player
                        new_player = Player(
                            player_id=player_id,
                            name=row.get('playerName', ''),
                            team_id=int(row.get('teamID', 0)) if row.get('teamID') else None,
                            position=row.get('pos', ''),
                            height=row.get('height', ''),
                            weight=self._safe_int(row.get('weight')),
                            birth_date=None,  # Not available in basic player list
                            jersey_number=row.get('jersey', ''),
                            is_active=True
                        )
                        session.add(new_player)

                except Exception as e:
                    self.logger.error(f"Error saving player data: {e}")
                    continue
            session.commit()

    def _save_teams_data(self, teams_df: pd.DataFrame):
        """Save teams data to database using upsert logic."""
        with get_db_session() as session:
            for _, row in teams_df.iterrows():
                try:
                    team_id = int(row.get('teamID', 0))

                    # Check if team already exists
                    existing_team = session.query(Team).filter(Team.team_id == team_id).first()

                    if existing_team:
                        # Update existing team
                        existing_team.abbreviation = row.get('teamAbv', '')
                        existing_team.name = row.get('teamName', '')
                        existing_team.conference = row.get('conference', '')
                        existing_team.division = row.get('division', '')
                    else:
                        # Create new team
                        new_team = Team(
                            team_id=team_id,
                            abbreviation=row.get('teamAbv', ''),
                            name=row.get('teamName', ''),
                            conference=row.get('conference', ''),
                            division=row.get('division', '')
                        )
                        session.add(new_team)

                except Exception as e:
                    self.logger.error(f"Error saving team data: {e}")
                    continue
            session.commit()

    def run_full_collection(self):
        """Run complete historical data collection."""
        self.logger.info("Starting full historical data collection")
        self.logger.info(f"Data will be saved to: {self.data_dir.absolute()}")
        self.logger.info(f"File format: {self.config.file_format}")
        self.logger.info(f"Database saving: {'enabled' if self.config.save_to_database else 'disabled'}")

        try:
            # Collect reference data first
            self.collect_teams_data()
            self.collect_players_data()

            # Collect in order of dependencies
            self.collect_games_for_date_range()
            self.collect_player_game_logs_for_date_range()
            self.collect_injury_data_for_date_range()
            self.collect_dfs_salaries_for_date_range()

            self.logger.info("Full historical data collection completed successfully!")

        except Exception as e:
            self.logger.error(f"Error during full collection: {e}")
            raise