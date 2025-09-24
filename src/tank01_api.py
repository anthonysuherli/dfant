import requests
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any

load_dotenv()

class NBAFantasyAPI:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('RAPIDAPI_KEY')
        if not self.api_key:
            raise ValueError("API key not found. Set RAPIDAPI_KEY in .env file or pass api_key parameter.")

        self.base_url = "https://tank01-fantasy-stats.p.rapidapi.com"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
        }

    def get_games_for_date(self, game_date: str) -> pd.DataFrame:
        url = f"{self.base_url}/getNBAGamesForDate"
        querystring = {"gameDate": game_date}

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_box_score(self, game_id: str, **kwargs) -> pd.DataFrame:
        url = f"{self.base_url}/getNBABoxScore"
        querystring = {"gameID": game_id}

        # Map of accepted keyword arguments to their API parameter names
        param_map = {
            "fantasy_points": "fantasyPoints",
            "pts": "pts",
            "stl": "stl",
            "blk": "blk",
            "reb": "reb",
            "ast": "ast",
            "tov": "TOV",
            "mins": "mins",
            "double_double": "doubleDouble",
            "triple_double": "tripleDouble",
            "quad_double": "quadDouble"
        }

        for key, api_param in param_map.items():
            if key in kwargs and kwargs[key] is not None:
                value = kwargs[key]
                # Special handling for fantasy_points to lowercase string
                if key == "fantasy_points":
                    querystring[api_param] = str(value).lower()
                else:
                    querystring[api_param] = str(value)

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_game_info(self, game_id: str) -> pd.DataFrame:
        """Get detailed game information for a specific game ID."""
        url = f"{self.base_url}/getNBAGameInfo"
        querystring = {"gameID": game_id}

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_betting_odds(self, game_date: str = None, game_id: str = None, 
                        item_format: str = "map") -> pd.DataFrame:
        """Get NBA betting odds for a specific date or game.
        
        Args:
            game_date: Date in YYYYMMDD format (e.g., "20240107"). Optional if game_id is provided.
            game_id: Specific game ID to get odds for. Optional if game_date is provided.
            item_format: Response format - can be "list" or "map". "map" gives games and lines 
                        in map/dictionary format, "list" gives them in list format. Defaults to "map"
        """
        url = f"{self.base_url}/getNBABettingOdds"
        querystring = {"itemFormat": item_format}
        
        if game_date:
            querystring["gameDate"] = game_date
        if game_id:
            querystring["gameID"] = game_id
            
        if not game_date and not game_id:
            raise ValueError("Either game_date or game_id must be provided")

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_teams(self, schedules: bool = False, rosters: bool = False, 
                  stats_to_get: str = "averages", top_performers: bool = True, 
                  team_stats: bool = True) -> pd.DataFrame:
        """Get NBA teams information with optional stats and details.
        
        Args:
            schedules: Set to True to add team schedules to the data returned, defaults to False
            rosters: Set to True to add team rosters to the data returned, defaults to False
            stats_to_get: Only active if rosters=True. Can be "averages" or "totals" to return 
                         player stats as either average per game or total for the season. 
                         Does not apply to team stats (which are always averages), defaults to "averages"
            top_performers: Set to True to add the team's stat leaders to the data returned, defaults to True
            team_stats: Set to True to add team stats to the data returned, defaults to True
        """
        url = f"{self.base_url}/getNBATeams"
        querystring = {
            "schedules": str(schedules).lower(),
            "rosters": str(rosters).lower(),
            "statsToGet": stats_to_get,
            "topPerformers": str(top_performers).lower(),
            "teamStats": str(team_stats).lower()
        }

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_games_for_player(self, player_id: str, game_id: str = None, season: str = None, 
                           number_of_games: str = None, **kwargs) -> pd.DataFrame:
        """Get NBA games and stats for a single player.
        
        Args:
            player_id: Player ID (required)
            game_id: Restrict results to one specific game (increases speed), optional
            season: Season in format YYYY, defaults to current season if gameID is not listed
            number_of_games: Restrict results to the most recent number of games
            **kwargs: Fantasy scoring parameters:
                fantasy_points: Include fantasy points calculation (bool as string)
                pts: Points scoring value (default: 1)
                reb: Rebounds scoring value (default: 1.25) 
                stl: Steals scoring value (default: 3)
                blk: Blocks scoring value (default: 3)
                ast: Assists scoring value (default: 1.5)
                tov: Turnovers scoring value (default: -1)
                mins: Minutes scoring value (default: 0)
                double_double: Double-double bonus (default: 0)
                triple_double: Triple-double bonus (default: 0)
                quad_double: Quad-double bonus (default: 0)
        """
        url = f"{self.base_url}/getNBAGamesForPlayer"
        querystring = {"playerID": player_id}
        
        # Add optional parameters
        if game_id:
            querystring["gameID"] = game_id
        if season:
            querystring["season"] = season
        if number_of_games:
            querystring["numberOfGames"] = number_of_games

        # Map of accepted keyword arguments to their API parameter names
        param_map = {
            "fantasy_points": "fantasyPoints",
            "pts": "pts",
            "stl": "stl",
            "blk": "blk",
            "reb": "reb",
            "ast": "ast",
            "tov": "TOV",
            "mins": "mins",
            "double_double": "doubleDouble",
            "triple_double": "tripleDouble",
            "quad_double": "quadDouble"
        }

        for key, api_param in param_map.items():
            if key in kwargs and kwargs[key] is not None:
                value = kwargs[key]
                # Special handling for fantasy_points to lowercase string
                if key == "fantasy_points":
                    querystring[api_param] = str(value).lower()
                else:
                    querystring[api_param] = str(value)

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_scores_only(self, game_date: str = None, game_id: str = None, 
                       top_performers: bool = True, lineups: bool = True) -> pd.DataFrame:
        """Get NBA daily scoreboard with live, real-time data.
        
        Args:
            game_date: Date in YYYYMMDD format (e.g., "20240108"). Either game_date or game_id is required.
            game_id: Specific game ID. Either game_date or game_id is required.
            top_performers: Set to True to add the game's top performers to each game. 
                           If game is in progress or completed, gives stats for that game.
                           If game is scheduled, gives season average for those players. Defaults to True.
            lineups: Set to True to include lineups. Defaults to True.
        """
        url = f"{self.base_url}/getNBAScoresOnly"
        querystring = {
            "topPerformers": str(top_performers).lower(),
            "lineups": str(lineups).lower()
        }
        
        if game_date:
            querystring["gameDate"] = game_date
        if game_id:
            querystring["gameID"] = game_id
            
        if not game_date and not game_id:
            raise ValueError("Either game_date or game_id must be provided")

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_team_roster(self, team_id: str = None, team_abv: str = None, 
                       archive_date: str = None, stats_to_get: str = "averages") -> pd.DataFrame:
        """Get NBA team roster with player stats.
        
        Args:
            team_id: Team ID (number 1-30). Either team_id or team_abv is required.
            team_abv: Team abbreviation (e.g., "SAC", "CHI", "BOS", "ATL"). Either team_id or team_abv is required.
            archive_date: Date in YYYYMMDD format for historical roster data, optional
            stats_to_get: Type of stats - either "totals" or "averages". 
                         Does not work with archive_date. Defaults to "averages".
        """
        url = f"{self.base_url}/getNBATeamRoster"
        querystring = {"statsToGet": stats_to_get}
        
        if team_id:
            querystring["teamID"] = team_id
        if team_abv:
            querystring["teamAbv"] = team_abv
        if archive_date:
            querystring["archiveDate"] = archive_date
            
        if not team_id and not team_abv:
            raise ValueError("Either team_id or team_abv must be provided")

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_injury_list(self, player_id: str = None, inj_date: str = None, 
                       beginning_inj_date: str = None, end_inj_date: str = None,
                       year: str = None, number_of_days: str = None) -> pd.DataFrame:
        """Get NBA injury list history with various filtering options.
        
        Args:
            player_id: Numerical player ID to restrict results to a specific player only, optional
            inj_date: Date in YYYYMMDD format to restrict results to only one date, optional
            beginning_inj_date: Date in YYYYMMDD format for start of date range (inclusive), optional
            end_inj_date: Date in YYYYMMDD format for end of date range (inclusive), optional
            year: Year in YYYY format to restrict results to one year, optional
            number_of_days: Number of days (1-30) to restrict results to most recent days, optional
        """
        url = f"{self.base_url}/getNBAInjuryList"
        querystring = {}
        
        if player_id:
            querystring["playerID"] = player_id
        if inj_date:
            querystring["injDate"] = inj_date
        if beginning_inj_date:
            querystring["beginningInjDate"] = beginning_inj_date
        if end_inj_date:
            querystring["endInjDate"] = end_inj_date
        if year:
            querystring["year"] = year
        if number_of_days:
            querystring["numberOfDays"] = number_of_days

        response = requests.get(url, headers=self.headers, params=querystring or None)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_player_info(self, player_id: str = None, player_name: str = None, 
                       stats_to_get: str = "averages") -> pd.DataFrame:
        """Get NBA player information and stats.
        
        Args:
            player_id: Numerical player ID. Either player_id or player_name is required.
                      If player_id is used, returns a map (single player result).
            player_name: Player name to search for. Either player_id or player_name is required.
                        If player_name is used, returns a list of players that match the name.
            stats_to_get: Type of stats - either "totals" or "averages". 
                         Works for current season only. Defaults to "averages".
        """
        url = f"{self.base_url}/getNBAPlayerInfo"
        querystring = {"statsToGet": stats_to_get}
        
        if player_id:
            querystring["playerID"] = player_id
        if player_name:
            querystring["playerName"] = player_name
            
        if not player_id and not player_name:
            raise ValueError("Either player_id or player_name must be provided")

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_player_list(self) -> pd.DataFrame:
        """Get complete list of all NBA players.
        
        Returns:
            DataFrame containing all NBA players in the system.
        """
        url = f"{self.base_url}/getNBAPlayerList"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_team_schedule(self, team_id: str = None, team_abv: str = None, 
                         season: str = "2025") -> pd.DataFrame:
        """Get NBA team schedule for a specific season.
        
        Args:
            team_id: Team ID (1-30). Either team_id or team_abv is required.
            team_abv: Team abbreviation (e.g., "DEN", "CHI", "BOS", "ATL"). Either team_id or team_abv is required.
            season: Season year in YYYY format. For NBA season 2023-24, use "2024". 
                   Only works for seasons 2022 and future. Defaults to "2025".
        """
        url = f"{self.base_url}/getNBATeamSchedule"
        querystring = {"season": season}
        
        if team_id:
            querystring["teamID"] = team_id
        if team_abv:
            querystring["teamAbv"] = team_abv
            
        if not team_id and not team_abv:
            raise ValueError("Either team_id or team_abv must be provided")

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_dfs_salaries(self, date: str) -> pd.DataFrame:
        """Get NBA DFS (Daily Fantasy Sports) salaries for a specific date.
        
        Args:
            date: Date in YYYYMMDD format (e.g., "20250120") - required
        """
        url = f"{self.base_url}/getNBADFS"
        querystring = {"date": date}

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_adp(self, adp_date: str = None, combine_guards: bool = False, 
               combine_forwards: bool = False, combine_fc: bool = False) -> pd.DataFrame:
        """Get NBA ADP (Average Draft Position) data.
        
        Args:
            adp_date: Date for historical ADP data. ADP goes back to 20240924. Optional, defaults to current.
            combine_guards: Set to True to remove PG and SG and designate every guard as G. Defaults to False.
            combine_forwards: Set to True to remove SF and PF and designate every forward as F. Defaults to False.
            combine_fc: Set to True to remove SF, PF, and C, and designate every forward and center as FC. Defaults to False.
        """
        url = f"{self.base_url}/getNBAADP"
        querystring = {}
        
        if adp_date:
            querystring["adpDate"] = adp_date
        if combine_guards:
            querystring["combineGuards"] = str(combine_guards).lower()
        if combine_forwards:
            querystring["combineForwards"] = str(combine_forwards).lower()
        if combine_fc:
            querystring["combineFC"] = str(combine_fc).lower()

        response = requests.get(url, headers=self.headers, params=querystring or None)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_depth_charts(self) -> pd.DataFrame:
        """Get NBA depth charts for all teams.
        
        Returns:
            DataFrame containing depth chart information showing player positions 
            and depth rankings for each NBA team.
        """
        url = f"{self.base_url}/getNBADepthCharts"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_news(self, player_id: str = None, team_id: str = None, team_abv: str = None,
                top_news: bool = False, recent_news: bool = False, fantasy_news: bool = False,
                max_items: str = "10") -> pd.DataFrame:
        """Get NBA news and headlines with various filtering options.
        
        Args:
            player_id: Player ID to get news for specific player, optional
            team_id: Team ID (integer) to get news for specific team, optional
            team_abv: Team abbreviation to get news for specific team, optional
            top_news: Set to True to get top news stories, defaults to False
            recent_news: Set to True to get recent news stories, defaults to False
            fantasy_news: Set to True to get fantasy-relevant news, defaults to False
            max_items: Maximum number of news items to return, defaults to "10"
        """
        url = f"{self.base_url}/getNBANews"
        querystring = {"maxItems": max_items}
        
        if player_id:
            querystring["playerID"] = player_id
        if team_id:
            querystring["teamID"] = team_id
        if team_abv:
            querystring["teamAbv"] = team_abv
        if top_news:
            querystring["topNews"] = str(top_news).lower()
        if recent_news:
            querystring["recentNews"] = str(recent_news).lower()
        if fantasy_news:
            querystring["fantasyNews"] = str(fantasy_news).lower()

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def get_projections(self, num_of_days: str = "7", **kwargs) -> pd.DataFrame:
        """Get NBA fantasy point projections with customizable scoring parameters.
        
        Args:
            num_of_days: Number of days for projections - can be "7" or "14", defaults to "7"
            **kwargs: Fantasy scoring parameters:
                pts: Points scoring value (default: 1)
                reb: Rebounds scoring value (default: 1.25)
                tov: Turnovers scoring value (default: -1)
                stl: Steals scoring value (default: 3)
                blk: Blocks scoring value (default: 3)
                ast: Assists scoring value (default: 1.5)
                mins: Minutes scoring value (default: 0)
        """
        url = f"{self.base_url}/getNBAProjections"
        querystring = {"numOfDays": num_of_days}
        
        # Map of accepted keyword arguments to their API parameter names
        param_map = {
            "pts": "pts",
            "reb": "reb",
            "tov": "TOV",
            "stl": "stl",
            "blk": "blk",
            "ast": "ast",
            "mins": "mins"
        }

        for key, api_param in param_map.items():
            if key in kwargs and kwargs[key] is not None:
                querystring[api_param] = str(kwargs[key])

        response = requests.get(url, headers=self.headers, params=querystring)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame.from_dict(data['body'])

    # Wrapper methods for CSV collection compatibility
    def get_injuries_for_date(self, date_str: str) -> pd.DataFrame:
        """Wrapper method to get injuries for a specific date.

        Args:
            date_str: Date in YYYYMMDD format
        """
        return self.get_injury_list(inj_date=date_str)

    def get_dfs_salaries_for_date(self, date_str: str) -> pd.DataFrame:
        """Wrapper method to get DFS salaries for a specific date.

        Args:
            date_str: Date in YYYYMMDD format
        """
        return self.get_dfs_salaries(date=date_str)

    def get_betting_odds_for_date(self, date_str: str) -> pd.DataFrame:
        """Wrapper method to get betting odds for a specific date.

        Args:
            date_str: Date in YYYYMMDD format
        """
        return self.get_betting_odds(game_date=date_str)

    def get_player_game_logs(self, game_id: str) -> pd.DataFrame:
        """Wrapper method to get player game logs (box score data) for a specific game.

        Args:
            game_id: Game ID to get player stats for
        """
        return self.get_box_score(game_id, fantasy_points=True)