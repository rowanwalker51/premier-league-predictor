import pandas as pd
import numpy as np



def load_season_csv(url: str,
                    season: str) -> pd.DataFrame:
    """
    Reads data from saved location per season.
    
    Parameters:
    ----------
    url : str
        Saved data location
    season : str
        Season in "YYYY-YY" format
    
    Returns:
    -------
    df : pd.DataFrame
        Output DataFrame.
    """
    df = pd.read_csv(url)

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    df['season'] = season
    
    return df

    

def build_master_df(seasons: dict,
                    save_path: str = "../data/all_seasons_data.csv"):
    """
    Loops through relevant seasons and concatenates data.
    
    Parameters:
    ----------
    seasons : dict
        Seasons dictionary with saved locations
    save_path : str
        Location to save total dataset
    
    Returns:
    -------
    master : pd.DataFrame
        Output final DataFrame.
    """
    all_seasons = []
    
    for season, src in seasons.items():
        print(f"Loading {season}...")
        df_season = load_season_csv(src, season)
        all_seasons.append(df_season)
    
    master = pd.concat(all_seasons, ignore_index=True)
    
    master = master.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'])
    
    master = master.sort_values('Date').reset_index(drop=True)
    
    # Save
    master.to_csv(save_path, index=False)
    print(f"Saved master dataset to {save_path} (Total matches: {len(master)}).")
    
    return master



def calculate_elo(df: pd.DataFrame, 
                  k_factor: int = 20, 
                  home_adv: int = 100, 
                  start_rating: int = 1500) -> (dict, pd.DataFrame):
    """
    Calculate Elo ratings.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame.
    k_factor : int
        Elo K-factor (higher = ratings change faster).
    home_adv : int
        Elo home advantage in points.
    start_rating : int
        Starting rating for teams not yet seen.
    
    Returns:
    -------
    final_ratings : dict
        Dictionary of final Elo ratings per team.
    elo_history : pd.DataFrame
        DataFrame with Elo ratings before and after each match.
    """

    # Load data
    df = df[['date', 'home_team', 'away_team', 'home_goals', 'away_goals']]

    # Ratings store
    elo = {}

    def get_elo(team):
        return elo.get(team, start_rating)

    def expected_score(rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    elo_history = []

    for _, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        home_goals, away_goals = row['home_goals'], row['away_goals']

        home_rating = get_elo(home)
        away_rating = get_elo(away)

        # Expected scores with home advantage
        exp_home = expected_score(home_rating + home_adv, away_rating)
        exp_away = 1 - exp_home

        # Match result
        if home_goals > away_goals:
            score_home, score_away = 1, 0
        elif home_goals < away_goals:
            score_home, score_away = 0, 1
        else:
            score_home, score_away = 0.5, 0.5

        # Update ratings
        home_new = home_rating + k_factor * (score_home - exp_home)
        away_new = away_rating + k_factor * (score_away - exp_away)

        elo[home] = home_new
        elo[away] = away_new

        # Save match history
        elo_history.append({
            'date': row['date'],
            'home_team': home,
            'away_team': away,
            'home_elo_before': home_rating,
            'away_elo_before': away_rating,
            'home_elo_after': home_new,
            'away_elo_after': away_new,
            'home_goals': home_goals,
            'away_goals': away_goals
        })

    return elo, pd.DataFrame(elo_history)



def add_rest_days(df: pd.DataFrame,
                  cap: int = 14) -> pd.DataFrame:
    '''
    Adds rest days for home and away teams.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame.
    clipped : int
        Upper value to cap the rest days given diminishing returns (default 14)
    '''
    rested_df = pd.DataFrame(columns=['home_team', 'away_team', 'days_rest_home_team', 'days_rest_away_team'])
    for team in df['home_team'].unique():
        temp_df = df.loc[(df['home_team'] == team) | 
                         (df['away_team'] == team)]
        temp_df.sort_values(by='date', inplace=True)
        home = temp_df['home_team'] == team
        away = temp_df['away_team'] == team
        temp_df["days_rested"] = temp_df['date'].diff().dt.days.clip(upper=cap)
        temp_df['days_rest_home_team'] = np.where(home, temp_df["days_rested"], np.nan)
        temp_df['days_rest_away_team'] = np.where(away, temp_df["days_rested"], np.nan)
        rested_df = pd.concat([rested_df, temp_df])

    rested_df_dropped = pd.DataFrame()
    rested_df_dropped['days_rest_home_team'] = rested_df.groupby(['home_team', 'away_team', 'date'])['days_rest_home_team'].max().reset_index(drop=True)
    rested_df_dropped['days_rest_away_team'] = rested_df.groupby(['home_team', 'away_team', 'date'])['days_rest_away_team'].max().reset_index(drop=True)
    df = df.sort_values(by=['home_team', 'away_team']).reset_index(drop=True)
    df = pd.merge(df, rested_df_dropped, right_index=True, left_index=True).sort_values(by='date').reset_index(drop=True)

    return df