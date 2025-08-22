import pandas as pd



def load_season_csv(url, season):

    df = pd.read_csv(url)

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    df['season'] = season
    
    return df

    

def build_master_df(seasons, save_path="./data/all_seasons_data.csv"):
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