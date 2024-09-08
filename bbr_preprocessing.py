import pandas as pd
import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup

# Read in preprocessed file from 1982-2023, this will always be the same
data_1982_to_2023 = pd.read_csv('FINAL_bbr_1982_to_2023.csv')

# Scrape BBR website to get per game data for 2024
pg_url = "https://www.basketball-reference.com/leagues/NBA_2024_per_game.html"
pg_html = urlopen(pg_url)
pg_soup = BeautifulSoup(pg_html, features="lxml")

pg_headers = [th.getText() for th in pg_soup.findAll('tr', limit=2)[0].findAll('th')]
pg_headers.remove("Rk")

pg_rows = pg_soup.findAll('tr')[1:]
pg_rows_data = [[td.getText() for td in pg_rows[i].findAll('td')] for i in range(len(pg_rows))]

per_game_2024 = pd.DataFrame(pg_rows_data, columns=pg_headers)

# Scrape BBR website to get advanced data for 2024
adv_url = "https://www.basketball-reference.com/leagues/NBA_2024_advanced.html"
adv_html = urlopen(adv_url)
adv_soup = BeautifulSoup(adv_html, features="lxml")

adv_headers = [th.getText() for th in adv_soup.findAll('tr', limit=2)[0].findAll('th')]
adv_headers.remove("Rk")

adv_rows = adv_soup.findAll('tr')[1:]
adv_rows_data = [[td.getText() for td in adv_rows[i].findAll('td')] for i in range(len(adv_rows))]

adv_2024 = pd.DataFrame(adv_rows_data, columns=adv_headers)

per_game_mappings = {"MP":"mp_per_g", "FG":"fg_per_g", "FGA":"fga_per_g", "FG%":"fg_pct", "3P":"fg3_per_g",
                     "3PA":"fg3a_per_g", "3P%":"fg3_pct", "2P":"fg2_per_g", "2PA":"fg2a_per_g", "2P%":"fg2_pct",
                     "eFG%":"efg_pct", "FT":"ft_per_g", "FTA":"fta_per_g", "FT%":"ft_pct", "ORB":"orb_per_g",
                     "DRB":"drb_per_g", "TRB":"trb_per_g", "AST":"ast_per_g", "STL":"stl_per_g", "BLK":"blk_per_g",
                     "TOV":"tov_per_g", "PF":"pf_per_g", "PTS":"pts_per_g", "Player":"player", "Pos":"pos",
                     "Age":"age", "Tm":"team_id", "G":"g", "GS":"gs"}
advanced_mappings = {"Player":"player", "Pos":"pos", "Age":"age", "Tm":"team_id",
            "G":"g", "GS":"gs" ,"MP":"mp", "PER":"per", "TS%":"ts_pct", "3PAr":"fg3a_per_fga_pct", "FTr":"fta_per_fga_pct",
            "ORB%":"orb_pct", "DRB%":"drb_pct", "TRB%":"trb_pct", "AST%":"ast_pct", "STL%":"stl_pct", "BLK%":"blk_pct",
            "TOV%":"tov_pct", "USG%":"usg_pct", "OWS":"ows", "DWS":"dws", "WS":"ws", "WS/48":"ws_per_48", "OBPM":"obpm",
            "DBPM":"dbpm", "BPM":"bpm", "VORP":"vorp"}

# Rename all columns
per_game_2024 = per_game_2024.rename(columns=per_game_mappings)
adv_2024 = adv_2024.rename(columns=advanced_mappings)

# Merge per game and advanced data together
data_2024 = pd.merge(per_game_2024, adv_2024, how='left')
data_2024['season'] = 2024

# Keep only TOT rows for players with multiple rows
player_counts = data_2024.groupby('player').size()
multi_row_players = player_counts[player_counts > 1].index
multi_row_tot_rows = data_2024[(data_2024['player'].isin(multi_row_players)) & (data_2024['team_id'] == 'TOT')]
single_row_players = player_counts[player_counts == 1].index
single_row_data = data_2024[data_2024['player'].isin(single_row_players)]
data_2024 = pd.concat([multi_row_tot_rows, single_row_data])

# Drop empty rows
data_2024.drop(columns=data_2024.columns[47], inplace=True)
data_2024.drop(columns=data_2024.columns[42], inplace=True)

data_2024 = data_2024.replace('', np.nan)

columns_to_float64 = ['mp_per_g', 'fg_per_g', 'fga_per_g', 'fg_pct', 'fg3_per_g', 'fg3a_per_g', 'fg3_pct',
                      'fg2_per_g', 'fg2a_per_g', 'fg2_pct', 'efg_pct', 'ft_per_g','fta_per_g', 'ft_pct',
                      'orb_per_g', 'drb_per_g', 'trb_per_g','ast_per_g', 'stl_per_g', 'blk_per_g',
                      'tov_per_g', 'pf_per_g','pts_per_g', 'per', 'ts_pct', 'fg3a_per_fga_pct','fta_per_fga_pct',
                      'orb_pct', 'drb_pct', 'trb_pct', 'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct',
                      'dws', 'ws', 'ws_per_48', 'obpm', 'dbpm', 'bpm', 'vorp']
columns_to_int64 = ['age', 'g', 'gs', 'mp']

data_2024[columns_to_float64] = data_2024[columns_to_float64].astype('float64')
data_2024[columns_to_int64] = data_2024[columns_to_int64].astype('int64')

nba_data = pd.merge(data_2024, data_1982_to_2023, how='outer')

min_season_dict = nba_data.groupby('player')['season'].min().to_dict()

def calculate_experience(row):
    return row['season'] - min_season_dict[row['player']] + 1

nba_data['experience'] = nba_data.apply(calculate_experience, axis=1)

nba_data.to_csv('bbr_preprocessed.csv')