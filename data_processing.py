import pandas as pd
import os
from util import calc_distance, calc_distance_rank, sideline_dist, defender_in_front, calc_avg_bc_dist, calc_rel_angle, calc_rel_speed, is_dlineman, is_linebacker, is_secondary, is_pass, is_rush, is_bc_wr, is_bc_te, is_bc_rb, is_bc_qb, endzone_dist

# Load dataset from "data" folder

# Get current working directory
cwd = os.getcwd()

# Load data
games = pd.read_csv(cwd + '\data' + '\\games.csv')
players = pd.read_csv(cwd + '\data' + '\\players.csv')
plays = pd.read_csv(cwd + '\data' + '\\plays.csv')
tackles = pd.read_csv(cwd + '\data' + '\\tackles.csv')

tracking_weekly = []
for i in range(1, 10):
    tracking_weekly.append(pd.read_csv(cwd + '\data' + '\\tracking_week_' + str(i) + '.csv'))
tracking_conc = pd.concat(tracking_weekly)

# Data Cleaning

# Remove plays nullified by penalties
official_plays = plays[plays.playNullifiedByPenalty=='N'].copy()
# Make ball carrier column in official plays 
official_plays['ball_carrier'] = 1
# Collect frames where event identifies player to be tackled (handoff, run, pass_arrived)
tracking_filter = tracking_conc[tracking_conc.event.isin(['handoff', 'run', 'pass_arrived'])].copy()
# In plays with multiple handoffs/passes select the final one
filter_mask = tracking_filter.groupby(['gameId', 'playId']).frameId.max()
tracking_filter = tracking_filter.merge(filter_mask, left_on=['gameId', 'playId', 'frameId'], right_on=['gameId', 'playId', 'frameId']).copy()
# Add tackle participation column to tackles dataframe for players who participated in a tackle
tackles['tackle_participant'] = 0
tackles.tackle_participant[(tackles.tackle==1) | (tackles.assist==1)] = 1

# Feature Engineering

# Merge weekly tracking data and tackles
tracking_merge_plays = tracking_filter.merge(official_plays.loc[:, ['gameId', 'playId','ballCarrierId','ball_carrier']].rename(columns={'ballCarrierId':'nflId'}), how='left')
tracking_merge_tackles = tracking_merge_plays.merge(tackles.loc[:, ['gameId', 'playId', 'nflId', 'tackle_participant']], how='left')

# Fill the tackle column with 0 and change to int 
tracking_merge_tackles['tackle_participant'] = tracking_merge_tackles['tackle_participant'].fillna(0).astype(int)
tracking_merge_tackles['ball_carrier'] = tracking_merge_tackles['ball_carrier'].fillna(0).astype(int)
                              
# Merge players 
tracking_merge_players = tracking_merge_tackles.merge(players.drop(columns=['displayName'], axis=1), how='left')

# Define whether a team was on offense or defense 
_p = official_plays.loc[:, ['gameId', 'playId', 'defensiveTeam']].rename(columns={'defensiveTeam':'club'}).copy()
_p['on_defense'] = 1
tracking_merge_def = tracking_merge_players.merge(_p, how='left')
tracking_merge_def['on_defense'] = tracking_merge_def['on_defense'].fillna(0).astype(int)

# Subset only players on defense
_def = tracking_merge_def.loc[tracking_merge_def['on_defense']==1].copy()

# Subset only ball carriers 
_bc = tracking_merge_def.loc[tracking_merge_def['ball_carrier']==1, ['gameId', 'playId', 'frameId', 'position', 'x', 'y', 's', 'dir']].rename(columns={'position': 'bc_pos', 'x':'bc_x','y':'bc_y','s':'bc_s', 'dir':'bc_dir'}).copy()

# Merge players on defense and the ball carrier dataframes so one row has information about both a defense player and the ball carrier 
tracking_final = _def.merge(_bc)

data_processed = pd.DataFrame()
# Iterate through each game
for game_id in tracking_final.gameId.unique():
    # Iterate through each play in a game
    for play_id in tracking_final[tracking_final.gameId==game_id].playId.unique():
        # Select current play given specific game and play id
        current_play = tracking_final[(tracking_final.gameId==game_id) & (tracking_final.playId==play_id)].copy()
        # Identify current play direction on field
        temp_direc = current_play.playDirection.unique()[0]
        # Apply util functions to add contextual features
        current_play['dist_to_bc'] = calc_distance(current_play)
        current_play['dist_to_bc_avg'] = calc_avg_bc_dist(current_play)
        current_play['dist_rank'] = calc_distance_rank(current_play)
        current_play['defender_in_front'] = defender_in_front(temp_direc, current_play)
        current_play['sideline_dist'] = sideline_dist(current_play)
        current_play['endzone_dist'] = endzone_dist(temp_direc, current_play)
        current_play['rel_angle'] = calc_rel_angle(current_play)
        current_play['rel_speed'] = calc_rel_speed(current_play)
        current_play['is_dlineman'] = is_dlineman(current_play)
        current_play['is_linebacker'] = is_linebacker(current_play)
        current_play['is_secondary'] = is_secondary(current_play)
        current_play['is_pass'] = is_pass(current_play)
        current_play['is_rush'] = is_rush(current_play)
        current_play['is_bc_wr'] = is_bc_wr(current_play)
        current_play['is_bc_te'] = is_bc_te(current_play)
        current_play['is_bc_rb'] = is_bc_rb(current_play)
        current_play['is_bc_qb'] = is_bc_qb(current_play)
        # Add each plays new features to a common dataframe
        data_processed = pd.concat([data_processed,current_play])

# Save final dataframe as CSV file        
data_processed.to_csv('processed_data')
