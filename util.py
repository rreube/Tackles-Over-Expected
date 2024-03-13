from scipy.spatial import distance
import numpy as np

def calc_distance(player_loc):
    '''
    Calculate distance between defender and designated ball carrier
    Input: Dataframe of individual play with position data of all 11 defenders
    Output: Series of distance calculations for 11 defenders at time of handoff/pass arrival
    '''
    temp_bc_x = player_loc.bc_x.max()
    temp_bc_y = player_loc.bc_y.max()
    
    return distance.cdist(player_loc[['x', 'y']], np.array([[temp_bc_x, temp_bc_y]]))

def calc_distance_rank(player_loc):
    '''
    Calculate distance rank among 11 defenders on each play
    Input: Dataframe of individual play with position data of 11 defenders
    Output: Series of ranks based on proximity to ball carrier for each defender
    '''
    return player_loc.dist_to_bc.rank().astype(int)

def calc_avg_bc_dist(player_loc):
    '''
    Calculate average distance to ball carrier among defenders
    Input: Dataframe of individual defender's tracking data
    Output: Series with avg. distance to ball carrier
    '''
    return player_loc.dist_to_bc.median()

def defender_in_front(play_direc, player_loc):
    '''
    Flag for whether defender was in front or behind ball carrier
    Input: play direction, dataframe of defender locations
    Output: Series of positional flags for each defender
    '''
    if play_direc == 'left':
        return (player_loc.x < player_loc.bc_x).astype(int)
    else:
        return (player_loc.x > player_loc.bc_x).astype(int)
    
def calc_rel_angle(player_loc):
    '''
    Calculate difference between player movement angle and ball carrier movement angle
    Input: dataframe of defender locations/speeds
    Output: Series of defender angles to ball carrier
    '''
    # Calculate relative angle
    theta_temp = player_loc.dir - player_loc.bc_dir
    return np.where(theta_temp.between(-180, 180), np.abs(theta_temp), 360 - np.abs(theta_temp))
    
def calc_rel_speed(player_loc):
    '''
    Calculate relative difference between ball carrier and defender speeds
    Input: dataframe of defender locations/speeds
    Output: Series of defender speeds (relative to ball carrier)
    '''
    # Law of cosines
    return np.sqrt(
            np.power(player_loc.bc_s, 2)
            + np.power(player_loc.s, 2)
            - 2 * player_loc.bc_s * player_loc.s * np.cos(np.pi / 180 * player_loc.rel_angle))

def sideline_dist(player_loc):
    '''
    Calculate ball carrier minimum distance (absolute) to sideline (0, 53.3)
    Input: Dataframe of defenseive player tracking data
    Output: Series of constant min distance to sideline values
    '''
    bc_y = player_loc.bc_y.unique()

    return float(min(bc_y - 0, 53.3 - bc_y))

def endzone_dist(play_direc, player_loc):
    '''
    Calculate ball carrier distance to endzone
    Input: Dataframe of defenseive player tracking data
    Output: Series of constant min distance to endzone values
    '''
    if play_direc == 'left' and player_loc.bc_x.unique()[0] <= 10:
        return player_loc.bc_x * 0
    elif play_direc == 'left' and player_loc.bc_x.unique()[0] > 10:
        return player_loc.bc_x - 10
    elif play_direc == 'right' and player_loc.bc_x.unique()[0] >= 110:
        return player_loc.bc_x * 0
    else:
        return 110 - player_loc.bc_x

def is_dlineman(player_loc):
    '''
    Flag for if defender is DE, DT, NT
    '''
    return player_loc.position.isin(['DE', 'DT', 'NT']).astype(int)

def is_linebacker(player_loc):
    '''
    Flag for if defender is ILB, OLB, MLB
    '''
    return player_loc.position.isin(['ILB', 'OLB', 'MLB']).astype(int)

def is_secondary(player_loc):
    '''
    Flag for if defender is CB, FS, SS, DB
    '''
    return player_loc.position.isin(['CB', 'FS', 'SS', 'DB']).astype(int)

def is_pass(player_loc):
    '''
    Flag for if play is pass
    '''
    return player_loc.event.isin(['pass_arrived']).astype(int)

def is_rush(player_loc):
    '''
    Flag for if play is handoff/rush
    '''
    return player_loc.event.isin(['handoff', 'run']).astype(int)

def is_bc_wr(player_loc):
    '''
    Flag for if ballcarrier is WR
    '''
    return player_loc.bc_pos.isin(['WR']).astype(int)

def is_bc_te(player_loc):
    '''
    Flag for if ballcarrier is TE
    '''
    return player_loc.bc_pos.isin(['TE']).astype(int)

def is_bc_rb(player_loc):
    '''
    Flag for if ballcarrier is RB
    '''
    return player_loc.bc_pos.isin(['RB', 'FB']).astype(int)

def is_bc_qb(player_loc):
    '''
    Flag for if ballcarrier is QB
    '''
    return player_loc.bc_pos.isin(['QB']).astype(int)