# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:19:41 2022

@author: andub
"""

import numpy as np
from multiprocessing import Pool
import os
import copy
import traceback


# There are three folders
folders = ['Centralised', 'Decentralised', 'Hybrid']

# Get the files in each of these folders
centr_regs = [x for x in os.listdir(folders[0]) if 'REGLOG' in x]
decentr_regs = [x for x in os.listdir(folders[1]) if 'REGLOG' in x]
hybrid_regs = [x for x in os.listdir(folders[2]) if 'REGLOG' in x]

# Create input array
input_arr = []
for scen in centr_regs:
    input_arr.append([folders[0], scen])
    
for scen in decentr_regs:
    input_arr.append([folders[1], scen])
    
for scen in hybrid_regs:
    input_arr.append([folders[2], scen])
    

# We only need to feed one log, and the rest will have the same name except for 
# the name of the actual log. So just filter out the REGLOGS and feed them to
# the pool.

# Make a function that accepts concept + log file as an input

CENTER_LAT = 48.20499787612939
CENTER_LON = 16.362249993868282

def kwikdist_matrix(lata, lona, latb, lonb):
    """
    Quick and dirty dist [nm]
    In:
        lat/lon, lat/lon vectors [deg]
    Out:
        dist vector [nm]
    """
    re      = 6371000.  # readius earth [m]
    dlat    = np.radians(latb - lata.T)
    dlon    = np.radians(((lonb - lona.T)+180)%360-180)
    cavelat = np.cos(np.radians(lata + latb.T) * 0.5)

    dangle  = np.sqrt(np.multiply(dlat, dlat) +
                      np.multiply(np.multiply(dlon, dlon),
                                  np.multiply(cavelat, cavelat)))
    dist    = re * dangle
    return dist

def process_logs(args):
    concept = args[0]
    reg_log_file = args[1]
    # Get the name of the file so we can load other logs from it as well
    log_file = reg_log_file.replace('REGLOG_', '')
    # Get the names of the other logs
    flst_log_file = 'FLSTLOG_' + log_file
    conf_log_file = 'CONFLOG_' + log_file
    los_log_file = 'LOSLOG_' + log_file
    geo_log_file = 'GEOLOG_' + log_file
    
    # Let's load this guy's flight intention file
    uncertain = ['W1', 'W3', 'W5', 'R1', 'R2', 'R3']
    if any([x in log_file for x in uncertain]):
        intention_file = log_file[:-25] + '.csv'
    else:
        intention_file = log_file[:-22] + '.csv'
    intention_data = np.genfromtxt('Intentions/' + intention_file, dtype = str, delimiter = ',')
    
    # Create a new array of origin destination data for aircraft
    orig_dest_data = np.zeros((len(intention_data), 4))
    
    for i, ac_data in enumerate(intention_data):
        # The flight number is just the row index + 1, so we don't need that info
        #origin lat
        orig_dest_data[i,0] = float(ac_data[5].replace(')"', ''))
        #origin lon
        orig_dest_data[i,1] = float(ac_data[4].replace('"(', ''))
        #dest lat
        orig_dest_data[i,2] = float(ac_data[7].replace(')"', ''))
        #dest lon
        orig_dest_data[i,3] = float(ac_data[6].replace('"(', ''))
        
    # Delete the data
    #del intention_data
    
    # Load the reglog so we check for drifting aircraft or aircraft sitting on
    # top of destination
    with open(concept + '/' + reg_log_file) as f:
        # Read the lines
        reg_lines = f.readlines()
        
    # Ok so now we need to go time step by time step, check if aircraft are 
    # close to their destinations, and remember them
    bouncy_ac_dict = dict()
    far_ac_dict = dict()
    idx = 9
    time = 30
    for acid_line in reg_lines[9::4]:   
        if ',' not in acid_line:
            continue
        # Get the ACIDs, lats, lons as numpy arrays
        acid = np.array(acid_line.split(','))[1:]
        try:
            now_lats = np.array(reg_lines[idx+2].split(','), dtype = float)[1:]
            now_lons = np.array(reg_lines[idx+3].split(','), dtype = float)[1:]
        except:
            # Scenario is missing some lines, eh
            continue
                
        # Find if there are rogues, and delete em
        rogue_locations = np.where(np.logical_or.reduce((acid == 'R0', acid == 'R1', acid == 'R2')))[0]

        if rogue_locations.size > 0:
            acid = np.delete(acid, rogue_locations)
            now_lats = np.delete(now_lats, rogue_locations)
            now_lons = np.delete(now_lons, rogue_locations)
                
        # Convert all those ACIDs into ACIDX
        acidx = np.char.replace(acid, 'D', '')
        # Convert these to int
        acidx = acidx.astype(np.int32)-1
        
        # Get the lats and lons of the destinations
        dest_lats = orig_dest_data[:,2][acidx]
        dest_lons = orig_dest_data[:,3][acidx]
        
        # Get the distances
        current_distances = kwikdist_matrix(now_lats, now_lons, dest_lats, dest_lons)
        
        # Get the index where current distances is smaller than 5m
        ac_strikes = np.where(current_distances < 5)[0]
        
        # Also get the indexes where aircraft are too far from the centre
        center_lats = CENTER_LAT + np.zeros(len(now_lats))
        center_lons = CENTER_LON + np.zeros(len(now_lats))
        dist_from_centre = kwikdist_matrix(now_lats, now_lons, center_lats, center_lons)
        
        # Get where aircraft are outside airspace
        far_away = np.where(dist_from_centre > 8500)[0]
        
        if ac_strikes.size>0:
            # Go through each entry, add number of strikes and times
            for num in ac_strikes:
                ac = acidx[num]
                if ac not in bouncy_ac_dict:
                    bouncy_ac_dict[ac] = [0]
                bouncy_ac_dict[ac][0] += 1
                bouncy_ac_dict[ac].append(time)
                
        if far_away.size>0:
            # Go through each entry, add time
            for num in far_away:
                ac = acidx[num]
                if ac not in far_ac_dict:
                    far_ac_dict[ac] = []
                far_ac_dict[ac]
                
        idx += 4
        time += 30
        
    # Filter out aircraft in ac_dict that do not have more than 4 entries
    for ac in copy.copy(bouncy_ac_dict):
        if bouncy_ac_dict[ac][0] < 3:
            bouncy_ac_dict.pop(ac)
        
    # We need to delete the aircraft in the dicts in all the other logs after
    # the second time in their entries
    
    # Return the amount of aircraft
    print(f'Scenario {log_file} of {concept[0]} had {len(bouncy_ac_dict)} bouncies and {len(far_ac_dict)} far aways.')
    return
    
                
def main():
    # process_logs(input_arr[326])
    # return
    # create_dill(input_arr[0])
    pool = Pool(4)
    try:
        _ = pool.map(process_logs, input_arr)
    except Exception as err:
        pool.close()
        print(traceback.format_exc())
        print(err)
    pool.close()
    
if __name__ == '__main__':
    main()