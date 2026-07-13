#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:33:37 2026

@author: kellymckeon
"""

import time
code_start = time.time() #begin timing code

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 
import random
import networkx as nx          # graph library: nodes = deposits, edges = overlaps
from collections import defaultdict  # dictionary that auto-initializes missing keys
rng = np.random.default_rng()

mac = r"/Users/kellymckeon/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Bahamas/Codes/For_Publication"
pc = r"C:/Users/kelly/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Bahamas/Codes/For_Publication"

# CHANGE DIRECTORY TO WHICHEVER COMPUTER YOU ARE ON
os.chdir(mac)

#%% Load Age Data

#### AM2
#median ages
AM2_ages_med = np.genfromtxt(r"AM2_CoreData/median_ages.csv", delimiter=",", skip_header=0)
AM2_event_times = np.genfromtxt(r"AM2_CoreData/event_times.csv", delimiter=",", skip_header=0)
AM2_event_times_pub = np.genfromtxt(r"AM2_CoreData/event_times_pub.csv", delimiter=",", skip_header=0)
#distributions
age_dist = np.genfromtxt(r"AM2_CoreData/AM2_Age_Distributions.csv", delimiter=",", skip_header=1)
age_dist = age_dist[:,1:]
AM2_age_dist = 1950 - age_dist
#annual years
AM2_allyears = np.genfromtxt(r"AM2_CoreData/all_years.csv", delimiter=",", skip_header=0)

#################################################

#### AM4
#median ages
AM4_ages_med = np.genfromtxt(r"AM4_CoreData/median_ages.csv", delimiter=",", skip_header=0)
AM4_event_times = np.genfromtxt(r"AM4_CoreData/event_times.csv", delimiter=",", skip_header=0)
AM4_event_times_pub = np.genfromtxt(r"AM4_CoreData/event_times_pub.csv", delimiter=",", skip_header=0)
#distributions
age_dist = np.genfromtxt(r"AM4_CoreData/AM4_Age_Distributions.csv", delimiter=",", skip_header=1)
age_dist = age_dist[:,1:]
AM4_age_dist = 1950 - age_dist
#annual years
AM4_allyears = np.genfromtxt(r"AM4_CoreData/all_years.csv", delimiter=",", skip_header=0)

#################################################

#### AM5
#median ages
AM5_ages_med = np.genfromtxt(r"AM5_CoreData/median_ages.csv", delimiter=",", skip_header=0)
AM5_event_times = np.genfromtxt(r"AM5_CoreData/event_times.csv", delimiter=",", skip_header=0)
AM5_event_times_pub = np.genfromtxt(r"AM5_CoreData/event_times_pub.csv", delimiter=",", skip_header=0)
#distributions
age_dist = np.genfromtxt(r"AM5_CoreData/AM5_Age_Distributions.csv", delimiter=",", skip_header=1)
age_dist = age_dist[:,1:]
AM5_age_dist = 1950 - age_dist
#annual years
AM5_allyears = np.genfromtxt(r"AM5_CoreData/all_years.csv", delimiter=",", skip_header=0)

#################################################

#### AM7
#median ages
AM7_ages_med = np.genfromtxt(r"AM7_CoreData/median_ages.csv", delimiter=",", skip_header=0)
AM7_event_times = np.genfromtxt(r"AM7_CoreData/event_times.csv", delimiter=",", skip_header=0)
#distributions
age_dist = np.genfromtxt(r"AM7_CoreData/AM7_Age_Distributions.csv", delimiter=",", skip_header=1)
age_dist = age_dist[:,1:]
AM7_age_dist = 1950 - age_dist
#annual years
AM7_allyears = np.genfromtxt(r"AM7_CoreData/all_years.csv", delimiter=",", skip_header=0)

#%% Get Event Age Percentiles

#### AM2
AM2_events16 = np.percentile(AM2_age_dist, 16, axis=0)
AM2_events84 = np.percentile(AM2_age_dist, 84, axis=0)
AM2_events50 = np.median(AM2_age_dist, axis=0)
#to plot error bars the error has to be in relation to the center value
#so subtract the median value from the percentiles to be able to plot percentiles w/error
lowerr = AM2_events50 - AM2_events16
higherr = AM2_events84 - AM2_events50
xerr= np.stack((lowerr, higherr), axis=1)
xerr_AM2 = xerr.T


#### AM4
AM4_events16 = np.percentile(AM4_age_dist, 16, axis=0)
AM4_events84 = np.percentile(AM4_age_dist, 84, axis=0)
AM4_events50 = np.median(AM4_age_dist, axis=0)
#to plot error bars the error has to be in relation to the center value
#so subtract the median value from the percentiles to be able to plot percentiles w/error
lowerr = AM4_events50 - AM4_events16
higherr = AM4_events84 - AM4_events50
xerr= np.stack((lowerr, higherr), axis=1)
xerr_AM4 = xerr.T

#### AM5
AM5_events16 = np.percentile(AM5_age_dist, 16, axis=0)
AM5_events84 = np.percentile(AM5_age_dist, 84, axis=0)
AM5_events50 = np.median(AM5_age_dist, axis=0)
#to plot error bars the error has to be in relation to the center value
#so subtract the median value from the percentiles to be able to plot percentiles w/error
lowerr = AM5_events50 - AM5_events16
higherr = AM5_events84 - AM5_events50
xerr= np.stack((lowerr, higherr), axis=1)
xerr_AM5 = xerr.T

#### AM7
AM7_events16 = np.percentile(AM7_age_dist, 16, axis=0)
AM7_events84 = np.percentile(AM7_age_dist, 84, axis=0)
AM7_events50 = np.median(AM7_age_dist, axis=0)
#to plot error bars the error has to be in relation to the center value
#so subtract the median value from the percentiles to be able to plot percentiles w/error
lowerr = AM7_events50 - AM7_events16
higherr = AM7_events84 - AM7_events50
xerr= np.stack((lowerr, higherr), axis=1)
xerr_AM7 = xerr.T

#%% Add Up Age Pdfs

#sort by number of deposits
#AM4 has 77
#AM5 has 41
#AM2 has 31
#AM7 has 21

record_dists = []
record_dists.append(AM4_age_dist)
record_dists.append(AM5_age_dist)
record_dists.append(AM2_age_dist)
record_dists.append(AM7_age_dist)
 
#%% Define Functions for Excluding Indices

#excludes multiple indexes
def indices_except(data, excluders):
    return [item for i, item in enumerate(data) if i not in excluders]

#excludes a single index
def indices_except1(data, excluded_index):
    return[i for i in range(len(data)) if i != excluded_index]

#%% Get Event Bounds
AM2_eventbounds = np.nan * np.zeros((np.size(AM2_age_dist, axis=1),2))
for i in range(0,len(AM2_eventbounds)):
    #get minimum and maximum ages of each deposit from the distributions
    AM2_eventbounds[i,0] = int(np.min(AM2_age_dist[:,i])) #minimum age
    AM2_eventbounds[i,1] = int(np.max(AM2_age_dist[:,i])) #maximum age

AM4_eventbounds = np.nan * np.zeros((np.size(AM4_age_dist, axis=1),2))
for i in range(0,len(AM4_eventbounds)):
    #get minimum and maximum ages of each deposit from the distributions
    AM4_eventbounds[i,0] = int(np.min(AM4_age_dist[:,i])) #minimum age
    AM4_eventbounds[i,1] = int(np.max(AM4_age_dist[:,i])) #maximum age
    
AM5_eventbounds = np.nan * np.zeros((np.size(AM5_age_dist, axis=1),2))
for i in range(0,len(AM5_eventbounds)):
    #get minimum and maximum ages of each deposit from the distributions
    AM5_eventbounds[i,0] = int(np.min(AM5_age_dist[:,i])) #minimum age
    AM5_eventbounds[i,1] = int(np.max(AM5_age_dist[:,i])) #maximum age  
    
AM7_eventbounds = np.nan * np.zeros((np.size(AM7_age_dist, axis=1),2))
for i in range(0,len(AM7_eventbounds)):
    #get minimum and maximum ages of each deposit from the distributions
    AM7_eventbounds[i,0] = int(np.min(AM7_age_dist[:,i])) #minimum age
    AM7_eventbounds[i,1] = int(np.max(AM7_age_dist[:,i])) #maximum age    
    
#%% Define Function to Get Overlapping Coordinates

def get_overlap_coordinates(interval1, interval2):
    start_overlap = max(interval1[0], interval2[0])
    end_overlap = min(interval1[1], interval2[1])
    
    if start_overlap < end_overlap:
        return (start_overlap, end_overlap)
    else:
        return None


#%% Determing Independent Events

start = time.time()
sim_len = 4000
perc_thresh = 0.68
record_names = ["AM4", "AM5", "AM2", "AM7"] #sort by record lenght

#going to be 5 loops here
#outer 2 loops loop through the matrix of cores (setting which 2 cores to compare on each loop)
#all deposits in each core will be compared individually to all deposits in the other cores

#this loop sets the anchor core
#each item in record_dists represents a core, 
# within the item in the list is a matrix of the annually binned PDFs of each deposit in that core
for i in range(0, len(record_dists)): 
    
    #makes a list called AM2, AM4, etc "overlap matrix" that has numpy arrays embedded of
    #the indices of overlaps with the 3 other records that are not the title record
    varname = record_names[i] + '_overlap_matrix'
    locals()[varname] = [None] * len(record_dists)
    
    
    #get the record you will compare all the other records to
    #search through every deposit in Record 1 (the title record)
    record1 = record_dists[i] #matrix of ages of each deposit, size is 4000 x # of events
    
    #compare every deposit in Record 1 with all other deposits in the other records
    #no need to compare it to itself, so cut that index out
    #AM2 is 0, AM4 is 1, AM5 is 2, AM7 is 3
    records_to_compare = indices_except1(np.arange(0,len(record_dists)), i)
    
    #this loop sets the core to compare to
    for ii in range(0, len(records_to_compare)):
        
        #index of record to compare to (this way ensures that you're never comparing a record to itself)
        record2_ind = records_to_compare[ii]
        #record to compare to
        record2 = record_dists[record2_ind] #matrix of ages of each deposit, size is 4000 x # of events
    
    
    ####### OK NOW WE HAVE OUR 2 MATRIXES TO COMPARE SO LETS NEST A COUPLE MORE LOOPS
    #the inner 3 loops will loop through every event in the anchor core and compare it to every event in the other 3 cores
    
        #where to save the indices to exclude
        #this will be a vector of indexes in the records that are NOT the title record
        #the loop will create 3 of these
        #indices to exclude variable is all the indices that are overlapping with the associated index in the title record
        #(it's not identifying overlaps per se, just identifying those not flagged as keepers)
        #in this case keepers means they are an independent distribution to be kept as is
        #keepers are only flagged when one age is older more than 68% of the iterations
        r2_exclude = []
        
        
        #now get all the deposits in the first record that will be individually compared with all the deposits in the other 3
        
        for j in range(0, np.size(record1,1)):
            r2_keep = [] #indices from record 2 to keep will be appended here 
            #keepers will be appended here only if there is no age overlap with r1
            
            for jj in range(0, np.size(record2,1)):
                r1_dist = record1[:,j]
                r2_dist = record2[:,jj]
                
                r1_older = []
                r2_older = []
                yrs_tied = []
                
                #permute distributions to compare ages randomly
                r1_distp = rng.permutation(r1_dist)
                r2_distp = rng.permutation(r2_dist)
                
                #compare each possible age to see which is older
                for p in range(0, sim_len):
                    #get the year associated with the iteration you're on
                    r1_yr = r1_distp[p]
                    r2_yr = r2_distp[p]
                    
                    #test which year is older
                    if r1_yr > r2_yr:
                        r1_older.append(p)
                    elif r2_yr > r1_yr:
                        r2_older.append(p)
                    elif r1_yr == r2_yr:
                        yrs_tied.append(p)
                        
                        
                #if one deposit is older than the other more than 68% of the time, it's kept as an independent ('keeper')
                #if the deposits are the same age, you'd expect a 50/50 split on which random draws are older        
                if len(r1_older) > sim_len*perc_thresh or len(r2_older) > sim_len*perc_thresh:
                    r2_keep.append(jj)                   
                
            #now get the indexes to "exclude" from record 2 by taking all the Record 2 inds that are NOT keepers
            r2_allinds = np.arange(0, np.size(record2,1))
            r2_ind = indices_except(r2_allinds, r2_keep)
            
            # if the number of indexes that overlap with this event in record one is NONzero, 
            # append those indexes to the record2 exclude list (which remember has the shape of record 1 because the format is:
            # every deposit in record 2 is looped past every deposit in record 1, and those that overlap are saved in the row of
            # whichever record 1 deposit the loop is on)
            # this will append once for each depsosit in record one
            # output is a list, of the size of record 1, with the indexes of each record 2 deposit that overlap with each record 1 deposit
            if len(r2_ind) > 0:
                r2_exclude.append(np.array(r2_ind))
                
            elif len(r2_ind) == 0:
                r2_exclude.append([])
                
            
        #now we have all the indices of the second record that overlap with the first record going through the loop
        #add these to a matrix that we will nest called "corenameX_overlap_matrix"
        #these overlapping indices will be used in the next step to determine which distributions get compiled
        locals()[varname][record2_ind]=r2_exclude

print('time to compare event beds is ', (time.time() - start)/60, ' minutes')


#%% Identifying Events to Compile
sorder_start = time.time()

# this section is going to create a "network graph" using the networkx python package
# it will map all of the overlapping deposits by setting each deposit as a "node"
# every deposit to compile will be connected as an "edge" - graphically this is like drawing a line between them
# this code will loop through every set of overlapping deposits and either preserve deposits as they are, or combine their distributions
# the outcome is as follows:
    # 1) stratigraphic order is preserved (if four deposits from one core overlap with three deposits from another core, 
    # four distributions are used. The distributions of the three deposits are merged with the 3 deposits of the four that are closest in age, 
    # the fourth deposit that is farthest in age from the overlapping three will be kept independent)
    # 2) no distributions are discarded, they are either merged or kept as is
    # 3) no "edge" in the graph may ever be connected to two deposits from the same core because these are stratigraphically distinct
    # 4) no transitive overlaps, all merged distributions must overlap with each other
    
# first create mega list of all the event bounds
# event bounds defined above as the range of the ENTIRE event distribution (min to max)
# format is a list with 4 elements for each core, each list item has a np.array inside of it
# dimensions of the np arrays are # deposits in core x 2 where the first column is minimum age and second column is max age

eventbounds = [AM4_eventbounds, AM5_eventbounds,
               AM2_eventbounds, AM7_eventbounds] #sort by # of deposits

# create mega list of all deposit medians
# output in same format as eventbounds
core_medians = [[float(np.median(record_dists[ci][:, di]))
     for di in range(np.size(record_dists[ci], axis=1))]  # loop over all deposits in core ci
    for ci in range(len(record_dists))]  # loop over all cores


# make set of directly overlapping pairs to be checked later to make sure no transitive overlaps are made
# If A overlaps B and B overlaps C but C doesn't overlap A, C won't be combined with B
direct_overlap_set = set()

for ci in range(len(record_dists)): #ci is first core
    for cj in range(ci + 1, len(record_dists)): #cj is second core
        rname_ci  = record_names[ci] + '_overlap_matrix'
        rname_cj  = record_names[cj] + '_overlap_matrix'
        matrix_ci = locals()[rname_ci]
        matrix_cj = locals()[rname_cj]
        
        #get overlapping pairs in both directions
        ovlp = set()
        if matrix_ci[cj] is not None:
            for di, olist in enumerate(matrix_ci[cj]):
                for dj in olist:
                    ovlp.add((di, int(dj))) #list all the overlaps associated with this deposit for this core
        if matrix_cj[ci] is not None:
            for dj, olist in enumerate(matrix_cj[ci]):
                for di in olist:
                    ovlp.add((int(di), dj)) #list them backwards now to make sure they are symmetrical
        
        #make every di,dj pair into a "frozenset" which is like a set but can't be changed
        #later in the code we will sort through the "sets" of overlapping pairs and eliminate sets based on priorities for distribtuion merging
        #will cross check the final sets to this frozen set to make sure they all have direct overlaps
        for di, dj in ovlp:
            direct_overlap_set.add(frozenset([(ci, di), (cj, dj)])) 

#make empty network graph object
tie_graph = nx.Graph()

#add every deposit as a node
for ci in range(len(record_dists)):
    for di in range(np.size(record_dists[ci], axis=1)):
        tie_graph.add_node((ci, di))

#### Loop Through All Pairs 

#outer loop is going to go through every possible pair of cores
#keep ci < cj to make sure that pairs are not going both ways (don't want to match 0-1 and 1-0 becuase it will duplicate everything)
# so pairs will be (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)

for ci in range(len(record_dists)): # ci: index of first core in pair
    for cj in range(ci + 1, len(record_dists)): # cj: index of second core in pair
    
    #get the overlap matrixes for each core pair (created in the section above)
        rname_ci  = record_names[ci] + '_overlap_matrix'
        rname_cj  = record_names[cj] + '_overlap_matrix'
        matrix_ci = locals()[rname_ci]
        matrix_cj = locals()[rname_cj]
        
        #get the overlapping deposit pairs from each core pair
        #use python set data type because it prevents duplicates
        #i.e. (0,2) only appears once instead of also as (2,0)
        #make empty set
        overlap_set = set()

        #do it in the first direction (core ci is the anchor core)
        #loop through every deposit in the overlap matrix
        if matrix_ci[cj] is not None: #make sure there are overlaps with that core
        # enumerate returns index and values, so "overlapping" is the numpy array of deposits that overlap in the
        #second core with the deposit you're on in the anchor core 
            for di, overlapping in enumerate(matrix_ci[cj]):
                for dj in overlapping:
                    overlap_set.add((di, int(dj))) #store overlaps as tuple   

        #loop through in other direction to make sure there are no assymetric pairs produced from the montecarlo sequence
        #this would only happen if deposits were very borderline overlapping such that a few random comparisons made 
        #the difference in meeting the threshold or not
        if matrix_cj[ci] is not None: #core 2 is outer loop now
            for dj, overlapping in enumerate(matrix_cj[ci]):
                for di in overlapping:
                    overlap_set.add((int(di), dj))
                    
        #if no overlaps, skip this pair
        if not overlap_set:
            print(f"\n{record_names[ci]} vs {record_names[cj]}: no overlaps")
            continue

        #sort overlapping pairs by distance between them (closest in age first)
        #a lambda function repeats the same step for every item in the array so is just an alternative to a loop
        #this bit makes it so that overlaps between deposits closest in age are consolidated first
        overlap_pairs_sorted = sorted(overlap_set,
            key=lambda p: ( #p is like the iterator
                abs(core_medians[ci][p[0]] - core_medians[cj][p[1]]),  # get distance between ages
                p[0],# deposit 1 (di)
                p[1]))  # deposit 2 (dj)
        
        #now this loop is going to match the deposits that need to be merged
        #so like if 3 overlap in AM2 with 3 in AM4, it is going to pair the youngest, middle, and oldest from each core
        #make emtpy sets, these will be reset in each outer loop 
        #so basically new sets for every core-pair so that 
        #deposits can be matched with a different core as well even if they match in this core-pair set
        #this will populate as each deposit is looped through for each core
        matched_ci = set() 
        matched_cj = set() 

        #print which core set we're on for sanity
        print(f"\n{record_names[ci]} vs {record_names[cj]} "
              f"({len(overlap_set)} overlapping pairs):")
        
        #di is the first number of the tuple, dj is the second number of the tuple
        for di, dj in overlap_pairs_sorted:

            #each deposit is only allowed one partner in this specific core pair comparison
            #so if either deposit is already in the matched list, skip it
            if di in matched_ci or dj in matched_cj:
                continue

            #make the node labels
            node1 = (ci, di) #core, deposit
            node2 = (cj, dj)

            #make sure deposits are not circularly connecting within the same core (they're separate!! stratigraphic order!!)
            #nx.node_connect_component is built into network graphing function to return node labels
            #here it will be a list of all the deposits that would be merged with node1 (deposit we're on in this loop)
            #if the edge of node2 is added to the graph (so like, if node2 is AM4 deposit1 and it has already been connected to AM3 deposit1,
            #if we connect it to AM3 deposit 2 as well, now we're merging two distributions from the same core) -- we don't want this
            # rule here is that each deposit can only have one deposit pair in each CORE-PAIR
            cluster1 = set(nx.node_connected_component(tie_graph, node1)) #set of tuples (core #, dep#) that are connected to node1
            # if the deposit being compared is already connected to the node (i.e. in the cluster), skip (prevent circular mega clustering)
            if node2 in cluster1:
                continue
            cluster2 = set(nx.node_connected_component(tie_graph, node2)) #
            
            #merge the components to see if it would make any bad combos
            merged = cluster1 | cluster2 #this is a python "set" variable "union" which is combining the sets

            #check here if the merged component would contain two deposits from the same core
            #if true, skip this pair
            conflict = False #defalult is conflict is false
            core_count = defaultdict(int) #function to count the deposits per core
            for c, d in merged:  #core labels
                core_count[c] += 1
                if core_count[c] > 1: #if more than one c label in this count, means deposit being double matched
                    conflict = True #label this pair as a conflict, exit loop
                    break
                
            #if conflict labeled true, skip this pair with continue
            if conflict:
                print(f"  SKIP  {record_names[ci]}[{di}]"
                      f"({core_medians[ci][di]:.0f}) ↔ "
                      f"{record_names[cj]}[{dj}]"
                      f"({core_medians[cj][dj]:.0f})"
                      f"  ← same-core conflict")
                continue

            #check here to see if any transitive overlap would be created 
            #this is if A-B overlap and B-C overlap but A-C don't overlap
            merged_list = list(merged)
            for ia in range(len(merged_list)): #loop through all possible merges associated with this deposit1
                for ib in range(ia + 1, len(merged_list)): #loop through all possible merges associated with deposit we're potentially connecting to
                    na = merged_list[ia]
                    nb = merged_list[ib]
                    if frozenset([na, nb]) not in direct_overlap_set: #compare to the frozen set of overlaps at the beginning
                        conflict = True #if not a direct match skip this pair
                        break
                if conflict:
                    break

            #print transitive overlaps
            if conflict:
                print(f"  SKIP  {record_names[ci]}[{di}]"
                      f"({core_medians[ci][di]:.0f}) ↔ "
                      f"{record_names[cj]}[{dj}]"
                      f"({core_medians[cj][dj]:.0f})"
                      f"  ← transitive overlap (not all members overlap directly)")
                continue

            #if there's no conflict, add the edge and label the deposits as matched
            tie_graph.add_edge(node1, node2)
            matched_ci.add(di) # add to already matched list
            matched_cj.add(dj) 
            
            #get distance in median ages between matched pairs
            age_gap = abs(core_medians[ci][di] - core_medians[cj][dj])  
            
            #print all the overlapping pairs to be merged and their age distances
            print(f"  MATCH {record_names[ci]}[{di}]"
                  f"({core_medians[ci][di]:.0f}) ↔ "
                  f"{record_names[cj]}[{dj}]"
                  f"({core_medians[cj][dj]:.0f})"
                  f"  [Δ = {age_gap:.0f} yr]")

print('time to determine overlapping distributions is ',
      (time.time() - sorder_start) / 60, ' minutes')
#%% Create Compiled Distributions

#empty array to put the new distributions in
combined_dists = []  
             
#make an empty list of each core name that will fill with the deposit indexes to be kept from each core 
#on each iteration of the loop below
core_keepers   = defaultdict(list) 

for comp in nx.connected_components(tie_graph):
    nodes = list(comp)

    # if there is no matching deposit, let this distribution stay as is
    if len(nodes) == 1:
        ci, di = nodes[0]         
        core_keepers[ci].append(di)
        continue

    #if there is a matching deposit, combine distributions
    
    #ci is core iterator, di is deposit iteratior
    temp_dists = np.nan * np.zeros(sim_len)
    for ci, di in nodes:                            
        temp_dists = np.vstack((temp_dists, record_dists[ci][:, di])) #get distribution of the specific deposit we're on                    
    cmean = np.nanmean(temp_dists, axis=0)

    # Concatenate Dists
    combined_dists.append(cmean)
    
    #print which deposits combined and what the median ages were
    node_labels = [(record_names[ci], di) for ci, di in sorted(nodes)]  
    node_meds   = [f"{core_medians[ci][di]:.0f}" for ci, di in sorted(nodes)]  
    print(f"Combined ({len(nodes)} deposits): {node_labels}  "
          f"medians: {node_meds}")

            
#%% Save Distributions for Rate Parameter Analysis      

AM4_keepers = np.sort(np.array(core_keepers.get(0, []), dtype=int))
AM5_keepers = np.sort(np.array(core_keepers.get(1, []), dtype=int))
AM2_keepers = np.sort(np.array(core_keepers.get(2, []), dtype=int))
AM7_keepers = np.sort(np.array(core_keepers.get(3, []), dtype=int))

#print how many it's keeping from each analysis
print(f"AM2 keepers ({len(AM2_keepers)}): {AM2_keepers}")
print(f"AM4 keepers ({len(AM4_keepers)}): {AM4_keepers}")
print(f"AM5 keepers ({len(AM5_keepers)}): {AM5_keepers}")
print(f"AM7 keepers ({len(AM7_keepers)}): {AM7_keepers}")
print(f"Combined distributions: {len(combined_dists)}")          

#all distributions have 4000 members
iterations=4000

#make sure there are combinations
if combined_dists:
    tied_matrix = np.zeros((iterations, len(combined_dists))) #size of 4000 x # combined distributions
    
    for i in range(0, len(combined_dists)):
        tied_matrix[:, i] = combined_dists[i]
        
else: #if no combinations just make this empty and we'll ignore it for the final compilation
    tied_matrix = np.empty((iterations, 0))
    
#### make sure there is data to keep
#keeper matrixes can be empty if all deposits from that core get combined, or there are no combinations

#define function to keep empty array if no deposits kept from the core
def safe_slice(age_dist, keepers):
    if len(keepers)==0:
        return np.empty((age_dist.shape[0],0)) #make empty array
    return age_dist[:,keepers]

#get final distributions
AM2_dist_keep = safe_slice(AM2_age_dist, AM2_keepers)
AM4_dist_keep = safe_slice(AM4_age_dist, AM4_keepers)
AM5_dist_keep = safe_slice(AM5_age_dist, AM5_keepers)
AM7_dist_keep = safe_slice(AM7_age_dist, AM7_keepers)

#stack combined distributions
composite_distributions = np.hstack((tied_matrix, AM2_dist_keep, AM4_dist_keep, AM5_dist_keep, AM7_dist_keep))
np.savetxt(r"Compilation_Data/composite_dist68.csv", composite_distributions, delimiter=',')

n_kept = (len(AM2_keepers) + len(AM4_keepers) +
          len(AM5_keepers) + len(AM7_keepers))

#### print summary of what was kept
print(f"\nComposite : {composite_distributions.shape[1]} total events")
print(f"  {len(combined_dists)} combined  +  {n_kept} kept unchanged")
print(f"    AM2={len(AM2_keepers)}, AM4={len(AM4_keepers)}, "
      f"AM5={len(AM5_keepers)}, AM7={len(AM7_keepers)}")
print(f"\nTotal runtime : {(time.time() - code_start)/60:.2f} min")

#%% Get Percentiles For Plotting


#### AM2
#percentiles
AM2_keepers16 = np.percentile(AM2_dist_keep, 16, axis=0)
AM2_keepers84 = np.percentile(AM2_dist_keep, 84, axis=0)
AM2_keepers50 = np.median(AM2_dist_keep, axis=0)
#errors
lowerr = AM2_keepers50 - AM2_keepers16
higherr = AM2_keepers84 - AM2_keepers50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_AM2_keep = xerr.T

#### AM4
#percentiles
AM4_keepers16 = np.percentile(AM4_dist_keep, 16, axis=0)
AM4_keepers84 = np.percentile(AM4_dist_keep, 84, axis=0)
AM4_keepers50 = np.median(AM4_dist_keep, axis=0)
#errors
lowerr = AM4_keepers50 - AM4_keepers16
higherr = AM4_keepers84 - AM4_keepers50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_AM4_keep = xerr.T

#### AM5
#percentiles
AM5_keepers16 = np.percentile(AM5_dist_keep, 16, axis=0)
AM5_keepers84 = np.percentile(AM5_dist_keep, 84, axis=0)
AM5_keepers50 = np.median(AM5_dist_keep, axis=0)
#errors
lowerr = AM5_keepers50 - AM5_keepers16
higherr = AM5_keepers84 - AM5_keepers50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_AM5_keep = xerr.T

#### AM7
#percentiles
AM7_keepers16 = np.percentile(AM7_dist_keep, 16, axis=0)
AM7_keepers84 = np.percentile(AM7_dist_keep, 84, axis=0)
AM7_keepers50 = np.median(AM7_dist_keep, axis=0)
#errors
lowerr = AM7_keepers50 - AM7_keepers16
higherr = AM7_keepers84 - AM7_keepers50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_AM7_keep = xerr.T

#### Ties
#percentiles
ties50 = np.median(tied_matrix, axis=0)
ties16 = np.percentile(tied_matrix, 16, axis=0)
ties84 = np.percentile(tied_matrix, 84, axis=0)
#errors
lowerr = ties50 - ties16
higherr = ties84 - ties50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_ties = xerr.T

#%% Plot Median Compilations

#get y axis values
AM2_y = 4 * np.ones(len(AM2_events50))
AM4_y = 3 * np.ones(len(AM4_events50))
AM5_y = 2 * np.ones(len(AM5_events50))
AM7_y = 1 * np.ones(len(AM7_events50))

AM2_y2 = 0 * np.ones(len(AM2_keepers50))
AM4_y2 = 0 * np.ones(len(AM4_keepers50))
AM5_y2 = 0 * np.ones(len(AM5_keepers50))
AM7_y2 = 0 * np.ones(len(AM7_keepers50))
ties_y2 = 0 * np.ones(len(ties50))


plt.figure(figsize=(20,5), dpi=150)

plt.errorbar(ties50, ties_y2, yerr=None, xerr=xerr_ties, ms=12, capsize=6, fmt='o', label='ties', color='slategray', alpha=0.7)
plt.errorbar(AM2_keepers50, AM2_y2, yerr=None, xerr=xerr_AM2_keep, ms=12, capsize=6, fmt='o', color='cornflowerblue', alpha=0.7) 
plt.errorbar(AM4_keepers50, AM4_y2, yerr=None, xerr=xerr_AM4_keep, ms=12, capsize=6, fmt='o', color='mediumblue', alpha=0.7) 
plt.errorbar(AM5_keepers50, AM5_y2, yerr=None, xerr=xerr_AM5_keep, ms=12, capsize=6, fmt='o', color='blueviolet', alpha=0.7) 
plt.errorbar(AM7_keepers50, AM7_y2, yerr=None, xerr=xerr_AM7_keep, ms=12, capsize=6, fmt='o', color='violet', alpha=0.7) 
plt.errorbar(AM2_events50, AM2_y, yerr=None, xerr=xerr_AM2, ms=12, capsize=6, fmt='o', label='AM2', color='cornflowerblue', alpha=0.7)
plt.errorbar(AM4_events50, AM4_y, yerr=None, xerr=xerr_AM4, ms=12, capsize=6, fmt='o', label='AM4', color='mediumblue', alpha=0.7)
plt.errorbar(AM5_events50, AM5_y, yerr=None, xerr=xerr_AM5, ms=12, capsize=6, fmt='o', label='AM5', color='blueviolet', alpha=0.7)
plt.errorbar(AM7_events50, AM7_y, yerr=None, xerr=xerr_AM7, ms=12, capsize=6, fmt='o', label='AM7', color='violet', alpha=0.7)
plt.legend()
plt.xlim(min(AM4_allyears), max(AM4_allyears))
plt.ylim(-0.6,4.6)
plt.grid(True)
plt.show()
plt.xlabel('Year (CE)')
plt.title('68% MC Threshold')


#%% Plot the TieGraph with all Overlaps
import colorsys

# axis positions
y_pos      = {0: 4, 1: 3, 2: 2, 3: 1}
y_compile  = 0
row_labels = {4: 'AM4', 3: 'AM5', 2: 'AM2', 1: 'AM7', 0: 'Compilation'}

ovlp_mats = [globals()[name + '_overlap_matrix'] for name in record_names]

#### Make Color Palette
DISTINCT_PALETTE = [
    '#e6194b',   # vivid red
    '#3cb44b',   # vivid green
    '#4363d8',   # vivid blue
    '#f58231',   # vivid orange
    '#911eb4',   # vivid purple
    '#42d4f4',   # vivid cyan
    '#f032e6',   # vivid magenta
    '#bfef45',   # vivid lime
    '#469990',   # vivid teal
    '#9A6324',   # vivid brown
    '#800000',   # maroon
    '#000075',   # navy
    '#aaffc3',   # mint
    '#808000',   # olive
    '#dcbeff',   # lavender
    '#fffac8',   # cream yellow
    '#ffd8b1',   # apricot
    '#a9a9a9',   # grey
]

def get_distinct_colors(n):
    if n == 0:
        return []
    if n <= len(DISTINCT_PALETTE):
        return DISTINCT_PALETTE[:n]
    return ['#%02x%02x%02x' % tuple(int(c * 255) for c in
            colorsys.hsv_to_rgb(i / n, 0.85, 0.85))
            for i in range(n)]

#### Get Unique Colors 
components  = list(nx.connected_components(tie_graph)) #get all connections
multi_comps = [c for c in components if len(c) > 1] #connected nodes
solo_comps  = [c for c in components if len(c) == 1] #isolated nodes

n_groups  = len(multi_comps) #number of groups we need
group_clr = get_distinct_colors(n_groups) #color each group

node_color = {}
comp_pts   = []

# multi-node components
for k, comp in enumerate(multi_comps):
    clr = group_clr[k]
    for node in comp:
        node_color[node] = clr
    stacked = np.vstack([record_dists[ci][:, di] for ci, di in comp])
    cmean   = np.mean(stacked, axis=0)
    med     = float(np.median(cmean))
    lo      = float(med - np.percentile(cmean,  5))
    hi      = float(np.percentile(cmean, 95) - med)
    comp_pts.append(dict(x=med, lo=lo, hi=hi, color=clr))

# single-node components
for comp in solo_comps:
    node   = next(iter(comp))
    node_color[node] = 'black'
    ci, di = node
    samp   = record_dists[ci][:, di]
    med    = float(np.median(samp))
    lo     = float(med - np.percentile(samp,  5))
    hi     = float(np.percentile(samp, 95) - med)
    comp_pts.append(dict(x=med, lo=lo, hi=hi, color='black'))

#### Get All Overlap Lines
line_pairs = []

for ci in range(len(record_dists)):
    for cj in range(ci + 1, len(record_dists)):
        mat_ci = ovlp_mats[ci]
        mat_cj = ovlp_mats[cj]
        ovlp   = set()

        if mat_ci[cj] is not None:
            for di, olist in enumerate(mat_ci[cj]):
                for dj in olist:
                    ovlp.add((di, int(dj)))

        if mat_cj[ci] is not None:
            for dj, olist in enumerate(mat_cj[ci]):
                for di in olist:
                    ovlp.add((int(di), dj))

        for di, dj in ovlp:
            line_pairs.append(((ci, di), (cj, dj)))

#### Get Preserved Edges
final_line_pairs = []   # list of ((ci,di), (cj,dj), colour)

for node1, node2 in tie_graph.edges():
    clr = node_color.get(node1, 'black')   # both nodes share the same colour
    final_line_pairs.append((node1, node2, clr))

#### Get Error Bars
dep_err = {}
for ci in range(len(record_dists)):
    for di in range(np.size(record_dists[ci], axis=1)):
        samp = record_dists[ci][:, di]
        med  = float(core_medians[ci][di])
        dep_err[(ci, di)] = (float(med - np.percentile(samp,  5)),
                             float(np.percentile(samp, 95) - med))

#### Plot
fig, ax = plt.subplots(figsize=(24, 9), dpi=150)

for y in range(5):
    ax.axhline(y, color='lightgray', lw=0.5, zorder=0)

ax.axhline(0.5, color='gray', lw=1.0, ls='--', alpha=0.6, zorder=0)

#plot all overlaps gray
for (ci, di), (cj, dj) in line_pairs:
    ax.plot(
        [float(core_medians[ci][di]), float(core_medians[cj][dj])],
        [y_pos[ci],                   y_pos[cj]],
        color='black', lw=0.7, alpha=0.25, zorder=1
    )

#plot final connections in group colors
for (ci, di), (cj, dj), clr in final_line_pairs:
    ax.plot(
        [float(core_medians[ci][di]), float(core_medians[cj][dj])],
        [y_pos[ci],                   y_pos[cj]],
        color=clr, lw=2.5, alpha=0.85, zorder=2
    )

#plot error bars in group colors
for ci in range(len(record_dists)):
    y = y_pos[ci]
    for di in range(np.size(record_dists[ci], axis=1)):
        x   = float(core_medians[ci][di])
        clr = node_color.get((ci, di), 'black')
        lo, hi = dep_err[(ci, di)]
        ax.errorbar(x, y, xerr=[[lo], [hi]],
                    fmt='none', color=clr, capsize=4,
                    lw=1.2, alpha=0.55, zorder=3)

for pt in comp_pts:
    ax.errorbar(pt['x'], y_compile,
                xerr=[[pt['lo']], [pt['hi']]],
                fmt='none', color=pt['color'], capsize=4,
                lw=1.2, alpha=0.55, zorder=3)

#plot median deposit ages
for ci in range(len(record_dists)):
    y = y_pos[ci]
    for di in range(np.size(record_dists[ci], axis=1)):
        x   = float(core_medians[ci][di])
        clr = node_color.get((ci, di), 'black')
        ax.plot(x, y, 'o', color=clr, ms=10,
                markeredgewidth=0, zorder=4)

for pt in comp_pts:
    ax.plot(pt['x'], y_compile, 'o', color=pt['color'],
            ms=10, markeredgewidth=0, zorder=4)

#labels
ytick_vals = sorted(row_labels)
ax.set_yticks(ytick_vals)
ax.set_yticklabels([row_labels[k] for k in ytick_vals], fontsize=12)
ax.set_xlabel('Year (CE)', fontsize=12)
ax.set_title('68% Threshold', fontsize=13)
ax.set_xlim(min(AM4_allyears), max(AM4_allyears))
ax.set_ylim(-0.6, 4.6)
ax.grid(True, axis='x', alpha=0.3)

# legend
legend_elems = [
    Line2D([0], [0], marker='o', color='w',
            markerfacecolor=group_clr[k], markersize=10,
            markeredgewidth=0,
            label=f'Compilation group {k + 1}')
    for k in range(n_groups)
]
legend_elems += [
    Line2D([0], [0], marker='o', color='w',
            markerfacecolor='black', markersize=10,
            markeredgewidth=0,
            label='Independent (no overlap)'),
    Line2D([0], [0], color='black', lw=0.8, alpha=0.35,
            label='Possible overlap (all pairs)'),
    Line2D([0], [0], color='dimgray', lw=2.5, alpha=0.85,
            label='Final matched connection'),
]
ax.legend(handles=legend_elems, loc='upper left',
          fontsize=9, framealpha=0.8, ncol=2)

plt.tight_layout()


#%% Plot Distributions of Modern Deposits

#### make distributions 
#sort composite distributions by medians
meds = np.median(composite_distributions, axis=0)
sorted_inds = np.argsort(meds)
sorted_dists = composite_distributions[:,sorted_inds]

modern_dist = np.zeros((200,6)) #dimensions are years x # of modern deposits (6)
for i in range(1,7):
    event = sorted_dists[:,-i]
    #bin annualy
    min_age = 1815
    max_age = 2014
    bin_edges = np.arange(min_age-1, max_age+1) #edges of bins for the histogram
    count, bins_count = np.histogram(event, bins=bin_edges, density=True) #density true normalizes to one
    bins = bins_count[:-1] #remove last bin edge so can plot 1:1 with counts
    modern_dist[:,i-1] = count

modern_years = np.arange(1815,2015)
colors=['blueviolet', 'violet', 'slategray', 'mediumblue', 'mediumblue', 'blueviolet']

modern_meds = np.flip(np.median(sorted_dists[:,-6:],axis=0))


plt.figure(dpi=150)
for i in range(0,6):
    plt.plot(modern_years, modern_dist[:,i], color=colors[i])
    plt.fill_between(modern_years, 0, modern_dist[:,i], color=colors[i], alpha=0.3)
    plt.stem(modern_meds[i], max(modern_dist[:,i]), colors[i], markerfmt =" ", basefmt=" ")
plt.xlim(1800,2000)
plt.grid(axis='both', linestyle='--', alpha=0.4)
plt.xlabel('Year (CE)')
plt.ylabel('normalized probability density')
plt.title('68% MC Threshold')

#%% Define Rate Parameter Function

def rate_parameter(allyrs, eventyrs, h):
    #initialize the rate parameter
    l = np.nan*(np.zeros(len(allyrs))) #l is lambda as in Mudelsee Eq 3.16
    
    for i in range(0,len(allyrs)): #loop through every year of the dataset
        l[i] = 0 
        for j in range(0, len(eventyrs)):
            y = (allyrs[i] - eventyrs[j]) / h
            l[i] = l[i] + (np.power((2*np.pi), -0.5) * np.exp((-y**2)/2))/h
    return(l*100)

#%% Load Published Frequency Data

#composite distribution (if not running compilation above)
comp_dist = np.genfromtxt(r"Compilation_Data/composite_dist68.csv", delimiter=",", skip_header=0)
eventyrs = np.median(comp_dist, axis=0)

strmyrs = np.isin(AM4_allyears, eventyrs.astype(int))

#published data
pub_data = np.genfromtxt(r"Compilation_Data/Published_Freq.csv", delimiter=",", skip_header=1)
pub_AM4_year = pub_data[:,5]
pub_AM4_freq = pub_data[:,6]
pub_AM_stack = pub_data[:,7]
pub_stack_year = pub_data[:,0]


#%% Bootstrap Composite

start_time = time.time()

#iterations 
iterations = 1000
#window size
h = 50 
#initalize empty matrix
l_matrix_comp = np.zeros((iterations, len(AM4_allyears))) #use AM4 because it's longest

#set age dist to draw from
age_dist = comp_dist

for i in range(0, iterations):
    
    #draw a random age from the probability distributions
    amax = len(age_dist) #maximum index
    amin = 0 # minimum index
    age_ind = np.random.randint(amin, high=amax, size=1) #get random index
    
    #draw random sample of event years from the random iteration
    eventyrs_temp = age_dist[age_ind,:]
    eventyrs_int = np.squeeze(eventyrs_temp.astype(int)) #convert to integer
    
    #create new set of event years with replacement, keeping the same total number of events and preserving the stratigraphic order
    rmax = np.size(age_dist, axis=1) #maximum event number index
    rmin = 0 #minimum event number index
    event_ind = np.random.randint(rmin, high=rmax, size=np.size(age_dist, axis=1))
    eventyrs_boot = eventyrs_int[event_ind]
    
    strmyrs_boot = np.isin(AM4_allyears, eventyrs_boot) #boolean index of event years
    
    #rate parameter analyis on the new timeseries of event years
    l_temp = rate_parameter(AM4_allyears, eventyrs_boot, h)
    
    #save lambda parameters
    l_matrix_comp[i,:] = l_temp
    
np.savetxt(r"Compilation_Data/l_matrix_comp68.csv", l_matrix_comp, delimiter=',')

print('time to run bootstrapping is ', (time.time() - start_time)/60, ' minutes')  

#%% Load bootstrap data if not bootstrapping

l_matrix_comp = np.genfromtxt(r"Compilation_Data/l_matrix_comp68.csv", delimiter=',', skip_header=0)

# individual rate parameter matrixes
AM2_lmatrix = np.genfromtxt(r"AM2_CoreData/l_matrix.csv", delimiter=",", skip_header=0)
AM4_lmatrix = np.genfromtxt(r"AM4_CoreData/l_matrix.csv", delimiter=",", skip_header=0)
AM5_lmatrix = np.genfromtxt(r"AM5_CoreData/l_matrix.csv", delimiter=",", skip_header=0)
AM7_lmatrix = np.genfromtxt(r"AM7_CoreData/l_matrix.csv", delimiter=",", skip_header=0)

#%% Plot 100yr freq

######## Get statistics

#### AM2
mean_rate_AM2 = np.median(AM2_lmatrix, axis=0)
low_rate2_AM2 = np.percentile(AM2_lmatrix, 5, axis=0)
low_rate1_AM2 = np.percentile(AM2_lmatrix, 16, axis=0)
high_rate2_AM2 = np.percentile(AM2_lmatrix, 95, axis=0)
high_rate1_AM2 = np.percentile(AM2_lmatrix, 84, axis=0)

#### AM4
mean_rate_AM4 = np.median(AM4_lmatrix, axis=0)
low_rate2_AM4 = np.percentile(AM4_lmatrix, 5, axis=0)
low_rate1_AM4 = np.percentile(AM4_lmatrix, 16, axis=0)
high_rate2_AM4 = np.percentile(AM4_lmatrix, 95, axis=0)
high_rate1_AM4 = np.percentile(AM4_lmatrix, 84, axis=0)

#### AM5
mean_rate_AM5 = np.median(AM5_lmatrix, axis=0)
low_rate2_AM5 = np.percentile(AM5_lmatrix, 5, axis=0)
low_rate1_AM5 = np.percentile(AM5_lmatrix, 16, axis=0)
high_rate2_AM5 = np.percentile(AM5_lmatrix, 95, axis=0)
high_rate1_AM5 = np.percentile(AM5_lmatrix, 84, axis=0)


#### AM7
mean_rate_AM7 = np.median(AM7_lmatrix, axis=0)
low_rate2_AM7 = np.percentile(AM7_lmatrix, 5, axis=0)
low_rate1_AM7 = np.percentile(AM7_lmatrix, 16, axis=0)
high_rate2_AM7 = np.percentile(AM7_lmatrix, 95, axis=0)
high_rate1_AM7 = np.percentile(AM7_lmatrix, 84, axis=0)

#### Compilation
mean_rate_C = np.median(l_matrix_comp, axis=0)
low_rate2_C = np.percentile(l_matrix_comp, 5, axis=0)
low_rate1_C = np.percentile(l_matrix_comp, 16, axis=0)
high_rate2_C = np.percentile(l_matrix_comp, 95, axis=0)
high_rate1_C = np.percentile(l_matrix_comp, 84, axis=0)




#%% Final Figure

#### Plot
fig,ax = plt.subplots(3,1, figsize=(9,9), dpi=150)
ax0 = ax[0]
ax1 = ax[1]
ax2 = ax[2]


#individual records
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major',zorder=0)
ax0.plot(AM2_allyears, mean_rate_AM2, label='AM2', color='cornflowerblue')
ax0.plot(AM4_allyears, mean_rate_AM4, label='AM4', color='mediumblue')
ax0.plot(AM5_allyears, mean_rate_AM5, label='AM5', color='blueviolet')
ax0.plot(AM7_allyears, mean_rate_AM7, label='AM7', color='violet')
ax0.plot(AM4_allyears, mean_rate_C, color='k', lw=1, linestyle='--', label='Compilation')
ax0.legend(loc='lower left')
ax0.set_ylabel(r'Storm Frequency ($\lambda$)')
ax0.set_xlim(350, 2050)

#compilation
ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major',zorder=0)
ax1.plot(AM4_allyears, mean_rate_C, color='k', lw=2, label='Compilation')
ax1.fill_between(AM4_allyears, low_rate2_C, high_rate2_C, color = 'k', alpha=0.1, label=r'2$\sigma$')
ax1.fill_between(AM4_allyears, low_rate1_C, high_rate1_C, color='k', alpha=0.3, label = r'1$\sigma$')
ax1.stem(AM4_allyears, strmyrs, 'k', markerfmt =" ", basefmt=" ")
ax1.set_ylabel(r'Storm Frequency ($\lambda$)')
ax1.legend()
ax1.set_xlim(350, 2050)

#Compilation w/previously published AM record
ax2.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major',zorder=0)
ax2.plot(AM4_allyears, mean_rate_C, color='k', lw=2, label='Compilation')
ax2.fill_between(AM4_allyears, low_rate2_C, high_rate2_C, color = 'k', alpha=0.1, label=r'2$\sigma$')
ax2.fill_between(AM4_allyears, low_rate1_C, high_rate1_C, color='k', alpha=0.3, label = r'1$\sigma$')
ax2.plot(pub_AM4_year, pub_AM4_freq, color='mediumblue', label='Wallace et al., 2019, AM4')
ax2.plot(pub_stack_year, pub_AM_stack, color='royalblue', label='Winkler et al., 2020, AM Stack')
ax2.set_ylabel(r'Storm Frequency ($\lambda$)')
ax2.legend()
ax2.set_xlabel('Year (CE)')
ax2.set_xlim(350, 2050)

plt.suptitle('68% MC Threshold')


#### Get statistics
#average storm frequency from 1200-2000 this study
fmean = np.mean(mean_rate_C[794:])
flow = np.percentile(mean_rate_C[794:], 16)
fhigh = np.percentile(mean_rate_C[794:],84)
fstd = np.std(mean_rate_C)

#average storm frequency wallace AM4
fmean_wall = np.nanmean(pub_AM4_freq[:721])
flow_wall = np.nanpercentile(pub_AM4_freq[:721],16)
fhigh_wall = np.nanpercentile(pub_AM4_freq[:721],84)
fstd_wall = np.nanstd(pub_AM4_freq[:721])

#average storm frequency winkler AM stack
fmean_wink = np.nanmean(pub_AM_stack[:820])
flow_wink = np.nanpercentile(pub_AM_stack[:820],16)
fhigh_wink = np.nanpercentile(pub_AM_stack[:820],84)
fstd_wink = np.nanstd(pub_AM_stack)