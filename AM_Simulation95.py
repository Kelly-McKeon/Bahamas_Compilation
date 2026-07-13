#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:44:34 2025

@author: kellymckeon
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import random
import time
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 
import networkx as nx          # graph library: nodes = deposits, edges = overlaps
from collections import defaultdict  # dictionary that auto-initializes missing keys
import pandas as pd
rng = np.random.default_rng()

mac = r"/Users/kellymckeon/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Bahamas/Codes/For_Publication"
pc = r"C:/Users/kelly/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Bahamas/Codes/For_Publication"

# CHANGE DIRECTORY TO WHICHEVER COMPUTER YOU ARE ON
os.chdir(mac)

start_code = time.time()

#%% Load Data

#composite rate parameter
l_matrix_comp = np.genfromtxt(r"Compilation_Data/l_matrix_comp.csv", delimiter=',', skip_header=0)
mean_rate_C = np.median(l_matrix_comp, axis=0)

#composite distributions
comp_dist = np.genfromtxt(r"Compilation_Data/composite_dist.csv", delimiter=",", skip_header=0)

#%% Define Rate Parameter Functions

def rate_parameter(allyrs, eventyrs, h):
    #initialize the rate parameter
    l = np.nan*(np.zeros(len(allyrs))) #l is lambda as in Mudelsee Eq 3.16
    
    for i in range(0,len(allyrs)): #loop through every year of the dataset
        l[i] = 0 
        for j in range(0, len(eventyrs)):
            y = (allyrs[i] - eventyrs[j]) / h
            l[i] = l[i] + (np.power((2*np.pi), -0.5) * np.exp((-y**2)/2))/h
    return(l*100)



#%% Define Functions 

#### Get Overlapping Coordinates
def get_overlap_coordinates(interval1, interval2):
    start_overlap = max(interval1[0], interval2[0])
    end_overlap = min(interval1[1], interval2[1])
    
    if start_overlap < end_overlap:
        return (start_overlap, end_overlap)
    else:
        return None
    
#### Excluding deposits

#excludes multiple indexes
def indices_except(data, excluders):
    return [item for i, item in enumerate(data) if i not in excluders]

#excludes a single index
def indices_except1(data, excluded_index):
    return[i for i in range(len(data)) if i != excluded_index]

#### RMSE
def rmse(observed, predicted):
    residuals = predicted - observed
    residuals_sq = residuals**2
    mse = np.mean(residuals_sq)
    rmse = np.sqrt(mse)
    
    return(rmse)

#%% Simulate the Underlying Storm Process

t = np.arange(1200,2020) #800 year record
lmean = np.mean(mean_rate_C[:800]) / 100  #use average rate parameter from compilation
lstd = np.std(mean_rate_C[:800]) /100 # use average std of rate parameter from compilation
tau = 150 #use 150 as the lambda time variation b/c roughly corresponds to the size of active intervals in the original AM4 record

#simulate underlying storm process
lsim = lmean+lstd*np.sin(2*np.pi*t/tau)

#randomly sample the underlying process for storm years
strmyrs_sim = []
for i in range(0,len(lsim)):
    strmyrs_sim.append(np.random.poisson(lsim[i])) #draw from a poisson distribution of the sinusoid
strmyrs_sim_bool=np.array(strmyrs_sim)    #convert to array not list

storm_years = t[np.argwhere(strmyrs_sim_bool > 0)]

#plot for sanity check

h=50
fake_rate = rate_parameter(t,storm_years,h)

#Plot Simulated Gaussian Record As Sanity Check
plt.figure(dpi=150)
plt.stem(t,strmyrs_sim, 'k', markerfmt =" ", basefmt=" ", label='known storm years')
plt.plot(t,lsim*100,'tab:blue', linewidth=2, label='underlying storm process')
plt.plot(t, fake_rate, 'dimgray', linewidth=2, label='known rate parameter')
plt.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
plt.ylabel(r'Storm Frequency ($\lambda$)')
plt.xlabel('Year (CE)')
plt.legend()


#%% Make 4 Storm Records

#### make 4 empty storm records
b1= np.zeros(len(storm_years))
b2= np.zeros(len(storm_years))
b3= np.zeros(len(storm_years))
b4= np.zeros(len(storm_years))

#### get indices of deposits that will show up in each core
for i in range(0, len(storm_years)):
    b1[i] = random.random() < 0.8
    b2[i] = random.random() < 0.7
    b3[i] = random.random() < 0.6
    b4[i] = random.random() < 0.5

#use average standard deviation of storm deposits in composite record
dstd = np.mean(np.std(comp_dist,axis=0)) #about 15 years

b1_stormyears = np.squeeze(storm_years[np.argwhere(b1 > 0)])
b2_stormyears = np.squeeze(storm_years[np.argwhere(b2 > 0)])
b3_stormyears = np.squeeze(storm_years[np.argwhere(b3 > 0)])
b4_stormyears = np.squeeze(storm_years[np.argwhere(b4 > 0)])

#### perturb storm years to get the "observed" ages
#perturb storm years by sampling from random uniform distributions assuming the mean is the simulated storm year 
#and width is the AM compilation std
b1_pert = []
b2_pert = []
b3_pert = []
b4_pert = []

#record 1
for i in range(0, len(b1_stormyears)):
    b1_pert.append(int(np.random.uniform(b1_stormyears[i],2*dstd))) #save as integers
b1_pert=np.array(b1_pert)
#record 2
for i in range(0, len(b2_stormyears)):
    b2_pert.append(int(np.random.uniform(b2_stormyears[i],2*dstd))) #save as integers
b2_pert=np.array(b2_pert)
#record 3
for i in range(0, len(b3_stormyears)):
    b3_pert.append(int(np.random.uniform(b3_stormyears[i],2*dstd))) #save as integers
b3_pert=np.array(b3_pert)
#record 4
for i in range(0, len(b4_stormyears)):
    b4_pert.append(int(np.random.uniform(b4_stormyears[i],2*dstd))) #save as integers
b4_pert=np.array(b4_pert)


##### make age distributions for boostrapping
b1_pert_dist = np.nan*np.zeros((1000, len(b1_stormyears)))
b2_pert_dist = np.nan*np.zeros((1000, len(b2_stormyears)))
b3_pert_dist = np.nan*np.zeros((1000, len(b3_stormyears)))
b4_pert_dist = np.nan*np.zeros((1000, len(b4_stormyears)))

#record 1
for i in range(0,len(b1_stormyears)):
    sw = np.random.uniform(1,2,1)
    b1_pert_dist[:,i] = np.random.normal(b1_stormyears[i],dstd*sw,1000)
#record 2
for i in range(0,len(b2_stormyears)):
    sw = np.random.uniform(1,2,1)
    b2_pert_dist[:,i] = np.random.normal(b2_stormyears[i],dstd*sw,1000)
#record 3
for i in range(0,len(b3_stormyears)):
    sw = np.random.uniform(1,2,1)
    b3_pert_dist[:,i] = np.random.normal(b3_stormyears[i],dstd*sw,1000)
#record 4
for i in range(0,len(b4_stormyears)):
    sw = np.random.uniform(1,2,1)
    b4_pert_dist[:,i] = np.random.normal(b4_stormyears[i],dstd*sw,1000)
    
#%% Get medians and percentiles of perturbed distributions for plotting

#### b1
#percentiles
b1_events50 = np.median(b1_pert_dist, axis=0)
b1_events5 = np.percentile(b1_pert_dist, 5, axis=0)
b1_events95 = np.percentile(b1_pert_dist, 95, axis=0)
#errors
lowerr = b1_events50 - b1_events5
higherr = b1_events95 - b1_events50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_b1 = xerr.T

#### b2
#percentiles
b2_events50 = np.median(b2_pert_dist, axis=0)
b2_events5 = np.percentile(b2_pert_dist, 5, axis=0)
b2_events95 = np.percentile(b2_pert_dist, 95, axis=0)
#errors
lowerr = b2_events50 - b2_events5
higherr = b2_events95 - b2_events50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_b2 = xerr.T

#### b3
#percentiles
b3_events50 = np.median(b3_pert_dist, axis=0)
b3_events5 = np.percentile(b3_pert_dist, 5, axis=0)
b3_events95 = np.percentile(b3_pert_dist, 95, axis=0)
#errors
lowerr = b3_events50 - b3_events5
higherr = b3_events95 - b3_events50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_b3 = xerr.T

#### b4
#percentiles
b4_events50 = np.median(b4_pert_dist, axis=0)
b4_events5 = np.percentile(b4_pert_dist, 5, axis=0)
b4_events95 = np.percentile(b4_pert_dist, 95, axis=0)
#errors
lowerr = b4_events50 - b4_events5
higherr = b4_events95 - b4_events50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_b4 = xerr.T
    
    
#%% Get Storm Frequency For Each Simulated Record Individually

start_boot = time.time()

iterations = 1000

#### b1
start_b1 = time.time()

l_matrix_b1 = np.zeros((iterations, len(t)))

for ii in range(0,iterations):
    
    #draw a random event age from the perturbed age probability distributions
    amax = len(b1_pert_dist) #maximum index
    amin = 0 #minimum index
    age_ind = np.random.randint(amin, high=amax, size=1) #get random index
    
    #make alternative set of event years from the simulated distributions
    eventyrs_temp = b1_pert_dist[age_ind,:]
    eventyrs_int = np.squeeze(eventyrs_temp.astype(int)) #convert to integer
    
    #create a new set of event years with replacement
    rmax = len(eventyrs_int) #maximum index
    rmin = 0 #minimum index
    event_ind = np.random.randint(rmin, high=rmax, size=len(b1_pert)) 
    eventyrs_boot = eventyrs_int[event_ind]
    
    #perform rate parameter analysis on the new timeseries of event years
    l_temp = rate_parameter(t, eventyrs_boot, h)
    
    l_matrix_b1[ii,:] = l_temp

mean_rate_b1 = np.mean(l_matrix_b1, axis=0)
print('time to bootstrap b1 is ', (time.time() - start_b1)/60, ' minutes')

#### b2
start_b2 = time.time()

l_matrix_b2 = np.zeros((iterations, len(t)))

for ii in range(0,iterations):
    
    #draw a random event age from the perturbed age probability distributions
    amax = len(b2_pert_dist) #maximum index
    amin = 0 #minimum index
    age_ind = np.random.randint(amin, high=amax, size=1) #get random index
    
    #make alternative set of event years from the simulated distributions
    eventyrs_temp = b2_pert_dist[age_ind,:]
    eventyrs_int = np.squeeze(eventyrs_temp.astype(int)) #convert to integer
    
    #create a new set of event years with replacement
    rmax = len(b2_pert) #maximum index
    rmin = 0 #minimum index
    event_ind = np.random.randint(rmin, high=rmax, size=len(b2_pert)) 
    eventyrs_boot = eventyrs_int[event_ind]
    
    #perform rate parameter analysis on the new timeseries of event years
    l_temp = rate_parameter(t, eventyrs_boot, h)
    
    l_matrix_b2[ii,:] = l_temp

mean_rate_b2 = np.mean(l_matrix_b2, axis=0)

print('time to bootstrap b2 is ', (time.time() - start_b2)/60, ' minutes')

#### b3
start_b3 = time.time()

l_matrix_b3 = np.zeros((iterations, len(t)))

for ii in range(0,iterations):
    
    #draw a random event age from the perturbed age probability distributions
    amax = len(b3_pert_dist) #maximum index
    amin = 0 #minimum index
    age_ind = np.random.randint(amin, high=amax, size=1) #get random index
    
    #make alternative set of event years from the simulated distributions
    eventyrs_temp = b3_pert_dist[age_ind,:]
    eventyrs_int = np.squeeze(eventyrs_temp.astype(int)) #convert to integer
    
    #create a new set of event years with replacement
    rmax = len(b3_pert) #maximum index
    rmin = 0 #minimum index
    event_ind = np.random.randint(rmin, high=rmax, size=len(b3_pert)) 
    eventyrs_boot = eventyrs_int[event_ind]
    
    #perform rate parameter analysis on the new timeseries of event years
    l_temp = rate_parameter(t, eventyrs_boot, h)
    
    l_matrix_b3[ii,:] = l_temp

mean_rate_b3 = np.mean(l_matrix_b3, axis=0)

print('time to bootstrap b3 is ', (time.time() - start_b3)/60, ' minutes')

#### b4
l_matrix_b4 = np.zeros((iterations, len(t)))

for ii in range(0,iterations):
    
    #draw a random event age from the perturbed age probability distributions
    amax = len(b4_pert_dist) #maximum index
    amin = 0 #minimum index
    age_ind = np.random.randint(amin, high=amax, size=1) #get random index
    
    #make alternative set of event years from the simulated distributions
    eventyrs_temp = b4_pert_dist[age_ind,:]
    eventyrs_int = np.squeeze(eventyrs_temp.astype(int)) #convert to integer
    
    #create a new set of event years with replacement
    rmax = len(b4_pert) #maximum index
    rmin = 0 #minimum index
    event_ind = np.random.randint(rmin, high=rmax, size=len(b4_pert))  
    eventyrs_boot = eventyrs_int[event_ind]
    
    #perform rate parameter analysis on the new timeseries of event years
    l_temp = rate_parameter(t, eventyrs_boot, h)
    
    l_matrix_b4[ii,:] = l_temp

mean_rate_b4 = np.mean(l_matrix_b4, axis=0)

print('time to bootstrap individually is ', (time.time() - start_boot)/60, ' minutes')


#%% Annually Bin PDFs

#create empty lists
record_years = []
record_dists = []

# Record 1

#make count matrix of size #of years in record by # of events
b1_count_matrix = np.zeros((len(t), np.size(b1_pert_dist,1)))
for i in range(0, np.size(b1_pert_dist,1)):
    event = b1_pert_dist[:,i]
    # make the pdfs into annual bins
    min_age = int(min(t)) #minimum year
    max_age = int(max(t)) #maximum year
    bin_edges = np.arange(min_age-1, max_age+1) #edges of bins for the histogram
    count, bins_count = np.histogram(event, bins=bin_edges, density=True) #density true normalizes to one
    bins = bins_count[:-1] #remove last bin edge so can plot 1:1 with counts
    b1_count_matrix[:,i] = count #PDF

record_years.append(t)
record_dists.append(b1_pert_dist)

# get probability of event at each year by summing the count matrix across the annual dimension
b1_prob = np.sum(b1_count_matrix, axis=1)


#################################################

# Record 2

#make count matrix of size #of years in record by # of events
b2_count_matrix = np.zeros((len(t), np.size(b2_pert_dist,1)))
for i in range(0, np.size(b2_pert_dist,1)):
    event = b2_pert_dist[:,i]
    # make the pdfs into annual bins
    min_age = int(min(t)) #minimum year
    max_age = int(max(t)) #maximum year
    bin_edges = np.arange(min_age-1, max_age+1) #edges of bins for the histogram
    count, bins_count = np.histogram(event, bins=bin_edges, density=True) #density true normalizes to one
    bins = bins_count[:-1] #remove last bin edge so can plot 1:1 with counts
    b2_count_matrix[:,i] = count #PDF

record_years.append(t)
record_dists.append(b2_pert_dist)

# get probability of event at each year by summing the count matrix across the annual dimension
b2_prob = np.sum(b1_count_matrix, axis=1)


#################################################

# Record 3

#make count matrix of size #of years in record by # of events
b3_count_matrix = np.zeros((len(t), np.size(b3_pert_dist,1)))
for i in range(0, np.size(b3_pert_dist,1)):
    event = b3_pert_dist[:,i]
    # make the pdfs into annual bins
    min_age = int(min(t)) #minimum year
    max_age = int(max(t)) #maximum year
    bin_edges = np.arange(min_age-1, max_age+1) #edges of bins for the histogram
    count, bins_count = np.histogram(event, bins=bin_edges, density=True) #density true normalizes to one
    bins = bins_count[:-1] #remove last bin edge so can plot 1:1 with counts
    b3_count_matrix[:,i] = count #PDF


record_years.append(t)
record_dists.append(b3_pert_dist)

# get probability of event at each year by summing the count matrix across the annual dimension
b3_prob = np.sum(b1_count_matrix, axis=1)

#################################################

# Record 4

#make count matrix of size #of years in record by # of events
b4_count_matrix = np.zeros((len(t), np.size(b4_pert_dist,1)))
for i in range(0, np.size(b4_pert_dist,1)):
    event = b4_pert_dist[:,i]
    # make the pdfs into annual bins
    min_age = int(min(t)) #minimum year
    max_age = int(max(t)) #maximum year
    bin_edges = np.arange(min_age-1, max_age+1) #edges of bins for the histogram
    count, bins_count = np.histogram(event, bins=bin_edges, density=True) #density true normalizes to one
    bins = bins_count[:-1] #remove last bin edge so can plot 1:1 with counts
    b4_count_matrix[:,i] = count #PDF

record_years.append(t)
record_dists.append(b4_pert_dist)

# get probability of event at each year by summing the count matrix across the annual dimension
b4_prob = np.sum(b1_count_matrix, axis=1)


#### Get Event Bounds   
b1_eventbounds = np.nan * np.zeros((np.size(b1_pert_dist, axis=1),2))
for i in range(0,len(b1_eventbounds)):
    #get minimum and maximum ages of each deposit from the distributions
    b1_eventbounds[i,0] = int(np.min(b1_pert_dist[:,i])) #minimum age
    b1_eventbounds[i,1] = int(np.max(b1_pert_dist[:,i])) #maximum age

b2_eventbounds = np.nan * np.zeros((np.size(b2_pert_dist, axis=1),2))
for i in range(0,len(b2_eventbounds)):
    #get minimum and maximum ages of each deposit from the distributions
    b2_eventbounds[i,0] = int(np.min(b2_pert_dist[:,i])) #minimum age
    b2_eventbounds[i,1] = int(np.max(b2_pert_dist[:,i])) #maximum age
    
b3_eventbounds = np.nan * np.zeros((np.size(b3_pert_dist, axis=1),2))
for i in range(0,len(b3_eventbounds)):
    #get minimum and maximum ages of each deposit from the distributions
    b3_eventbounds[i,0] = int(np.min(b3_pert_dist[:,i])) #minimum age
    b3_eventbounds[i,1] = int(np.max(b3_pert_dist[:,i])) #maximum age 
    
b4_eventbounds = np.nan * np.zeros((np.size(b4_pert_dist, axis=1),2))
for i in range(0,len(b4_eventbounds)):
    #get minimum and maximum ages of each deposit from the distributions
    b4_eventbounds[i,0] = int(np.min(b4_pert_dist[:,i])) #minimum age
    b4_eventbounds[i,1] = int(np.max(b4_pert_dist[:,i])) #maximum age  


#%% Winkler et al., 2023 Method
# sum probability of event in each year in 100 year windows

#### Add PDFs across records
pdf_sum = b1_prob + b2_prob + b3_prob + b4_prob

#### sum PDF values over 100 year windows
window_size = 100
#convert array to pandas series
pdf_series = pd.Series(pdf_sum)

#get values for each window
windows = pdf_series.rolling(window_size, min_periods=1, center=True)
mov = windows.sum()
#convert back to numpy
pdf_100 = mov.to_numpy()  


#%% Winkler et al., 2020 Method
#average individual storm frequencies at each timestep

rmean_comp = np.mean(np.column_stack((mean_rate_b1, mean_rate_b2, mean_rate_b3, mean_rate_b4)), axis=1)
    
#%% Determing Overlapping Event Beds
start_cdf = time.time()
sim_len = 1000
perc_thresh = 0.95
record_names = ["b1", "b2", "b3", "b4"]

#going to be 5 loops here
#outer 2 loops loop through the matrix of cores (setting which 2 cores to compare on each loop)
#all deposits in each core will be compared individually to all deposits in the other cores

#this loop sets the anchor core
for i in range(0, len(record_dists)): 
    
    #makes a list called AM2, AM4, etc "overlap matrix" that has numpy arrays embedded of
    #the indeces of overlaps with the 3 other records that are not the title record
    varname = record_names[i] + '_overlap_matrix'
    locals()[varname] = [None] * len(record_dists)
    
    
    #get the record you will compare all the other records to
    #search through every deposit in Record 1 (the title record)
    record1 = record_dists[i] #matrix of ages of each deposit, size is 1000 x # of events
    
    #compare every deposit in Record 1 with all other deposits in the other records
    #no need to compare it to itself, so cut that index out
    #AM2 is 0, AM4 is 1, AM5 is 2, AM7 is 3
    records_to_compare = indices_except1(np.arange(0,len(record_dists)), i)
    
    #this loop sets the core to compare to
    for ii in range(0, len(records_to_compare)):
        
        #index of record to compare to (this way ensures that you're never comparing a record to itself)
        record2_ind = records_to_compare[ii]
        #record to compare to
        record2 = record_dists[record2_ind] #matrix of ages of each deposit, size is 1000 x # of events
    
    
    ####### OK NOW WE HAVE OUR 2 MATRIXES TO COMPARE SO LETS NEST A COUPLE MORE LOOPS
    #the inner 3 loops will loop through every event in the anchor core and compare it to every event in the other 3 cores
    
        #where to save the indices to exclude
        #this will be a vector of indexes in the records that are NOT the title record
        #the loop will create 3 of these
        #indices to exclude variable is all the indices that are overlapping with the associated index in the title record
        #(it's not identifying overlaps per se, just identifying those not flagged as keepers)
        #in this case keepers means they are an independent distribution to be kept as is
        #keepers are only flagged when one age is older more than 95% of the iterations
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
                
                #permute distributions to compare agest randomly
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
                        
                        
                #if one deposit is older than the other more than 95% of the time, it's kept as an independent ('keeper')
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

print('time to run comparison loop is ', (time.time() - start_cdf)/60, ' minutes')

#%% Determine Age Distributions to Be Compiled
#see compilation code for commented description of how this words

sorder_start = time.time()

eventbounds = [b1_eventbounds, b2_eventbounds,
               b3_eventbounds, b4_eventbounds] 

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
        
        #make every di,dj pair into a "frozenset"
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
        overlap_set = set()

        #do it in the first direction (core ci is the anchor core)
        #loop through every deposit in the overlap matrix
        if matrix_ci[cj] is not None: #make sure there are overlaps with that core
        # enumerate returns index and values, so "overlapping" is the numpy array of deposits that overlap in the
        #second core with the deposit you're on in the anchor core 
            for di, overlapping in enumerate(matrix_ci[cj]):
                for dj in overlapping:
                    overlap_set.add((di, int(dj))) #store overlaps as tuple   

        #loop through in other direction to make sure there are no assymetric pairs
        if matrix_cj[ci] is not None: #core 2 is outer loop now
            for dj, overlapping in enumerate(matrix_cj[ci]):
                for di in overlapping:
                    overlap_set.add((int(di), dj))
                    
        #if no overlaps, skip this pair
        if not overlap_set:
            print(f"\n{record_names[ci]} vs {record_names[cj]}: no overlaps")
            continue

        #sort overlapping pairs by distance between them (closest in age first)
        overlap_pairs_sorted = sorted(overlap_set,
            key=lambda p: ( #p is like the iterator
                abs(core_medians[ci][p[0]] - core_medians[cj][p[1]]),  # get distance between ages
                p[0],# deposit 1 (di)
                p[1]))  # deposit 2 (dj)
        
        #now this loop is going to match the deposits that need to be merged
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

            #make sure deposits are not circularly connecting within the same core
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
            merged_list = list(merged)
            for ia in range(len(merged_list)): #loop through all possible merges associated with this deposit
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
            
            #get distances between matched pairs
            age_gap = abs(core_medians[ci][di] - core_medians[cj][dj])  
            
            #print all the overlapping pairs to be merged and their age distances
            print(f"  MATCH {record_names[ci]}[{di}]"
                  f"({core_medians[ci][di]:.0f}) ↔ "
                  f"{record_names[cj]}[{dj}]"
                  f"({core_medians[cj][dj]:.0f})"
                  f"  [Δ = {age_gap:.0f} yr]")

print('time to determine distributions to be compiled is ', (time.time() - sorder_start)/60, ' minutes')

#%% Create Combined Distributions 

#empty array to put the new distributions in
combined_dists = []  
             
#make an empty list of each core name 
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

    # Concatenate 
    combined_dists.append(cmean)
    
    #print which deposits combined and what the median ages were
    node_labels = [(record_names[ci], di) for ci, di in sorted(nodes)]  
    node_meds   = [f"{core_medians[ci][di]:.0f}" for ci, di in sorted(nodes)]  
    print(f"Combined ({len(nodes)} deposits): {node_labels}  "
          f"medians: {node_meds}")


#%% Save Distributions for Rate Parameter Analysis      
b1_keepers = np.sort(np.array(core_keepers.get(0, []), dtype=int))
b2_keepers = np.sort(np.array(core_keepers.get(1, []), dtype=int))
b3_keepers = np.sort(np.array(core_keepers.get(2, []), dtype=int))
b4_keepers = np.sort(np.array(core_keepers.get(3, []), dtype=int))
        

#all distributions have 1000 members
iterations=1000

#make sure there are combinations
if combined_dists:
    tied_matrix = np.zeros((iterations, len(combined_dists))) #size of 1000 x # combined distributions
    
    for i in range(0, len(combined_dists)):
        tied_matrix[:, i] = combined_dists[i]
        
else: #if no combinations just make this empty and we'll ignore it for the final compilation
    tied_matrix = np.empty((iterations, 0))
    
#### make sure there is data to keep
#keeper matrixes can be empty if all deposits from that core get combined, or there are no combinations

def safe_slice(age_dist, keepers):
    if len(keepers)==0:
        return np.empty((age_dist.shape[0],0)) #make empty array
    return age_dist[:,keepers]

#get final distributions
b1_dist_keep = safe_slice(b1_pert_dist, b1_keepers)
b2_dist_keep = safe_slice(b2_pert_dist, b2_keepers)
b3_dist_keep = safe_slice(b3_pert_dist, b3_keepers)
b4_dist_keep = safe_slice(b4_pert_dist, b4_keepers)

#stack combined distributions
composite_distributions = np.hstack((tied_matrix, b1_dist_keep, b2_dist_keep, b3_dist_keep, b4_dist_keep))        


#%% Winkler et al. 2023c Method

#bin combined distributions annually and sum
comp_count_matrix = np.zeros((len(t), np.size(composite_distributions,1)))
for i in range(0, np.size(composite_distributions,1)):
    event = composite_distributions[:,i]
    # make the pdfs into annual bins
    min_age = int(min(t)) #minimum year
    max_age = int(max(t)) #maximum year
    bin_edges = np.arange(min_age-1, max_age+1) #edges of bins for the histogram
    count, bins_count = np.histogram(event, bins=bin_edges, density=True) #density true normalizes to one
    bins = bins_count[:-1] #remove last bin edge so can plot 1:1 with counts
    comp_count_matrix[:,i] = count

cpdf_sum = np.sum(comp_count_matrix, axis=1)
wsize = 100
cpdf_series = pd.Series(cpdf_sum)
    
#get values for each window
windows = cpdf_series.rolling(wsize, min_periods=1, center=True)
mov = windows.sum()
#convert back to numpy
cpdf_100 = mov.to_numpy()


#%% Get Percentiles For Plotting

#### b1
#percentiles
b1_keepers5 = np.percentile(b1_dist_keep, 5, axis=0)
b1_keepers95 = np.percentile(b1_dist_keep, 95, axis=0)
b1_keepers50 = np.median(b1_dist_keep, axis=0)
#errors
lowerr = b1_keepers50 - b1_keepers5
higherr = b1_keepers95 - b1_keepers50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_b1_keep = xerr.T

#### b2
#percentiles
b2_keepers5 = np.percentile(b2_dist_keep, 5, axis=0)
b2_keepers95 = np.percentile(b2_dist_keep, 95, axis=0)
b2_keepers50 = np.median(b2_dist_keep, axis=0)
#errors
lowerr = b2_keepers50 - b2_keepers5
higherr = b2_keepers95 - b2_keepers50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_b2_keep = xerr.T

#### b3
#percentiles
b3_keepers5 = np.percentile(b3_dist_keep, 5, axis=0)
b3_keepers95 = np.percentile(b3_dist_keep, 95, axis=0)
b3_keepers50 = np.median(b3_dist_keep, axis=0)
#errors
lowerr = b3_keepers50 - b3_keepers5
higherr = b3_keepers95 - b3_keepers50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_b3_keep = xerr.T

#### b4
#percentiles
b4_keepers5 = np.percentile(b4_dist_keep, 5, axis=0)
b4_keepers95 = np.percentile(b4_dist_keep, 95, axis=0)
b4_keepers50 = np.median(b4_dist_keep, axis=0)
#errors
lowerr = b4_keepers50 - b4_keepers5
higherr = b4_keepers95 - b4_keepers50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_b4_keep = xerr.T

#### Ties
#percentiles
ties50 = np.median(tied_matrix, axis=0)
ties5 = np.percentile(tied_matrix, 5, axis=0)
ties95 = np.percentile(tied_matrix, 95, axis=0)
#errors
lowerr = ties50 - ties5
higherr = ties95 - ties50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_ties = xerr.T

#%% Reconstruct Compilation Frequency with Bootstrapping 
#initialize empty matrix
lp_matrix = np.zeros((iterations, len(t)))

# start_b2 = time.time()
for ii in range(0,iterations):
    
    #draw a random event age from the Bacon probability distributions
    amax = len(composite_distributions) #maximum index
    amin = 0 #minimum index
    age_ind = np.random.randint(amin, high=amax, size=1) #get random index
    
    #make alternative set of event years from the random distribution iteration
    eventyrs_temp = composite_distributions[age_ind,:]
    eventyrs_int = np.squeeze(eventyrs_temp.astype(int)) #convert to integer
    
    #create a new set of event years with replacement
    rmax = len(eventyrs_int) #maximum index
    rmin = 0 #minimum index
    event_ind = np.random.randint(rmin, high=rmax, size=len(eventyrs_int)) 
    eventyrs_boot = eventyrs_int[event_ind]
    
    #perform rate parameter analysis on the new timeseries of event years
    l_temp = rate_parameter(t, eventyrs_boot, h)
    
    lp_matrix[ii,:] = l_temp
    
#Get statistics
mean_crate = np.mean(lp_matrix, axis=0)
low_crate2 = np.percentile(lp_matrix, 5, axis=0)
low_crate1 = np.percentile(lp_matrix, 16, axis=0)
high_crate2 = np.percentile(lp_matrix, 95, axis=0)
high_crate1 = np.percentile(lp_matrix, 84, axis=0)
    

#%% Plot One Simulation

fig,ax = plt.subplots(4,1, dpi=150, constrained_layout=True)
ax0 = ax[0]
ax1 = ax[1]
ax2 = ax[2]
ax3 = ax[3]

####Plot the Known Storm Process
ax0.stem(t,strmyrs_sim, 'k', markerfmt =" ", basefmt=" ", label='simulated storm years')
ax0.plot(t,lsim*100, 'tab:blue', linewidth=2, label='underlying storm process')
ax0.plot(t, fake_rate, 'dimgray', linewidth=2, label='true rate parameter')
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax0.set_ylabel(r'Storm Frequency ($\lambda$)')
ax0.legend()

####Plot Event Deposit Distributions
b1_y = 4 * np.ones(len(b1_pert))
b2_y = 3 * np.ones(len(b2_pert))
b3_y = 2 * np.ones(len(b3_pert))
b4_y = 1 * np.ones(len(b4_pert))

b1_y2 = 0 * np.ones(len(b1_keepers50))
b2_y2 = 0 * np.ones(len(b2_keepers50))
b3_y2 = 0 * np.ones(len(b3_keepers50))
b4_y2 = 0 * np.ones(len(b4_keepers50))
ties_y2 = 0 * np.ones(len(ties50))

true_y = -1 * np.ones(len(storm_years))

ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='both')
ax1.errorbar(ties50, ties_y2, yerr=None, xerr=xerr_ties, fmt='o', capsize=3,label='ties', color='slategray', alpha=0.7)
ax1.errorbar(b1_keepers50, b1_y2, yerr=None, xerr=xerr_b1_keep, fmt='o',capsize=3, color='cornflowerblue', alpha=0.7) 
ax1.errorbar(b2_keepers50, b2_y2, yerr=None, xerr=xerr_b2_keep, fmt='o', capsize=3,color='mediumblue', alpha=0.7) 
ax1.errorbar(b3_keepers50, b3_y2, yerr=None, xerr=xerr_b3_keep, fmt='o', capsize=3,color='blueviolet', alpha=0.7) 
ax1.errorbar(b4_keepers50, b4_y2, yerr=None, xerr=xerr_b4_keep, fmt='o', capsize=3,color='violet', alpha=0.7) 
ax1.errorbar(b1_events50, b1_y, yerr=None, xerr=xerr_b1, fmt='o', capsize=3,label='r1', color='cornflowerblue', alpha=0.7)
ax1.errorbar(b2_events50, b2_y, yerr=None, xerr=xerr_b2, fmt='o', capsize=3,label='r2', color='mediumblue', alpha=0.7)
ax1.errorbar(b3_events50, b3_y, yerr=None, xerr=xerr_b3, fmt='o', capsize=3,label='r3', color='blueviolet', alpha=0.7)
ax1.errorbar(b4_events50, b4_y, yerr=None, xerr=xerr_b4, fmt='o', capsize=3,label='r4', color='violet', alpha=0.7)
ax1.scatter(storm_years, true_y, label='Simulated Event Times', color='k')
ax1.legend()
# ax1.set_xlim(min(t), max(t))
ax1.set_ylim(-1.5,4.5)


#### Plot Frequency Curves Individually
ax2.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax2.plot(t, mean_rate_b1, color='royalblue', lw=2, label='r1')
ax2.plot(t, mean_rate_b2, color='mediumblue', lw=2, label='r2')
ax2.plot(t, mean_rate_b3, color='blueviolet', lw=2, label='r3')
ax2.plot(t, mean_rate_b4, color='violet', lw=2, label='r4')
ax2.plot(t, mean_crate, color='k', ls='--', lw=2, label='compilation')
# ax2.set_ylim(0,10)
ax2.legend()

#### Plot Final Compilation W/Known Storm Process on Top
ax3.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax3.stem(t,strmyrs_sim, 'k', markerfmt =" ", basefmt=" ", label='simulated storm years')
ax3.plot(t,lsim*100, 'tab:blue', linewidth=2, label='underlying storm process')
ax3.plot(t, fake_rate, 'navy', linewidth=2, label='true rate parameter')
ax3.plot(t, mean_crate, color='k', lw=2, label='compilation')
# ax3.plot(t, pdf_100, color='mediumblue', lw=2, label='Winkler et al., 2023')
ax3.plot(t, cpdf_100, color='mediumblue', linestyle='--',lw=2, label='Winkler et al., 2023, csum')
ax3.plot(t, rmean_comp, color='blueviolet', lw=2, label='Winkler et al., 2020')
ax3.fill_between(t, low_crate2, high_crate2, color = 'k', alpha=0.1, label=r'2$\sigma$')
ax3.fill_between(t, low_crate1, high_crate1, color='k', alpha=0.3, label = r'1$\sigma$')
ax3.set_ylabel(r'Storm Frequency ($\lambda$)')
ax3.legend()
ax3.set_xlabel('Year (CE)')

plt.suptitle('95% MC Threshold')

#%% Bootstrap Psuedostorm Simulation

print('time to start montecarlo on the bootstrapping is ', time.ctime())

start_mc_time = time.time()

# outer loop is the number of simulations (only do 100 simulations but bootstrap 1000 times within the sim)
iterations1 = 100

#### initialize empty arrays
dif_deposit_number = np.nan * np.zeros(iterations1)
dif_deposit_perc = np.nan * np.zeros(iterations1)
RMSE_timeseries = np.nan * np.zeros(iterations1)
RMSE_simulated = np.nan * np.zeros(iterations1)
RMSE_w2020 = np.nan * np.zeros(iterations1)
RMSE_w2023 = np.nan * np.zeros(iterations1)
RMSE_w2023c = np.nan * np.zeros(iterations1)
RMSE_w2020_sim = np.nan * np.zeros(iterations1)
RMSE_w2023_sim = np.nan * np.zeros(iterations1)
RMSE_w2023c_sim = np.nan * np.zeros(iterations1)
faker_outside = np.nan * np.zeros(iterations1)
lsim_outside = np.nan * np.zeros(iterations1)
mean_sim_rate = np.nan * np.zeros((iterations1, len(t)))


iterations = 1000

#### Monte Carlo Simulation
for nn in range(0, iterations1):

    start_time1 = time.time()
    
    #randomly choose storm years for the record
    strmyrs_sim = []
    for i in range(0,len(lsim)):
        strmyrs_sim.append(np.random.poisson(lsim[i])) #draw from a the underlying storm process
    strmyrs_sim_bool=np.array(strmyrs_sim)    #convert to array not list
    
    #get storm years
    storm_years = t[np.argwhere(strmyrs_sim_bool > 0)]

    #simulate psuedostorm frequency
    h=50
    fake_rate = rate_parameter(t,storm_years,h) #Gaussian
    
    #### make 4 empty storm records
    b1= np.zeros(len(storm_years))
    b2= np.zeros(len(storm_years))
    b3= np.zeros(len(storm_years))
    b4= np.zeros(len(storm_years))

    #### get indices of deposits that will show up in each core
    for i in range(0, len(storm_years)):
        b1[i] = random.random() < 0.8
        b2[i] = random.random() < 0.7
        b3[i] = random.random() < 0.6
        b4[i] = random.random() < 0.5

    #use average standard deviation of storm deposits in composite record
    dstd = np.mean(np.std(comp_dist,axis=0)) #about 15 years

    b1_stormyears = np.squeeze(storm_years[np.argwhere(b1 > 0)])
    b2_stormyears = np.squeeze(storm_years[np.argwhere(b2 > 0)])
    b3_stormyears = np.squeeze(storm_years[np.argwhere(b3 > 0)])
    b4_stormyears = np.squeeze(storm_years[np.argwhere(b4 > 0)])

    #### perturb storm years to get the "observed" ages
    b1_pert = []
    b2_pert = []
    b3_pert = []
    b4_pert = []

    #record 1
    for i in range(0, len(b1_stormyears)):
        b1_pert.append(int(np.random.uniform(b1_stormyears[i],2*dstd))) #save as integers
    b1_pert=np.array(b1_pert)
    #record 2
    for i in range(0, len(b2_stormyears)):
        b2_pert.append(int(np.random.uniform(b2_stormyears[i],2*dstd))) #save as integers
    b2_pert=np.array(b2_pert)
    #record 3
    for i in range(0, len(b3_stormyears)):
        b3_pert.append(int(np.random.uniform(b3_stormyears[i],2*dstd))) #save as integers
    b3_pert=np.array(b3_pert)
    #record 4
    for i in range(0, len(b4_stormyears)):
        b4_pert.append(int(np.random.uniform(b4_stormyears[i],2*dstd))) #save as integers
    b4_pert=np.array(b4_pert)


    ##### make age distributions for boostrapping
    b1_pert_dist = np.nan*np.zeros((1000, len(b1_stormyears)))
    b2_pert_dist = np.nan*np.zeros((1000, len(b2_stormyears)))
    b3_pert_dist = np.nan*np.zeros((1000, len(b3_stormyears)))
    b4_pert_dist = np.nan*np.zeros((1000, len(b4_stormyears)))

    #record 1
    for i in range(0,len(b1_stormyears)):
        sw = np.random.uniform(1,2,1)
        b1_pert_dist[:,i] = np.random.normal(b1_stormyears[i],dstd*sw,1000)
    #record 2
    for i in range(0,len(b2_stormyears)):
        sw = np.random.uniform(1,2,1)
        b2_pert_dist[:,i] = np.random.normal(b2_stormyears[i],dstd*sw,1000)
    #record 3
    for i in range(0,len(b3_stormyears)):
        sw = np.random.uniform(1,2,1)
        b3_pert_dist[:,i] = np.random.normal(b3_stormyears[i],dstd*sw,1000)
    #record 4
    for i in range(0,len(b4_stormyears)):
        sw = np.random.uniform(1,2,1)
        b4_pert_dist[:,i] = np.random.normal(b4_stormyears[i],dstd*sw,1000)
        
    
    #### Integrate Age PDFs to Create CDFS
    #here we will integrate manually by binning annually and summing

    #create empty lists
    record_years = []
    record_dists = []
    
    # Record 1
    
    #make count matrix of size #of years in record by # of events
    b1_count_matrix = np.zeros((len(t), np.size(b1_pert_dist,1)))
    for i in range(0, np.size(b1_pert_dist,1)):
        event = b1_pert_dist[:,i]
        # make the pdfs into annual bins
        min_age = int(min(t)) #minimum year
        max_age = int(max(t)) #maximum year
        bin_edges = np.arange(min_age-1, max_age+1) #edges of bins for the histogram
        count, bins_count = np.histogram(event, bins=bin_edges, density=True) #density true normalizes to one
        bins = bins_count[:-1] #remove last bin edge so can plot 1:1 with counts
        b1_count_matrix[:,i] = count #PDF
    
    record_years.append(t)
    record_dists.append(b1_pert_dist)
    
    # get probability of event at each year by summing the count matrix across the annual dimension
    b1_prob = np.sum(b1_count_matrix, axis=1)
    
    
    #################################################
    
    # Record 2
    
    #make count matrix of size #of years in record by # of events
    b2_count_matrix = np.zeros((len(t), np.size(b2_pert_dist,1)))
    for i in range(0, np.size(b2_pert_dist,1)):
        event = b2_pert_dist[:,i]
        # make the pdfs into annual bins
        min_age = int(min(t)) #minimum year
        max_age = int(max(t)) #maximum year
        bin_edges = np.arange(min_age-1, max_age+1) #edges of bins for the histogram
        count, bins_count = np.histogram(event, bins=bin_edges, density=True) #density true normalizes to one
        bins = bins_count[:-1] #remove last bin edge so can plot 1:1 with counts
        b2_count_matrix[:,i] = count #PDF
    
    record_years.append(t)
    record_dists.append(b2_pert_dist)
    
    # get probability of event at each year by summing the count matrix across the annual dimension
    b2_prob = np.sum(b1_count_matrix, axis=1)
    
    
    #################################################
    
    # Record 3
    
    #make count matrix of size #of years in record by # of events
    b3_count_matrix = np.zeros((len(t), np.size(b3_pert_dist,1)))
    for i in range(0, np.size(b3_pert_dist,1)):
        event = b3_pert_dist[:,i]
        # make the pdfs into annual bins
        min_age = int(min(t)) #minimum year
        max_age = int(max(t)) #maximum year
        bin_edges = np.arange(min_age-1, max_age+1) #edges of bins for the histogram
        count, bins_count = np.histogram(event, bins=bin_edges, density=True) #density true normalizes to one
        bins = bins_count[:-1] #remove last bin edge so can plot 1:1 with counts
        b3_count_matrix[:,i] = count #PDF
    
    record_years.append(t)
    record_dists.append(b3_pert_dist)
    
    # get probability of event at each year by summing the count matrix across the annual dimension
    b3_prob = np.sum(b1_count_matrix, axis=1)
    
    #################################################
    
    # Record 4
    
    #make count matrix of size #of years in record by # of events
    b4_count_matrix = np.zeros((len(t), np.size(b4_pert_dist,1)))
    for i in range(0, np.size(b4_pert_dist,1)):
        event = b4_pert_dist[:,i]
        # make the pdfs into annual bins
        min_age = int(min(t)) #minimum year
        max_age = int(max(t)) #maximum year
        bin_edges = np.arange(min_age-1, max_age+1) #edges of bins for the histogram
        count, bins_count = np.histogram(event, bins=bin_edges, density=True) #density true normalizes to one
        bins = bins_count[:-1] #remove last bin edge so can plot 1:1 with counts
        b4_count_matrix[:,i] = count #PDF
    
    record_years.append(t)
    record_dists.append(b4_pert_dist)
    
    # get probability of event at each year by summing the count matrix across the annual dimension
    b4_prob = np.sum(b1_count_matrix, axis=1)
    
    
    #### Get Event Bounds   
    b1_eventbounds = np.nan * np.zeros((np.size(b1_pert_dist, axis=1),2))
    for i in range(0,len(b1_eventbounds)):
        #get minimum and maximum ages of each deposit from the distributions
        b1_eventbounds[i,0] = int(np.min(b1_pert_dist[:,i])) #minimum age
        b1_eventbounds[i,1] = int(np.max(b1_pert_dist[:,i])) #maximum age
    
    b2_eventbounds = np.nan * np.zeros((np.size(b2_pert_dist, axis=1),2))
    for i in range(0,len(b2_eventbounds)):
        #get minimum and maximum ages of each deposit from the distributions
        b2_eventbounds[i,0] = int(np.min(b2_pert_dist[:,i])) #minimum age
        b2_eventbounds[i,1] = int(np.max(b2_pert_dist[:,i])) #maximum age
        
    b3_eventbounds = np.nan * np.zeros((np.size(b3_pert_dist, axis=1),2))
    for i in range(0,len(b3_eventbounds)):
        #get minimum and maximum ages of each deposit from the distributions
        b3_eventbounds[i,0] = int(np.min(b3_pert_dist[:,i])) #minimum age
        b3_eventbounds[i,1] = int(np.max(b3_pert_dist[:,i])) #maximum age 
        
    b4_eventbounds = np.nan * np.zeros((np.size(b4_pert_dist, axis=1),2))
    for i in range(0,len(b4_eventbounds)):
        #get minimum and maximum ages of each deposit from the distributions
        b4_eventbounds[i,0] = int(np.min(b4_pert_dist[:,i])) #minimum age
        b4_eventbounds[i,1] = int(np.max(b4_pert_dist[:,i])) #maximum age  
     
        
     
        
    #### Winkler et al., 2023 Method 
    # sum probability of event in each year in 100 year windows
    # Add PDFs across records
    pdf_sum = b1_prob + b2_prob + b3_prob + b4_prob

    # sum PDF values over 100 year windows
    window_size = 100
    #convert array to pandas series
    pdf_series = pd.Series(pdf_sum)

    #get values for each window
    windows = pdf_series.rolling(window_size, min_periods=1, center=True)
    mov = windows.sum()
    #convert back to numpy
    pdf_100 = mov.to_numpy()  
    
    #### Winkler et al., 2020 Method
    #average individual storm frequencies at each timestep
    rmean_comp = np.mean(np.column_stack((mean_rate_b1, mean_rate_b2, mean_rate_b3, mean_rate_b4)), axis=1)
        
    
    
    
    #### Compare all Age Distributions
    start_cdf = time.time()
    sim_len = 1000
    perc_thresh = 0.95
    record_names = ["b1", "b2", "b3", "b4"]

    #this loop sets the anchor core
    for i in range(0, len(record_dists)): 
        
        varname = record_names[i] + '_overlap_matrix'
        locals()[varname] = [None] * len(record_dists)
        
        
        #get the record you will compare all the other records to
        #search through every deposit in Record 1 (the title record)
        record1 = record_dists[i] #matrix of ages of each deposit, size is 1000 x # of events
        
        #compare every deposit in Record 1 with all other deposits in the other records
        records_to_compare = indices_except1(np.arange(0,len(record_dists)), i)
        
        #this loop sets the core to compare to
        for ii in range(0, len(records_to_compare)):
            
            #index of record to compare to (this way ensures that you're never comparing a record to itself)
            record2_ind = records_to_compare[ii]
            #record to compare to
            record2 = record_dists[record2_ind] #matrix of ages of each deposit, size is 1000 x # of events
        
        
        ####### OK NOW WE HAVE OUR 2 MATRIXES TO COMPARE SO LETS NEST A COUPLE MORE LOOPS
        #the inner 3 loops will loop through every event in the anchor core and compare it to every event in the other 3 cores
        
            #where to save the indices to exclude
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
                    
                    #permute distributions to compare agest randomly
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
                            
                            
                    #if one deposit is older than the other more than 95% of the time, it's kept as an independent ('keeper')
                    if len(r1_older) > sim_len*perc_thresh or len(r2_older) > sim_len*perc_thresh:
                        r2_keep.append(jj)                   
                    
                #now get the indexes to "exclude" from record 2 by taking all the Record 2 inds that are NOT keepers
                r2_allinds = np.arange(0, np.size(record2,1))
                r2_ind = indices_except(r2_allinds, r2_keep)
                
                # if the number of indexes that overlap with this event in record one is NONzero, 
                # append those indexes to the record2 exclude list
                if len(r2_ind) > 0:
                    r2_exclude.append(np.array(r2_ind))
                    
                elif len(r2_ind) == 0:
                    r2_exclude.append([])
                    
                
            #now we have all the indices of the second record that overlap with the first record going through the loop
            #add these to a matrix that we will nest
            locals()[varname][record2_ind]=r2_exclude
            
    print('time to run comparison loop is ', (time.time() - start_cdf)/60, ' minutes')
    
    
    
    #### Preserve Stratigraphic Order
    sorder_start = time.time()
        
    # first create mega list of all the event bounds
    eventbounds = [b1_eventbounds, b2_eventbounds,
                   b3_eventbounds, b4_eventbounds] 
    
    # create mega list of all medians
    core_medians = [[float(np.median(record_dists[ci][:, di]))
         for di in range(np.size(record_dists[ci], axis=1))]  # loop over all deposits in core ci
        for ci in range(len(record_dists))]  # loop over all cores
    
    
    # make set of directly overlapping pairs to be checked later to make sure no transitive overlaps are made
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
    for ci in range(len(record_dists)): # ci: index of first core in pair
        for cj in range(ci + 1, len(record_dists)): # cj: index of second core in pair
        
        #get the overlap matrixes for each core pair (created in the section above)
            rname_ci  = record_names[ci] + '_overlap_matrix'
            rname_cj  = record_names[cj] + '_overlap_matrix'
            matrix_ci = locals()[rname_ci]
            matrix_cj = locals()[rname_cj]
            
            #get the overlapping deposit pairs from each core pair
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
    
            #loop through in other direction to make sure there are no assymetric pairs
            if matrix_cj[ci] is not None: #core 2 is outer loop now
                for dj, overlapping in enumerate(matrix_cj[ci]):
                    for di in overlapping:
                        overlap_set.add((int(di), dj))
                        
            #if no overlaps, skip this pair
            if not overlap_set:
                print(f"\n{record_names[ci]} vs {record_names[cj]}: no overlaps")
                continue
    
            #sort overlapping pairs by distance between medians (closest in age first)
            overlap_pairs_sorted = sorted(overlap_set,
                key=lambda p: ( #p is like the iterator
                    abs(core_medians[ci][p[0]] - core_medians[cj][p[1]]),  # get distance between ages
                    p[0],# deposit 1 (di)
                    p[1]))  # deposit 2 (dj)
            
            #now this loop is going to match the deposits that need to be merged
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
    
                #make sure deposits are not circularly connecting within the same core
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
                merged_list = list(merged)
                for ia in range(len(merged_list)): #loop through all possible merges associated with this deposit
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
                
                #get distances between matched pairs
                age_gap = abs(core_medians[ci][di] - core_medians[cj][dj])  
                
                #print all the overlapping pairs to be merged and their age distances
                print(f"  MATCH {record_names[ci]}[{di}]"
                      f"({core_medians[ci][di]:.0f}) ↔ "
                      f"{record_names[cj]}[{dj}]"
                      f"({core_medians[cj][dj]:.0f})"
                      f"  [Δ = {age_gap:.0f} yr]")

    
    print('time to  determine distributions to compile is ', (time.time() - sorder_start)/60, ' minutes')
    
    #### Create combined distributions 
    
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

        # Concatenate 
        combined_dists.append(cmean)
        
        #print which deposits combined and what the median ages were
        node_labels = [(record_names[ci], di) for ci, di in sorted(nodes)]  
        node_meds   = [f"{core_medians[ci][di]:.0f}" for ci, di in sorted(nodes)]  
        print(f"Combined ({len(nodes)} deposits): {node_labels}  "
              f"medians: {node_meds}")    

    
    #### Save Distributions for Rate Parameter Analysis      
    b1_keepers = np.sort(np.array(core_keepers.get(0, []), dtype=int))
    b2_keepers = np.sort(np.array(core_keepers.get(1, []), dtype=int))
    b3_keepers = np.sort(np.array(core_keepers.get(2, []), dtype=int))
    b4_keepers = np.sort(np.array(core_keepers.get(3, []), dtype=int))
            
    
    #subsample so all distributions have 1000 members
    iterations=1000

    #make sure there are combinations
    if combined_dists:
        tied_matrix = np.zeros((iterations, len(combined_dists))) #size of 1000 x # combined distributions
        
        for i in range(0, len(combined_dists)):
            tied_matrix[:, i] = combined_dists[i]
            
    else: #if no combinations just make this empty and we'll ignore it for the final compilation
        tied_matrix = np.empty((iterations, 0))
        
    #### make sure there is data to keep
    
    def safe_slice(age_dist, keepers):
        if len(keepers)==0:
            return np.empty((age_dist.shape[0],0)) #make empty array
        return age_dist[:,keepers]
    
    #get final distributions
    b1_dist_keep = safe_slice(b1_pert_dist, b1_keepers)
    b2_dist_keep = safe_slice(b2_pert_dist, b2_keepers)
    b3_dist_keep = safe_slice(b3_pert_dist, b3_keepers)
    b4_dist_keep = safe_slice(b4_pert_dist, b4_keepers)
    
    #stack combined distributions
    composite_distributions = np.hstack((tied_matrix, b1_dist_keep, b2_dist_keep, b3_dist_keep, b4_dist_keep))        
   
    
   
    
#### Winkler et al. 2023c 
#bin combined distributions annually and sum
    comp_count_matrix = np.zeros((len(t), np.size(composite_distributions,1)))
    for i in range(0, np.size(composite_distributions,1)):
        event = composite_distributions[:,i]
        # make the pdfs into annual bins
        min_age = int(min(t)) #minimum year
        max_age = int(max(t)) #maximum year
        bin_edges = np.arange(min_age-1, max_age+1) #edges of bins for the histogram
        count, bins_count = np.histogram(event, bins=bin_edges, density=True) #density true normalizes to one
        bins = bins_count[:-1] #remove last bin edge so can plot 1:1 with counts
        comp_count_matrix[:,i] = count
    
    cpdf_sum = np.sum(comp_count_matrix, axis=1)
    wsize = 100
    cpdf_series = pd.Series(cpdf_sum)
    
    #get values for each window
    windows = cpdf_series.rolling(wsize, min_periods=1, center=True)
    mov = windows.sum()
    #convert back to numpy
    cpdf_100 = mov.to_numpy()  

    
#### Reconstruct Compilation Frequency with Bootstrapping 
    #initialize empty matrix
    lp_matrix = np.zeros((iterations, len(t)))
    
    # start_b2 = time.time()
    for ii in range(0,iterations):
        
        #draw a random event age from the Bacon probability distributions
        amax = len(composite_distributions) #maximum index
        amin = 0 #minimum index
        age_ind = np.random.randint(amin, high=amax, size=1) #get random index
        
        #make alternative set of event years from the random distribution iteration
        eventyrs_temp = composite_distributions[age_ind,:]
        eventyrs_int = np.squeeze(eventyrs_temp.astype(int)) #convert to integer
        
        #create a new set of event years with replacement
        rmax = np.size(composite_distributions, axis=1) #maximum index
        rmin = 0 #minimum index
        event_ind = np.random.randint(rmin, high=rmax, size=np.size(composite_distributions, axis=1)) #rmax is exclusive of the final element so add 1
        eventyrs_boot = eventyrs_int[event_ind]
        
        #perform rate parameter analysis on the new timeseries of event years
        l_temp = rate_parameter(t, eventyrs_boot, h)
        
        lp_matrix[ii,:] = l_temp
        
    #Get statistics
    mean_prate = np.mean(lp_matrix, axis=0)
    low_prate2 = np.percentile(lp_matrix, 5, axis=0)
    low_prate1 = np.percentile(lp_matrix, 16, axis=0)
    high_prate2 = np.percentile(lp_matrix, 95, axis=0)
    high_prate1 = np.percentile(lp_matrix, 84, axis=0)
        
    
#### Get Offsets from Original Storm Process

    ### Underlying Storm Process
    #this study
    RMSE_timeseries[nn] = rmse(fake_rate, mean_prate)
    RMSE_simulated[nn] = rmse(lsim, mean_prate)
    #winkler 2023 using these distributions
    RMSE_w2023c[nn] = rmse(fake_rate, cpdf_100)
    RMSE_w2023c_sim[nn] = rmse(lsim, cpdf_100)
    #winkler 2023 raw
    RMSE_w2023[nn] = rmse(fake_rate, pdf_100)
    RMSE_w2023_sim[nn] = rmse(lsim, pdf_100)
    #winkler 2020 freq average
    RMSE_w2020[nn] = rmse(fake_rate, rmean_comp)
    RMSE_w2020_sim[nn] = rmse(lsim, rmean_comp)
    
    #### difference in deposit  numbers
    dif_deposit_number[nn] = len(storm_years) - np.size(composite_distributions, axis=1)
    dif_deposit_perc[nn] = dif_deposit_number[nn]/len(storm_years) *100

  #### Estimate how often the known record falls outside confidence intervals from compilation
    lsim_out = []
    faker_gauss_out = []
    
    ###underlying storm process
    for ii in range(0, len(lsim)):
        if lsim[ii]*100 < low_prate2[ii] or lsim[ii]*100 > high_prate2[ii]:
            lsim_out.append(1)
    
    lsim_outside[nn] = np.sum(lsim_out)/len(lsim)*100
    
    ###simulated gaussian psuedostorm frequency
    for ii in range(0,len(fake_rate)):
        if fake_rate[ii] < low_prate2[ii] or fake_rate[ii] > high_prate2[ii]:
            faker_gauss_out.append(1)
    
    faker_outside[nn] = np.sum(faker_gauss_out)/len(fake_rate)*100
    

    #### Save all compiled frequencies
    mean_sim_rate[nn,:] = mean_prate 
    print('time to run one bootstrap is ', (time.time() - start_time1)/60, ' minutes')  
    print('now on iteration', nn) 
    
print('time to run monte carlo bootstrapping is ', (time.time() - start_mc_time)/60, ' minutes')  


#%% Get Stats

#deposit number
dnum_mean = np.mean(dif_deposit_number)
dnum_std = np.std(dif_deposit_number)

#deposit percent
dper_mean = np.mean(dif_deposit_perc)
dper_std = np.std(dif_deposit_perc)

#idealized frequency outside reconstructed error bars
sim_out_mean = np.mean(lsim_outside)
sim_out_std = np.std(lsim_outside)

#known frequency outside reconstructed error bars
fr_out_mean = np.mean(faker_outside)
fr_out_std = np.std(faker_outside)

#RMSE
RMSE_faker_mean = np.mean(RMSE_timeseries)
RMSE_faker_std = np.std(RMSE_timeseries)
RMSE_sim_mean = np.mean(RMSE_simulated)
RMSE_sim_std = np.std(RMSE_simulated)

#other studies with fake freq
RMSE_w2020_mean = np.mean(RMSE_w2020)
RMSE_w2020_std = np.std(RMSE_w2020)

RMSE_w2023_mean = np.mean(RMSE_w2023)
RMSE_w2023_std = np.std(RMSE_w2023)

RMSE_w2023c_mean = np.mean(RMSE_w2023c)
RMSE_w2023c_std = np.std(RMSE_w2023c)

#other studies with simulated freq
RMSE_w2020_sim_mean = np.mean(RMSE_w2020_sim)
RMSE_w2020_sim_std = np.std(RMSE_w2020)

RMSE_w2023_sim_mean = np.mean(RMSE_w2023_sim)
RMSE_w2023_sim_std = np.std(RMSE_w2023_sim)

RMSE_w2023c_sim_mean = np.mean(RMSE_w2023c_sim)
RMSE_w2023c_sim_std = np.std(RMSE_w2023c_sim)

#%% Save Simulation Data


np.savetxt(r"Simulation_Data/Dep_Number.csv", dif_deposit_number, delimiter=',')
np.savetxt(r"Simulation_Data/Dep_Perc.csv", dif_deposit_perc, delimiter=',')
np.savetxt(r"Simulation_Data/RMSE_Timeseries.csv", RMSE_timeseries, delimiter=',')
np.savetxt(r"Simulation_Data/RMSE_Simulated.csv", RMSE_simulated, delimiter=',')
np.savetxt(r"Simulation_Data/RMSE_w2020.csv", RMSE_w2020, delimiter=',')
np.savetxt(r"Simulation_Data/RMSE_w2023.csv", RMSE_w2023, delimiter=',')
np.savetxt(r"Simulation_Data/RMSE_w2023c.csv", RMSE_w2023c, delimiter=',')
np.savetxt(r"Simulation_Data/RMSE_w2020_sim.csv", RMSE_w2020_sim, delimiter=',')
np.savetxt(r"Simulation_Data/RMSE_w2023_sim.csv", RMSE_w2023_sim, delimiter=',')
np.savetxt(r"Simulation_Data/RMSE_w2023c_sim.csv", RMSE_w2023c_sim, delimiter=',')
np.savetxt(r"Simulation_Data/FakeR_Outside.csv", faker_outside, delimiter=',')
np.savetxt(r"Simulation_Data/Lsim_Outside.csv", lsim_outside, delimiter=',')


print('time to run entire code  is ', (time.time() - start_code)/60, ' minutes')  