#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:09:12 2025

@author: kellymckeon
"""

import time
code_start = time.time() #begin timing code

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


mac = r"/Users/kellymckeon/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Bahamas/Codes/For_Publication"
pc = r"C:/Users/kelly/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Bahamas/Codes/For_Publication"

# CHANGE DIRECTORY TO WHICHEVER COMPUTER YOU ARE ON
os.chdir(mac)


#%% Load Data
#### grainsize data 
AM5 = np.genfromtxt(r"AM5_CoreData/AM5_GS.csv", delimiter=",", skip_header=1)
GS_depth = AM5[:, 0]
gs63 = AM5[:,1]

#### age data
#choose age model slumps based on the threshold set in the next section
#if you have not run the age model yet, comment out this section and return to it 
#after running the section below

ages = np.genfromtxt(r"AM5_CoreData/AM5_ages.txt", skip_header=1)
ages_depth = ages[:,0]
ind = np.isin(ages_depth, GS_depth)

#ages in CE
ages_med = 1950 - ages[ind,3]
ages_mean = 1950 - ages[ind,4]
ages_min = 1950 - ages[ind,1]
ages_max = 1950 - ages[ind,2]

#%% Look at GrainSize Data

#### Histogram
plt.figure(dpi=150)
plt.grid(color='lightgrey', ls=':', lw=0.7, zorder=0)
plt.hist(gs63, bins=50, histtype='bar', ec='k')
plt.xlabel(r'% coarse (> 63 $\mu$m)')
plt.ylabel('counts')
plt.title('AM5 Coarse Distribution')
plt.show()

#save counts
count, bins_count = np.histogram(gs63, bins=100)

#### PDF/CDF

#calculate PDF
pdf = count / sum(count)

#plot
plt.figure(dpi=150)
plt.grid(color='lightgrey', ls=':', lw=0.7, zorder=0)
plt.plot(bins_count[1:], pdf, color="red")
plt.axvline(np.percentile(gs63, 68), color='k', linestyle='dashed', label='68 percentile')
plt.axvline(np.percentile(gs63, 95), color='gray', linestyle='dashed', label='95 percentile')
plt.xlabel(r'% coarse (> 63 $\mu$m)')
plt.title('AM5')
plt.legend()
plt.show()


#%% Create Slump Threshold

# Wallace et al. 2019 uses a threshold of 10% to exclude from the moving average
# Here use 95th percentile instead to standardize across cores

#get summary stats of coarse fraction
sd63 = np.std(gs63)
med63 = np.median(gs63)
avg63 =np.mean(gs63)
thr63 = np.percentile(gs63,95)


# get points of indices exceeding threshold
sd2 = np.argwhere(gs63 >= thr63) #save indices greater than the threshold
gs63_2 = np.copy(gs63) #make a new array 
gs63_2[sd2] = avg63 #replace values above the threshold with the timeseries mean


#### Get Event Depths to Slump in Age Model
event_depths = GS_depth[sd2]
event_gs = gs63[sd2]

#plot for sanity check
fig,ax = plt.subplots(figsize=(4,8),dpi=150)
ax.plot(gs63, GS_depth, label = '%coarse')
ax.scatter(event_gs, event_depths, marker = '*', color='k')
ax.plot(gs63_2, GS_depth, color='k', label= "%coarse < 95th percentile")
plt.axvline(x=thr63, label='threshold',color='k', linestyle = 'dashed')
ax.invert_yaxis()
ax.set_ylabel('depth (cm)')
ax.set_xlabel('%coarse')
ax.set_title('slump threshold (AM5)')
ax.legend()
plt.show()
plt.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)

# save slump starts and slump ends for easy typing into age model
iterator = np.arange(np.size(event_depths))
starts = np.zeros(np.size(iterator)) * np.nan
ends = np.zeros(np.size(iterator)) * np.nan
slump_start = []
slump_end = []


for i in iterator[:-1]:
    starts[i] = event_depths[i+1] - event_depths[i]

for i in iterator:
    if i == 0:
        pass
    else:
        if starts[i] == 1 and starts[i-1] > 1 :
           slump_start.append(event_depths[i])
        if starts[i] == 1 and starts[i+1] > 1:
            slump_end.append(event_depths[i+1])

            
################################################
# Now go into R and add slumps for the depths identified here and run the age model
################################################
            
#%% Get Grainsize Anomaly

window_size = 10 #this is decadal avg assuming 1cm/yr acc rate

# make moving mean
gs63_series = pd.Series(gs63_2) #use the timeseries with the outliers removed so they do not skew the moving mean
windows = gs63_series.rolling(window_size, min_periods=1, center=True)
mov = windows.mean()
#fill the nans left out of the moving average with the ts median
moving_averages = mov.fillna(med63)
  
# Convert pandas series back to numpy 
gs63_mov = moving_averages.to_numpy()

#### Get Anomaly
anomaly63 = gs63-gs63_mov #use the full timeseries not the one with outliers removed

#### Plot 
fig, ax = plt.subplots(1,2, figsize=(6,8),dpi=150,constrained_layout=True)
ax0=ax[0]
ax1=ax[1]

#moving mean
ax0.plot(gs63, GS_depth, label = 'AM5 %Coarse')
ax0.plot(gs63_mov, GS_depth, label = '10pt mov mean')
ax0.invert_yaxis()
ax0.set_ylabel('depth (cm)')
ax0.set_xlabel('% coarse')
ax0.set_title('All % Coarse (AM5)')
ax0.legend()
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')

# anomaly with optional thresholds
ax1.plot(anomaly63, GS_depth)
ax1.axvline(0, color='k', label='zero') #plot zero line
ax1.axvline(np.median(anomaly63)+np.std(anomaly63), color='k', linestyle = 'dashed', label='med+1SD')
ax1.axvline(np.median(anomaly63)+2*np.std(anomaly63), color='k', linestyle = 'dotted', label='med+2SD')
ax1.axvline(np.percentile(anomaly63, 68), color='gray', linestyle='dashed', label='68 prctl')
ax1.axvline(np.percentile(anomaly63, 95), color='gray', linestyle='dotted', label='95 prctl')
ax1.axvline(np.percentile(anomaly63, 85), color='gray', label='85 prctl')
ax1.invert_yaxis()
ax1.set_ylabel('depth (cm)')
ax1.set_xlabel('Coarse Anomaly (%)')
ax1.set_title('Coarse Anomaly (AM5)')
ax1.legend()
ax1.set_xlim(0,25)
ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
            
#%% Create Event Threshold

# Wallace et al., 2019 states the event threshold is the 68th percentile of the anomaly CDF over the historical interval 
# It is more likely that they meant the 68th percentile of the raw % coarse 
# The 68th percentile of the anomaly over the historic interval is nearly zero (not 2.77 as stated in the paper)
# Here we will use the median + 1sd of the entire % coarse anomaly timeseries 
# This metric is easy to standardize across cores


#### get summary statistics
sd63a = np.std(anomaly63)
med63a = np.median(anomaly63)
avg63a =np.mean(anomaly63)
uppera = np.percentile(anomaly63, 95)
lowera = np.percentile(anomaly63, 68) #vary this one to choose event threshold
thresh = med63a + sd63a
athresh = anomaly63 >= thresh


#### Plot anomaly with threshold
fig, ax = plt.subplots(1,2,figsize=(6,8),constrained_layout=True,dpi=150)
ax0=ax[0]
ax1=ax[1]

#moving mean with slump depths marked
ax0.plot(gs63, GS_depth, label='%coarse')
ax0.plot(gs63_mov, GS_depth, label = '10pt filter')
ax0.scatter(event_gs, event_depths, marker = '*', color='k', label='slump depths')
ax0.invert_yaxis()
ax0.set_ylabel('depth (cm)')
ax0.set_xlabel('% coarse')
ax0.legend()
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')

#anomaly with points over the threshold marked
ax1.plot(anomaly63, ages_med, label = 'anomaly')
ax1.scatter(anomaly63[athresh], ages_med[athresh], marker = '*', color='k', label='events')
# ax1.invert_yaxis()
ax1.axvline(thresh, color = 'k', linestyle = 'dashed', label= 'event threshold')
ax1.set_xlabel('coarse anomaly (% > 63)')
ax1.set_ylabel('median age (CE)')
ax1.legend()
plt.show()
plt.xlim(0,30)
plt.suptitle('AM5 Events, % coarse')     
plt.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')



            
#%% Make Booelan Event Index
#this block removes consecutive depths over the threshold as individual events

new_array = np.zeros(np.size(anomaly63)) #make array that will populate with boolean index

event_depth_array = [] #make empty depth array to save event depths in
event_gs_array = [] #make empty gs array to save event grainsizes in
event_anom_array = [] #make empty anomaly array
event_time_array = [] #make empty time array

iterator = np.arange(0,np.size(anomaly63))

#### Set Each Depth as Event or Not
#use the lower threshold from above 

for i in iterator:
    if anomaly63[i] < thresh:
        new_array[i] = 0
        
    #here's the important one
    elif anomaly63[i] >= thresh and anomaly63[i-1] < thresh and anomaly63[i-2] < thresh: #only code it as an event if the cm before was NOT an event
        new_array[i] = 1
        event_depth_array.append(GS_depth[i])
        event_gs_array.append(gs63[i])
        event_anom_array.append(anomaly63[i])
        event_time_array.append(ages_med[i])
        
    elif anomaly63[i] >= thresh and anomaly63[i-1] >= thresh: #if both consecutive cm are events, only code the first one as an event
        new_array[i] = 0
        
    elif anomaly63[i] >= thresh and anomaly63[i+1] >= thresh:
        new_array[i] = 0


bool_anom = new_array #make boolean array 


#### plot as sanity check
fig, ax = plt.subplots(1,2,figsize=(6,8),constrained_layout=True,dpi=150)
ax0=ax[0]
ax1=ax[1]

#plot raw GS with events marked (should be just the first cm of event)
ax0.plot(gs63, GS_depth, label='% coarse')
ax0.invert_yaxis()
ax0.set_ylabel('depth (cm)')
ax0.set_xlabel('% coarse')
ax0.scatter(event_gs_array, event_depth_array, marker='*', color='k', label='events')
ax0.axvline(np.min(event_gs_array), color = 'k', linestyle = 'dashed', label='minimum event % coarse')
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax0.legend()

#plot anomaly with events marked at median ages
ax1.plot(anomaly63, ages_med, label = 'anomaly')
ax1.scatter(event_anom_array, event_time_array, marker='*', color='k', label='events')
ax1.axvline(thresh, color = 'k', linestyle = 'dashed', label= 'event threshold')
ax1.set_xlabel('coarse anomaly (%)')
ax1.set_ylabel('median age (CE)')
ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax1.legend()
plt.xlim(0,30)
plt.show()

plt.suptitle('AM5 Events')       


#%% Get Published Event Depths
published = np.genfromtxt(r"AM5_CoreData/AM5_pubFreq.csv", delimiter=",", skip_header=1)
allyrs_pub = published[:,0]
strmyrs_pub = published[:,1]
freq_array_pub = published[:,2]

event_depth_pub=[]
for i in range(0, len(strmyrs_pub)):
    if strmyrs_pub[i] == True:
        event_depth_pub.append(GS_depth[i]) #start populating new list with event years based on median of age distributions

ind=[]
for i in range(0, len(event_depth_pub)):
    ind.append(np.where(GS_depth == event_depth_pub[i]))

event_time_pub=[]
event_gs_pub=[]
for i in range(0, len(ind)):
    event_time_pub.append(ages_med[ind[i]])
    event_gs_pub.append(gs63[ind[i]])
        

#%% Plot Moving Threshold With New Events and Published Events

mov_anomaly = gs63_mov + thresh      

fig,ax = plt.subplots(1,1, figsize=(4,2), dpi=150)
ax.plot(ages_med, gs63, label='%>63', zorder=1)
ax.plot(ages_med[11:], mov_anomaly[11:], linestyle='--', linewidth=1, color='k',label='threshold', zorder=2)
ax.plot(ages_med[11:], gs63_mov[11:], linewidth=1, color='k',label='moving mean', zorder=3)
ax.scatter(event_time_array, event_gs_array, marker='*', s=50, color='k', label='events', zorder=4)
ax.scatter(event_time_pub, event_gs_pub, marker='o', s=15, color='r', label='published events', zorder=5)
ax.set_xlabel('median age (YBP)')
ax.set_ylabel('% coarse')
plt.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax.legend()
ax.set_title('AM5')



#%% Save Data 

## data for comparison plot
np.savetxt(r"AM5_CoreData/threshold.csv", mov_anomaly, delimiter=',')
np.savetxt(r"AM5_CoreData/moving_mean.csv", gs63_mov, delimiter=',')
np.savetxt(r"AM5_CoreData/median_ages.csv", ages_med, delimiter=',')
np.savetxt(r"AM5_CoreData/event_times.csv", event_time_array, delimiter=',')
np.savetxt(r"AM5_CoreData/event_gs.csv", event_gs_array, delimiter=',')    
np.savetxt(r"AM5_CoreData/event_times_pub.csv", event_time_pub, delimiter=',')  
np.savetxt(r"AM5_CoreData/event_gs_pub.csv", event_gs_pub, delimiter=',')      

## event depths to export to R
np.savetxt(r"AM5_CoreData/event_depth.csv", event_depth_array, delimiter=',')
np.savetxt(r"AM5_CoreData/event_depth_pub.csv", event_depth_pub, delimiter=',')
#once you have these go into Bacon and save the PDFs for each of these depths to get age ranges on each event

#now export them in R, output distributions from Bacon, and save them

#%% Load Age Distributions from Bacon

age_dist = np.genfromtxt(r"AM5_CoreData/AM5_Age_Distributions.csv", delimiter=",", skip_header=1)
# columns represent events so the PDF for event 2 is indexed [:,3]
age_dist = age_dist[:,1:] #remove first index column of nans
age_dist_CE = 1950 - age_dist

#%% Make Dataset Annual

eventyrs=[]
for i in range(0, len(bool_anom)):
    if bool_anom[i] == True:
        eventyrs.append(ages_med[i]) #start populating new list with event years based on median of age distributions
        
eventyrs = np.array(eventyrs) #convert to array
eventyrs = eventyrs.astype(int) #round to nearest year (no decimals)

#make new array with evently spaced years
allyrs = np.arange(min(ages_med), max(ages_med), 1, dtype=int) #boundaries are the minimum and maximum median ages, stepping by 1 year ata  time
strmyrs = np.isin(allyrs, eventyrs) #this creates a boolean index of the same size as allyrs, where if the year in allyrs is a storm year (in yrs) it's coded as 1

#make stem plot from boolean index of years with storms
plt.figure(figsize=(6,4), dpi=150)
plt.stem(allyrs,strmyrs, 'k', basefmt=" ")
plt.xlabel('year (CE)')
plt.ylabel('storm')
plt.title('AM5')

# save storm years
np.savetxt(r"AM5_CoreData/storm_years.csv", strmyrs, delimiter=',')
np.savetxt(r"AM5_CoreData/all_years.csv", allyrs, delimiter=',')



#%% Kernel Rate Estimation Using Median Ages of Storms
import time
start_time = time.time()
#set window size
h = 50 #h represents the standard deviation of the sliding bell curve so make it half of your important time period

#### Calculate rate parameter
#create a function for easy bootstrapping later

# input parameters are all the years in your dataset, the event years, and the window size 
def rate_parameter(allyrs, eventyrs, h):
    #initialize the rate parameter
    l = np.nan*(np.zeros(len(allyrs))) #l is lambda as in Mudelsee Eq 3.16
    
    for i in range(0,len(allyrs)): #loop through every year of the dataset
        l[i] = 0 
        for j in range(0, len(eventyrs)):
            y = (allyrs[i] - eventyrs[j]) / h
            l[i] = l[i] + (np.power((2*np.pi), -0.5) * np.exp((-y**2)/2))/h
    return(l*100)

#experiment with some different window sizes
l25 = rate_parameter(allyrs, eventyrs, 25)
l50 = rate_parameter(allyrs, eventyrs, 50)
l100 = rate_parameter(allyrs, eventyrs, 100)

print('time to run kernel rate estimation is ', (time.time() - start_time)/60, ' minutes')  


#%% Plot Rate Parameters With Different Bandwidths

plt.figure(dpi=150)
plt.stem(allyrs,strmyrs, 'k', markerfmt =" ", basefmt=" ")
plt.plot(allyrs, l25, color='tab:blue', linewidth=2, label='25')
plt.plot(allyrs, l50, color='tab:green', linewidth=2, label='50')
plt.plot(allyrs, l100, color='tab:orange',linewidth=2, label='100')
plt.xlabel('Year (CE)')
plt.ylabel(r'Storm Frequency ($\lambda$)')
plt.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)

plt.legend()



#%% Bootstrap Kernel Rate Estimation with Age Uncertainties

start_time = time.time()

#set number of iterations
iterations = 1000
#set window size
h = 50
#initialize empty matrix
l_matrix = np.zeros((iterations, len(allyrs)))
event_matrix_bool = np.nan * np.zeros((iterations, len(allyrs)))

for ii in range(0,iterations):
    
    #draw a random event age from the Bacon probability distributions
    amax = len(age_dist_CE) #maximum index
    amin = 0 #minimum index
    age_ind = np.random.randint(amin, high=amax, size=1) #get random index
    
    #make alternative set of event years from the random age model iteration
    eventyrs_temp = age_dist_CE[age_ind,:]
    eventyrs_int = np.squeeze(eventyrs_temp.astype(int)) #convert to integer
    
    #create a new set of event years with replacement from the alternate event years 
    #keeping the same total number of events
    #preserving stratigraphic order
    rmax = len(eventyrs) #maximum index
    rmin = 0 #minimum index
    event_ind = np.random.randint(rmin, high=rmax, size=len(eventyrs)) #rmax is exclusive of the final element so add 1
    eventyrs_boot = eventyrs_int[event_ind]
    strmyrs_boot = np.isin(allyrs, eventyrs_boot) #boolean index of event years
    
    #perform rate parameter analysis on the new timeseries of event years
    l_temp = rate_parameter(allyrs, eventyrs_boot, h)
    
    l_matrix[ii,:] = l_temp
    event_matrix_bool[ii,:] = strmyrs_boot

print('time to run bootstrapping is ', (time.time() - start_time)/60, ' minutes')    


    
    
#%% Plot Rate Parameter with Age Uncertainties   
# take statistics of rate parameter for plotting
mean_rate_ages = np.median(l_matrix, axis=0)
low_rate2_ages = np.percentile(l_matrix, 5, axis=0)
low_rate1_ages = np.percentile(l_matrix, 16, axis=0)
high_rate2_ages = np.percentile(l_matrix, 95, axis=0)
high_rate1_ages = np.percentile(l_matrix, 84, axis=0)


fig,ax = plt.subplots(figsize=(12,4), dpi=150)
ax.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major',zorder=0)
ax.plot(allyrs, mean_rate_ages, color='k', label = 'bootstrapped mean')
ax.fill_between(allyrs, low_rate2_ages, high_rate2_ages, color = 'k', alpha=0.3, label=r"2$\sigma$")
ax.fill_between(allyrs, low_rate1_ages, high_rate1_ages, color='k', alpha=0.5, label = r"1$\sigma$")
ax.plot(allyrs, l50, color='gray',label='median ages')
ax.legend()
ax.set_xlabel('Years (CE)')
ax.set_ylabel(r'Storm Frequency ($\lambda$)')
ax.set_title('Rate Parameter with Age Uncertainty (AM5)')


#%% Save kernel data for comparison later 
np.savetxt(r"AM5_CoreData/l_matrix.csv", l_matrix, delimiter=',')
np.savetxt(r"AM5_CoreData/event_matrix_bool.csv", event_matrix_bool, delimiter=',')


print('time to run code is ', (time.time() - code_start)/60, ' minutes')  
