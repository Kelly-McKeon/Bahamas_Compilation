#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:24:27 2026

@author: kellymckeon
"""

import time
code_start = time.time() #begin timing code

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 

mac = r"/Users/kellymckeon/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Bahamas/Codes/For_Publication"
pc = r"C:/Users/kelly/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Bahamas/Codes/For_Publication"

# CHANGE DIRECTORY TO WHICHEVER COMPUTER YOU ARE ON
os.chdir(mac)


#%% Load Grainsize Data

#### AM2
#raw data
AM2 = np.genfromtxt(r"AM2_CoreData/AM2_GS.csv", delimiter=",", skip_header=1)
AM2_GS_depth = AM2[:, 0]
AM2_gs63 = AM2[:,1]

#grainsize threshold
AM2_thresh = np.genfromtxt(r"AM2_CoreData/threshold.csv", delimiter=",", skip_header=0)
AM2_movmean = np.genfromtxt(r"AM2_CoreData/moving_mean.csv", delimiter=",", skip_header=0)

#events
AM2_event_gs = np.genfromtxt(r"AM2_CoreData/event_gs.csv", delimiter=",", skip_header=0)
AM2_event_depth = np.genfromtxt(r"AM2_CoreData/event_depth.csv", delimiter=",", skip_header=0)
AM2_event_gs_pub = np.genfromtxt(r"AM2_CoreData/event_gs_pub.csv", delimiter=",", skip_header=0)
AM2_event_depth_pub = np.genfromtxt(r"AM2_CoreData/event_depth_pub.csv", delimiter=",", skip_header=0)

#################################################

#### AM4
#raw data
AM4 = np.genfromtxt(r"AM4_CoreData/AM4_GS.csv", delimiter=",", skip_header=1)
AM4_GS_depth = AM4[:, 0]
AM4_gs63 = AM4[:,1]

#grainsize threshold
AM4_thresh = np.genfromtxt(r"AM4_CoreData/threshold.csv", delimiter=",", skip_header=0)
AM4_movmean = np.genfromtxt(r"AM4_CoreData/moving_mean.csv", delimiter=",", skip_header=0)

#events
AM4_event_gs = np.genfromtxt(r"AM4_CoreData/event_gs.csv", delimiter=",", skip_header=0)
AM4_event_depth = np.genfromtxt(r"AM4_CoreData/event_depth.csv", delimiter=",", skip_header=0)
AM4_event_gs_pub = np.genfromtxt(r"AM4_CoreData/event_gs_pub.csv", delimiter=",", skip_header=0)
AM4_event_depth_pub = np.genfromtxt(r"AM4_CoreData/event_depth_pub.csv", delimiter=",", skip_header=0)

#################################################

#### AM5
#raw data
AM5 = np.genfromtxt(r"AM5_CoreData/AM5_GS.csv", delimiter=",", skip_header=1)
AM5_GS_depth = AM5[:, 0]
AM5_gs63 = AM5[:,1]

#grainsize threshold
AM5_thresh = np.genfromtxt(r"AM5_CoreData/threshold.csv", delimiter=",", skip_header=0)
AM5_movmean = np.genfromtxt(r"AM5_CoreData/moving_mean.csv", delimiter=",", skip_header=0)

#events
AM5_event_gs = np.genfromtxt(r"AM5_CoreData/event_gs.csv", delimiter=",", skip_header=0)
AM5_event_depth = np.genfromtxt(r"AM5_CoreData/event_depth.csv", delimiter=",", skip_header=0)
AM5_event_gs_pub = np.genfromtxt(r"AM5_CoreData/event_gs_pub.csv", delimiter=",", skip_header=0)
AM5_event_depth_pub = np.genfromtxt(r"AM5_CoreData/event_depth_pub.csv", delimiter=",", skip_header=0)

#################################################

#### AM7
#raw data
AM7 = np.genfromtxt(r"AM7_CoreData/AM7_GS.csv", delimiter=",", skip_header=1)
AM7_GS_depth = AM7[:, 0]
AM7_gs250 = AM7[:,1]

#grainsize threshold
AM7_thresh = np.genfromtxt(r"AM7_CoreData/threshold.csv", delimiter=",", skip_header=0)
AM7_movmean = np.genfromtxt(r"AM7_CoreData/moving_mean.csv", delimiter=",", skip_header=0)

#events
AM7_event_gs = np.genfromtxt(r"AM7_CoreData/event_gs.csv", delimiter=",", skip_header=0)
AM7_event_depth = np.genfromtxt(r"AM7_CoreData/event_depth.csv", delimiter=",", skip_header=0)



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

#full age model
ages = np.genfromtxt(r"AM2_CoreData/AM2_ages.txt", skip_header=1)
ages_depth = ages[:,0]
ind = np.isin(ages_depth, AM2_GS_depth)
#get ages in CE
AM2_ages_mean = 1950 - ages[ind,4]
AM2_ages_min = 1950 - ages[ind,1]
AM2_ages_max = 1950 - ages[ind,2]

#c14
AM2_c14 = np.genfromtxt(r"AM2_CoreData/AM2_Calibrated.csv", delimiter=",", skip_header=1)
AM2_c14_depth = AM2_c14[:,0]
AM2_c14_med = AM2_c14[:,1]
AM2_c14_min = AM2_c14[:,2]
AM2_c14_max = AM2_c14[:,3]

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

#full age model
ages = np.genfromtxt(r"AM4_CoreData/AM4_ages.txt", skip_header=1)
ages_depth = ages[:,0]
ind = np.isin(ages_depth, AM4_GS_depth)
#get ages in CE
AM4_ages_mean = 1950 - ages[ind,4]
AM4_ages_min = 1950 - ages[ind,1]
AM4_ages_max = 1950 - ages[ind,2]

#c14
AM4_c14 = np.genfromtxt(r"AM4_CoreData/AM4_Calibrated.csv", delimiter=",", skip_header=1)
AM4_c14_depth = AM4_c14[:,0]
AM4_c14_med = AM4_c14[:,1]
AM4_c14_min = AM4_c14[:,2]
AM4_c14_max = AM4_c14[:,3]

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

#full age model
ages = np.genfromtxt(r"AM5_CoreData/AM5_ages.txt", skip_header=1)
ages_depth = ages[:,0]
ind = np.isin(ages_depth, AM5_GS_depth)
#get ages in CE
AM5_ages_mean = 1950 - ages[ind,4]
AM5_ages_min = 1950 - ages[ind,1]
AM5_ages_max = 1950 - ages[ind,2]

#c14
AM5_c14 = np.genfromtxt(r"AM5_CoreData/AM5_Calibrated.csv", delimiter=",", skip_header=1)
AM5_c14_depth = AM5_c14[:,0]
AM5_c14_med = AM5_c14[:,1]
AM5_c14_min = AM5_c14[:,2]
AM5_c14_max = AM5_c14[:,3]

#################################################

#### AM7
#median ages
AM7_ages_med = np.genfromtxt(r"AM7_CoreData/median_ages.csv", delimiter=",", skip_header=0)
AM7_event_times = np.genfromtxt(r"AM7_CoreData/event_times.csv", delimiter=",", skip_header=0)
#distributions
age_dist = np.genfromtxt(r"AM7_CoreData/AM7_Age_Distributions.csv", delimiter=",", skip_header=1)
age_dist = age_dist[:,1:]
AM7_age_dist = 1950 - age_dist

#full age model
ages = np.genfromtxt(r"AM7_CoreData/AM7_ages.txt", skip_header=1)
ages_depth = ages[:,0]
ind = np.isin(ages_depth, AM7_GS_depth)
#get ages in CE
AM7_ages_mean = 1950 - ages[ind,4]
AM7_ages_min = 1950 - ages[ind,1]
AM7_ages_max = 1950 - ages[ind,2]

#c14
AM7_c14 = np.genfromtxt(r"AM7_CoreData/AM7_Calibrated.csv", delimiter=",", skip_header=1)
AM7_c14_depth = AM7_c14[:,0]
AM7_c14_med = AM7_c14[:,1]
AM7_c14_min = AM7_c14[:,2]
AM7_c14_max = AM7_c14[:,3]

#%% Make Event Threshold Plot

fig, ax = plt.subplots(4,1, figsize=(7,10), constrained_layout=True, dpi=150)
ax0 = ax[0]
ax1 = ax[1]
ax2 = ax[2]
ax3 = ax[3]

#### AM2
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax0.plot(AM2_ages_med, AM2_gs63, label='% > 63',zorder=1)
ax0.plot(AM2_ages_med, AM2_thresh, linestyle='--', lw=1, color='k', label='threshold', zorder=2)
ax0.plot(AM2_ages_med, AM2_movmean, lw=1, color='k', label='moving median', zorder=3)
ax0.scatter(AM2_event_times, AM2_event_gs, marker='*', s=75, color='k', label='events', zorder=4)
ax0.scatter(AM2_event_times_pub, AM2_event_gs_pub, marker='o', s=10, color='tab:red', label='events (Wallace et al., 2019)', zorder=5)
# ax0.set_xlabel('Median Age (CE)')
ax0.set_ylabel(r'% coarse > 63$\mu$m')
ax0.legend()
ax0.set_title('AM2')
ax0.set_xlim(400,2020)

#### AM4
ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax1.plot(AM4_ages_med, AM4_gs63, label='% > 63',zorder=1)
ax1.plot(AM4_ages_med, AM4_thresh, linestyle='--', lw=1, color='k', label='threshold', zorder=2)
ax1.plot(AM4_ages_med, AM4_movmean, lw=1, color='k', label='moving median', zorder=3)
ax1.scatter(AM4_event_times, AM4_event_gs, marker='*', s=75, color='k', label='events', zorder=4)
ax1.scatter(AM4_event_times_pub, AM4_event_gs_pub, marker='o', s=10, color='tab:red', label='events (Wallace et al., 2019)', zorder=5)
# ax1.set_xlabel('Median Age (CE)')
ax1.set_ylabel(r'% coarse > 63$\mu$m')
ax1.legend()
ax1.set_title('AM4')
ax1.set_xlim(400,2020)


#### AM5
ax2.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax2.plot(AM5_ages_med, AM5_gs63, label='% > 63',zorder=1)
ax2.plot(AM5_ages_med, AM5_thresh, linestyle='--', lw=1, color='k', label='threshold', zorder=2)
ax2.plot(AM5_ages_med, AM5_movmean, lw=1, color='k', label='moving median', zorder=3)
ax2.scatter(AM5_event_times, AM5_event_gs, marker='*', s=75, color='k', label='events', zorder=4)
ax2.scatter(AM5_event_times_pub, AM5_event_gs_pub, marker='o', s=10, color='tab:red', label='events (Wallace et al., 2019)', zorder=5)
# ax2.set_xlabel('Median Age (CE)')
ax2.set_ylabel(r'% coarse > 63$\mu$m')
ax2.legend()
ax2.set_title('AM5')
ax2.set_xlim(400,2020)

#### AM2
ax3.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax3.plot(AM7_ages_med, AM7_gs250, label='% > 250',zorder=1)
ax3.plot(AM7_ages_med, AM7_thresh, linestyle='--', lw=1, color='k', label='threshold', zorder=2)
ax3.plot(AM7_ages_med, AM7_movmean, lw=1, color='k', label='moving median', zorder=3)
ax3.scatter(AM7_event_times, AM7_event_gs, marker='*', s=75, color='k', label='events', zorder=4)
ax3.set_xlabel('Median Age (CE)')
ax3.set_ylabel(r'% coarse > 250$\mu$m')
ax3.legend()
ax3.set_title('AM7')
ax3.set_xlim(400,2020)


#%% Make Age Model Plot

fig, ax = plt.subplots(1,4, constrained_layout=True, dpi=150)
ax0 = ax[0]
ax1 = ax[1]
ax2 = ax[2]
ax3 = ax[3]

#### AM2
ax0.plot(AM2_ages_med, AM2_GS_depth,color='k', label='Median Age')
ax0.fill_betweenx(AM2_GS_depth, AM2_ages_min, AM2_ages_max,color='k', alpha=0.1, linewidth=1)
ax0.set_ylabel('Depth (cm)')
ax0.set_xlabel('Calendar Year (CE)')
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax0.tick_params(axis='both', direction='in')
xerror = np.vstack((AM2_c14_max-AM2_c14_med, AM2_c14_med-AM2_c14_min))
ax0.errorbar(AM2_c14_med, AM2_c14_depth, xerr=xerror, fmt='o', color='tab:red', markersize=3.5, capsize=3,label='Radiocarbon Date')
ax0.set_ylim(-10,1800)
ax0.set_xlim(300,2050)
ax0.invert_yaxis()
ax0.legend()
ax0.set_title('AM2')

#### AM4
ax1.plot(AM4_ages_med, AM4_GS_depth,color='k', label='Median Age')
ax1.fill_betweenx(AM4_GS_depth, AM4_ages_min, AM4_ages_max,color='k', alpha=0.1, linewidth=1)
ax1.set_ylabel('Depth (cm)')
ax1.set_xlabel('Calendar Year (CE)')
ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax1.tick_params(axis='both', direction='in')
xerror = np.vstack((AM4_c14_max-AM4_c14_med, AM4_c14_med-AM4_c14_min))
ax1.errorbar(AM4_c14_med, AM4_c14_depth, xerr=xerror, fmt='o', color='tab:red', markersize=3.5, capsize=3,label='Radiocarbon Date')
ax1.set_ylim(-10,1800)
ax1.set_xlim(300,2050)
ax1.invert_yaxis()
ax1.legend()
ax1.set_title('AM4')

#### AM5
ax2.plot(AM5_ages_med, AM5_GS_depth,color='k', label='Median Age')
ax2.fill_betweenx(AM5_GS_depth, AM5_ages_min, AM5_ages_max,color='k', alpha=0.1, linewidth=1)
ax2.set_ylabel('Depth (cm)')
ax2.set_xlabel('Calendar Year (CE)')
ax2.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax2.tick_params(axis='both', direction='in')
xerror = np.vstack((AM5_c14_max-AM5_c14_med, AM5_c14_med-AM5_c14_min))
ax2.errorbar(AM5_c14_med, AM5_c14_depth, xerr=xerror, fmt='o', color='tab:red', markersize=3.5, capsize=3,label='Radiocarbon Date')
ax2.set_ylim(-10,1800)
ax2.set_xlim(300,2050)
ax2.invert_yaxis()
ax2.legend()
ax2.set_title('AM5')


#### AM7
ax3.plot(AM7_ages_med, AM7_GS_depth,color='k', label='Median Age')
ax3.fill_betweenx(AM7_GS_depth, AM7_ages_min, AM7_ages_max,color='k', alpha=0.1, linewidth=1)
ax3.set_ylabel('Depth (cm)')
ax3.set_xlabel('Calendar Year (CE)')
ax3.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax3.tick_params(axis='both', direction='in')
xerror = np.vstack((AM7_c14_max-AM7_c14_med, AM7_c14_med-AM7_c14_min))
ax3.errorbar(AM7_c14_med, AM7_c14_depth, xerr=xerror, fmt='o', color='tab:red', markersize=3.5, capsize=3,label='Radiocarbon Date')
ax3.set_ylim(-10,1800)
ax3.set_xlim(300,2050)
ax3.invert_yaxis()
ax3.legend()
ax3.set_title('AM7')




