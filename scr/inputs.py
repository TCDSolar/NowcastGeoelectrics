#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:16:58 2017

Input values for modelling and plotting Geoelectric fields realtime

Author: Joan Campnaya, John Malone-Leigh
"""
import numpy as np

###############################################################################
# 1) Input variables

# 1.1) Main Paths
#main_path = '/mnt/data.magie.ie/Python_Scripts/Geo_Electrics_realtime/'
main_path = r'C:\Users\Dunsink\Documents\Python Scripts/Geo_Electrics_realtime_houdini/'

#1.2) padding length for realtime
padding=105 #setting padding length


# 1.2.2) if 1 will not compute interpolated magnetic fields
avoid_comp_secs = 0

# 1.3) Periods of interest
# 1.3.1) maximum period to analyse (seconds)
hi = 10 ** 4.2  

# 1.3.2) minimum period to analyse (seconds)
low = 10 #**2

# 1.4) Sampling rate (seconds)
samp = 60.

# 1.5) Time series properties 
# 1.5.1) length of the time series
#Will have to read to magnetometer reading script

# 1.5.2) Starting point for the analysis
mint= 0

# 1.5.3) End point for the analysis
maxt= -1  

# 1.6) Area of interest for SECS interpolation
secswest, secseast, secssouth, secsnorth = -15, 15, 43, 65  

# 1.7) Ref. magnetic sites to compute e_fields (Approach #2)    


rmf=np.loadtxt(main_path +'in/Observatories.dat', usecols=[0], dtype='str')
 

try: #testing for only one site
    rmf_test=rmf[0]
except:
    rmf=[str(rmf)]
#rmf = ['ARM'] 
#rmf=['BIR']
###############################################################################
# 2) Additional inputs
# No need to modify them if following the suggested structure and parameters 
# from Campanya et al., 2018

# 2.1) Errors
# 2.1.1) Error floor for the Non-plane wave approximation
e_nvpwa = 1 / np.sqrt(10.0 ** (10.0 / 10.0)) 

# 2.1.2) Error floor for the MT and quasi-MT tensor relationships
ef_tf = 10e-2
 
# 2.1.3)  Error floor for H tensor relationship
ef_h = 2e-2

# 2.2) Statistics for error propagation
stat=1000


# 2.3) Paths of interest 
# 2.3.1) Folder with data from a particluar geomagnetic storm
data_path = main_path + 'in/data/realtime/' 


# 2.3.3) Folder with electric field time series
e_path = data_path + 'E/'  

# 2.3.4) Folder with input parameters
in_path = main_path + 'in/' 

# 2.3.5) Folder with output parameters
out_path = main_path + 'out/'
                          

# 2.3.7) Folder with inputs - outputs for SECS interpolation
secs_path =  out_path + 'SECS/'

# 2.4) Files with sites of interest
# 2.4.1) Magnetic observatories
obs_f = 'Observatories.dat'

# 2.4.2) Sites where to calculate the electric fields
sit_f = 'sites_interest2.dat'

#2.5 Correction curve

correction_c=np.loadtxt(main_path+'scr/corrections.csv',usecols=0)

#2.6 Length of the video (1 frame = 1 minute)
video_length=800 #no of frames(minutes) for video

#2.7 Mode of electric field map plots

p_mode='efield' #options are 'efield', 'galvanic' 'std', 'galvanicstd' std=standard devaiation
