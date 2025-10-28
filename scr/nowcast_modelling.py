#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Algorithm to modelled geoelectric fields at a particular site based on data 
from magnetic observatories and geophysical parameters 
 to measured data.

 Original author: Joan Campanya i Llovet (Trinity College Dublin) [joan.campanya@tcd.ie]
 08 - 2018, Dublin, Ireland.
 
 Adapated for real-time plotting by John Malone-Leigh [jmalone@cp.dias.ie]

 Inputs ("in" folder)
   1) Sites of interest where to measure E fields ('sites_interest.dat')
   2) Magnetometers to be used for for modelling E_fields('Observatories.dat')
   3) Electromagnetic transfer functions (TF folder)
   4) Magnetic time series at the magnetic observatories (B folder within
      the folder for each storm)

 Input values are specified in the input.py file in the "scr" folder. 
 The program will read the input from this file and look for the corresponsing
 files in the "in" folder

 Outputs (in "out" folder)
   1) Modelled E fields at the sites of interest
   2) Results inputs and outputs for the Spherical elementary current systems 
     (SECS folder)
     
Altgorithm related to Campanya et al. 2018 publication in Space Weather AGU Journal
##############################################################################
Realtime Script v1:

Script by J Malone-Leigh updated from script by Campanya 2019 to work in real time

Changes:
1) Reads in MagIE file format and INTERMAGNET format
    o Converts MagIE second data to minute data
    o Reads in two most recent days of mag data( and end of third)
    o Removes bad magnetic field data (B=99999)
2) Added 105 minutes of padding to magnetics to reduce error at end
3) Read in last two hours of three days ago to reduce error at start
4) Created new plots for realtime E
   
                                                          
Geopandas install instructions: 
    o Geopandas currently cannot being installed properly at the moment
    o Current version only works on Windows (and sometimes Mac).
    o Install instructions https://hatarilabs.com/ih-en/how-to-install-python-geopandas-on-anaconda-in-windows-tutorial (download files in currect order)
    o Alternatively, use Basemap or other mapping modules, with Basemap alternative function nowcast_mapping_basemap provided

"""
import os
#labelling Proj_Lib to prevent Basemap error
os.environ['PROJ_LIB'] = ""
import numpy as np
import matplotlib.pyplot as plt
import functions_EM_modelling as femm
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap
import matplotlib
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
from scipy import interpolate
import matplotlib.gridspec as gridspec
from shapely import speedups
import time
import ffmpy
import mapping_geo_library as mpl
import cv2
import os
from os.path import isfile, join
import shutil
import inputs as inp
# disable speedups to make shapely not break later, when making maps
speedups.disable() 
plt.close('all')
plt.style.use('classic')
plt.rc('font', size=11)
########################################################################
#Recording Load times
nowz=datetime.datetime.utcnow()
year_str=str(nowz.year)
month_str="%02d" %(nowz.month)
day_str="%02d" %(nowz.day)
date_str= day_str+'/'+month_str+'/'+year_str
#######################################################################

#defining variables from input file
in_path=inp.in_path
obs_f=inp.obs_f
rmf=inp.rmf
secs_path=inp.secs_path
samp=inp.samp
hi=inp.hi
low=inp.low
padding=inp.padding
sit_f=inp.sit_f
stat=inp.stat
#tf_path=inp.tf_path
avoid_comp_secs=inp.avoid_comp_secs
ef_tf=inp.ef_tf
e_nvpwa=inp.e_nvpwa
out_path=inp.out_path
ef_h=inp.ef_h
video_length=inp.video_length
correction_c=inp.correction_c
secswest, secseast, secssouth, secsnorth = (inp.secswest,
                                            inp.secseast,
                                            inp.secssouth,
                                            inp.secsnorth)
main_path=inp.main_path
p_mode=inp.p_mode
fname = "SECS_"

if p_mode=='efield' or p_mode=='std':
    tf_path = main_path + 'in/data/TF/'#+'no_correction_galvanic_distortion/' 
else:
    #with galvanic tensors
    tf_path = main_path + 'in/data/TF/no_correction_galvanic_distortion/' 
#######################################################################

# 2) Read and Compute B fields
print('Read and compute B fields')

total_activity=[]

 #2.1   
#downloading magnetic field data from MagIE website
folders,value_test,MagIE=femm.save_magnetics(in_path,str(obs_f),rmf)
#comment out to use MagIe data


#manually setting to folder to work with older sites
#comment out to use live sites for MagIe website
"""
folders=[]
value_test=[1,1,1]
MagIE=[1,1,0]
for i in rmf:
    folders1=[]
    folder=in_path+'data/2017-09/'
    folders1.append(folder+str(i).lower()+'20170907.txt')
    folders1.append(folder+str(i).lower()+'20170908.txt')
    folders1.append(folder+str(i).lower()+'20170909.txt')
    for j in folders1:
        
        folders.append(j)"""

#loading magnetic field data
mh_obs, h_obs, length, DATE, HOUR = femm.load_magnetics(
            in_path,
            str(obs_f),
            secs_path,
            samp,
            hi,
            low,
            1,rmf,folders,value_test,MagIE,padding
            )
len_record=length

###################################################
#3) Using Spherical Elementary Current Systems
#What this section does is:
#   o Checks which magnetometers have data available
#   o If 2 or more performs an SECS interpolation for the magnetic fields
#   o saves files based on sites electric fields are calculated at

# Open observatories datafile and select the names of the sites
name_secs = np.loadtxt(in_path + str(sit_f),
                      usecols = (0,),
                      unpack = True,
                      skiprows = 0,
                      dtype = str
                      )
#loading latif=tude andlongitude of sites
#set magnetometer sites to use here, edit in in/observatories.dat
lats_secs, lons_secs = np.loadtxt(in_path + str(sit_f),
                                usecols=(1, 2),
                                unpack=True,
                                skiprows = 0)
#  Computing SECS


if avoid_comp_secs != 1: # In case it was computed already
    print('interpolate magnetic fields using SECS')
    
    if len(rmf) >1:
        exec(open("SECS_interpolation.py").read())
        #from SECS_interpolation_houdini import SECS_Interpolation
        #obs_bx_secs,obs_by_secs=SECS_Interpolation(out_path,secswest,secssouth,secsnorth,secseast,
        #               length,lons_secs,lats_secs)
        obs_bx_secs = np.array(obs_bx_secs)
        obs_by_secs = np.array(obs_by_secs)

    else:
        #If only one magnetics site present, SECS will not be used
        print('Only 1 magnetics, No SECS')
        #Using same magnetics for all of Ireland

            
        obs_bx_secs=np.loadtxt(secs_path+fname+rmf[0]+'_realtime.dat',usecols=[0])
        obs_by_secs=np.loadtxt(secs_path+fname+rmf[0]+'_realtime.dat',usecols=[1])

    #Try loops added to deal with a) only one magnetic site
    #                             b) only one electric field site
    
    #Saving SECS data
    #try loop added in case only one site present
    try:
        #Adding one more try loop if only 1 magnetics used for multiple electrics
        try:
            for value, ip in enumerate(name_secs):
                #opening file to save to
                secsx_id = open(secs_path + fname + name_secs[value]+"_magBx.dat", 'w+')
                #saving
                np.savetxt(secsx_id, obs_bx_secs[:,value], fmt=[ '%10.3f'])
        
                secsx_id.close()
                #opening fiel to save to
                secsy_id = open(secs_path + fname+name_secs[value]+ "_magBy.dat", 'w+')
                #saving
                np.savetxt(secsy_id, obs_by_secs[:,value], fmt=[ '%10.3f'])
        
                secsy_id.close()
        except: #When only one sites Electrics is present
            #opening file to save to
            secsx_id = open(secs_path + fname + str(name_secs)+ "_magBx.dat", 'w+')
            #saving
            np.savetxt(secsx_id, obs_bx_secs, fmt=[ '%10.3f'])
        
            secsx_id.close()
            #opening file to save to
            secsy_id = open(secs_path + fname+ str(name_secs)+ "_magBy.dat", 'w+')
            #saving
            np.savetxt(secsy_id, obs_by_secs, fmt=[ '%10.3f'])
        
            secsy_id.close()
    except:
        #opening file to save to
        secsx_id = open(secs_path + fname + rmf[0]+ "_magBx.dat", 'w+')
        #saving
        np.savetxt(secsx_id, obs_bx_secs, fmt=[ '%10.3f'])
    
        secsx_id.close()
    
        secsy_id = open(secs_path + fname+ rmf[0]+ "_magBy.dat", 'w+')
    
        np.savetxt(secsy_id, obs_by_secs, fmt=[ '%10.3f'])
    
        secsy_id.close()
        
        print('One Magnetics, Multiple Electrics')
        pass

#######################################################################
# 4) Deffine new variables for computing the e_fields
#defining empty lists to be filled with E field data
print('Defining new variables for computing electric fields')
# 4.1) Read sites where we want to compute the E fields
e_site, e_lat, e_lon = femm.read_co(in_path + str(sit_f))


# 4.2) Deffine size of several variables needed to compute the E fields
size_a = ([length-1, len(e_site), 2])
size_b = [length-1, len(e_site), len(rmf), 2]

secs_e = np.zeros(size_a)
stdsecs_e = np.zeros(size_a)
std_error = np.zeros(size_a)
av_e_fields = np.zeros(size_a)
std_reg_e = np.zeros(size_a)
av_reg_e = np.zeros(size_a)
std_secs_e = np.zeros(size_a)
av_secs_e = np.zeros(size_a)
std_loc_e = np.zeros(size_a)
av_loc_e = np.zeros(size_a)
c_reg_e = np.zeros(size_b)
c_std_reg_e = np.zeros(size_b)
d_rmf_e = np.zeros(size_b)
std_rmf_e = np.zeros(size_b)

# 4.3) Define variables that will be used for error propagation, including stat

secs_e_fields = np.zeros([length-1, len(e_site), 2, stat])
reg_e_fields = np.zeros([length-1, len(e_site), 2, len(rmf), stat])


size_a = ([length, len(e_site), 2])
size_b = [length, len(e_site), len(rmf), 2]
loc_emt = np.zeros(size_a)
std_loc_emt = np.zeros(size_a)
e_fields = np.zeros([length-1, len(e_site), 2, stat])
comp_e_fields = np.zeros([length, len(e_site), 2, stat])

#######################################################################
# 5) Compute electric fields at the sites of interest
print('Computing electric fields')
for v1, ip1 in enumerate(e_site): # Sites where to calculate the e fields

    # 5.1) Read magnetics computed from the interpolation approach
    try:
        
        if len(rmf)>1:
            
            rmfb = femm.read_rmfb(secs_path + fname ,e_site[v1])
        
        else:
            rmfb=femm.read_rmfb(secs_path + fname ,rmf[0])
            #If only one magnetics use for all Ireland
        #print(e_site+' E fields')
        print(ip1)
        
    except:
        rmfb=femm.read_rmfb(secs_path + fname ,rmf[0])
    # 5.2) Compute the error associated with the interpolation of the magnetic
    # fields (Based on Figure 7, Campanya et al. 2018 )

    error_bf = femm.error_secs_interpolation(
            e_lat[v1],
            e_lon[v1],
            in_path,
            1,
            obs_f
            )
    #print(error_bf)
    # 5.3) Compute the electric fields unsing results from SECS
    ex_secs, ey_secs, std_ex_secs, std_ey_secs = femm.compute_e_fields_secs(
            rmfb[0:len(secs_e)],
            tf_path,
            ip1,
            samp,
            hi,
            low,
            error_bf,
            ef_tf,
            e_nvpwa,
            stat
            )

    sax = np.array([ex_secs]).T;
    say = np.array([ey_secs]).T;
    sa1 = np.array([std_ex_secs]).T
    sa2 = np.array([std_ey_secs]).T

    # 5.4) Deffine the vectors with electric fields including statistics
    secs_e[:,v1,:] = np.hstack((sax, say))
    stdsecs_e[:,v1,:] = np.hstack((sa1, sa2))

    # 5.5) Compute mean and std of electric fields
    # 5.5.1) Following Approach #1
    #if mode == 1:
        # Define the E vector using several values within the error bars
    for ip in range (0,stat):
        #e_fields = np.zeros([length-1, len(e_site), 2, stat])
        #if ip ==0:
        
        #    e_fields[:,v1,:,ip] = (secs_e[:,v1,:]
        #        + np.random.standard_normal([length,2])
        #        * stdsecs_e[:,v1,:])
        #else:
        e_fields[:,v1,:,ip] = (secs_e[:,v1,:]
    + np.random.standard_normal([length-1,2])
    * stdsecs_e[:,v1,:])
# Deffine the E vector
ce_fields = e_fields



#######################################################################
# 6) Calculate the average/representative E_field, and the standard deviation
print('Calculating mean and std of the computed electric fields')
for v4, ip4 in enumerate(e_site):
    for ip2 in range(0,2):
        for ip3 in range(0,length-1):
            #calculating average e field over no of calcualations
            #The standard devaition of the modelled E fields
            std_error[ip3,v4,ip2] = np.std(ce_fields[ip3,v4,ip2,:])
            #The modelled E field
            av_e_fields[ip3,v4,ip2] = np.average(ce_fields[ip3,v4,ip2,:])


#######################################################################
# 7 Write E fields in the out folder
print('Saving the electric fields')
for vi, ppi in enumerate(e_site):
    # Total electric field
    ex_path_id = (out_path
                 + str(ppi)
                 + '_Ex_'
                 + 'realtime'
                 + '.dat')

    ey_path_id = (out_path
                 + str(ppi)
                 + '_Ey_'
                 + 'realtime'
                 + '.dat')

    ex = np.array([av_e_fields[:,vi,0], std_error[:,vi,0]])
    ey = np.array([av_e_fields[:,vi,1], std_error[:,vi,1]])

    np.savetxt(ex_path_id, ex.T, fmt=['%15.5f' , '%15.5f'])
    np.savetxt(ey_path_id, ey.T, fmt=['%15.5f', '%15.5f'])


for i in range(0,len(ex[0])):
    #snr ratio of X and Y components    
    snr_x=sum(ex[0])/sum(ex[1])
    snr_y=sum(ey[0])/sum(ey[1])


###########################################
# 8) now creating a Map of E fields

#Plotting data
DATE,HOUR=femm.nowcast_mapping(in_path,main_path,DATE,HOUR,p_mode,av_e_fields,
                            video_length,e_site,correction_c,std_error,
                            mh_obs,total_activity,padding,rmf)     

#uncomment bottom to use basemap
# DATE,HOUR=femm.nowcast_mapping_basemap(in_path,main_path,DATE,HOUR,p_mode)  

#-----------------
#creating histogram of storm activity
femm.hist_plot(total_activity)
#Plotting video of data
femm.video_maker(main_path)
#Cleaning old image files
femm.video_fixer(main_path)
#Closing figures
plt.close('all')
