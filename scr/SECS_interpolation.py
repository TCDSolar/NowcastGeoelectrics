#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
# Code for computing the Spherical Elementary Current Systems (SECS) to interpolate magnetic field variations.
# Original Author: C.D.Beggan (adapted from AJmcK) [ciar@bgs.ac.uk]
# Ported from MATLAB to Python by Sean Blake (Trinity College Dublin) [blakese@tcd.ie]
# Modifed for Electric field model by John Malone-Leigh(Dublin Institute for Advanced Studies) [jmalonel@tcd.ie] and Joan Campanya
# Date: June 2020
"""


import matplotlib.dates as mdates
from multiprocessing import Pool

import inputs as iEM
import numpy as np
from scipy.interpolate import griddata
from secs_pre import cart2sph_matlab, sph2cart_matlab, pole_common2source,pole_source2common, sourcecords, secsmatrix_XYonly
    


datafolder =  str(out_path)
###############################################################################
    
in_path = iEM.in_path
Samp = iEM.samp

 ################################################################################
# DATA INPUT
################################################################################
# Declare the constants
earthrad, ionorad = 6371000.0, 6481000.0    #6371000.0, 6481000.0
Samp_P_day =int( 86400 / Samp) # Number of samples pr day !!!!
# Define grid: uniform in lat and long
#secswest, secseast, secssouth, secsnorth = -13, 10, 48, 63
#secswest, secseast, secssouth, secsnorth = -15, 15, 40, 67  #!!!!!

lonmin, latmin = secswest, secssouth

dlon = 0.5 # spacing !!!!

lonpts = int((secseast - secswest)*(1/dlon)+1)
latpts = int((secsnorth - secssouth)*(1/dlon)+1)


ncol = latpts * lonpts * 2

         
         

numb_of_days =int( length * Samp / 86400)   # Days recording !!!!
numb_of_days =int( length * Samp / 86400)   # Days recording !!!!

print ("Forming Grid:", lonpts, "x", latpts, "[lon x lat grid]")
lon, lat = np.meshgrid(np.arange(lonmin, lonmin+dlon*lonpts-dlon + 0.1, dlon),
    np.arange(latmin, latmin + dlon*latpts-dlon+0.1, dlon))

# Folder which contains all of your inputs
#datafolder =  str(out_path)

# site data (name, lat, lon) are in file: .../datafolder/sites.txt
datafolder =  str(out_path)
#length=len(data)
# site data (name, lat, lon) are in file: .../datafolder/sites.txt
sitesfile = str(in_path) + "Observatories.dat"
sitenames = np.loadtxt(sitesfile, usecols = (0,), unpack = True, skiprows =0, dtype = str)
sitelat, sitelon = np.loadtxt(sitesfile, usecols = (1,2), unpack = True, skiprows =0)

nsites = len(sitelat)
allBmeas = np.zeros((length, 2*nsites))
allBxmeas, allBymeas = [], []




# Load data from files- varying Bx and By values, one per minute
for index, value in enumerate(sitenames):
    filename = datafolder + "SECS/SECS_"  + value +"_realtime.dat"
    X ,Y = np.loadtxt(filename, usecols = (0,1), unpack = True, skiprows = 0)

    X = X[0:length]
    Y = Y[0:length]
    allBmeas[:,index*2] = X
    allBmeas[:,index*2 + 1] = Y

'''
sitesfile = str(in_path) + "Observatories.dat"
sitenames = np.loadtxt(sitesfile, usecols = (0,), unpack = True, skiprows =0, dtype = str)
sitelat, sitelon = np.loadtxt(sitesfile, usecols = (1,2), unpack = True, skiprows =0)'''

#reading number of sites
#Try loop added in case only one site present
try:
    nsites = len(sitelat)
except:
    nsites=1
# Load data from files- varying Bx and By values, one per minute
#for index, value in enumerate(sitenames):
#filename = datafolder + "SECS/" + str(fname)+ "ARM_realtime.dat"
'''
index=0


X ,Y = np.loadtxt(filename, usecols = (0,1), unpack = True, skiprows = 0)

allBmeas = np.zeros((len(X), 2*nsites))
allBxmeas, allBymeas = [], []'''
#X = X[0:numb_of_days*Samp_P_day]
#Y = Y[0:numb_of_days*Samp_P_day]
'''
allBmeas[:,index*2] = X
allBmeas[:,index*2 + 1] = Y'''

#####################################le###########################################
# Construct the SECS matrix
Amatrix_ext = np.zeros((nsites*2, int(latpts*lonpts)))

for n in np.arange(0, nsites, 1):
    nm = (2*n)
    try:
        Tex, Tey = secsmatrix_XYonly(latpts, lonpts, ncol, sitelat[n], sitelon[n],
          lat, lon, earthrad, ionorad)

    except: #except loop accounts if only one site present
        Tex, Tey = secsmatrix_XYonly(latpts, lonpts, ncol, sitelat, sitelon,
          lat, lon, earthrad, ionorad)        

    Amatrix_ext[nm] = Tex
    Amatrix_ext[nm+1] = Tey

    testx = Tex.flatten()
    testy = Tey.flatten()

    for i in testx:

        if np.isnan(i) == True:
            print (n)
            Tex_backup = Tex
    for j in testx:
        if np.isnan(j) == True:
            Tey_backup = Tey
            print (n)

# make sure there are no Nan's in the data
Amatrix_ext = np.nan_to_num(Amatrix_ext)

# Perform a Singular Value Decomposition of the SECS matrix
Ue, We, Ve = np.linalg.svd(Amatrix_ext, full_matrices = False)
Ve = Ve.T

svdthresh = We[0]/100.0
for index, value  in enumerate(We):
    if value < svdthresh:
        We[index] = 0

truncate = len(We)
for i in We:
    if i < svdthresh:
        truncate = truncate - i


print ("Calculating B throughout the grid")
TmatrixX = np.zeros((lonpts*latpts, lonpts*latpts))
TmatrixY = np.zeros((latpts*lonpts, latpts*lonpts))

for m in np.arange(0, int(lonpts), 1):
    grdlat = lat[:,m]
    grdlon = lon[:,m]
    zzz = np.arange(int(m*latpts), int((m*latpts) + latpts), 1)

    Tx, Ty = secsmatrix_XYonly(latpts, lonpts, ncol, grdlat, grdlon, lat,
        lon, earthrad, ionorad)

    TmatrixX[:, zzz] = Tx
    TmatrixY[:, zzz] = Ty

TmatrixX = TmatrixX * -1
TmatrixY = TmatrixY * -1


print ("Calculating B-Fields")

# Arbitrarily calculates every 7th minute over the 2 day period
#for minute in np.arange(0, numb_of_days* Samp_P_day, 1):
A = np.matrix(Ve[:, range(0, truncate)])
B = np.matrix(np.diag(1/We[:truncate]))
C = np.matrix(Ue[:, range(0, truncate)].T)
Tmatrix = np.vstack((TmatrixX, TmatrixY))
print('Tmatrix created' )
Tmatrix = np.nan_to_num(Tmatrix)



def single_calc(minute):
    Bmeas = allBmeas[minute]

    D = np.matrix(Bmeas).T
    Ecurr = A*B*C*D
    Bfield_XYonly = Tmatrix*Ecurr

    Bx = Bfield_XYonly[0:latpts*lonpts]
    Bx = np.reshape(Bx, (lonpts, latpts), order = 'C')
    Bx = Bx.T
    Bx = np.array(Bx)
    Bx_1D = Bx.flatten()

    By = Bfield_XYonly[latpts*lonpts:]

    By = np.reshape(By, (lonpts, latpts), order = 'C')
    By = By.T
    By = np.array(By)
    By_1D = By.flatten()

    # interpolating the data
    lon_1D = lon.flatten()
    lat_1D = lat.flatten()

    points =(lon_1D, lat_1D)
    otherpoints = []
#    for i in np.arange(0, len(longicc), 1):
#        otherpoints.append((longicc[i][0], latgicc[i][0]))

    otherpoints =(lons_secs, lats_secs)



    bx_interp = griddata(points, Bx_1D, otherpoints, method = 'cubic')
    by_interp = griddata(points, By_1D, otherpoints, method = 'cubic')


    #print (minute)


    

    return bx_interp, by_interp

print("creating the output")

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
# The commented lines are for using parallel processing
#print(os.cpu_count) #Use to show no of cpu on machine
'''
pool = Pool(2) # Add number of cpu you want to use
output = pool.map(single_calc, np.arange(0, numb_of_days* Samp_P_day, 1))
output = np.array(output)'''
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*

output = []
for i in range(0,length-1,1):
    print(i)
    output.append(single_calc(i))
    
#Only works for multiple sites
print(output)
try:
    obs_bx_secs = np.array(output)[:,0,:]
    obs_by_secs = np.array(output)[:,1,:]
except:
    obs_bx_secs = np.array(output)[:,0]
    obs_by_secs = np.array(output)[:,1]
    

    
    
def SECS_Interpolation(out_path,secswest,secssouth,secsnorth,secseast,
                       length,lons_secs,lats_secs):
    #only importing SECS modules if secs are used
    from secs_pre_now import cart2sph_matlab, sph2cart_matlab, pole_common2source,pole_source2common, sourcecords, secsmatrix_XYonly
    
    datafolder =  str(out_path)
    ###############################################################################
        
    in_path = iEM.in_path
    Samp = iEM.samp
    
     ################################################################################
    # DATA INPUT
    ################################################################################
    # Declare the constants
    earthrad, ionorad = 6371000.0, 6481000.0    #6371000.0, 6481000.0
    Samp_P_day =int( 86400 / Samp) # Number of samples pr day !!!!
    # Define grid: uniform in lat and long
    #secswest, secseast, secssouth, secsnorth = -13, 10, 48, 63
    #secswest, secseast, secssouth, secsnorth = -15, 15, 40, 67  #!!!!!
    
    lonmin, latmin = secswest, secssouth
    
    dlon = 0.5 # spacing !!!!
    
    lonpts = int((secseast - secswest)*(1/dlon)+1)
    latpts = int((secsnorth - secssouth)*(1/dlon)+1)
    
    
    ncol = latpts * lonpts * 2
    
             
             
    
    numb_of_days =int( length * Samp / 86400)   # Days recording !!!!
    numb_of_days =int( length * Samp / 86400)   # Days recording !!!!
    
    print ("Forming Grid:", lonpts, "x", latpts, "[lon x lat grid]")
    lon, lat = np.meshgrid(np.arange(lonmin, lonmin+dlon*lonpts-dlon + 0.1, dlon),
        np.arange(latmin, latmin + dlon*latpts-dlon+0.1, dlon))
    
    # Folder which contains all of your inputs
    #datafolder =  str(out_path)
    
    # site data (name, lat, lon) are in file: .../datafolder/sites.txt
    datafolder =  str(out_path)
    #length=len(data)
    # site data (name, lat, lon) are in file: .../datafolder/sites.txt
    sitesfile = str(in_path) + "Observatories.dat"
    sitenames = np.loadtxt(sitesfile, usecols = (0,), unpack = True, skiprows =0, dtype = str)
    sitelat, sitelon = np.loadtxt(sitesfile, usecols = (1,2), unpack = True, skiprows =0)
    
    nsites = len(sitelat)
    allBmeas = np.zeros((length, 2*nsites))
    allBxmeas, allBymeas = [], []
    
    
    
    
    # Load data from files- varying Bx and By values, one per minute
    for index, value in enumerate(sitenames):
        filename = datafolder + "SECS/SECS_"  + value +"_realtime.dat"
        X ,Y = np.loadtxt(filename, usecols = (0,1), unpack = True, skiprows = 0)
    
        X = X[0:length]
        Y = Y[0:length]
        allBmeas[:,index*2] = X
        allBmeas[:,index*2 + 1] = Y
    
    '''
    sitesfile = str(in_path) + "Observatories.dat"
    sitenames = np.loadtxt(sitesfile, usecols = (0,), unpack = True, skiprows =0, dtype = str)
    sitelat, sitelon = np.loadtxt(sitesfile, usecols = (1,2), unpack = True, skiprows =0)'''
    
    #reading number of sites
    #Try loop added in case only one site present
    try:
        nsites = len(sitelat)
    except:
        nsites=1
    # Load data from files- varying Bx and By values, one per minute
    #for index, value in enumerate(sitenames):
    #filename = datafolder + "SECS/" + str(fname)+ "ARM_realtime.dat"
    '''
    index=0
    
    
    X ,Y = np.loadtxt(filename, usecols = (0,1), unpack = True, skiprows = 0)
    
    allBmeas = np.zeros((len(X), 2*nsites))
    allBxmeas, allBymeas = [], []'''
    #X = X[0:numb_of_days*Samp_P_day]
    #Y = Y[0:numb_of_days*Samp_P_day]
    '''
    allBmeas[:,index*2] = X
    allBmeas[:,index*2 + 1] = Y'''
    
    #####################################le###########################################
    # Construct the SECS matrix
    Amatrix_ext = np.zeros((nsites*2, int(latpts*lonpts)))
    
    for n in np.arange(0, nsites, 1):
        nm = (2*n)
        try:
            Tex, Tey = secsmatrix_XYonly(latpts, lonpts, ncol, sitelat[n], sitelon[n],
              lat, lon, earthrad, ionorad)
    
        except: #except loop accounts if only one site present
            Tex, Tey = secsmatrix_XYonly(latpts, lonpts, ncol, sitelat, sitelon,
              lat, lon, earthrad, ionorad)        
    
        Amatrix_ext[nm] = Tex
        Amatrix_ext[nm+1] = Tey
    
        testx = Tex.flatten()
        testy = Tey.flatten()
    
        for i in testx:
    
            if np.isnan(i) == True:
                print (n)
                Tex_backup = Tex
        for j in testx:
            if np.isnan(j) == True:
                Tey_backup = Tey
                print (n)
    
    # make sure there are no Nan's in the data
    Amatrix_ext = np.nan_to_num(Amatrix_ext)
    
    # Perform a Singular Value Decomposition of the SECS matrix
    Ue, We, Ve = np.linalg.svd(Amatrix_ext, full_matrices = False)
    Ve = Ve.T
    
    svdthresh = We[0]/100.0
    for index, value  in enumerate(We):
        if value < svdthresh:
            We[index] = 0
    
    truncate = len(We)
    for i in We:
        if i < svdthresh:
            truncate = truncate - i
    
    
    print ("Calculating B throughout the grid")
    TmatrixX = np.zeros((lonpts*latpts, lonpts*latpts))
    TmatrixY = np.zeros((latpts*lonpts, latpts*lonpts))
    
    for m in np.arange(0, int(lonpts), 1):
        grdlat = lat[:,m]
        grdlon = lon[:,m]
        zzz = np.arange(int(m*latpts), int((m*latpts) + latpts), 1)
    
        Tx, Ty = secsmatrix_XYonly(latpts, lonpts, ncol, grdlat, grdlon, lat,
            lon, earthrad, ionorad)
    
        TmatrixX[:, zzz] = Tx
        TmatrixY[:, zzz] = Ty
    
    TmatrixX = TmatrixX * -1
    TmatrixY = TmatrixY * -1
    
    
    print ("Calculating B-Fields")
    
    # Arbitrarily calculates every 7th minute over the 2 day period
    #for minute in np.arange(0, numb_of_days* Samp_P_day, 1):
    A = np.matrix(Ve[:, range(0, truncate)])
    B = np.matrix(np.diag(1/We[:truncate]))
    C = np.matrix(Ue[:, range(0, truncate)].T)
    Tmatrix = np.vstack((TmatrixX, TmatrixY))
    print('Tmatrix created' )
    Tmatrix = np.nan_to_num(Tmatrix)
    
    
    
    def single_calc(minute):
        Bmeas = allBmeas[minute]
    
        D = np.matrix(Bmeas).T
        Ecurr = A*B*C*D
        Bfield_XYonly = Tmatrix*Ecurr
    
        Bx = Bfield_XYonly[0:latpts*lonpts]
        Bx = np.reshape(Bx, (lonpts, latpts), order = 'C')
        Bx = Bx.T
        Bx = np.array(Bx)
        Bx_1D = Bx.flatten()
    
        By = Bfield_XYonly[latpts*lonpts:]
    
        By = np.reshape(By, (lonpts, latpts), order = 'C')
        By = By.T
        By = np.array(By)
        By_1D = By.flatten()
    
        # interpolating the data
        lon_1D = lon.flatten()
        lat_1D = lat.flatten()
    
        points =(lon_1D, lat_1D)
        otherpoints = []
    #    for i in np.arange(0, len(longicc), 1):
    #        otherpoints.append((longicc[i][0], latgicc[i][0]))
    
        otherpoints =(lons_secs, lats_secs)
    
    
    
        bx_interp = griddata(points, Bx_1D, otherpoints, method = 'cubic')
        by_interp = griddata(points, By_1D, otherpoints, method = 'cubic')
    
    
        #print (minute)
    
    
        
    
        return bx_interp, by_interp
    
    print("creating the output")
    
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
    # The commented lines are for using parallel processing
    #print(os.cpu_count) #Use to show no of cpu on machine
    '''
    pool = Pool(2) # Add number of cpu you want to use
    output = pool.map(single_calc, np.arange(0, numb_of_days* Samp_P_day, 1))
    output = np.array(output)'''
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
    
    output = []
    for i in range(0,length-1,1):
        print(i)
        output.append(single_calc(i))
        
    #Only works for multiple sites
    print(output)
    try:
        obs_bx_secs = np.array(output)[:,0,:]
        obs_by_secs = np.array(output)[:,1,:]
    except:
        obs_bx_secs = np.array(output)[:,0]
        obs_by_secs = np.array(output)[:,1]

    return obs_bx_secs,obs_by_secs
