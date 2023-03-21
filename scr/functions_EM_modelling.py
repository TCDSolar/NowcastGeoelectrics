#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on April 2017 - Joan Campanya
Modified June 2021 - John Malone-Leigh
Email queries to jmalonel@tcd.ie

Altgorithm adapted from Campanya et al. 2018 publication in AGU - Space Weather

Functions used to Model/Nowcast geoelectric fields are below.
Fucntions loaded in by EM_modelling.py



Set input magnetometer sites in observatories.dat
Set input MT sites in in/sites_interest.dat



Minor Functions
o time2float - converts datetime objects to floats
o float2time - converts floats times to datetime
o timedatez - converts columns with date and time to datetime object
o scr_fft - Calculates fourier transform with time series
o read_co - read coordinates from file
o read_rmfb - reads magnetics data sved from SECS interpolation
o save_magnetics - save magnetometer data from MagIE website
o nan_helper - deals with nans in time series
o mag_filter - Filters out noisy mag data, converts to nans
o minute_bin - bin data from second to minute data
o read_variable - reads variables from MT tensors
o site_tester - Tests if enough available real-time data is present


Major Functions
o Load_magnetics - loads magnetic inputs, sorts data and adds padding
o compute_e_fields_secs - computes the electric fields at each MT site... 
using secs as inputs


"""
import numpy as np
import scipy.signal
import seaborn as sns
import pandas as pd
from scipy.fftpack import ifft
import scipy.fftpack as fftpack
import geopy.distance
import datetime
from time import strptime
import urllib
import matplotlib.pyplot as plt



sns.set()
def time2float(x):

    """converts datetime to float, so that interpolation/smoothing can

       be performed"""

    if (type(x) == np.ndarray) or (type(x) == list):

        emptyarray = []

        for i in x:

            z = (i - datetime.datetime(1970, 1, 1, 0)).total_seconds()

            emptyarray.append(z)

        emptyarray = np.array([emptyarray])

        return emptyarray[0]

    else:

        return (x - datetime.datetime(1970, 1, 1, 0)).total_seconds()



##########################################################################

##########################################################################



def float2time(x):

    """converts array back to datetime so that it can be plotted with time

       on the axis"""

    if (type(x) == np.ndarray) or (type(x) == list):

        emptyarray = []

        for i in x:

            z = datetime.datetime.utcfromtimestamp(i)

            emptyarray.append(z)

        emptyarray = np.array([emptyarray])

        return emptyarray[0]

    else:

        return datetime.datetime.utcfromtimestamp(x)



##########################################################################

##########################################################################



def timedatez(date, timez):

    """creating 'timedate' array for date specified, from date + time columns"""

    timedate = [] 

    for i in range(0, len(date)):

        a = date[i] + timez[i]

        try:
            c = datetime.datetime(*strptime(a, "%d/%m/%Y%H:%M:%S")[0:6])    
        except:
            c=datetime.datetime(*strptime(a,"%Y-%m-%d%H:%M:%S.000")[0:6])
        timedate.append(c)

    return timedate




###########################################################################
def scr_fft(x, y, s):
    """ Calculate Fourier transform for two time series (x, y components)
		
		Parameters
		-----------
		x = time series 1
		y = time series 2
		s = sampling rate (in seconds)

		Returns
		-----------
		perd = periods in seconds
		x_fft = fft of time series 1
		y_fft = fft of time series 2

		-----------------------------------------------------------------
	"""
    
    w = scipy.signal.tukey(x.shape[0], 0.1) 
    x = x * w
    y = y * w
    freq = (np.fft.fftfreq(x.shape[0], d=float(s)))
    for i in range(0,len(freq)):
        if freq[i] == 0:
            freq[i] = 1e-99

    perd = freq ** (-1)
    x_fft = fftpack.fft(x)
    y_fft = fftpack.fft(y)

    return (perd, x_fft, y_fft)

##########################################################################
def read_co(path):
    
    """ Read name of the sites and coordinates of the site from the input
        files. Latitude and longitude should be in degrees.
		
		Parameters
		-----------
		path = path of the site with the name of the sites and coordinates

		Returns
		-----------
		name = Name of the site
		lat = latitude of the site (in degrees)
		lon = longitude of the site (in degrees)

		-----------------------------------------------------------------
    """
   
    a = pd.read_csv(path, 
                    header = None, 
                    skiprows = None, 
                    sep='\s+'
                    )
    
    a = np.array(a)
    name = a[:,0]
    lat =  a[:,1]
    lon =  a[:,2]
    
    return(name, lat, lon)

def hist_plot(total_activity):
    #Plots histogram of all activity for storm

    plt.style.use('classic')
    plt.figure()
    plt.hist(total_activity,bins=[0,50,100,150,200,250,300,350,400,450,500])
    plt.yscale('log')
    plt.ylabel('Counts')
    plt.xlabel('Electric field (mV/km)')
    plt.ylim([1,10e5])
    
##########################################################################
def read_rmfb(path, name):
    
    """ Read interpolated magnetic fields from SECS
		
		Parameters
		-----------
		path = path for the SECS output files
         storm = Name of the selected storm  
         name = Name of the site  


		Returns
		-----------
		rmfb = Interpolated magnetic fields from SECS

		-----------------------------------------------------------------
	"""

    f = open(path 
             +name
             + "_magBx.dat" , 
             'r'
             )
    
    rmf_bx = np.loadtxt(f, skiprows = 0)
    f.close()
        
    f = open(path 
             + name 
             + "_magBy.dat",
             'r'
             )
    
    rmf_by = np.loadtxt(f, skiprows = 0)
    f.close()
    
    rmfb = np.array([rmf_bx, rmf_by]).T
    
    return(rmfb)



def minute_bin(timedate_float, bx, by, bz, n):



    # Gets the start of the day in seconds

    day_seconds = int(timedate_float[0])-int(timedate_float[0])%(24*3600)



    # Creates array of minutes

    minutes = np.arange(0, n * 1440)

    minutes = (minutes * 60) + day_seconds



    # master is numpy array with columns for bx, by, bz, count and times

    master = np.zeros((n*1440, 5))

    master[:,-1] = minutes



    # loop over times

    for i, v in enumerate(timedate_float):

        # check which master row it belongs to

        index = int((v - day_seconds)/60) #- 1

        # add to each column

        try:

            master[index][3] += 1

            master[index][0] += bx[i]

            master[index][1] += by[i]

            master[index][2] += bz[i]

        except:

            continue



    # now make empty arrays which will be filled

    minute_bx, minute_by, minute_bz, minute_time = [], [], [], []

    for i, v in enumerate(master):

        if v[3] == 0:   # if count = 0, ignore

            continue

        else:           # otherwise, add average to respective array

            minute_bx.append(v[0]/v[3])

            minute_by.append(v[1]/v[3])

            minute_bz.append(v[2]/v[3])

            minute_time.append(v[4])

    

    return minute_time, minute_bx, minute_by, minute_bz

def minute_bin2(timedate_float, bx, by, bz,n):
    #A quick fix if minute bin doesn't work on version of python
    
    time_len=int(len(timedate_float))
    minute_time=[]
    minute_bx=[]
    minute_by=[]
    minute_bz=[]
    for i in range(0,time_len):
        
        minute_time1=timedate_float[i]
        minute_bx2=bx[60*i:60*i+60]
        minute_by2=by[60*i:60*i+60]
        minute_bz2=bz[60*i:60*i+60]
        
        minute_bx1=np.mean(minute_bx2)
        minute_by1=np.mean(minute_by2)
        minute_bz1=np.mean(minute_bz2)
        minute_time.append(minute_time1)
        minute_bx.append(minute_bx1)
        minute_by.append(minute_by1)
        minute_bz.append(minute_bz1)
    minute_time.append(minute_time1)
    minute_bx.append(minute_bx1)
    minute_by.append(minute_by1)
    minute_bz.append(minute_bz1)        

    
    return minute_time,minute_bx,minute_by,minute_bz
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

            

def read_variable(n, index, data):
    k = 1
    val = []
    for j in range(0, n):
        if k < n:
            val_1 = np.loadtxt([data[index + j + 1]])
            val = np.append(val, val_1)
            k = len(val)
            val = np.array(val)
    return(val)

##########################################################################

def save_magnetics( in_path, sites,rmf):
    """
    Purpose: Loads live magnetometer data from MagIE.ie
    Loads data for last 3 days
    Creates a list with filenames which can then be loaded
    """
    # Read the time series for these magnetic fields
    nowz=datetime.datetime.utcnow()
    year_str=str(nowz.year)
    month_str="%02d" %(nowz.month)
    day_str="%02d" %(nowz.day)
    one_day=nowz-datetime.timedelta(days=1)
    two_day=nowz-datetime.timedelta(days=2)

    days=[two_day,one_day,nowz]


    #for ip in range(0,len(s)):
    archive_list=[]
    value_list=[]
    MagIE_list=[]
    for ip, site in enumerate(rmf):        
        print(site)
        site_l=site.lower()
        #Strs used for real listed below
        #Reading in last 3 magnetometer files
    

        website="https://data.magie.ie/"
        file=urllib.request.URLopener()
        save_folder=r'C:\Users\Dunsink\Documents\Python Scripts\Geo_Electrics_realtime_houdini/Data/'
        
        for i in range(0,3):
            day_str="%02d" %(nowz.day-i)
            
            if day_str =="00":
                time=nowz-datetime.timedelta(days=i)
                day_str="%02d" %(time.day)
                month_str="%02d" %(time.month)
                #month_str=months[i]
                year_str="%02d" %(time.year)
            
            date_str= year_str+'/'+month_str+'/'+day_str+'/txt/'
            date_str2= site_l+year_str+month_str+day_str+'.txt'
            
            site_day=website+date_str+date_str2
           # print(site_day)
            #uncomment to turn on
            try:
                file.retrieve(site_day,save_folder+site_l+str(i)+'.txt')
            except:
                print('***No File present***')
        l=2
        for i in days:
            
            year_str=str(i.year)
            month_str="%02d" %(i.month)
            day_str="%02d" %(i.day)
            
            if site=="ARM" or site=="BIR" or site=="DUN":

                archive_str=save_folder+str(site_l)+str(l)+'.txt'

                val=1 #number 
                MagIE=1
            else:
                #Else to deal with VAL intermagnet sites
            
                archive_str=save_folder+"val"+str(l)+'.txt'
                val=1
                MagIE=0
            archive_list.append(archive_str)
            value_list.append(val)
            MagIE_list.append(MagIE)
            l=l-1
        
        #testing sites to see if they're running close to realtime
    """
    sites_true=site_tester(archive_list,sites)
    
    archive_t=[]
    value_list_t=[]
    MagIE_list_t=[]
    
    for i in sites_true:
        archive_t.append(archive_list[i])
        value_list_t.append(value_list_t[i])
        MagIE_list_t.append(MagIE_list_t[i])"""
            
    #return archive_t, value_list_t,MagIE_list_t   
    return archive_list, value_list, MagIE_list


def load_dates(archive_list,sites):
    
    #Module used to read in time data from latest file
    #
    dtim_list=[]
    for j in sites:
        dtim2=[]
        print(archive_list)
        archive_list2=[]
        
        for i in archive_list:
            if j.lower() in i:
                
                
                DATE=np.genfromtxt(i,dtype=str,usecols=[0])
                Time=np.genfromtxt(i,dtype=str,usecols=[1])
                DATE=DATE[1:]
                Time=Time[1:]
                DT=[]
                for a,b in list(zip(DATE,Time)):
                    DT.append(a+' '+b)
                #converting from sting to datetime
                dtim=[datetime.datetime.strptime(j, "%d/%m/%Y %H:%M:%S") for j in DT]
                
                dtim2.append(dtim)
           
            #dtim2=([k for sublist in dtim2 for k in sublist])
           
        dtim_list.append(dtim2)
    
    return dtim_list

def site_tester(archive_list,sites):
    """
    Inputs
    Mag data
    
    Purpose of this module is to test site data

    #Tests if data is constantly stremed for past tow days
    #sets cropping length to nearest real-time for all "good" sites

    Returns
    names of good sites
    """
    
    #time data for all data
    dtim_list=load_dates(archive_list,sites)
    #print(dtim_list)
    
    nowz=datetime.datetime.utcnow()     
    twodays=nowz-datetime.timedelta(days=2) 


    t0=datetime.datetime(twodays.year,twodays.month,twodays.day,
                         0,0)
    tf=datetime.datetime(nowz.year,nowz.month,nowz.day,
                         nowz.hour,nowz.minute)
    time_delta=tf-t0 #number of minutes between now and start of dates

    
    minutes_delta=int((time_delta.seconds+time_delta.days*3600*24)/60)
    
    dates=[]
    for i in range(minutes_delta):
        dates.append(t0+datetime.timedelta(minutes=i))
    recs=[]
    for j in dtim_list:
        
        l=0
        for k in j:
            if k==dates[l]:
                rec=1
                pass
            else:
                print('Errorsome data')
                rec=0
                recs.append(rec)
                break
            
            l=l+1
        if rec==1:
            print('Good Site')
        recs.append(rec)
    #rec=0 is bad site, rec =1 is good site
    
    #now comparing length of good files
    
    lengths=[len(i) for i in dtim_list]
    l=0
    for i in sites:
        if i=='arm' or 'bir':
            #accounting for second data
            lengths[l]=int(lengths[l]/60)
        l=l+1     
    true_len=len(dates)
    l=0
    #checking that file is running close enough to real-time ...i.e within an hour
    #print(lengths)
    for i in lengths:
        if true_len-60< i:
            recs[l]=0
        l=l+1
    
    sites_good=[]
    l=0
    for i in recs:
        if i==1:
            
            sites_good.append(l)
            l=l+1
    
    return sites_good
    
    #now compare live dates to new dates
def mag_filter(Bx,By,Bz):
    """
    Filters out noise sharp peaks in Magnetic field time seroes
    """
    if len(Bx)==len(By) and len(Bx)==len(Bz):
        dF=[]#rate of change of full field
        for i in range(0,len(Bx)-1):

            dF.append(abs(Bx[i+1]-Bx[i])+abs(By[i+1]-By[i])+abs(Bz[i+1]-Bz[i]))
        Bxnew=[]
        Bynew=[]
        
        Bznew=[]
        for j in range(0,len(dF)-60,60):
            if max(dF[j:j+60])>10: #nT/sec=10 for 1 sec
                array=np.full(shape=60,fill_value=99999.99,dtype=np.float)
                for k in array:
                    Bxnew.append(k)
                    Bynew.append(k)
                    Bznew.append(k)
            else:
                for l in range(0,60):
                    Bxnew.append(Bx[j+l])
                    Bynew.append(By[j+l])  
                    Bznew.append(Bz[j+l])  
        #need to include last values, i.e the last <60 secs
        if len(dF)>len(Bxnew):
            if max(dF[len(Bxnew):len(dF)])>10: #nT/sec=10 for 1 sec
                array=np.full(shape=len(dF)-len(Bxnew)+1,fill_value=99999.99,dtype=np.float)
                #+1 account for last value
                for k in array:
                    Bxnew.append(k)
                    Bynew.append(k)
                    Bznew.append(k)
                
            else:
                length=len(Bxnew)
                #length of list lacking
                for l in range(0,len(dF)-length):
                 
                    Bxnew.append(Bx[length+l])
                    Bynew.append(By[length+l])  
                    Bznew.append(Bz[length+l])
                #appending last value
                Bxnew.append(Bx[length+l])
                Bynew.append(By[length+l])  
                Bznew.append(Bz[length+l])
        
    else:
        print('Error in Mag Filter length')
        
    return Bxnew,Bynew,Bznew    
    
    #all dates should include strs 
    
def load_magnetics( in_path, sites, secs_path, samp, hi, low, 
                    var,rmf, archive_list, val_list, MagIE_list,
                    padding):

    s, s_lat, s_lon = read_co(in_path + str(sites))
    bx_second=[]
    by_second=[]
    bz_second=[]
    magie_second=[]
    val2=1
    
    for ip, site in enumerate(rmf):  
        
        archive_list2=archive_list[int(3*ip+0):int(3+3*ip)]
        
        MagIE=MagIE_list[ip]
        val=val_list[ip]
        
        count=0
        #print(archive_list)
        for i in archive_list2:
            fff = i
            f = open(fff, 'r')
            data = f.readlines()
            f.close()
        
            # Read the number of lines to be skipped
            #val=1 if MagIE site, varies for intermag
            for index, line in enumerate(data):
                if line.startswith('DATE'):
                    val2 = index
        
        for i in archive_list2:
            print(archive_list)
            #print(i)
            df = pd.read_csv(i, sep = '\s+', skiprows = int(val2)) # Check the rows that need to be skiped
        
            count = count + 1
            #Concatenating previous lists in archive lists together
            if count == 1:
                df_t = np.array(df)
                print(df_t.shape)
            else:        
                df_t = np.row_stack((df_t, df)) #appending data
                print(df_t.shape)
        
        Date = df_t[:,0]
        Hour = df_t[:,1]  
        bx = pd.to_numeric(df_t[:,3])
        by = pd.to_numeric(df_t[:,4])
        bz = pd.to_numeric(df_t[:,5])
        
        bx_second.append(bx)
        by_second.append(by)
        bz_second.append(bz)
        magie_second.append(MagIE)
        timedate = timedatez(Date, Hour) 
        
        #converting dates and hours to datetime objects
        timedate_float = time2float(timedate)

    ############################################################      
                
    #Determining noise from magnetic series data data
    
    #calculating variation in B

    
    ###################################################
    
    
    for ip, site in enumerate(rmf):    
        
        #if site=='VAL':
        #    MagIE=0#1
        #else:
        #    MagIE=1
        bx=bx_second[ip]
        #print(len(bx))
        by=by_second[ip]
        bz=bz_second[ip]
        
        bx,by,bz=mag_filter(bx,by,bz)
            

        bx=np.array(bx)
       
        by=np.array(by)
        bz=np.array(bz)
        
        #Determining whether Bx is errorsome
        bx[bx >= 80000.0] = 'nan'
        bx[bx == 'infs'] = 'nan'
        bx[bx <= 0.0] = 'nan'
        nans, x = nan_helper(bx)
        bx[nans]= np.interp(x(nans), x(~nans), bx[~nans])
            
        
        by[by >= 50000.0] = 'nan'
        by[by == 'infs'] = 'nan'
        by[by <= -10000.0] = 'nan'
        nans, x = nan_helper(by)
        by[nans]= np.interp(x(nans), x(~nans), by[~nans])
    
        bz[bz >= 80000.0] = 'nan'
        bz[bz == 'infs'] = 'nan'
        bz[bz <= 10000.0] = 'nan'
        nans, x = nan_helper(bz)
        bz[nans]= np.interp(x(nans), x(~nans), bz[~nans])
        
       
        #Converting MagIE files from second data to minute
        if MagIE==1:
            minute_time, minute_bx, minute_by, minute_bz= minute_bin(timedate_float, bx, by, bz, 5)
        if MagIE==0:
            minute_time, minute_bx, minute_by, minute_bz= timedate_float, bx, by, bz
            
            # Remove detrend
        
        MagIE=1
        if MagIE==0:
            Bx =minute_bx * np.cos(minute_by*3.141592/(180*60))
            By = minute_bx * np.sin(minute_by*3.141592/(180*60))
        
            dif_bx = scipy.signal.detrend(Bx)
            dif_by = scipy.signal.detrend(By)
        else:
            dif_bx = scipy.signal.detrend(minute_bx)
            dif_by = scipy.signal.detrend(minute_by)
       
        #dif_bx = scipy.signal.detrend(minute_bx)
        #dif_by = scipy.signal.detrend(minute_by)

        ################################################################
        #removing first day


        #Using only the last 48 hours of data
        if site=='ARM' or site=='DUN':
            Date=Date[-2880*60:-1]
            
            Hour=Hour[-2880*60:-1]
        else:
            Date=Date[-2880:-1]
            
            Hour=Hour[-2880:-1]
        dif_bx=np.array(dif_bx[-2880:-1])
        dif_by=np.array(dif_by[-2880:-1])

        
        
        last_bx=np.array(dif_bx[-1])
        last_by=np.array(dif_by[-1])  
        
        #zero-end padding

        flatbx=np.linspace(last_bx,0,padding)
        flatby=np.linspace(last_by,0,padding)
        
        #zero padding at end  of series
        #flat=np.array(np.zeros(105))
        #print(dif_bx)

        dif_bx=np.append(dif_bx,flatbx)
        dif_by=np.append(dif_by,flatby)
            
        ############################################################
    
        length=len(dif_bx)
        if ip==0:
            #list to save magnetics
            mag_s = np.zeros([length,len(s),2])
        
        # compute FFT
        perd, dif_bx_fft, dif_by_fft = scr_fft(dif_bx,dif_by,samp)
            
        # Select periods of interest
        factor = np.ones(perd.shape[0])        
        for i, v in enumerate (perd):
            if (v < low) or (v > hi):
                factor[i] = 0
    
        bx_cfft = (factor * dif_bx_fft) 
        by_cfft = (factor * dif_by_fft) 
        
        # Compute ifft
        bx_c = np.real(np.array(
                ifft(bx_cfft + np.conj(np.roll(bx_cfft[::-1], 1)))))
        
        by_c = np.real(np.array(
                ifft(by_cfft + np.conj(np.roll(by_cfft[::-1], 1)))))
    
        # Deffine the vector of magnetic field data
        ax = np.array([bx_c]).T
        ay = np.array([by_c]).T
    
        mag_s[:,ip,:] = np.hstack((ax, ay))
    
        # Write the magnetic data to be used by the SECS interpolation algorithm
        if var == 1:
    
            mag = str(s[ip])

            w_path = (str(secs_path) 
                     + "SECS_" 
                     + str(mag) 
                     + "_realtime.dat")
            
            f_id = open(w_path,'w+')
            np.savetxt(f_id, mag_s[:,ip,:], fmt=['%15.5f', '%15.5f'])
            print(mag_s)
            f_id.close()
             
    return(mag_s, s,length, Date, Hour)

def read_magnetics( in_path, sites, mag_path, secs_path, samp, hi, low, 
                     var,rmf):
    
    """ Read magnetic time series from input folder
		
		Parameters
		-----------
		in_path = Folder with input parameters
		sites = Name of the site
         mag_path = Folder with magnetic fields time series
         secs_path = Folder with inputs - outputs for SECS interpolation
         samp = Sampling rate (seconds)
         hi = maximum period to analyse (seconds)
         low = minimum period to analyse (seconds)
         length = length of the time series
         var = if (1) write the magnetic time series

		Returns
		-----------
		mag_s = Magnetic time series
		s = Name of the site

        Here data is loaded
        Can change padding length (set to 0 for non realtime)
        

		-----------------------------------------------------------------
	"""
    ############################################
    #Dowloading Magnetic field data
    ############################################
    # Check the magnetic observatories used in the experiment
    s, s_lat, s_lon = read_co(in_path + str(sites))
    # Read the time series for these magnetic fields
    nowz=datetime.datetime.utcnow()
    year_str=str(nowz.year)
    month_str="%02d" %(nowz.month)
    day_str="%02d" %(nowz.day)
    one_day=nowz-datetime.timedelta(days=1)
    two_day=nowz-datetime.timedelta(days=2)

    days=[two_day,one_day,nowz]


    #for ip in range(0,len(s)):
    bx_second=[]
    by_second=[]
    bz_second=[]
    magie_second=[]
    archive_list=[]
    for ip, site in enumerate(rmf):        
        
        site_l=site.lower()
        #Strs used for real listed below
        #Reading in last 3 magnetometer files
    
        print('rmf',rmf)
        website="https://data.magie.ie/"
        file=urllib.request.URLopener()
        save_folder=r'C:\Users\Dunsink\Documents\Python Scripts\Geo_Electrics_realtime_houdini/Data/'
        
        for i in range(0,3):
            day_str="%02d" %(nowz.day-i)
            
            if day_str =="00":
                time=nowz-datetime.timedelta(days=i)
                day_str="%02d" %(time.day)
                month_str="%02d" %(time.month)
                #month_str=months[i]
                year_str="%02d" %(time.year)
            
            date_str= year_str+'/'+month_str+'/'+day_str+'/txt/'
            date_str2= site_l+year_str+month_str+day_str+'.txt'
            
            site_day=website+date_str+date_str2
            
            #uncomment to turn on
            file.retrieve(site_day,save_folder+site_l+str(i)+'.txt')
        l=2
        for i in days:
            
            year_str=str(i.year)
            month_str="%02d" %(i.month)
            day_str="%02d" %(i.day)
            
            if site=="ARM" or site=="BIR" or site=="DUN":
                #archive_str=mag_path+year_str+'/'+month_str+'/'+day_str+'/txt/'+str(site_l)+year_str+month_str+day_str+".txt"

                archive_str=save_folder+str(site_l)+str(l)+'.txt'

                val=1 #number 
                MagIE=1
            else: #Else to deal with VAL intermag sites
            
                archive_str=save_folder+"val"+str(l)+'.txt'
                val=1
                MagIE=0
            archive_list.append(archive_str)
            l=l-1
        #archive_list=[archive_str1,archive_str2,archive_str3]
        #########################################################
        #Now loading maagnetic field data
        #########################################################
        count=0
        
        for i in archive_list:
            fff = i
            f = open(fff, 'r')
            data = f.readlines()
            f.close()
        
            # Read the number of lines to be skipped
            #val=1 if MagIE site, varies for intermag
            for index, line in enumerate(data):
                if line.startswith('DATE'):
                    val = index
        
        for i in archive_list:
            df = pd.read_csv(i, sep = '\s+', skiprows = int(val)) # Check the rows that need to be skiped
        
            count = count + 1
            #Concatenating previous lists in archive lists together
            if count == 1:
                df_t = np.array(df)
                #print(df_t.shape)
            else:        
                df_t = np.row_stack((df_t, df)) #appending data
                #print(df_t.shape)
        
        Date = df_t[:,0]
        Hour = df_t[:,1]  
        bx = pd.to_numeric(df_t[:,3])
        by = pd.to_numeric(df_t[:,4])
        bz = pd.to_numeric(df_t[:,5])
        
        bx_second.append(bx)
        by_second.append(by)
        bz_second.append(bz)
        magie_second.append(MagIE)
        timedate = timedatez(Date, Hour) 
        
        #converting dates and hours to datetime objects
        timedate_float = time2float(timedate)
    
    
    ###################################################
    
        
    for ip, site in enumerate(rmf):    
        
        #if site=='VAL':
        #    MagIE=0#1
        #else:
        #    MagIE=1
        bx=bx_second[ip]
        by=by_second[ip]
        bz=bz_second[ip]

        #Determining whether Bx is errorsome
        bx[bx >= 80000.0] = 'nan'
        bx[bx == 'infs'] = 'nan'
        bx[bx <= 0.0] = 'nan'
        nans, x = nan_helper(bx)
        bx[nans]= np.interp(x(nans), x(~nans), bx[~nans])
            
        
        by[by >= 50000.0] = 'nan'
        by[by == 'infs'] = 'nan'
        by[by <= -10000.0] = 'nan'
        nans, x = nan_helper(by)
        by[nans]= np.interp(x(nans), x(~nans), by[~nans])
    
        bz[bz >= 80000.0] = 'nan'
        bz[bz == 'infs'] = 'nan'
        bz[bz <= 10000.0] = 'nan'
        nans, x = nan_helper(bz)
        bz[nans]= np.interp(x(nans), x(~nans), bz[~nans])

        #Converting MagIE files from second data to minute
        if MagIE==1:
            minute_time, minute_bx, minute_by, minute_bz= minute_bin(timedate_float, bx, by, bz, 5)
        if MagIE==0:
            minute_time, minute_bx, minute_by, minute_bz= timedate_float, bx, by, bz
            
            # Remove detrend
        #print(minute_bx)
        MagIE=1
        if MagIE==0:
            Bx =minute_bx * np.cos(minute_by*3.141592/(180*60))
            By = minute_bx * np.sin(minute_by*3.141592/(180*60))
        
            dif_bx = scipy.signal.detrend(Bx)
            dif_by = scipy.signal.detrend(By)
        else:
            dif_bx = scipy.signal.detrend(minute_bx)
            dif_by = scipy.signal.detrend(minute_by)
        
        #dif_bx = scipy.signal.detrend(minute_bx)
        #dif_by = scipy.signal.detrend(minute_by)

        ################################################################
        #removing first day to reduce run time
        #only keeping last 2 hours of first day for interpolation
        #To reduce runspeed
        #dif_bx = dif_bx[1440-120:-1]
        #dif_by = dif_by[1440-120:-1]
        
        dif_bx=np.array(dif_bx)
        dif_by=np.array(dif_by)

        last_bx=np.array(dif_bx[-1])
        last_by=np.array(dif_by[-1])  
        #zero-end padding
        #change third fifgure to chnage length
        flatbx=np.linspace(last_bx,0,105)
        flatby=np.linspace(last_by,0,105)
        #zero padding at end  of series
        #flat=np.array(np.zeros(105))
        #print(dif_bx)

        dif_bx=np.append(dif_bx,flatbx)
        dif_by=np.append(dif_by,flatby)
            
        ############################################################
    
        length=len(dif_bx)
        mag_s = np.zeros([length,len(s),2])
        
        # compute FFT
        perd, dif_bx_fft, dif_by_fft = scr_fft(dif_bx,dif_by,samp)
            
        # Select periods of interest
        factor = np.ones(perd.shape[0])        
        for i, v in enumerate (perd):
            if (v < low) or (v > hi):
                factor[i] = 0
    
        bx_cfft = (factor * dif_bx_fft) 
        by_cfft = (factor * dif_by_fft) 
        
        # Compute ifft
        bx_c = np.real(np.array(
                ifft(bx_cfft + np.conj(np.roll(bx_cfft[::-1], 1)))))
        
        by_c = np.real(np.array(
                ifft(by_cfft + np.conj(np.roll(by_cfft[::-1], 1)))))
    
        # Deffine the vector of magnetic field data
        ax = np.array([bx_c]).T
        ay = np.array([by_c]).T
    
        mag_s[:,ip,:] = np.hstack((ax, ay))
    
        # Write the magnetic data to be used by the SECS interpolation algorithm
        if var == 1:
    
            mag = str(s[ip])

            w_path = (str(secs_path) 
                     + "SECS_" 
                     + str(mag) 
                     + "_realtime.dat")
            
            f_id = open(w_path,'w+')
            np.savetxt(f_id, mag_s[:,ip,:], fmt=['%15.5f', '%15.5f'])
            f_id.close()
             
    return(mag_s, s,length, Date, Hour)

###############################################################################


##########################################################################
def error_secs_interpolation(e_lat, e_lon, in_path, mode, obs):

    """ Compute the errors caused by SECS interpolation (based on Figure 7, 
        Campanya et al 2018)
		
		Parameters
		-----------
         e_lat = latitude sites to compute electric fields
         e_lon = longitude sites to compute magnetic fields
         in_path = Folder with input parameters
         mode = (1) for Approach #1 and (2) for Approach #2
         obs_f = Name of the file with name and coordinated of the 
                 magnetic observatories

		Returns
		-----------
         error_bf = error associated with the SECS interpolation approach
		-----------------------------------------------------------------
	"""
    
    # Compute the distance of each site to the magnetic observatories

    Obs, Obs_lat, Obs_lon = read_co(in_path + str(obs))
    dist = np.zeros(len(Obs))

    for i in range(0,len(Obs)):
        coo_a = (e_lat, e_lon)
        coo_b = (Obs_lat[i], Obs_lon[i])        
        dist[i] = geopy.distance.vincenty(coo_a, coo_b).km
    
    if mode == 1: # Approach 1   
        snr = -1.75e-2 * dist.min() + 12.81                         

    if mode == 2: # Approach 2
        snr = -1.48e-2 * dist.min() + 9.70
    # Compute the error                             
    error_bf = 1.0/np.sqrt([10**(snr/10)])
    
    return(error_bf)
def compute_e_fields_secs(sb, tf_path, e_site, samp, hi, low, error_bf,
                          ef_tf, e_nvpwa, stat):

    """ Compute E fields using magnetic time series from SECS
		
		Parameters
		-----------
         sb = magnetic time series from SECS
         tf_path = Folder with electromagnetic tensor relationships
         e_site = Name of the site to compute electric fields
         samp = Sampling rate (seconds)
         hi = Maximum period to analyse (seconds)
         low = Minimum period to analyse (seconds)
         error_bf = Error associated with the SECS interpolation approach
         ef_tf = Error floor for the MT and quasi-MT tensor relationships
         e_ncpwa =  Error floor for the Non-plane wave approximation
         stat = Statistics for error propagation

         
		Returns
		-----------
         tf_ex = electric time series x component (mean value)
         tf_ey = electric time series y component (mean value)
         std_ex = standard deviation of the electric time series x component
         std_ey = standard deviation of the electric time series y component
         
         -----------------------------------------------------------
	"""
    
    s_bx = sb[:,0]*1e-9  # convert to nT
    s_by = sb[:,1]*1e-9  # convert to nT

    # Remove detrend
    s_bx = scipy.signal.detrend(s_bx)
    s_by = scipy.signal.detrend(s_by)
    
    # Tukey window to avoid instabilities at the edges of the time series
    window = scipy.signal.tukey(s_by.shape[0], 0) 
    s_bx = window * s_bx 
    s_by = window * s_by
    
    # get Frequencies / periods
    freqB = np.fft.fftfreq(s_bx.shape[0], d = samp)
    for i in range(0,len(freqB)):
        if freqB[i] == 0:
            freqB[i] = 1e-99

    perB = freqB ** -1
    
    # Compute fft
    s_bx_fft = fftpack.fft(s_bx)
    s_by_fft = fftpack.fft(s_by)

    
    # Read tensors
    file_format = -1
    try:
        filename = (str(tf_path) 
                    + "E"
                    + str(e_site) 
                    + "B"
                    + str(e_site) 
                    + "_s.j")
        

        f = open(filename, 'r')
        data = f.readlines()
        f.close()

        file_format = 0
    except:
        check = 0

    try:
         filename = (str(tf_path) 
                     + "E"
                     + str(e_site) 
                     + "B"
                     + str(e_site)
                     + "_s.edi")
         
         f = open(filename, 'r')
         data = f.readlines()
         f.close()

         file_format = 1
    except:
        check = 1

    if file_format == 0:     
        try:
            for index, line in enumerate(data):
                if line.startswith("ZXX"):
                    index_zxx = index
                if line.startswith("ZXY"):
                    index_zxy = index
                if line.startswith("ZYX"):
                    index_zyx = index
                if line.startswith("ZYY"):
                    index_zyy = index
                if line.startswith("TZX"):
                    index_tzx = index
                if line.startswith("TZY"):
                    index_tzy = index
                  
            data_zxx = data[index_zxx + 2 : index_zxy]
            zxx = np.loadtxt(data_zxx)    
            data_zxy = data[index_zxy + 2 : index_zyx]
            zxy = np.loadtxt(data_zxy)     
            data_zyx = data[index_zyx + 2 : index_zyy]
            zyx = np.loadtxt(data_zyx)    
            data_zyy = data[index_zyy + 2 : index_tzx]
            zyy = np.loadtxt(data_zyy)  
            per_z = zxx[:,0]
            
            zxx[:,1:3] = (1) * zxx[:,1:3]
            zxy[:,1:3] = (1) * zxy[:,1:3]
            zyx[:,1:3] = (1) * zyx[:,1:3]
            zyy[:,1:3] = (1) * zyy[:,1:3]


        except:
            check = 2     
            
    if file_format == 1:     
        try:
            for index, line in enumerate(data):
                if 'NFREQ' in line[:5]:
                    for j in range(0, len(line)):
                        if line[j] == '=':
                            n_freq = int(line[j+1::])
        
                if 'NPER' in line[:5]:
                    for j in range(0, len(line)):
                        if line[j] == '=':
                            n_freq = int(line[j+1::])
        
                if '>FREQ' in line[:5]:
                    freq = read_variable(n_freq, index, data)
                    
                if '>PERI' in line[:5]:
                    per = read_variable(n_freq, index, data)
                    freq = 1./per
                    
                if '>ZROT' in line[:5]:
                    zrot = read_variable(n_freq, index, data)
        
                if '>ZXXR' in line[:5]:
                    zxxr = read_variable(n_freq, index, data)
                    
                if '>ZXXI' in line[:5]:
                    zxxi = read_variable(n_freq, index, data)
        
                if '>ZXX.V' in line[:6]:
                    zxxv = read_variable(n_freq, index, data)
                    zxxstd = np.sqrt(zxxv)
    
                if '>ZXYR' in line[:5]:
                    zxyr = read_variable(n_freq, index, data)
        
                if '>ZXYI' in line[:5]:
                    zxyi = read_variable(n_freq, index, data)
        
                if '>ZXY.V' in line[:6]:
                    zxyv = read_variable(n_freq, index, data)
                    zxystd = np.sqrt(zxyv)
    
                if '>ZYXR' in line[:5]:
                    zyxr = read_variable(n_freq, index, data)
                if '>ZYXI' in line[:5]:
                    zyxi = read_variable(n_freq, index, data)
                    
                if '>ZYX.V' in line[:6]:
                    zyxv = read_variable(n_freq, index, data)
                    zyxstd = np.sqrt(zyxv)
    
                if '>ZYYR' in line[:5]:
                    zyyr = read_variable(n_freq, index, data)
                    
                if '>ZYYI' in line[:5]:
                    zyyi = read_variable(n_freq, index, data)
                    
                if '>ZYY.V' in line[:6]:
                    zyyv = read_variable(n_freq, index, data)
                    zyystd = np.sqrt(zyyv)
            try:
                periods = 1./freq
            except:
                periods = per    
            
            zxx = np.column_stack([periods, -1*zxxr, -1*zxxi, zxxstd])
            zxy = np.column_stack([periods, -1*zxyr, -1*zxyi, zxystd])
            zyx = np.column_stack([periods, -1*zyxr, -1*zyxi, zyxstd])
            zyy = np.column_stack([periods, -1*zyyr, -1*zyyi, zyystd])
            per_z = zxx[:,0]
    
            if per_z[0] > per_z[1]:
                per_z = per_z[::-1]
                zxx = zxx[::-1]
                zxy = zxy[::-1]
                zyx = zyx[::-1]
                zyy = zyy[::-1]

        except:
            check = 3
            
    if file_format == -1:
        print('Cannot read the MT impedance tensor for site:' + str(e_site))
        print('MT impdeance tensor must be j. or edi. file')

    # Select the periods of interest
    factor = np.ones(perB.shape[0])
    zxx_int=np.zeros([perB.shape[0],3])
    zxy_int=np.zeros([perB.shape[0],3])
    zyx_int=np.zeros([perB.shape[0],3])
    zyy_int=np.zeros([perB.shape[0],3])
    
    for i, v in enumerate (perB):
        if (v < low) or (v > hi):
           factor[i] = 0
    
    for i in range (0,3):
        zxx_int[:,i] = np.interp(perB, per_z, zxx[:,i+1])*factor
        zxy_int[:,i] = np.interp(perB, per_z, zxy[:,i+1])*factor
        zyx_int[:,i] = np.interp(perB, per_z, zyx[:,i+1])*factor
        zyy_int[:,i] = np.interp(perB, per_z, zyy[:,i+1])*factor

    # Deffine Variables
    ex_calc=np.zeros([s_bx.shape[0],stat])
    ey_calc=np.zeros([s_bx.shape[0],stat])
   
    # Deffine Error floor
    zzz_det = np.sqrt(np.abs(((zxy_int[:,0] + zxy_int[:,1]*1j)
              * (zyx_int[:,0] + zyx_int[:,1]*1j)) 
              - ((zxx_int[:,0] + zxx_int[:,1]*1j) 
              * (zyy_int[:,0] + zyy_int[:,1]*1j))))

    for ik in range(0,len(zxx_int)):
        if zxx_int[ik,2] <= zzz_det[ik]*ef_tf:
            zxx_int[ik,2] = zzz_det[ik]*ef_tf 
        if zxy_int[ik,2] <= zzz_det[ik]*ef_tf:
            zxy_int[ik,2] = zzz_det[ik]*ef_tf 
        if zyx_int[ik,2] <= zzz_det[ik]*ef_tf:
            zyx_int[ik,2] = zzz_det[ik]*ef_tf 
        if zyy_int[ik,2] <= zzz_det[ik]*ef_tf:
            zyy_int[ik,2] = zzz_det[ik]*ef_tf 
                       
    # Compute electric fields    
    for i in range(0,stat):
        ex_1 = ((((zxx_int[:,0] + np.random.standard_normal()*zxx_int[:,2])
              + (zxx_int[:,1] + np.random.standard_normal()*zxx_int[:,2])*1j) 
              * (s_bx_fft + s_bx_fft*np.random.standard_normal()*(error_bf))) 
              + (((zxy_int[:,0]+np.random.standard_normal()*zxy_int[:,2])
              + (zxy_int[:,1]+np.random.standard_normal()*zxy_int[:,2])*1j) 
              * (s_by_fft + s_by_fft*np.random.standard_normal()*(error_bf))))
    
        ey_1 = ((((zyx_int[:,0]+np.random.standard_normal()*zyx_int[:,2])
              + (zyx_int[:,1]+np.random.standard_normal()*zyx_int[:,2])*1j) 
              * (s_bx_fft + s_bx_fft*np.random.standard_normal()*(error_bf))) 
              + (((zyy_int[:,0]+np.random.standard_normal()*zyy_int[:,2])
              + (zyy_int[:,1]+np.random.standard_normal()*zyy_int[:,2])*1j) 
              * (s_by_fft + s_by_fft*np.random.standard_normal()*(error_bf))))
        
        # Add error associated with the non-validity of the planewave approx.
        ex_1 = ex_1 + np.random.standard_normal() * ex_1 * e_nvpwa
        ey_1 = ey_1 + np.random.standard_normal() * ey_1 * e_nvpwa

        # Compute ifft
        ex_calc[:,i] = (np.real(ifft(ex_1 + np.conj(np.roll(ex_1[::-1],1)))) 
                       * (1000. / (4*np.pi*1e-7)))
        ey_calc[:,i] = (np.real(ifft(ey_1 + np.conj(np.roll(ey_1[::-1],1)))) 
                       * (1000. / (4*np.pi*1e-7)))

    # Define mean value
    mex=np.zeros(s_bx.shape[0])
    mey=np.zeros(s_bx.shape[0])
    
    for i in range(s_bx.shape[0]):
        mex[i]=ex_calc[i,:].mean()
        mey[i]=ey_calc[i,:].mean()
        
    # Calculate standard deviation (errorbars)
    ex_s=np.zeros(s_bx.shape[0])
    ey_s=np.zeros(s_bx.shape[0])
    
    for i in range(s_bx.shape[0]):
        ex_s[i]=ex_calc[i,:].std()
        ey_s[i]=ey_calc[i,:].std()
    

        
    # Deffine the outputs
    tf_ex = np.array(np.copy(mex))
    tf_ey = np.array(np.copy(mey))
    std_ex = np.array(np.copy(ex_s))
    std_ey = np.array(np.copy(ey_s))

    return (tf_ex, tf_ey, std_ex, std_ey)
