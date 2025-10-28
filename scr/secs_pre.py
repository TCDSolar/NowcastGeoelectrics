"""
# Functions to be used in the Spherical Elementary Current Systems program
# (SECS_interpolation.py)
#
# Original Author: C.D.Beggan (adapted from AJmcK) [ciar@bgs.ac.uk]
# Ported from MATLAB to Python by Sean Blake (Trinity College Dublin) [blakese@tcd.ie]
# Modified by Joan Campanya (Trinity College Dublin) [joan.campanya@tcd.ie] 
# Date: 31-Aug-2018
"""

import numpy as np

def cart2sph_matlab(x, y, z, theta, phi):
    """ Cartesian to Spherical """
    sint = np.sin(theta)
    cost = np.cos(theta)
    sinp = np.sin(phi)
    cosp = np.cos(phi)

    rhat = x*sint*cosp + y*sint*sinp + z*cost
    thetahat = x*cost*cosp + y*cost*sinp - z*sint
    phihat = y*cosp - x*sinp

    return (rhat, thetahat, phihat)

def sph2cart_matlab(x, y, z, theta, phi):
    """ Spherical to Cartesian """
    sint = np.sin(theta)
    cost = np.cos(theta)
    sinp = np.sin(phi)
    cosp = np.cos(phi)

    xhat = x*sint*cosp + y*cost*cosp-z*sinp
    yhat = x*sint*sinp + y*cost*sinp + z*cosp
    zhat = x*cost - y*sint

    return (xhat, yhat, zhat)

def pole_common2source(xx, yy, zz, theta, phi):

    rtheta = theta
    rphi = (np.pi/2. + phi)

    sint = np.sin(rtheta)
    cost = np.cos(rtheta)
    sinp = np.sin(rphi)
    cosp = np.cos(rphi)

    dum = xx * cosp + yy*sinp
    yy = -xx*sinp + yy*cosp
    xx = dum

    dum = yy*cost + zz*sint
    zz = -yy*sint + zz*cost
    yy = dum

    return(xx, yy, zz)

def pole_source2common(xx, yy, zz, theta, phi):

    rtheta = theta
    rphi = (np.pi/2. + phi)
    sint = np.sin(rtheta)
    cost = np.cos(rtheta)
    sinp = np.sin(rphi)
    cosp = np.cos(rphi)

    dum = yy*cost - zz*sint
    zz = yy*sint + zz*cost
    yy = dum

    dum = xx*cosp - yy*sinp
    yy = xx*sinp + yy*cosp
    xx = dum

    return(xx, yy, zz)

def sourcecords(theta, phi, stheta, sphi):
    sint = np.sin(theta)
    cost = np.cos(theta)
    sinp = np.sin(phi)
    cosp = np.cos(phi)

    xx = sint * cosp
    yy = sint * sinp
    zz = cost

    x, y, z = pole_common2source(xx, yy, zz, stheta, sphi)

    stheta = np.arccos(z/ (np.sqrt(x**2 + y**2 + z**2)))

    sphi = np.arctan2(y, x)

    return (stheta, sphi)

def secsmatrix_XYonly(latpts, lonpts, nsecs, sitelat, sitelon, 
    lat, lon, measrad, srcrad):
    
    def sourcecords(theta, phi, stheta, sphi):
        sint = np.sin(theta)
        cost = np.cos(theta)
        sinp = np.sin(phi)
        cosp = np.cos(phi)
    
        xx = sint * cosp
        yy = sint * sinp
        zz = cost
    
        x, y, z = pole_common2source(xx, yy, zz, stheta, sphi)
    
        stheta = np.arccos(z/ (np.sqrt(x**2 + y**2 + z**2)))
    
        sphi = np.arctan2(y, x)
    
        return (stheta, sphi)
    
    def pole_source2common(xx, yy, zz, theta, phi):

        rtheta = theta
        rphi = (np.pi/2. + phi)
        sint = np.sin(rtheta)
        cost = np.cos(rtheta)
        sinp = np.sin(rphi)
        cosp = np.cos(rphi)
    
        dum = yy*cost - zz*sint
        zz = yy*sint + zz*cost
        yy = dum
    
        dum = xx*cosp - yy*sinp
        yy = xx*sinp + yy*cosp
        xx = dum
    
        return(xx, yy, zz)
    
    def pole_common2source(xx, yy, zz, theta, phi):

        rtheta = theta
        rphi = (np.pi/2. + phi)
    
        sint = np.sin(rtheta)
        cost = np.cos(rtheta)
        sinp = np.sin(rphi)
        cosp = np.cos(rphi)
    
        dum = xx * cosp + yy*sinp
        yy = -xx*sinp + yy*cosp
        xx = dum
    
        dum = yy*cost + zz*sint
        zz = -yy*sint + zz*cost
        yy = dum
    
        return(xx, yy, zz)
    
    def sph2cart_matlab(x, y, z, theta, phi):
        """ Spherical to Cartesian """
        sint = np.sin(theta)
        cost = np.cos(theta)
        sinp = np.sin(phi)
        cosp = np.cos(phi)
    
        xhat = x*sint*cosp + y*cost*cosp-z*sinp
        yhat = x*sint*sinp + y*cost*sinp + z*cosp
        zhat = x*cost - y*sint
    
        return (xhat, yhat, zhat)
    
    def cart2sph_matlab(x, y, z, theta, phi):
        """ Cartesian to Spherical """
        sint = np.sin(theta)
        cost = np.cos(theta)
        sinp = np.sin(phi)
        cosp = np.cos(phi)
    
        rhat = x*sint*cosp + y*sint*sinp + z*cost
        thetahat = x*cost*cosp + y*cost*sinp - z*sint
        phihat = y*cosp - x*sinp
    
        return (rhat, thetahat, phihat)
    
    """ For a given grid position, calculate the geometrical factors which 
        relate an elementary current placed in the ionosphere to the magnetic 
        field at the grid position. """

    Tx, Ty = [], []

    testa, testb, testc = [], [], []

    d2rad = np.pi/180.
    muzero = np.pi * 4e-7
    currconst = muzero/float((4*np.pi*measrad))

    roverR = measrad/float(srcrad)
    roverRsq = (roverR)**2

    grdlat = sitelat * d2rad
    grdlon = sitelon * d2rad
    grdcolat = (90 - sitelat) * d2rad

    
    for j in np.arange(0, lonpts, 1):
        for i in np.arange(0, latpts, 1):
            ij = (j-1)*latpts + i
            srclat = lat[i,j] * d2rad
            srclon = lon[i, j] * d2rad

            srccolat = (90.0 - lat[i, j]) * d2rad
            londiff = srclon - grdlon
            latdiff = srclat - grdlat


            #colat of grid point
            colat = (np.cos(srccolat) * np.cos(grdcolat)) + (np.sin(srccolat)* np.sin(grdcolat) * np.cos(londiff))
            costheta = colat

            #colat = math.acos(colat)
            colat = np.arccos(colat)
            sintheta = np.sin(colat)


            denom = (np.sqrt((1-(roverR * 2 * costheta) + roverRsq)))

            Btheta = (roverR - costheta)/denom + costheta
            Btheta = -1*currconst*Btheta/sintheta


            Br = (1/denom) - 1
            Br = currconst * Br

            # calculate co-lat and long of point in source co-ordinates
            srccolat, srclon = sourcecords(grdcolat, grdlon, srccolat, srclon)

            #transform spherical to cartesian
            Bphi = 0
            Br, Btheta, Bphi = sph2cart_matlab(Br, Btheta, Bphi, srccolat, srclon)

            #rotate pole from source co-ords to common co-ords
            #first find colat and lon of pole in common co-ordinates

            srccolat = (90.0 - lat[i, j])*d2rad
            srclon = lon[i, j]*d2rad
            Br, Btheta, Bphi = pole_source2common(Br, Btheta, Bphi, srccolat, srclon)

            # transform back 
            Br, Btheta, Bphi = cart2sph_matlab(Br, Btheta, Bphi, grdcolat, grdlon)

            
            # calculate weights to account for change in
            # gridpoint area with colat
            colat = 90 - lat[i, j]

            wgt = np.sin((colat)*d2rad)

            # Local geomagnetic co-ordinates
            Tx.append(-1*Btheta*wgt)
            Ty.append(Bphi*wgt)

    Tx = np.array(Tx)
    Ty = np.array(Ty)
    return Tx, Ty

##########################################################################
##########################################################################
