#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/03/17 at 3:35 PM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from collections import OrderedDict


# =============================================================================
# Define variables
# =============================================================================
# set file paths
WORKSPACE = '/Astro/Projects/RayPaul_Work/SuperWASP/'
DPATH = WORKSPACE + '/Data/Elodie/'
# -----------------------------------------------------------------------------
# SID = 'ARG_54'
SID = 'BPC_46A'
# -----------------------------------------------------------------------------
# sigma clipping options
# size in pixels (actual size will be median separation between points * SIZE)
SIZE = 100
# sigma of the clip i.e. median(DATACOL) + sigma*std(DATACOL)
SIGMA = 3.0
# whether to use a weighted median based on uncertainties
WEIGHTED = True

# -----------------------------------------------------------------------------
# whether data is binned (changes col names and file names)
BINDATA = True
BINSIZE = 0.01
# -----------------------------------------------------------------------------
# Column info
if BINDATA:
    TIMECOL = 'time'
    DATACOL = 'mag'
    EDATACOL = 'emag'
else:
    TIMECOL = 'HJD'
    DATACOL = 'MAG2'
    EDATACOL = 'MAG2_ERR'


# =============================================================================
# Define functions
# =============================================================================
def quantile_1D(data, weights, quantile):
    """
    Compute the weighted quantile of a 1D numpy array.

    Taken from:
    https://github.com/nudomarinero/wquantiles/blob/master/wquantiles.py

    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.
    Returns
    -------
    quantile_1D : float
        The output value.
    """
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if ((quantile > 1.) or (quantile < 0.)):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    # TODO: Check that the weights do not sum zero
    #assert Sn != 0, "The sum of the weights must not be zero"
    Pn = (Sn-0.5*sorted_weights)/np.sum(sorted_weights)
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)


def sigma_clip(y=None, x=None, ey=None, sigma=3.0, boxsize=10,
               weighted=False, log=False):
    """
    Performs a sigma clip on all y points (weighted if requested)
    uses a box of size median(dx)*boxsize and a clip of:
       median - sigma*stdev < data < median + sigma*stdev

    where median and stdev are weighted if "weighted"=True

    :param y: numpy array, array to be sigma clipped

    :param x: numpy array or None, if not none used for non uniform data
              where dx != 1, if x is None dx = 1

    :param ey: numpy array, uncertainties associated with y (required for
               weighted median)

    :param sigma: float, how many sigmas away from the median to perform the
                  sigma clip to

    :param boxsize: int, number of pixels for size of box (total size of the
                    box is boxsize * dx (centered on the y-element in question

    :param weighted: boolean, if True and ey is not None, performs a weighted
                     median (and weighted standard deviation) based on
                     quantile_1D

                     i.e. standard deviation = mean(q[84]-q[50], q[50]-q[16])
                          median = q[50]

    :param log: boolean, if True print progress messages to screen

    :return good: numpy array of booleans, mask of the points that are kept
                  by the sigma clipping
    """
    # if x is none use a index grid
    # work out average pixel separation (for box size)
    dx = np.median(x[1:] - x[:-1])
    # loop around each row
    good = np.ones_like(x, dtype=bool)
    for row in __tqdmlog__(range(len(x)), log):
        # need box that is boxsize * dx
        xbox = boxsize // 2 * dx
        # mask for points inside box
        mask = (x > (x[row] - xbox)) & (x < (x[row] + xbox))
        # if use weighted median
        if weighted and (ey is not None):
            weights = 1.0 / ey[mask] ** 2
            median = quantile_1D(y[mask], weights, 0.50)
            upper = quantile_1D(y[mask], weights, 0.682689492137/2.0 + 0.5)
            lower = quantile_1D(y[mask], weights, 0.5 - 0.682689492137/2.0)
            onestd = np.mean([upper - median, median-lower])
        # else use median and have no weighted error
        else:
            median = np.median(y[mask])
            onestd = np.std(y[mask])
        good[row] &= (y[row] > (median - sigma*onestd))
        good[row] &= (y[row] < (median + sigma*onestd))
    # return match of good values (True where good)
    return good


def __tqdmlog__(x_input, log):
    """
    Private function for dealing with logging

    :param x_input:  any iterable object

    :param log: bool, if True and module tqdm exists use logging

    :return:
    """
    # deal with importing tqdm
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        tqdm = (lambda x: x)
    # deal with logging
    if log:
        rr = tqdm(x_input)
    else:
        rr = x_input
    return rr


def save_to_file(coldata, savename, savepath, exts=None):
    # ---------------------------------------------------------------------
    # Convert to astropy table
    atable = Table()
    for col in coldata:
        dtype = type(coldata[col][0])
        atable[col] = np.array(coldata[col], dtype=dtype)
    # ---------------------------------------------------------------------
    # Save as fits file
    print('\n Saving to file...')
    if exts is None:
        exts = ['.fits']
    formats = dict(fits='fits', dat='ascii',  csv='csv')
    for ext in exts:
        fmt = formats[ext]
        path = '{0}{1}.{2}'.format(savepath, savename, ext)
        atable.write(path, format=fmt, overwrite='True')


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # load data
    print("\n Loading data...")
    sargs = [SID, 'binsize={0}'.format(BINSIZE)]
    if BINDATA:
        dname = '{0}_lightcurve_{1}.fits'.format(*sargs)
    else:
        dname = '{0}_lightcurve.fits'.format(*sargs)
    lightcurve = fits.getdata(DPATH + dname)
    # -------------------------------------------------------------------------
    # get columns
    time_arr = np.array(lightcurve[TIMECOL])
    data_arr = np.array(lightcurve[DATACOL])
    edata_arr = np.array(lightcurve[EDATACOL])
    # -------------------------------------------------------------------------
    # perform sigma clipping with weighted median
    gmask1 = sigma_clip(data_arr, x=time_arr, ey=edata_arr, sigma=SIGMA,
                        boxsize=SIZE, weighted=True)
    # push back into dictionary
    pdata = OrderedDict(time=time_arr[gmask1],
                        data=data_arr[gmask1],
                        edata=edata_arr[gmask1])
    # compute save name
    sargs = [SID, 'binsize={0}'.format(BINSIZE), 'w{0}sigmaclip'.format(SIGMA)]
    if BINDATA:
        dname = '{0}_lightcurve_{1}_{2}'.format(*sargs)
    else:
        dname = '{0}_lightcurve_{2}'.format(*sargs)
    # save to file
    print('\n Saving light curve to file...')
    save_to_file(pdata, dname, DPATH, exts=['fits'])
    # ---------------------------------------------------------------------
    # perform sigma clipping without weighted median
    gmask2 = sigma_clip(data_arr, x=time_arr, sigma=SIGMA,
                        boxsize=SIZE, weighted=False)
    # push back into dictionary
    pdata = OrderedDict(time=time_arr[gmask2],
                        mag=data_arr[gmask2],
                        emag=edata_arr[gmask2])
    # compute save name
    sargs = [SID, 'binsize={0}'.format(BINSIZE), '{0}sigmaclip'.format(SIGMA)]
    if BINDATA:
        dname = '{0}_lightcurve_{1}_{2}'.format(*sargs)
    else:
        dname = '{0}_lightcurve_{2}'.format(*sargs)
    # save to file
    print('\n Saving light curve to file...')
    save_to_file(pdata, dname, DPATH, exts=['fits'])


# =============================================================================
# End of code
# =============================================================================
