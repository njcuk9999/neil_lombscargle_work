#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/03/17 at 2:49 PM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from collections import OrderedDict
from astropy import units as u
from tqdm import tqdm

# =============================================================================
# Define variables
# =============================================================================
# set file paths
WORKSPACE = '/Astro/Projects/RayPaul_Work/SuperWASP/'
DPATH = WORKSPACE + '/Data/Elodie/'
# -----------------------------------------------------------------------------
SID = 'ARG_54'
SID = 'BPC_46A'
# -----------------------------------------------------------------------------
# Column info
TIMECOL = 'HJD'
DATACOL = 'MAG2'
EDATACOL = 'MAG2_ERR'

# -----------------------------------------------------------------------------
# whether to bin data
BINDATA = True
BINSIZE = 0.1
# -----------------------------------------------------------------------------



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


def bin_data(time, data, edata=None, binsize=None, log=False):
    """
    Bin time and data vectors by binsize (using a median combine of points in
    each bin (weight median if edata is not None).

    :param time: numpy array or list, input time(independent) vector

    :param data: numpy array or list, input dependent vector

    :param edata: None or numpy array, uncertainties associated with "data"

    :param binsize: float, size of each bin (in units of "time")

    :param log: boolean, if True prints progress to standard output
                         if False silent

    :return binnedtime: numpy array, binned "time" array

    :return binneddata: numpy array, binned "data" array

    """
    # Deal with bin size, if None, rebin to 1000 elements or don't bin
    # if len(time) is less than 1000
    if binsize is None:
        maxbins = np.min(len(time), 1000)
        bins = np.linspace(min(time), max(time), maxbins)
    else:
        bins = np.arange(min(time), max(time), binsize)

    # Now bin the data
    binnedtime = []
    binneddata = []
    binnederror = []
    # Loop round each bin and median the time and the data for all values
    # within that bin
    if log:
        print('\n\t Binning data...')
    for bin in __tqdmlog__(bins, log):
        # mask values within this iteration bin
        mask = (time >= bin) & (time < bin+binsize)
        # if there are no values in this bin do not bin it
        if np.sum(mask) == 0:
            continue
        # if there are values in this bin take the median or weighted median
        # if we have uncertainties
        else:
            # No uncertainties with time so just take the median
            btime = np.median(time[mask])
            # We have no uncertainties don't weight points
            if edata is None:
                bdata = np.median(data[mask])
                berror = np.nan
            # We have uncertainties so weight the medians
            else:
                weights = 1.0 / edata[mask] ** 2
                bdata = quantile_1D(data[mask], weights, 0.50)
                berror = 1.0/np.sqrt(np.sum(weights))
                # Finally add the binned data to array
            binnedtime.append(btime)
            binneddata.append(bdata)
            binnederror.append(berror)

    return np.array(binnedtime), np.array(binneddata), np.array(binnederror)


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
    # ----------------------------------------------------------------------
    # load data
    print("\n Loading data...")
    lightcurve = fits.getdata(DPATH + '{0}_lightcurve.fits'.format(SID))
    # ----------------------------------------------------------------------
    # get columns
    time_arr = np.array(lightcurve[TIMECOL])
    data_arr = np.array(lightcurve[DATACOL])
    edata_arr = np.array(lightcurve[EDATACOL])
    # ----------------------------------------------------------------------
    # Bin data
    if BINDATA:
        kwargs = dict(binsize=BINSIZE, log=True)
        if edata_arr is None:
            res = bin_data(time_arr, data_arr, **kwargs)
        else:
            res = bin_data(time_arr, data_arr, edata_arr, **kwargs)
        time_arr, data_arr, edata_arr = res
    # ----------------------------------------------------------------------
    # push back into dictionary
    pdata = OrderedDict(time=time_arr, data=data_arr, edata=edata_arr)
    # ---------------------------------------------------------------------
    # Save as fits file
    sargs = [SID, 'binsize={0}'.format(BINSIZE)]
    dname = '{0}_lightcurve_{1}'.format(*sargs)
    print('\n Saving light curve to file...')
    save_to_file(pdata, dname, DPATH, exts=['fits'])


# =============================================================================
# End of code
# =============================================================================
