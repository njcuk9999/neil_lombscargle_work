#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Date 2017-01-12

@author: Neil Cook

Program description here

Version 0.0.1

Last modified 2017-01-12
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import MySQLdb
import pandas
from scipy.signal import lombscargle

# =============================================================================
# Define variables
# =============================================================================
# set file paths
workspace = '/Astro/Projects/RayPaul_Work/SuperWASP/'
savepath = workspace + '/Plots/raw_lightcurves/'
# set database settings
hostname = 'localhost'
username = 'root'
password = '1234'
database = 'swasp'
table = 'swasp_sep16_tab'
# -----------------------------------------------------------------------------
skip_done = True


# =============================================================================
# Define functions
# =============================================================================
def load_db(db_name, host, uname, pword):
    conn1 = MySQLdb.connect(host=host, user=uname, db=db_name,
                            connect_timeout=100000, passwd=pword)
    c1 = conn1.cursor()
    return c1, conn1


def get_list_of_objects_from_db(conn):
    # ----------------------------------------------------------------------
    # find all systemids
    print("\nGetting list of objects...")
    query1 = "SELECT CONCAT(c.systemid,c.comp)"
    query1 += " AS sid FROM {0} AS c".format(table)
    query1 += " where c.systemid is not null and c.systemid <> ''"
    rawdata = pandas.read_sql_query(query1, conn)
    rawsystemid = np.array(rawdata['sid'])
    # get list of unique ids (for selecting each as a seperate curve)
    sids = np.array(np.unique(rawsystemid), dtype=str)
    # return list of objects
    return sids


def compute_lombscargle(xs, ys):
    """
    Compute the lombscarlge periodogram from time data and flux/magnitude data
    :param xs: numpy array of floats, time series data
    :param ys: numpy array of floats, flux/magnitude data
    :return:
    """
    # normalise the time series (i.e. start from zero)
    xs = xs - xs.min()
    # define the range of time values (rangex) and separation between (dffx)
    rangex, diffx = abs(xs[-1] - xs[0]) / (2 * np.pi), xs[1] - xs[0]

    # compute the number of frequencies
    freqs = np.linspace(1.0 / rangex, 1.0 / diffx,
                        np.max([4 * len(xs), 10000]))
    # value to normalise the periodogram by
    normval = xs.shape[0]
    # scale y axis to be between -1 and 1
    yscale = (ys - ys.mean()) / ys.std()
    # compute the Lomb-Scargle periodogram
    pgram = lombscargle(x, yscale, freqs)
    npgram = np.sqrt(4 * (pgram / normval))
    return freqs, npgram


def fit_largest_peak(xs, ys):
    """
    Algorithm to "fit" the largest peak

    :param xs: numpy array of floats, time series data
    :param ys: numpy array of floats, power data
    :return:
    """
    # make sure time series data is sorted
    sort = np.argsort(xs)
    xs, ys = np.array(xs[sort]), np.array(ys[sort])
    # find highest y point (x and y)
    argmax = np.argmax(ys)
    xmax, ymax = xs[argmax], ys[argmax]
    # based error on the first stationary point after the maximum point
    exmaxl = xmax - num_first_stationary_point(xs, ys, 'lower', xmax)
    exmaxu = num_first_stationary_point(xs, ys, 'upper', xmax) - xmax
    return dict(xmax=xmax, ymax=ymax, exmaxl=exmaxl, exmaxu=exmaxu)


def num_first_stationary_point(xs, ys, k, mu):
    """
    Find the stationary point around a certain value (mu)
    :param xs: numpy array of floats, time series data
    :param ys: numpy array of floats, power data
    :param k: string, lower or upper depending which side of the mu it is
    :param mu: float, the value to look for stationary points around
    :return:
    """
    # mask data to only part interested in and flip if lower
    if k == 'lower':
        mask = xs <= mu
        xx, yy = np.array(xs[mask][::-1]), np.array(ys[mask])[::-1]
        if len(xx) == 0:
            return xs[0]
    else:
        mask = xs >= mu
        xx, yy = xs[mask], ys[mask]
        if len(xx) == 0:
            return xs[-1]
    # find stationary point (inflection in the data)
    i = 0
    while i in range(len(xx) - 1):
        mean = np.mean(yy[i: i + 5])
        if mean > yy[i]:
            break
        i += 1
    # sigma defined as half way between mean and stationary point
    return xx[i]


def plot_lombscargle(xarr, yarr, eyarr, freqs, npgram, spath, sname, fdata):
    """
    Plot a 3 sub plot plot of the light curve, lomb scargle periodogram, and
    phase curve

    :param xarr: numpy array of floats, numpy array of floats, time series data
    :param yarr: numpy array of floats, flux/magnitude data
    :param eyarr: numpy array of floats, uncertainties associated with yarr
    :param freqs: numpy array of floats, frequency data
    :param npgram: numpy array of floats, lomb-scargle periodogram data
    :param spath: string, folder location to save the graph to
    :param sname: string, filename to save the graph as (including .png)
    :param fdata: dictionary, fit parameters (xmax, ymax, exmaxl, exmaxu)
    :return:
    """
    # normalise the time series (i.e. start from zero)
    xarr = xarr - xarr.min()
    # extract values from fdata
    xmax, ymax = fdata['xmax'], fdata['ymax']
    exmaxl, exmaxu = fdata['exmaxl'], fdata['exmaxu']
    # --------------------------------------------------------------------------
    # set up the figure
    plt.close()
    fig, frames = plt.subplots(ncols=1, nrows=3)
    fig.set_size_inches(16, 12)
    # --------------------------------------------------------------------------
    # plot light curve
    frames[0].errorbar(xarr, yarr, yerr=eyarr,
                       ls='none', marker='o', markersize=2)
    frames[0].set_xlabel('Days from Start of observation')
    frames[0].set_ylabel('Corrected Flux')
    # --------------------------------------------------------------------------
    #  lombscargle
    if np.sum(np.isfinite(freqs)):
        frames[1].plot(2 * np.pi / freqs, npgram, color='k')
        frames[1].set_xscale('log')
    else:
        frames[1].text(0.5, 0.5, 'No finite numbers',
                       transform=frames[1].transAxes)
    frames[1].set_xlabel('Period (days)')
    frames[1].set_ylabel('Power')
    gymin, gymax = frames[1].get_ylim()
    frames[1].vlines(xmax, gymin, gymax, color='r', linestyles='dashed',
                     label='Period')
    frames[1].hlines(ymax/2, xmax - exmaxl, xmax + exmaxu,
                     color='r', linestyles='dashed')
    # --------------------------------------------------------------------------
    # phase curve
    mv = xmax
    xfold = ((xarr - xarr[0]) - np.floor((xarr - xarr[0]) / mv) * mv) / mv

    frames[2].plot(xfold, yarr, color='b', linestyle='none', marker='+')
    frames[2].set_xlabel('Phase (Period = {0:.3f} days)'.format(mv))
    frames[2].set_ylabel('Corrected Flux')
    frames[2].set_xlim(0.0, 1.0)
    frames[2].set_ylim(yarr.mean() - 3 * yarr.std(),
                       yarr.mean() + 3 * yarr.std())
    # --------------------------------------------------------------------------
    # add title save and close
    esargs = [xmax, exmaxl, exmaxu]
    errstr = '${{{0:.3f}}}^{{+{1:.3f}}}_{{-{2:.3f}}}$'.format(*esargs)
    plt.suptitle('Max period = {0} days'.format(errstr), fontsize=30)
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(spath + sname, bbox_inches='tight')
    plt.savefig(spath + sname.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # Main code here
    # ----------------------------------------------------------------------
    # load database
    # Must have database running
    # mysql -u root -p
    print("\nConnecting to database...")
    c, conn = load_db(database, hostname, username, password)
    # ----------------------------------------------------------------------
    # find all systemids
    sids = get_list_of_objects_from_db(conn)
    # ----------------------------------------------------------------------
    # loop round the system ids and plot a graph of the lightcurve
    print("\nPlotting graphs of lightcurves...")
    for sid in tqdm(sids):
        # skip done files
        savename = 'Lightcurve_{0}.png'.format(sid)
        if savename in os.listdir(savepath) and skip_done:
            continue
        print('\n\t Running for ID = {0}...'.format(sid))
        # get data using SQL query on database
        query2 = 'SELECT * FROM swasp_sep16_tab WHERE systemid = "{0}"'
        data = pandas.read_sql_query(query2.format(sid), conn)
        x, y = np.array(data['HJD']), np.array(data['FLUX2'])
        ey = np.array(data['FLUX2_ERR'])
        # sort into order (by x)
        sortx = np.argsort(x)
        x, y, ey = x[sortx], y[sortx], ey[sortx]
        # remove infinities
        m = np.isfinite(y) * np.isfinite(x)
        xm, ym, eym = x[m], y[m], ey[m]
        # do not even try to plot if there is no data
        if (len(xm) == 1) or (len(xm) != len(ym)):
            continue
        # compute the lombscarle
        print('\n\t Computing Lomb-Scarle...')
        freqs1, npgram1 = compute_lombscargle(xm, ym)
        # fit peaks
        print('\n\t Fitting periodogram...')
        fitdata = fit_largest_peak(2 * np.pi / freqs1, npgram1)
        # plot periodogram
        print('\n\t Plotting periodogram...')
        plot_lombscargle(xm, ym, eym, freqs1, npgram1, savepath, savename,
                         fitdata)


# =============================================================================
# End of code
# =============================================================================
