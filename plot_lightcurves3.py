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
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
from tqdm import tqdm
import MySQLdb
import pandas
from astropy.stats import LombScargle
from collections import OrderedDict
from astropy.table import Table
from astropy.io import fits


# =============================================================================
# Define variables
# =============================================================================
# set file paths
workspace = '/Astro/Projects/RayPaul_Work/SuperWASP/'
savepath = workspace + '/Plots/fitted_lightcurves_regions/'
dsavepath = workspace + '/Data/fitted_lightcurve_data_regions.fits'
rdatapath = workspace + '/Data/selected_lightcurve_data.fits'
# set database settings
hostname = 'localhost'
username = 'root'
password = '1234'
database = 'swasp'
table = 'swasp_sep16_tab'
# -----------------------------------------------------------------------------
# if plots exist assume that we don't need to reprocess
skip_done = True
# number of peaks to fit
npeak = 5
# maximum times to look for a peak before continuing
max_it_per_peak = 10000
# minimum width in pixels of peak
minimum_width_of_peak = 10


# =============================================================================
# Define functions
# =============================================================================
def find_sys_ids(conn):
    # load database
    # Must have database running
    # mysql -u root -p
    print("\nConnecting to database...")
    c, conn = load_db(database, hostname, username, password)
    # ----------------------------------------------------------------------
    # find all systemids
    sids = get_list_of_objects_from_db(conn)
    # load file if it exists (to save on repeating on exit)
    ddict, skips = OrderedDict(), 0
    if os.path.exists(dsavepath):
        atable = Table.read(dsavepath)
        for col in list(atable.colnames):
            ddict[col] = list(atable[col])
        del atable
        # ----------------------------------------------------------------------
        # skip sids if the are in the plot folder (assume this means they are done)
        do_sids = []
        for sid in sids:
            savename = 'Lightcurve_{0}_full.png'.format(sid)
            cond1 = savename in os.listdir(savepath)
            cond2 = sid in ddict['id']
            if cond1 and cond2 and skip_done:
                skips += 1
            else:
                do_sids.append(sid)
    else:
        do_sids = list(sids)
    print('\n Skipping {0} sids'.format(skips))
    return do_sids, ddict


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


def get_data(sid):
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
    return xm, ym, eym


def compute_lombscargle(xs, ys, eys):
    """
    Compute the lombscarlge periodogram from time data and flux/magnitude data
    :param xs: numpy array of floats, time series data
    :param ys: numpy array of floats, flux/magnitude data
    :return:
    """
    # compute the Lomb-Scargle periodogram
    freqs, pgram = LombScargle(xs, ys, eys).autopower()
    # npgram = np.sqrt(4 * (pgram / normval))
    return freqs, pgram


def fit_largest_peak(xs, ys, num):
    """
    Algorithm to "fit" the largest peak

    :param xs: numpy array of floats, time series data
    :param ys: numpy array of floats, power data
    :param num: integer, number of peaks to fit
    :return:
    """
    sort = np.argsort(xs)
    xs, ys = xs[sort], ys[sort]
    # if we have nans or infinites we cannot fit
    if np.sum(np.isfinite(xs)) != len(xs) or np.sum(np.isfinite(ys)) != len(ys):
        print('\n\t\t\t\t Warning nans present - cannot fit, skipping')
        flag = np.zeros(len(xs))

        xmax, ymax = np.repeat(-999.0, num), np.repeat(-999.0, num)
        exmaxl, exmaxu = np.repeat(-999.0, num), np.repeat(-999.0, num)
        return dict(xmax=xmax, ymax=ymax, exmaxl=exmaxl, exmaxu=exmaxu,
                    flag=flag)
    # set up new array to flag peaks (if element is False it means it has not
    # been flagged as part of a peak
    flag = np.zeros(len(xs))
    # loop round num of peaks to find, exclude data in those peaks in next fit
    xmax, ymax, exmaxl, exmaxu = [], [], [], []
    for n in range(num):
        print('\n\t\t\t\t Peak {0}'.format(n + 1))
        # find peak
        xmaxt, ymaxt = find_peaks(xs, ys, flag, num)
        # append maxima to lists
        xmax.append(xmaxt)
        ymax.append(ymaxt)
        # if we couldn't find peak set uncertainties to -999.0 and skip rest
        if xmaxt == -999.0 or ymaxt == -999.0:
            exmaxl.append(-999.0), exmaxu.append(-999.0)
            continue
        # based error on the first stationary point after the maximum point
        exmaxl.append(xmax[n] - find_s_point(xs, ys, 'l', xmax[n]))
        exmaxu.append(find_s_point(xs, ys, 'u', xmax[n]) - xmax[n])
        # flag up all data in this region with positive integer tag
        peak = (xs >= (xmax[n] - exmaxl[n])) & (xs <= (xmax[n] + exmaxu[n]))
        flag[peak] = n+1
    # inverse is true
    flag = flag[::-1]

    return dict(xmax=xmax, ymax=ymax, exmaxl=exmaxl, exmaxu=exmaxu, flag=flag)


def find_peaks(xs, ys, flag, num, dpix=None, maxit=None):
    """
    Finds a maximum peak
    :param xs: numpy array of floats, time series data
    :param ys: numpy array of floats, power data

    :param flag: numpy array of ints, length of xs, flags up which peak
                 each pixel belongs to (0 means a pixel belongs to no peak
                 and is the default value)
    :param num: integer, number of peaks to fit
    :param dpix: minimum width of the peak to be allowed as a peak
    :param maxit: maximum pixels to look at before giving up
    :return:
    """
    if dpix is None:
        dpix = minimum_width_of_peak
    if maxit is None:
        maxit = max_it_per_peak
    # work out distance away from peak required
    hw = int(dpix / 2.0)
    # set up while variables
    cond, its, bad_pixels = True, 0, np.zeros(len(xs), dtype=bool)
    # set initial values to bad peak values
    xmaxt, ymaxt = -999.0, -999.0
    while cond:
        # target only unflagged regions
        fm = (flag == 0) & (~bad_pixels)
        # if there are no unfladded regions return bad peak values
        if len(ys[fm]) == 0:
            return -999.0, -999.0
        # find highest y point in the unflagged data (store x and y)
        argmax = np.argmax(ys[fm])
        xmaxt = xs[fm][argmax]
        ymaxt = ys[fm][argmax]
        # find this location in the full data set
        argmaxf = np.where((xs == xmaxt) & (ys == ymaxt))[0][0]
        # maximum point must have at least 5 pixels either side (to rule out
        # edges or artifacts created from peak removal, but if we go over
        # maximum iterations stop
        psum = np.sum(~fm[argmaxf - hw: argmaxf + hw])
        # if sum is zero it means there are no used pixels within hw
        # pixels of the found peak
        if psum == 0:
            cond = False
        else:
            bad_pixels[argmaxf - hw: argmaxf + hw] = True
        if its > maxit:
            print('\n\t\t\t Warning max iterations exceed. Peak not fit.')
            cond = False
        its += 1
    return xmaxt, ymaxt


def find_s_point(xs, ys, k, mu):
    """
    Find the stationary point around a certain value (mu)
    :param xs: numpy array of floats, time series data
    :param ys: numpy array of floats, power data
    :param k: string, 'lower' or 'upper' depending which side of the mu it is
    :param mu: float, the value to look for stationary points around
    :return:
    """
    # mask data to only part interested in and flip if lower
    if 'l' in k:
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


def getfitfold(t, y, maxperiod):
    tfit = np.linspace(0, t.max(), 1000)
    yfit = LombScargle(t, y).model(tfit, 1.0/maxperiod)
    tfitfold = (tfit / maxperiod) % 1
    fsort = np.argsort(tfitfold)
    return tfitfold[fsort], yfit[fsort]


def plot_lombscargle(xall, yall, eyall, xarr, yarr, eyarr, freqs, npgram,
                     spath, sname, fdata):
    """
    Plot a 3 sub plot plot of the light curve, lomb scargle periodogram, and
    phase curve

    :param xall: numpy array of floats, numpy array of floats, time series data
    :param yall: numpy array of floats, flux/magnitude data
    :param eyall: numpy array of floats, uncertainties associated with yarr
    :param xarr: numpy array of floats, numpy array of floats, time series data
    :param yarr: numpy array of floats, flux/magnitude data
    :param eyarr: numpy array of floats, uncertainties associated with yarr
    :param freqs: numpy array of floats, frequency data
    :param npgram: numpy array of floats, lomb-scargle periodogram data
    :param spath: string, folder location to save the graph to
    :param sname: string, filename to save the graph as (including .png)
    :param fdata: dictionary, fit parameters (xmax, ymax, exmaxl, exmaxu, flag)
    :return:
    """
    # --------------------------------------------------------------------------
    # extract values from fdata
    xmaxt, ymaxt = fdata['xmax'], fdata['ymax']
    exmaxlt, exmaxut = fdata['exmaxl'], fdata['exmaxu']
    flag = fdata['flag']
    num = len(xmaxt)
    # adjust fit data to remove error values
    xmax, ymax, exmaxl, exmaxu = [], [], [], []
    for n in range(num):
        if xmaxt[n] != -999.0 and ymaxt[n] != -999.0:
            xmax.append(xmaxt[n]), ymax.append(ymaxt[n])
            exmaxl.append(exmaxlt[n]), exmaxu.append(exmaxut[n])
    num = len(xmax)

    # --------------------------------------------------------------------------
    # set the colours of the peaks
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=-1, vmax=num)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    colours = []
    for n in range(num):
        colours.append(scalarMap.to_rgba(n))
    # --------------------------------------------------------------------------
    # set up the figure
    plt.close()
    fig, frames = plt.subplots(ncols=1, nrows=3+num)
    fig.set_size_inches(16, 6*(num+2))
    # --------------------------------------------------------------------------
    # plot light curve
    frames[0].errorbar(xall, yall, yerr=eyall,
                       ls='none', marker='o', color='k', markersize=2)
    # plot light curve
    frames[0].errorbar(xarr, yarr, yerr=eyarr,
                       ls='none', marker='o', color='r', markersize=2)
    frames[0].set_xlabel('Days from Start of observation')
    frames[0].set_ylabel('Flux')
    frames[0].set_title('Raw light curve', fontsize=22)
    # plot light curve
    frames[1].errorbar(xarr, yarr, yerr=eyarr,
                       ls='none', marker='o', markersize=2)
    frames[1].set_xlabel('Days from Start of observation')
    frames[1].set_ylabel('Flux')
    frames[1].set_title('Raw light curve', fontsize=22)
    # --------------------------------------------------------------------------
    #  lombscargle
    if np.sum(np.isfinite(freqs)):
        frames[2].plot(1.0 / freqs, npgram, color='k', zorder=1)
        # plot the found peaks in colour
        for n in range(num):
            fm = flag == n+1
            frames[2].plot(1.0 / freqs[fm], npgram[fm], color=colours[n],
                           zorder=2, linewidth=1.5)
        ymin, ymax = frames[2].get_ylim()
        for n in range(num):
            frames[2].vlines(xmax[n], ymin, ymax, color='0.5',
                             linewidth=1, linestyles='dotted')

    else:
        frames[2].text(0.5, 0.5, 'No finite numbers',
                       transform=frames[2].transAxes)


    frames[2].set_xscale('log')
    frames[2].set_xlabel('Period (days)')
    frames[2].set_ylabel('Power')
    frames[2].set_title('Lomb Scargle periodogram', fontsize=22)

    # --------------------------------------------------------------------------
    # phase curve
    for n in range(num):
        # fold the xdata at given period
        xfold = (xarr/xmax[n]) % 1
        # commute the lomb-scargle model at given period
        t_fit, y_fit = getfitfold(xarr, yarr, xmax[n])

        frames[3+n].plot(xfold, yarr, color=colours[n], linestyle='none',
                         marker='+')
        frames[3+n].plot(t_fit, y_fit, color='k')
        frames[3+n].set_xlabel('Phase (Period = {0:.3f} days)'.format(xmax[n]))
        frames[3+n].set_ylabel('Flux')
        frames[3+n].set_xlim(0.0, 1.0)
        frames[3+n].set_ylim(np.median(yarr) - 3 * yarr.std(),
                             np.median(yarr) + 3 * yarr.std())

        esargs = [xmax[n], exmaxl[n], exmaxu[n]]
        errstr = '${{{0:.3f}}}^{{+{1:.3f}}}_{{-{2:.3f}}}$'.format(*esargs)
        pre = 'Phase-folded periogram, period = '
        frames[3 + n].set_title('{0} {1} days'.format(pre, errstr),
                                fontsize=20, color=colours[n], y=1.05)

    # --------------------------------------------------------------------------
    # add title
    # title = ['Max periods = ']
    # tcolours = ['k'] + colours + ['k']
    # for n in range(num):
    #     ext = '\n' if (n % 5 == 0 and n != 0) else ''
    #     esargs = [xmax[n], exmaxl[n], exmaxu[n], ext]
    #     errstr = '${{{0:.3f}}}^{{+{1:.3f}}}_{{-{2:.3f}}}$ {3}'.format(*esargs)
    #     title.append(errstr)
    # title += [' days']
    #
    # rtext(0.9, 0.5, title, tcolours, fontsize=20)

    # save and close
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(spath + sname, bbox_inches='tight')
    plt.savefig(spath + sname.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


def organsize_data(sname, fdata, num, start, end):
    """
    Take the fit data and flatten it ready for absorption into the final
    table

    :param sname: unique identification name for each object
    :param fdata: dictionary, fit parameters (xmax, ymax, exmaxl, exmaxu, flag)
    :param num: integer, number of peaks to fit
    :return:
    """

    cols = ['xmax', 'ymax', 'exmaxl', 'exmaxu']

    row = OrderedDict()
    row['id'] = sname
    row['start'] = start
    row['end'] = end
    # loop round each peak (1 to num + 1)
    for n in range(num):
        # loop round each column (xmax, ymax, exmaxl, exmaxu)
        for colt in cols:
            # make new column name based on peak and column
            col = '{0}_{1}'.format(colt, n+1)
            # get value
            value = fdata[colt][n]
            # if value is -999.0 set to nan
            if value == -999.0:
                value = np.nan
            # set the row col value
            row[col] = value

    return row


def run_fit_process(sname, xa, ya, eya, xm, ym, eym, npeak, savepath,
                    ddict, dsavepath, start, end):
    # compute the lombscarle
    print('\n\t\t\t Computing Lomb-Scarle...')
    freqs1, npgram1 = compute_lombscargle(xm, ym, eym)
    # fit peaks
    print('\n\t\t\t Fitting periodogram...')
    fitdata = fit_largest_peak(1.0 / freqs1, npgram1, npeak)
    # plot periodogram
    print('\n\t\t\t Plotting periodogram...')
    savename = 'Lightcurve_{0}.png'.format(sname)
    plot_lombscargle(xa, ya, eya, xm, ym, eym, freqs1, npgram1,
                     savepath, savename, fitdata)
    # save periodgram data to file
    print('\n\t\t\t Organising fit data for saving')
    rowdict = organsize_data(sname, fitdata, npeak, start, end)
    # all rows will have the same columns
    for col in rowdict.keys():
        if col not in ddict:
            ddict[col] = [rowdict[col]]
        else:
            ddict[col].append(rowdict[col])
    # create an astropy table from ddict
    atable = Table()
    for col in list(rowdict.keys()):
        atable[col] = ddict[col]
    print('\n\t\t Saving fit data table to file...')
    atable.write(dsavepath, overwrite=True)


def post_processing(dsavepath):
    # load fit data
    print('\n Running post processing...')
    data = Table.read(dsavepath)
    ids = np.array(data['id'])
    # Add a numerical ID number and mask columns
    it = 9000000
    used, nid, regions = dict(), [], []
    for rid in ids:
        sid = rid.replace(' ', '')
        group = sid.split('_')[0] + '_' + sid.split('_')[1]
        if group not in used:
            it += 1
        used[group] = it
        nid.append(it)
        if 'full' in rid:
            regions.append(0)
        else:
            try:
                regions.append(int(rid.strip()[-1]) + 1)
            except ValueError:
                regions.append(-1)

    data['nid'] = nid
    data['regions'] = regions
    # Add masks for each region
    for r in range(1, max(regions)+1):
        data['region{0}'.format(r)] = np.array(regions) == r

    data.write(dsavepath, overwrite=True)


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
    do_sids, ddict = find_sys_ids(conn)
    # ----------------------------------------------------------------------
    # Load selected regions
    regions = fits.getdata(rdatapath, ext=1)
    maxregions = int(0.5*(len(regions.columns)-1))
    # ----------------------------------------------------------------------
    # loop round the system ids and plot a graph of the lightcurve
    print("\nPlotting graphs of lightcurves...")
    rowdict = dict()
    for sid in tqdm(do_sids):
        print('\n\t Running for ID = {0}...'.format(sid))
        # ---------------------------------------------------------------------
        # Get the data from SQL database
        xm, ym, eym = get_data(sid)
        pos = np.where(regions['SID'] == sid)[0][0]
        # do not even try to plot if there is no data
        if (len(xm) == 1) or (len(xm) != len(ym)):
            continue
        # ---------------------------------------------------------------------
        # normalise the time series (i.e. start from zero)
        xm = xm - xm.min()
        # make sure time series data is sorted
        sort = np.argsort(xm)
        xm, ym = np.array(xm[sort]), np.array(ym[sort])
        # ---------------------------------------------------------------------

        # run for region in regions
        for r in range(maxregions):
            start = float(regions['start{0}'.format(r+1)][pos])
            end = float(regions['end{0}'.format(r+1)][pos])
            if np.isfinite(start) and np.isfinite(end):
                print('\n\t\t {0}'.format('=' * 50))
                print('\n\t\t {0}) region {1}...'.format(r+2, r+1))
                print('\n\t\t {0}'.format('=' * 50))
                mask = ((xm-xm.min()) >= start) & ((xm-xm.min()) <= end)
                sname = sid + '_r{0}'.format(r)
                run_fit_process(sname, xm, ym, eym, xm[mask], ym[mask],
                                eym[mask], npeak, savepath, ddict, dsavepath,
                                start, end)
        # ---------------------------------------------------------------------
        # run for full region
        print('\n\t\t {0}'.format('=' * 50))
        print('\n\t\t 1) full region...')
        print('\n\t\t {0}'.format('=' * 50))
        sname, start, end = sid + '_full', xm.min(), xm.max()
        run_fit_process(sname, xm, ym, eym, xm, ym, eym, npeak, savepath,
                        ddict, dsavepath, start, end)
        # input('Press enter to continue. Ctrl+C to cancel')

# -----------------------------------------------------------------------------
# As a post processing add numerical id columns and some flags
post_processing(dsavepath)

# =============================================================================
# End of code
# =============================================================================
