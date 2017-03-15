#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/02/17 at 11:28 AM

@author: neil

Program description here

Version 0.0.1
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import MySQLdb
import pandas
from collections import OrderedDict
from astropy.table import Table
from Rectangle_Selector import Select_Rectange

# =============================================================================
# Define variables
# =============================================================================
# set file paths
workspace = '/Astro/Projects/RayPaul_Work/SuperWASP/'
savepath = workspace + '/Plots/fitted_lightcurves/'
dsavepath = workspace + '/Data/selected_lightcurve_data.fits'
# set database settings
hostname = 'localhost'
username = 'root'
password = '1234'
database_name = 'swasp'
table = 'swasp_sep16_tab'
# -----------------------------------------------------------------------------
# if plots exist assume that we don't need to reprocess
skip_done = True
# write to database (else this is a test run)
write_to_database = False
# number of peaks to fit
npeak = 5
# maximum times to look for a peak before continuing
max_it_per_peak = 10000
# minimum width in pixels of peak
minimum_width_of_peak = 10


# =============================================================================
# Define functions
# =============================================================================
def load_db(db_name, host, uname, pword):
    """
    Load the SQL database

    :param db_name: string, database name
    :param host: string, host
    :param uname: string, username
    :param pword: string, password
    :return:
    """
    conn1 = MySQLdb.connect(host=host, user=uname, db=db_name,
                            connect_timeout=100000, passwd=pword)
    c1 = conn1.cursor()
    return c1, conn1


def load_pre_existing_data(dpath):
    """
    Load pre-exisiting data

    :param dpath: string, filepath to the pre-existing data
    :return:
    """
    print("\n Loading pre existing files...")
    atable = Table.read(dpath)
    done_ids = list(atable['SID'])
    max_s = int(0.5*(len(atable.colnames)-1))
    sarr, earr = OrderedDict(), OrderedDict()
    for s, s_id in tqdm(enumerate(done_ids)):
        for r in range(max_s):
            start = list(atable['start{0}'.format(r+1)])[s]
            end = list(atable['end{0}'.format(r+1)])[s]
            sarr, earr = sort_data_in_dict(s_id, sarr, earr, start, end)
    del atable
    # ---------------------------------------------------------------------
    # skip sids if the are in table (assume this means they are done)
    do_ids, skips = [], 0
    for s_id in sids:
        if s_id in done_ids:
            skips += 1
        else:
            do_ids.append(s_id)
    # Print statement for how many files skipped due to pre existing data
    print('\n Skipping {0} sids'.format(skips))
    return sarr, earr, max_s, do_ids, done_ids


def sort_data_in_dict(s_id, sarr, earr, start, end):
    """
    Sort the data based on the id and the start and end points (into the sarr
    and earr dictionaries

    :param s_id: string, ID of this object (will be used as key)
    :param sarr: dict, start point list (list of time series data for each
                 segments i.e. if there are two segments list would be
                 [x pos start of segment 1, x pos start of segment 2]
    :param earr: dict, end point list (list of time series data for each
                 segments i.e. if there are two segments list would be
                 [x pos end of segment 1, x pos end of segment 2]
    :param start: float, the x starting position of this segment
    :param end: float, the x ending position of this segment
    :return:
    """
    if s_id not in sarr:
        sarr[s_id], earr[s_id] = [start], [end]
    else:
        sarr[s_id].append(start), earr[s_id].append(end)
    return sarr, earr


def write_to_table(done_ids, max_s, sarr, earr):
    """
    Writes the start and end dictionaries to the table (overwrites pre-existing
    data, which should be loaded into the memory at the start of this run)

    :param done_ids: list of strings, the ID of those objects already processed
                     either in this run or from pre-existing data
    :param max_s: int, the maximum number of segments for any object
    :param sarr: dict, start point list (list of time series data for each
                 segments i.e. if there are two segments list would be
                 [x pos start of segment 1, x pos start of segment 2]
    :param earr: dict, end point list (list of time series data for each
                 segments i.e. if there are two segments list would be
                 [x pos end of segment 1, x pos end of segment 2]
    :return:
    """
    arraystarts = np.zeros((len(done_ids), max_s))
    arrayends = np.zeros((len(done_ids), max_s))
    for s in tqdm(range(len(done_ids))):
        s_id = done_ids[s]
        for row1 in range(max_s):
            if s_id not in sarr:
                arraystarts[s][row1] = np.nan
                arrayends[s][row1] = np.nan
            elif len(sarr[s_id]) <= row1:
                arraystarts[s][row1] = np.nan
                arrayends[s][row1] = np.nan
            else:
                arraystarts[s][row1] = sarr[s_id][row1]
                arrayends[s][row1] = earr[s_id][row1]

    newtable = Table()
    newtable['SID'] = np.array(done_ids, dtype=str)
    for row1 in range(max_s):
        newtable['start{0}'.format(row1 + 1)] = arraystarts[:, row1]
        newtable['end{0}'.format(row1 + 1)] = arrayends[:, row1]
    newtable.write(dsavepath, overwrite=True)


def plot_periodogram(s_id, x, y, ey=None):
    """
    Plots a periodogram with a rectangular selection interface, returns the
    object containing the chosen rectangular regions

    :param s_id: string, ID of this object
    :param x: numpy array, the time series data
    :param y: numpy array, the flux data
    :param ey: numpy array or None, the uncertainties on the flux data
    :return selection: instance of select_rectangle contains the data for the
                       selected rectangle regions in selection.data
                       in the form of a list of lists:

                       [[xA0, xA1, yA0, yA1], [xB0, xB1, yB0, yB1],  ...]

                        where A and B are two selected rectangles

                        and x0, y0 is the first selected corner of the rectangle
                            x1, y1 is the last selected corner of the rectangle
    """
    plt.close()
    # make days since first observation
    xarr = x - x.min()
    # Do plot
    fig, frame = plt.subplots(ncols=1, nrows=1)
    frame.errorbar(xarr, y, yerr=ey, ls='none', marker='o', markersize=2)
    frame.set_xlabel('Days from Start of observation')
    frame.set_ylabel('Flux')
    frame.set_title('Raw light curve of {0}'.format(s_id), fontsize=22)
    selection = Select_Rectange(frame)
    plt.show()
    plt.close()
    return selection


def print_to_screen(x, startx, endx):
    xarr = x - x.min()
    print('\n\tFull: {0} to {1}'.format(xarr.min(), xarr.max()))
    for it in range(len(startx)):
        args = it + 1, startx[it], endx[it]
        print('\n\tRegion {0}: {1} to {2}'.format(*args))

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # Main code here
    # -------------------------------------------------------------------------
    # load database
    # Must have database running
    # mysql -u root -p
    print("\nConnecting to database...")
    c, conn = load_db(database_name, hostname, username, password)
    # -------------------------------------------------------------------------
    # find all systemids
    print("\nGetting list of objects...")
    query1 = 'SELECT systemid FROM swasp_sep16_tab '
    query1 += " where systemid is not null and systemid <> ''"
    rawsystemids = pandas.read_sql_query(query1, conn)
    # get list of unique ids (for selecting each as a seperate curve)
    sids = np.unique(rawsystemids)
    # load file if it exists (to save on repeating on exit)
    if os.path.exists(dsavepath) and skip_done and write_to_database:
        ldata = load_pre_existing_data(dsavepath)
        starts, ends, maxsegments, do_sids, done_sids = ldata
    else:
        do_sids, done_sids, maxsegments = list(sids), [], 0
        starts, ends = OrderedDict(), OrderedDict()
    # -------------------------------------------------------------------------
    # loop round the system ids and plot a graph of the lightcurve
    print("\nPlotting graphs of lightcurves...")
    for sid in tqdm(do_sids):
        print('\n\t Running for ID:\n\n\t{0}'.format(sid))
        # get data using SQL query on database
        query2 = 'SELECT * FROM swasp_sep16_tab WHERE systemid = "{0}"'
        data = pandas.read_sql_query(query2.format(sid), conn)
        time, flux = np.array(data['HJD']), np.array(data['FLUX2'])
        eflux = np.array(data['FLUX2_ERR'])
        # sort into order (by x)
        sorttime = np.argsort(time)
        time, flux, eflux = time[sorttime], flux[sorttime], eflux[sorttime]
        # plot and show periodogram
        a = plot_periodogram(sid, time, flux, eflux)
        # Deal with data acquisition
        if maxsegments < len(a.data):
            maxsegments = len(a.data)
        for row in a.data:
            starts, ends = sort_data_in_dict(sid, starts, ends, row[0], row[1])
        done_sids.append(sid)
        # ----------------------------------------------------------------------
        # convert into fits table and save
        if write_to_database:
            print('\n Converting in to fits table...')
            write_to_table(done_sids, maxsegments, starts, ends)
        else:
            print('\n Selected region:')
            print_to_screen(time, starts[sid], ends[sid])

# =============================================================================
# End of code
# =============================================================================



