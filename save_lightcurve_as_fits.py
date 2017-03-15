#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017-02-24 15:42

@author: Neil Cook

Program description here

Version 0.0.0
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
WORKSPACE = '/Astro/Projects/RayPaul_Work/SuperWASP/'
DPATH = WORKSPACE + '/Data/Elodie/'

# set database settings
HOSTNAME = 'localhost'
USERNAME = 'root'
PASSWORD = '1234'
DATABASE = 'swasp'
TABLE = 'swasp_sep16_tab'

SID = 'ARG_54'
SID = 'BPC_46A'


# =============================================================================
# Define functions
# =============================================================================
def get_list_of_objects_from_db(table, conn):
    # ----------------------------------------------------------------------
    # find all systemids
    print("\nGetting list of objects...")
    query1 = "SELECT CONCAT(c.systemid, c.comp)"
    query1 += " AS sid FROM {0} AS c".format(table)
    query1 += " where c.systemid is not null and c.systemid <> ''"
    rawdata = pandas.read_sql_query(query1, conn)
    rawsystemid = np.array(rawdata['sid'])
    # get list of unique ids (for selecting each as a seperate curve)
    sids = np.array(np.unique(rawsystemid), dtype=str)
    # return list of objects
    return sids


def load_db(db_name, host, uname, pword):
    conn = MySQLdb.connect(host=host, user=uname, db=db_name,
                            connect_timeout=100000, passwd=pword)
    c = conn.cursor()
    return c, conn


def get_data(sid, table, conn):
    # get data using SQL query on database
    query2 = 'SELECT * FROM {0} AS c'.format(table)
    query2 += ' WHERE CONCAT(c.systemid, c.comp) = "{0}"'
    data = pandas.read_sql_query(query2.format(sid), conn)
    return data


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
    # Main code here
    # ----------------------------------------------------------------------
    # load database
    # Must have database running
    # mysql -u root -p
    print("\nConnecting to database...")
    c1, conn1 = load_db(DATABASE, HOSTNAME, USERNAME, PASSWORD)
    # ----------------------------------------------------------------------
    # find all systemids
    do_sids = get_list_of_objects_from_db(TABLE, conn1)
    # ----------------------------------------------------------------------
    print('\n\t Running for ID = {0}...'.format(SID))
    # ---------------------------------------------------------------------
    # Get the data from SQL database
    pdata = get_data(SID, TABLE, conn1)
    # ---------------------------------------------------------------------
    # sort by HJD column
    pdata = pdata.sort_values('HJD')
    # Replace infinities with nans
    pdata = pdata.replace([np.inf, -np.inf], np.nan)
    # ---------------------------------------------------------------------
    # save file
    save_to_file(pdata, '{0}_lightcurve', DPATH, exts=['fits', 'dat' ,'csv'])


# =============================================================================
# End of code
# =============================================================================
