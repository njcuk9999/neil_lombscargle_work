#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Date here

@author: Neil Cook

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# =============================================================================
# Define variables
# =============================================================================
workspace = '/Astro/Projects/RayPaul_Work/SuperWASP/'
dpath = workspace + 'Data/fitted_lightcurve_data_regions.fits'
# -----------------------------------------------------------------------------
# Number of peaks
npeak = 5
maxregions = 3


peakformat = ['o', '^', 's', 'v', 'd']
regioncolours = dict(full='k', r0='r', r1='b', r2='g')


graphtypes = ['log', 'normal']
graphylims = [(0.01, 1000), (-1, 25)]

# =============================================================================
# Define functions
# =============================================================================



# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    print('\n Loading data...')
    data = fits.getdata(dpath, ext=1)
    # Get columns
    nid = np.array(data['nid'])
    names = np.array(data['id'])
    # Get peak data
    xpeaks = []
    expeaksl = []
    expeaksu = []
    for n in range(npeak):
        xpeaks.append(np.array(data['xmax_{0}'.format(n+1)]))
        expeaksl.append(np.array(data['exmaxl_{0}'.format(n + 1)]))
        expeaksu.append(np.array(data['exmaxu_{0}'.format(n + 1)]))

    for g in range(len(graphtypes)):
        # Plot graph
        plt.close()
        fig, frame = plt.subplots(ncols=1, nrows=1)

        # Loop round each row
        print('\n Adding points to plot...')
        for row in tqdm(range(len(nid))):
            # identify region
            region = None
            for rkey in regioncolours:
                if rkey in names[row]:
                    region = rkey
            # Loop around each peak
            for n in range(npeak):
                # construct label
                label = 'Peak {0} - {1}'.format(n+1, region.upper())
                # construct keyword arguments
                kwargs = dict(mec=regioncolours[region], mfc='None',
                              ecolor=regioncolours[region],
                              marker=peakformat[n], label=label,
                              linestyle='None', ms=5, zorder=3)
                # plot errorbars for peak
                eyarr = [[expeaksl[n][row]], [expeaksu[n][row]]]
                frame.errorbar([nid[row]], [xpeaks[n][row]], yerr=eyarr, **kwargs)

        # plot a line at each nid
        print('\n Finalising data...')
        xmin, xmax, ymin, ymax = frame.axis()
        kwargs = dict(colors='0.5', alpha=0.25, zorder=0, linewidth=0.5)
        frame.vlines(nid, ymin, ymax, **kwargs)

        # finalise graph properties
        frame.set_xlabel('Object Number')
        frame.set_ylabel('Period / days')
        if 'log' in graphtypes[g]:
            frame.set_yscale('log')
        frame.set_xlim(nid.min()-1, nid.max()+1)
        frame.set_ylim(*graphylims[g])

        # add legend outside axis
        handles, labels = frame.get_legend_handles_labels()
        sort = np.argsort(labels)
        h, l = [], []
        for it in list(sort):
            if labels[it] not in l:
                l.append(labels[it]), h.append(handles[it])
        frame.legend(h, l, loc=6, scatterpoints=1,
                     numpoints=1, bbox_to_anchor=(1.05, 0.5))
        # Save show and close
        print('\n Saving graph...')
        plt.show()
        plt.close()

# =============================================================================
# End of code
# =============================================================================
