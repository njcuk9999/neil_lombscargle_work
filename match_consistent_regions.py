#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/03/17

@author: Neil Cook

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# =============================================================================
# Define variables
# =============================================================================
# Define paths
WS = '/Astro/Projects/RayPaul_Work/SuperWASP'
# loading data path
DPATH = WS + '/Data/'
DFILE = DPATH + 'fitted_lightcurve_data_regions.fits'
# saving data path
SFILE = DPATH + 'matched_periods_data_regions_match_{0}.fits'
PLOTPATH = WS + '/Plots/match_consistent_regions/'
# -----------------------------------------------------------------------------
# define a threshold fraction (i.e. 0.1 = 10%) to match periods
THRESHOLD = 0.05


# =============================================================================
# Define functions
# =============================================================================
def add_periods_to_groups(d, gm, num):
    periods, regions, peaks = [], [], []
    for npeak in range(num):
        col = 'xmax_{0}'.format(npeak + 1)
        periods = np.append(periods, d[col][gm])
        regions = np.append(regions, names[gm])
        peaks = np.append(peaks, np.repeat([npeak + 1], len(gmask[[gmask]])))
    return periods, regions, peaks


def find_combs_within_thres(gnum, periods, regions, peaks):
    # integer array for defining row number
    itperiods = np.array(range(len(periods)), dtype=int)
    # define number of unique regions a group has
    rnum = len(np.unique(regions))
    # storage containers
    gnums, rnums = [], []
    keepperiods1, keepregions1, keeppeaks1 = [], [], []
    keepperiods2, keepregions2, keeppeaks2 = [], [], []

    for c1 in range(len(periods)):
        itmask = (itperiods != c1) & (regions != regions[c1])
        keepmask = np.repeat([True], len(periods[itmask]))
        keepmask &= (periods[itmask] > periods[c1] * (1 - THRESHOLD))
        keepmask &= (periods[itmask] < periods[c1] * (1 + THRESHOLD))
        knum = len(keepmask[keepmask])
        # append keeps
        gnums = np.append(gnums, np.repeat([gnum], knum))
        rnums = np.append(rnums, np.repeat([rnum], knum))
        keepperiods1 = np.append(keepperiods1, np.repeat([periods[c1]], knum))
        keepregions1 = np.append(keepregions1, np.repeat([regions[c1]], knum))
        keeppeaks1 = np.append(keeppeaks1, np.repeat([peaks[c1]], knum))
        keepperiods2 = np.append(keepperiods2, periods[itmask][keepmask])
        keepregions2 = np.append(keepregions2, regions[itmask][keepmask])
        keeppeaks2 = np.append(keeppeaks2, peaks[itmask][keepmask])
    # Store all in a list for zipping return (in correct format)
    keeplist = []
    keeplist += [np.array(gnums, dtype=int)]
    keeplist += [np.array(rnums, dtype=int)]
    keeplist += [np.array(keepregions1, dtype=str)]
    keeplist += [np.array(keeppeaks1, dtype=int)]
    keeplist += [np.array(keepperiods1, dtype=float)]
    keeplist += [np.array(keepregions2, dtype=str)]
    keeplist += [np.array(keeppeaks2, dtype=int)]
    keeplist += [np.array(keepperiods2, dtype=float)]
    return list(zip(*keeplist))


def save_to_table(mlist):
    allmatches = np.array(mlist)
    table = Table()
    colnames = ['Group_id', 'N_regions', 'Main_id', 'Main_peak', 'Main_period',
                'matched_id', 'matched_peak', 'matched_period']
    colformats = [int, int, str, int, float, str, int, float]
    for c, col in enumerate(colnames):
        table[col] = np.array(allmatches[:, c], dtype=colformats[c])
    table.write(SFILE.format(THRESHOLD), format='fits', overwrite=True)
    return table


def plot_mean_diff_plot(table, gg, gmask):
    # -------------------------------------------------------------------------
    # sort out custom colormap
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = range(1, npeaks + 2)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # -------------------------------------------------------------------------
    # sort out data
    x = table['Main_period'][gmask]
    y = table['matched_period'][gmask]
    z = table['Main_peak'][gmask]
    mean, diff = 0.5 * (x + y), abs(x - y)
    ymin = 10 ** np.floor(np.log10(min(diff)))
    ymax = 10 ** np.ceil(np.log10(max(diff)))
    # -------------------------------------------------------------------------
    # plot the scatter plot
    plt.close()
    fig, frame = plt.subplots(ncols=1, nrows=1)
    fig.set_size_inches(8, 6)
    frame.set_title(group)
    cax = frame.scatter(mean, diff, c=z, cmap=cmap, norm=norm,
                        marker='x', s=100)
    kwargs = dict(cmap=cmap, norm=norm, spacing='proportional',
                  ticks=bounds, boundaries=bounds, format='%1i')
    cbar = fig.colorbar(cax, **kwargs)
    zticks = cbar.ax.get_yticks()
    zticks = zticks[:-1] + 0.5*(zticks[1] + zticks[0])
    zticklabels = np.array(bounds[:-1], dtype=str)
    cbar.ax.get_yaxis().set_ticks(zticks)
    cbar.ax.get_yaxis().set_ticklabels(zticklabels)

    cbar.ax.set_ylabel('Peak Number')

    frame.grid(b=True, which='major', color='0.5', linestyle='-', alpha=0.5)
    frame.grid(b=True, which='minor', color='0.25', linestyle='--', alpha=0.25)
    plt.minorticks_on()

    frame.set_yscale('log')
    frame.set_xscale('log')
    frame.set_xlabel('Mean period / days')
    frame.set_ylabel(r'$\Delta$(Period) between matches')
    frame.set_ylim(ymin, ymax)

    sname = 'Scatter_plot_{0}'.format(gg)
    plt.savefig(PLOTPATH + sname + '.png', bbox_inches='tight')
    plt.savefig(PLOTPATH + sname + '.pdf', bbox_inches='tight')
    plt.close()

# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Load data
    print('\n Loading data...')
    data = fits.getdata(DFILE, ext=1)
    # Get columns
    names = np.array(data['id'])
    cols = list(data.columns.names)
    # Find number of peaks
    npeaks = len(list(filter(lambda x: 'xmax_' in x, cols)))
    # ----------------------------------------------------------------------
    # Find the object names and group
    print('\n Finding all groups...')
    groups, groupnums, nids, nid = dict(), dict(), [], 0
    for name in names:
        # define the group name
        group = '{0}_{1}'.format(*name.split('_'))
        # define the group number
        if group not in groups:
            nid += 1
        # add all elements in group to group dictionaries
        groups[group] = list(filter(lambda x: group + '_' in x, names))
        groupnums[group] = nid
    # ----------------------------------------------------------------------
    # Loop around each object
    print('\n Finding all combinations in each group that match in period...')
    allmatches, numregions = [], []
    for group in tqdm(groups):
        # create a mask for this group
        gmask = np.array(np.in1d(names, groups[group]))
        # loop around peaks and add all periods in group to list
        pargs = add_periods_to_groups(data, gmask, npeaks)
        # get the number of regions for each group
        numregions.append(len(np.unique(pargs[1])))
        # Find any combinations where two periods lie within threshold
        kargs = [groupnums[group]] + list(pargs)
        klist = find_combs_within_thres(*kargs)
        # append klist
        for ki in range(len(klist)):
            allmatches.append(klist[ki])
    # ----------------------------------------------------------------------
    # Save as table
    print('\n Saving to table...')
    mtable = save_to_table(allmatches)
    # ----------------------------------------------------------------------
    # Stats
    groupid = np.sort(np.array(mtable['Group_id']))
    bins = range(0, len(groups)+1)
    hist, edges = np.histogram(groupid, bins=bins)
    nhist = hist/np.array(numregions, dtype=float)
    # target those that have matches
    # flagged_targets = []
    # flagged_periods = []
    print('\n Plotting stat graphs...')
    for group in tqdm(groups):
        if groupnums[group] in edges[1:][nhist > 0]:
            # flagged_targets.append(group)
            groupmask = groupid == groupnums[group]
            if np.sum(groupmask) == 0:
                continue
            plot_mean_diff_plot(mtable, group, groupmask)
            # periodpairs = zip(mtable['Main_period'][groupmask],
            #                   mtable['matched_period'][groupmask])
            # peakpairs = zip(mtable['Main_period'][groupmask],
            #                   mtable['matched_period'][groupmask])
            # regionpairs = zip(mtable['Main_id'][groupmask],
            #                   mtable['matched_id'][groupmask])


# =============================================================================
# End of code
# =============================================================================
