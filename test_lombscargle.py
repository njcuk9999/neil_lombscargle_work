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
from astropy.stats import LombScargle

# =============================================================================
# Define variables
# =============================================================================

# -----------------------------------------------------------------------------


# =============================================================================
# Define functions
# =============================================================================
def getfit(t, y, maxperiod):
    tfit = np.linspace(0, t.max(), 1000)
    yfit = LombScargle(t, y).model(tfit, 1.0/maxperiod)
    tfitfold = (tfit/maxperiod) % 1
    sort = np.argsort(tfitfold)
    return tfitfold[sort], yfit[sort]

# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    rand = np.random.RandomState(42)
    t = 100 * rand.rand(1000)
    y = 10*np.sin(2 * np.pi * 0.33*t) + 5*np.sin(2*np.pi*0.1*t) + rand.randn(1000)

    F = LombScargle(t, y)
    frequency, power = F.autopower()
    maxperiod = 1.0/frequency[np.argmax(power)]



    fig, frames = plt.subplots(ncols=1, nrows=3)
    frames[0].scatter(t, y)
    frames[1].plot(1.0/frequency, power)

    tfold = (t/maxperiod) % 1

    t_fit, y_fit = getfit(t, y, maxperiod)

    frames[2].scatter(tfold, y)
    frames[2].plot(t_fit, y_fit, color='r')
    plt.show()
    plt.close()


# =============================================================================
# End of code
# =============================================================================
