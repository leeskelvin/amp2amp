#!/usr/bin/env python3

# setup lsst_distrib
# ds9 &

# imports
import os
import sys
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.stats as stats
from collections import Counter as count
import pandas as pd
import pickle
import seaborn as sns

# LSST stack imports
import lsst.daf.persistence as dafPersist
import lsst.afw.display as afwDisplay
from lsst.ip.isr import IsrTask
import lsst.afw.detection as afwDetection
import lsst.afw.image as afwImage

# # interactive plotting setup
# matplotlib.use('Qt5Agg')
# plt.ion()


def update_progress(job_title, progress):
    """Progress bar."""
    length = 60  # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1:
        msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()


# setup
datadir = '/datasets/hsc/repo/rerun/RC/w_2020_03/DM-23121'
outdir = '/project/lskelvin/amp2amp/ampfind'
butler = dafPersist.Butler(datadir)

# find visits
coadd = butler.get("deepCoadd_calexp", tract=9813, patch="5,5", filter="HSC-R")   # middle of cosmos
visits = coadd.getInfo().getCoaddInputs().visits['id']
exptimes = np.zeros(len(visits))
for i, vis in enumerate(visits):
    exptimes[i] = butler.queryMetadata('raw', ['exptime'], visit=int(vis))[0]
hi = exptimes == np.max(exptimes)
lo = exptimes == np.min(exptimes)
print('visits hi: ' + str(visits[hi][0]) + ' (exptime=' + str(exptimes[hi][0]) + ')')
print('visits lo: ' + str(visits[lo][0]) + ' (exptime=' + str(exptimes[lo][0]) + ')')

# visits: 1202, 23692

# pixel match
datadir2 = '/datasets/hsc/repo/rerun/private/lskelvin/isrflatonhilo/'
butler2 = dafPersist.Butler(datadir2)
xoff = 109
yoff = 147

isrlo = butler2.get('postISRCCD', visit=23692, ccd=35)
isrhi = butler2.get('postISRCCD', visit=1202, ccd=35)
lotime = butler.queryMetadata('postISRCCD', ['exptime'], visit=23692)[0]
hitime = butler.queryMetadata('postISRCCD', ['exptime'], visit=1202)[0]
timeratio = hitime / lotime

display1 = afwDisplay.getDisplay(backend='ds9', frame=1)
display2 = afwDisplay.getDisplay(backend='ds9', frame=2)
display1.mtv(isrlo)
display2.mtv(isrhi)

loimage = isrlo.image.array[:-yoff,:-xoff]
lomask = isrlo.mask.array[:-yoff,:-xoff]
hiimage = isrhi.image.array[yoff:,xoff:]
himask = isrhi.mask.array[yoff:,xoff:]

display1.mtv(afwImage.ImageF(loimage))
display2.mtv(afwImage.ImageF(hiimage))

# stats (mean, median, std)
losclip = stats.sigma_clipped_stats(loimage[1000:-1000,500:-500], sigma=3, maxiters=5)
hisclip = stats.sigma_clipped_stats(hiimage[1000:-1000,500:-500], sigma=3, maxiters=5)

lodat = loimage - losclip[1]
hidat = hiimage - hisclip[1]

display1.mtv(afwImage.ImageF(lodat))
display2.mtv(afwImage.ImageF(hidat))

good = (hidat > 10) & (lomask == 0) & (himask == 0)

nbins = 100
xlo = np.log10(1e1)
xhi = np.log10(1e4)
ylo = 0.1
yhi = 2
x = np.log10(hidat[good])
y = (lodat[good]*timeratio) / hidat[good]
data = np.column_stack((a1,a2))
k = kde.gaussian_kde(data.T)
# xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
xi, yi = np.mgrid[xlo:xhi:nbins*1j, ylo:yhi:nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.clf()
# plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
plt.contour(xi, yi, zi.reshape(xi.shape), levels=25, linewidths=3)
plt.scatter(x, y, c='blue', s=1, alpha=0.2)
plt.xlim(np.log10(1e2), np.log10(1e4))
plt.ylim(0.6, 1.4)
# plt.xscale('log')
ygrids = np.arange(0.1, 2.1, 0.05)
for ypos in ygrids:
    plt.axhline(y=ypos, color='black', linestyle='-', alpha=0.5)
plt.axhline(y=1, color='black', linestyle='-', linewidth=5)
step = 0.2
xbins = np.arange(xlo, xhi, step) + (step/2)
ymeds = []
for xbin in xbins:
    xgood = (x >= (xbin-(step/2))) & (x <= (xbin+(step/2)))
    if(len(y[xgood]) > 0):
        ymeds.append(np.median(y[xgood]))
    else:
        ymeds.append(0)
plt.scatter(xbins, ymeds, c='red', s=100, label='binned medians')
plt.legend()
plt.xlabel('log counts (360s)')
plt.ylabel('flux ratio: (1.2 * 300s) / 360s')
plt.suptitle('scaled flux ratios for visits 23692 (300s) and 1202 (360s)')
