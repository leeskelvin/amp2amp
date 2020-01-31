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
    """progress bar"""
    length = 60  # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1:
        msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()


def rebin(arr, shape):
    """rebin 2D array"""
    if shape != arr.shape:
        new_shape = (int(shape[0]), arr.shape[0] // int(shape[0]),
                int(shape[1]), arr.shape[1] // int(shape[1]))
        return arr.reshape(new_shape).mean(-1).mean(1)
    else:
        return arr


def imprep(data, lo = None, hi = None, scaleLo = None, scaleHi = None, scale = 100, block = 1):
    """prepare image for plotting"""
    maplo = np.log10(0.5)
    maphi = np.log10(500 + 0.5)
    if lo is None:
        if scaleLo is None:
            scaleLo = (100 - scale) / 2
        lo = np.quantile(data, scaleLo / 100)
    if hi is None:
        if scaleHi is None:
            scaleHi = 100 - (100 - scale) / 2
        hi = np.quantile(data, scaleHi / 100)
    temp = (data - lo) / (hi - lo)
    temp = np.clip(temp, a_min = 0, a_max = 1)
    temp *= 500
    temp = np.log10(temp + 0.5)
    temp = (temp - maplo) / (maphi - maplo)
    new_shape = tuple(np.array(temp.shape) / block / block)
    temp = rebin(temp, shape=new_shape)
    return temp, lo, hi


# setup
visit = 1252
datadir = '/datasets/hsc/repo/rerun/private/erykoff/hscproc/deltaoverscan/run5_vanilla'
pickledir = '/project/lskelvin/amp2amp/ampscan'
outdir = '/project/lskelvin/amp2amp/ampanal'
butler = dafPersist.Butler(datadir)

# pickled data
picklefile = pickledir + '/v' + f'{visit:07}' + '.pickle'
allres = pd.read_pickle(picklefile)

# construct per-ccd data
visres = pd.DataFrame()
visfix = pd.DataFrame()
for ccd in np.unique(allres[allres.visit == visit].ccd):

    # progress
    update_progress("Visit %s" % visit, ccd / (len(np.unique(allres[allres.visit == visit].ccd)) + 1))

    # combine ccd amp data
    ccdres = allres[(allres.visit == visit) & (allres.ccd == ccd)]
    subres = pd.DataFrame([(ccd,
                           ccdres[ccdres.amp == 1].hpisrmedian.item(),
                           ccdres[ccdres.amp == 1].allisrmedian.item(),
                           ccdres[ccdres.amp == 1].hoisrmedian.item(),
                           ccdres[ccdres.amp == 2].hoisrmedian.item(),
                           ccdres[ccdres.amp == 2].allisrmedian.item(),
                           ccdres[ccdres.amp == 2].hpisrmedian.item(),
                           ccdres[ccdres.amp == 3].hpisrmedian.item(),
                           ccdres[ccdres.amp == 3].allisrmedian.item(),
                           ccdres[ccdres.amp == 3].hoisrmedian.item(),
                           ccdres[ccdres.amp == 4].hoisrmedian.item(),
                           ccdres[ccdres.amp == 4].allisrmedian.item(),
                           ccdres[ccdres.amp == 4].hpisrmedian.item()
                           )],
                          columns = ['ccd',
                                     'hp1', 'all1', 'ho1',
                                     'ho2', 'all2', 'hp2',
                                     'hp3', 'all3', 'ho3',
                                     'ho4', 'all4', 'hp4'
                                     ])
    visres = visres.append(subres)

    # generate per-ccd fix
    original = butler.get('postISRCCD', visit=int(visit), ccd=int(ccd))
    fixa = original.image.array.copy()
    fixb = original.image.array.copy()
    ped1a = 0
    ped2a = ped1a + (subres.all1 - subres.all2)
    if np.isnan(ped2a.item()):
        ped2a = 0
    ped3a = ped2a + (subres.all2 - subres.all3)
    if np.isnan(ped3a.item()):
        ped3a = ped2a
    ped4a = ped3a + (subres.all3 - subres.all4)
    if np.isnan(ped4a.item()):
        ped2a = ped3a
    ped1b = 0
    ped2b = ped1b + (subres.ho1 - subres.ho2)
    ped3b = ped2b + (subres.hp2 - subres.hp3)
    ped4b = ped3b + (subres.ho3 - subres.ho4)
    fixa[0:4176, 0:512] = ped1a
    fixa[0:4176, 512:1024] = ped2a
    fixa[0:4176, 1024:1536] = ped3a
    fixa[0:4176, 1536:2048] = ped4a
    fixb[0:4176, 0:512] = ped1b
    fixb[0:4176, 512:1024] = ped2b
    fixb[0:4176, 1024:1536] = ped3b
    fixb[0:4176, 1536:2048] = ped4b
    # display1 = afwDisplay.getDisplay(backend='ds9', frame=1)
    # display2 = afwDisplay.getDisplay(backend='ds9', frame=2)
    # display3 = afwDisplay.getDisplay(backend='ds9', frame=3)
    # pistonsa = afwImage.ImageF(fixa)
    # pistonsb = afwImage.ImageF(fixb)
    # repaireda = afwImage.ImageF(original.image.array + fixa)
    # repairedb = afwImage.ImageF(original.image.array + fixb)
    # display1.mtv(original)
    # display2.mtv(repaireda)
    # display3.mtv(repairedb)

    # generate image
    fontsize = 8
    fig = plt.figure(figsize=(5,3), dpi=150)
    gs = fig.add_gridspec(1,7)
    ax1 = fig.add_subplot(gs[0, 0:2], frameon=True)
    ax2 = fig.add_subplot(gs[0, 2:4], frameon=True)
    ax3 = fig.add_subplot(gs[0, 4:6], frameon=True)
    ax4 = fig.add_subplot(gs[0, 5:], frameon=True)
    ax1.axis('off'); ax1.set_title('CCD %s, vanilla' % ccd, fontsize=fontsize)
    ax2.axis('off'); ax2.set_title('CCD %s, with pistons' % ccd, fontsize=fontsize)
    ax3.axis('off'); ax3.set_title('pistons', fontsize=fontsize)
    ax4.axis('off')
    orig,lo,hi = imprep(original.image.array, scale=75, block=4)
    rep,u,v = imprep(original.image.array + fixa, scale=75, block=4)
    ax1.imshow(np.flip(orig,0), cmap="Greys_r")
    ax2.imshow(np.flip(rep,0), cmap="Greys_r")
    cbim = ax3.imshow(np.flip(fixa,0), cmap=plt.cm.get_cmap('RdYlBu', 13), vmin=-3.25, vmax=3.25)
    cb = fig.colorbar(cbim, ax=ax4, aspect=10)
    cb.ax.tick_params(labelsize = 8)
    outfile = outdir + '/v' + f'{visit:07}' + f'c{ccd:03}' + '.png'
    plt.savefig(outfile)

# finish up
update_progress("Visit %s" % visit, 1)
