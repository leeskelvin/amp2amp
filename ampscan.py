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


# setup
datadir = '/datasets/hsc/repo/rerun/private/erykoff/hscproc/deltaoverscan/run5_vanilla'
outdir = '/project/lskelvin/amp2amp/ampscan'
butler = dafPersist.Butler(datadir)

# parse folder structure
# visits = butler.queryMetadata('raw', ['visit'])
# print(len(visits))
unparsed_vf = os.listdir(datadir + '/postISRCCD')
visits = []
filters = []
ccds = []
for i,folder in enumerate(unparsed_vf):
    visits.append(int(folder.split('-fHSC')[0].split('v')[1].strip('0')))
    filters.append(folder.split('-f')[1])
    unparsed_c = sorted(os.listdir(datadir + '/postISRCCD/' + folder))
    ccds.append([])
    for file in unparsed_c:
        ccds[i-1].append(int(file.strip('.fits').strip('.c')))

# loop over each visit
for visnum, vis, filt in zip(range(len(visits)), visits, filters):

    # loop over each ccd
    visres = pd.DataFrame()
    for ccd in ccds[visnum]:

        # get amp data
        vfcraw = butler.get('raw', visit=vis, ccd=ccd)
        vfcisr = butler.get('postISRCCD', visit=vis, ccd=ccd)
        # vfcfile = os.path.abspath(datadir + '/postISRCCD/v' + str(vis).zfill(7) + '-f' + filt + '/c' + str(ccd).zfill(3) + '.fits')
        # display = afwDisplay.getDisplay(backend='ds9')
        # display.mtv(vfcraw)
        # display.mtv(vfcisr)

        # loop over each amplifier
        update_progress("Visit %s" % vis, ccd/(len(ccds[visnum])+1))
        ccdres = pd.DataFrame()
        for amp in [0, 1, 2, 3]:

            # setup
            inset1 = 25
            inset2 = -inset1

            # horizontal prescan coords
            hpylo = vfcraw.getDetector()[amp].getRawDataBBox().beginY
            hpyhi = vfcraw.getDetector()[amp].getRawDataBBox().endY
            # choose flip
            if amp in [0, 2]:
                hpx = vfcraw.getDetector()[amp].getRawDataBBox().beginX - 1
                edge1 = -1
                edge2 = 0
            else:
                hpx = vfcraw.getDetector()[amp].getRawDataBBox().endX
                edge1 = 0
                edge2 = -1

            # extract image rows/columns
            vpscan = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawPrescanBBox()].getArray()[-1, :]
            vpraw = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawDataBBox()].getArray()[(0+inset1), :]
            vpisr = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()[(0+inset1), :]
            voscan = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawVerticalOverscanBBox()].getArray()[0, :]
            voraw = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawDataBBox()].getArray()[(-1+inset2), :]
            voisr = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()[(-1+inset2), :]
            hpscan = vfcraw.getImage().getArray()[hpylo:hpyhi, hpx]
            hpraw = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawDataBBox()].getArray()[:, (edge2+inset2)]
            hpisr = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()[:, (edge2+inset2)]
            hoscan = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawHorizontalOverscanBBox()].getArray()[:, edge2]
            horaw = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawDataBBox()].getArray()[:, (edge1+inset1)]
            hoisr = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()[:, (edge1+inset1)]
            allraw = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawDataBBox()].getArray()
            allisr = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()

            # extract mask BA9876543210
            bitmask = int('101111101111', 2)
            # 0:BAD, 1:SAT, 2:INTRP, 3:CR, 4:EDGE, 5:DETECTED,
            # 6:DETECTED_NEGATIVE, 7:SUSPECT, 8:NO_DATA, 9:CROSSTALK,
            # 10:NOT_BLENDED, 11:UNMASKEDNAN
            vpmask = vfcisr.getMask()[vfcisr.getDetector()[amp].getBBox()].getArray()[(0+inset1), :]
            vomask = vfcisr.getMask()[vfcisr.getDetector()[amp].getBBox()].getArray()[(-1+inset2), :]
            hpmask = vfcisr.getMask()[vfcisr.getDetector()[amp].getBBox()].getArray()[:, (edge2+inset2)]
            homask = vfcisr.getMask()[vfcisr.getDetector()[amp].getBBox()].getArray()[:, (edge1+inset1)]
            vpgood = (vpmask & bitmask) == 0
            vogood = (vomask & bitmask) == 0
            hpgood = (hpmask & bitmask) == 0
            hogood = (homask & bitmask) == 0
            allmask = vfcisr.getMask()[vfcisr.getDetector()[amp].getBBox()].getArray()
            allgood = (allmask & bitmask) == 0

            # stats (mean, median, std)
            sigma = 3
            maxiters = 5
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                scvpscan = stats.sigma_clipped_stats(vpscan, sigma=sigma, maxiters=maxiters)
                scvpraw = stats.sigma_clipped_stats(vpraw[vpgood], sigma=sigma, maxiters=maxiters)
                scvpisr = stats.sigma_clipped_stats(vpisr[vpgood], sigma=sigma, maxiters=maxiters)
                scvoscan = stats.sigma_clipped_stats(voscan, sigma=sigma, maxiters=maxiters)
                scvoraw = stats.sigma_clipped_stats(voraw[vogood], sigma=sigma, maxiters=maxiters)
                scvoisr = stats.sigma_clipped_stats(voisr[vogood], sigma=sigma, maxiters=maxiters)
                schpscan = stats.sigma_clipped_stats(hpscan, sigma=sigma, maxiters=maxiters)
                schpraw = stats.sigma_clipped_stats(hpraw[hpgood], sigma=sigma, maxiters=maxiters)
                schpisr = stats.sigma_clipped_stats(hpisr[hpgood], sigma=sigma, maxiters=maxiters)
                schoscan = stats.sigma_clipped_stats(hoscan, sigma=sigma, maxiters=maxiters)
                schoraw = stats.sigma_clipped_stats(horaw[hogood], sigma=sigma, maxiters=maxiters)
                schoisr = stats.sigma_clipped_stats(hoisr[hogood], sigma=sigma, maxiters=maxiters)
                scallraw = stats.sigma_clipped_stats(allraw[allgood], sigma=sigma, maxiters=maxiters)
                scallisr = stats.sigma_clipped_stats(allisr[allgood], sigma=sigma, maxiters=maxiters)

            # return results
            ampres = pd.DataFrame(([(vis, filt, ccd, (amp+1)) +
                                   (scvpscan + scvpraw + scvpisr +
                                    scvoscan + scvoraw + scvoisr +
                                    schpscan + schpraw + schpisr +
                                    schoscan + schoraw + schoisr +
                                    scallraw + scallisr)]),
                                  columns = ['visit', 'filter', 'ccd', 'amp',
                                             'vpscanmean', 'vpscanmedian', 'vpscanstd',
                                             'vprawmean', 'vprawmedian', 'vprawstd',
                                             'vpisrmean', 'vpisrmedian', 'vpisrstd',
                                             'voscanmean', 'voscanmedian', 'voscanstd',
                                             'vorawmean', 'vorawmedian', 'vorawstd',
                                             'voisrmean', 'voisrmedian', 'voisrstd',
                                             'hpscanmean', 'hpscanmedian', 'hpscanstd',
                                             'hprawmean', 'hprawmedian', 'hprawstd',
                                             'hpisrmean', 'hpisrmedian', 'hpisrstd',
                                             'hoscanmean', 'hoscanmedian', 'hoscanstd',
                                             'horawmean', 'horawmedian', 'horawstd',
                                             'hoisrmean', 'hoisrmedian', 'hoisrstd',
                                             'allrawmean', 'allrawmedian', 'allrawstd',
                                             'allisrmean', 'allisrmedian', 'allisrstd'
                                             ])
            ccdres = ccdres.append(ampres)

        # return results
        visres = visres.append(ccdres)

    # finish up
    update_progress("Visit %s" % vis, 1)

    # pickle
    outfile = outdir + '/v' + f'{vis:07}' + '.pickle'
    visres.to_pickle(outfile)
