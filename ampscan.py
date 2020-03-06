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

# LSST stack imports
import lsst.daf.persistence as dafPersist
import lsst.afw.display as afwDisplay
from lsst.ip.isr import IsrTask
import lsst.afw.detection as afwDetection

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


def bestfit(xs, ys):
    """Simple best fit line."""
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
            ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    b = np.mean(ys) - m*np.mean(xs)
    return m, b


# setup
datadir1 = '/datasets/hsc/repo/rerun/private/erykoff/hscproc/deltaoverscan/run5_vanilla'
datadir2 = '/datasets/hsc/repo/rerun/private/lskelvin/isrnoflat1252/'
datadirflatoffhilo = '/datasets/hsc/repo/rerun/private/lskelvin/isrflatoffhilo/'
datadirflatonhilo = '/datasets/hsc/repo/rerun/private/lskelvin/isrflatonhilo/'
datadir = datadirflatonhilo
outdir = '/project/lskelvin/amp2amp/ampscan'
butler = dafPersist.Butler(datadir)
# butler = dafPersist.Butler(datadir2)
stripsize = 8

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
    visccds = []
    for file in unparsed_c:
        visccds.append(int(file.strip('.fits').strip('.c')))
    ccds.append(visccds)

# loop over each visit
for visnum, vis, filt in zip(range(len(visits)), visits, filters):

    # setup
    visdir = outdir + '/v' + f'{vis:07}'
    outreadme = visdir + '/README.md'
    if not os.path.exists(visdir):
        os.makedirs(visdir)
    if os.path.exists(outreadme):
        os.remove(outreadme)
    exptime = butler.queryMetadata('raw', ['exptime'], visit=vis)[0]

    # loop over each ccd
    vfcares = pd.DataFrame()
    vfcres = pd.DataFrame()
    for ccd in ccds[visnum]:

        # get ccd data
        vfcraw = butler.get('raw', visit=vis, ccd=ccd)
        vfcisr = butler.get('postISRCCD', visit=vis, ccd=ccd)
        hdrtemp = vfcraw.getMetadata().toDict()['T_CCDTV']
        # vfcfile = os.path.abspath(datadir + '/postISRCCD/v' + str(vis).zfill(7) + '-f' + filt + '/c' + str(ccd).zfill(3) + '.fits')
        # display1 = afwDisplay.getDisplay(backend='ds9', frame=1)
        # display2 = afwDisplay.getDisplay(backend='ds9', frame=2)
        # display1.mtv(vfcraw)
        # display1.mtv(vfcisr)

        # loop over each amplifier
        update_progress("Visit %s" % vis, ccd/(len(ccds[visnum])+1))
        ccdres = pd.DataFrame()
        for amp in [0, 1, 2, 3]:

            # vertical prescan (RawPrescan)
            vpscan = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawPrescanBBox()].getArray()[-stripsize:, :]
            vpraw = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawDataBBox()].getArray()[:stripsize, :]
            vpisr = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()[:stripsize, :]

            # vertical overscan (RawVerticalOverscan)
            voscan = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawVerticalOverscanBBox()].getArray()[:stripsize, :]
            voraw = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawDataBBox()].getArray()[-stripsize:, :]
            voisr = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()[-stripsize:, :]

            # horizontal prescan (not currently in butler)
            hpylo = vfcraw.getDetector()[amp].getRawDataBBox().beginY
            hpyhi = vfcraw.getDetector()[amp].getRawDataBBox().endY
            if amp in [0, 2]:
                hpxlo = vfcraw.getDetector()[amp].getRawDataBBox().beginX - stripsize
                hpxhi = vfcraw.getDetector()[amp].getRawDataBBox().beginX
                hpscan = vfcraw.getImage().getArray()[hpylo:hpyhi, hpxlo:hpxhi]
                hpraw = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawDataBBox()].getArray()[:, :stripsize]
                hpisr = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()[:, :stripsize]
            else:
                hpxlo = vfcraw.getDetector()[amp].getRawDataBBox().endX
                hpxhi = vfcraw.getDetector()[amp].getRawDataBBox().endX + stripsize
                hpscan = vfcraw.getImage().getArray()[hpylo:hpyhi, hpxlo:hpxhi]
                hpraw = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawDataBBox()].getArray()[:, -stripsize:]
                hpisr = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()[:, -stripsize:]

            # horizontal overscan (RawHorizontalOverscan)
            if amp in [0, 2]:
                hoscan = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawHorizontalOverscanBBox()].getArray()[:, :stripsize]
                horaw = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawDataBBox()].getArray()[:, -stripsize:]
                hoisr = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()[:, -stripsize:]
            else:
                hoscan = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawHorizontalOverscanBBox()].getArray()[:, -stripsize:]
                horaw = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawDataBBox()].getArray()[:, :stripsize]
                hoisr = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()[:, :stripsize]

            # global BB data
            allraw = vfcraw.getImage()[vfcraw.getDetector()[amp].getRawDataBBox()].getArray()
            allisr = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()

            # extract mask BA9876543210
            bitmask = int('101111101111', 2)
            # 0:BAD, 1:SAT, 2:INTRP, 3:CR, 4:EDGE, 5:DETECTED,
            # 6:DETECTED_NEGATIVE, 7:SUSPECT, 8:NO_DATA, 9:CROSSTALK,
            # 10:NOT_BLENDED, 11:UNMASKEDNAN
            vpmask = vfcisr.getMask()[vfcisr.getDetector()[amp].getBBox()].getArray()[:stripsize, :]
            vomask = vfcisr.getMask()[vfcisr.getDetector()[amp].getBBox()].getArray()[-stripsize:, :]
            if amp in [0, 2]:
                hpmask = vfcisr.getMask()[vfcisr.getDetector()[amp].getBBox()].getArray()[:, :stripsize]
                homask = vfcisr.getMask()[vfcisr.getDetector()[amp].getBBox()].getArray()[:, -stripsize:]
            else:
                hpmask = vfcisr.getMask()[vfcisr.getDetector()[amp].getBBox()].getArray()[:, -stripsize:]
                homask = vfcisr.getMask()[vfcisr.getDetector()[amp].getBBox()].getArray()[:, :stripsize]
            allmask = vfcisr.getMask()[vfcisr.getDetector()[amp].getBBox()].getArray()
            vpgood = (vpmask & bitmask) == 0
            vogood = (vomask & bitmask) == 0
            hpgood = (hpmask & bitmask) == 0
            hogood = (homask & bitmask) == 0
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

            # ancilliary info
            btlrgain = vfcraw.getDetector().getAmplifiers()[amp].getGain()
            btlrreadnoise = vfcraw.getDetector().getAmplifiers()[amp].getReadNoise()
            hdrgain = vfcraw.getMetadata().toDict()['T_GAIN' + str(amp+1)]

            # pixel offset from corner
            xv, yv = np.meshgrid(range(allgood.shape[1]), range(allgood.shape[0]))
            xyv = np.sqrt((xv**2) + (yv**2))

            # best fit
            sig = vfcisr.getImage()[vfcisr.getDetector()[amp].getBBox()].getArray()[allgood]
            var = vfcisr.getVariance()[vfcisr.getDetector()[amp].getBBox()].getArray()[allgood]
            xoff = xv[allgood]
            yoff = yv[allgood]
            xyoff = xyv[allgood]
            fitgain, fitintercept = bestfit(var, sig)

            # analysis plots
            outfile = visdir + '/v' + f'{vis:07}' + f'c{ccd:03}' + f'a{amp+1}' + '.png'
            x = np.linspace(np.min(var), np.max(var), 100)
            yb = btlrgain * (x - (btlrreadnoise**2))
            yf = fitgain * (x - (btlrreadnoise**2))
            yp = btlrgain * (var.flatten() - (btlrreadnoise**2))
            dsig = sig.flatten() - yp
            fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            fig.suptitle('visit ' + str(vis) + ', ccd ' + str(ccd) + ', amp ' + str(amp+1))
            ax1.set_title(str(filt) + ', ' + str(exptime) + 's, ' + str(hdrtemp) + 'C', fontsize=10)
            ax1.set_ylabel('signal (counts)')
            ax2.set_xlabel('variance (counts²)')
            ax2.set_ylabel('Δsignal (counts)')
            ax1.set_xscale('log'); ax1.set_yscale('log')
            ax1.scatter(var.flatten(), sig.flatten(), marker='.', c=xyoff)
            ax1.plot(x, yb, color='black', linestyle='--', label='butler gain = '+str(btlrgain))
            ax1.plot(x, yf, color='red', linestyle='--', label='assumed gain = '+str(fitgain))
            ax1.legend(loc='upper left')
            ax2.scatter(var.flatten(), dsig.flatten(), marker='.', c=xyoff)
            ax2.plot(x, yb - yb, color='black', linestyle='--', label='butler gain = '+str(btlrgain))
            ax2.plot(x, yf - yb, color='red', linestyle='--', label='assumed gain = '+str(fitgain))
            # fig.savefig(outfile)

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
        vfcares = vfcares.append(ccdres)

        # combine amp results into vfc results
        comccdres = pd.DataFrame(([(vis, filt, ccd) + (
                                   tuple(ccdres[ccdres.amp == 1].hpisrmedian) +
                                   tuple(ccdres[ccdres.amp == 1].allisrmedian) +
                                   tuple(ccdres[ccdres.amp == 1].hoisrmedian) +
                                   tuple(ccdres[ccdres.amp == 2].hoisrmedian) +
                                   tuple(ccdres[ccdres.amp == 2].allisrmedian) +
                                   tuple(ccdres[ccdres.amp == 2].hpisrmedian) +
                                   tuple(ccdres[ccdres.amp == 3].hpisrmedian) +
                                   tuple(ccdres[ccdres.amp == 3].allisrmedian) +
                                   tuple(ccdres[ccdres.amp == 3].hoisrmedian) +
                                   tuple(ccdres[ccdres.amp == 4].hoisrmedian) +
                                   tuple(ccdres[ccdres.amp == 4].allisrmedian) +
                                   tuple(ccdres[ccdres.amp == 4].hpisrmedian) +
                                   tuple(ccdres[ccdres.amp == 1].hpisrstd) +
                                   tuple(ccdres[ccdres.amp == 1].allisrstd) +
                                   tuple(ccdres[ccdres.amp == 1].hoisrstd) +
                                   tuple(ccdres[ccdres.amp == 2].hoisrstd) +
                                   tuple(ccdres[ccdres.amp == 2].allisrstd) +
                                   tuple(ccdres[ccdres.amp == 2].hpisrstd) +
                                   tuple(ccdres[ccdres.amp == 3].hpisrstd) +
                                   tuple(ccdres[ccdres.amp == 3].allisrstd) +
                                   tuple(ccdres[ccdres.amp == 3].hoisrstd) +
                                   tuple(ccdres[ccdres.amp == 4].hoisrstd) +
                                   tuple(ccdres[ccdres.amp == 4].allisrstd) +
                                   tuple(ccdres[ccdres.amp == 4].hpisrstd)
                                   )]),
                                 columns = ['visit', 'filter', 'ccd',
                                            'amp1a', 'amp1c', 'amp1b',
                                            'amp2a', 'amp2c', 'amp2b',
                                            'amp3a', 'amp3c', 'amp3b',
                                            'amp4a', 'amp4c', 'amp4b',
                                            'amp1astd', 'amp1cstd', 'amp1bstd',
                                            'amp2astd', 'amp2cstd', 'amp2bstd',
                                            'amp3astd', 'amp3cstd', 'amp3bstd',
                                            'amp4astd', 'amp4cstd', 'amp4bstd'
                                            ])
        vfcres = vfcres.append(comccdres)

    # finish up
    update_progress("Visit %s" % vis, 1)

    # pickle
    outres = [vfcres, vfcares]
    outname = visdir + '/v' + f'{vis:07}'  + '.pickle'
    outfile = open(outname, 'wb')
    pickle.dump(outres, file = outfile)
    outfile.close()
