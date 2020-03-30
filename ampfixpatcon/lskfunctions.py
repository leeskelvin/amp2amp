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


def tonemap(data, lo = None, hi = None, scaletype = "lin", scalepow = 0.5):
    """Map input dataset to scaled range."""
    # setup
    old_settings = np.seterr(all='ignore')
    if lo is None:
        lo = np.nanmin(data)
    if hi is None:
        hi = np.nanmax(data)
    # rescale input to (soft) range (0,1)
    data2 = (data - lo) / (hi - lo)
    # scaling functions
    if scaletype == "lin":
        maplo = 0
        maphi = 1
        data3 = data2
    elif scaletype == "log":
        maplo = np.log10(scalepow)
        maphi = np.log10(scalepow + 500)
        data3 = np.log10((data2 * 500) + scalepow)
    elif scaletype == "pow":
        maplo = 0
        maphi = 1
        data3 = data2**scalepow
    elif scaletype == "atan":
        maplo = 0
        maphi = np.arctan(5)
        data3 = np.arctan(data2 * 5)
    elif scaletype == "asinh":
        maplo = 0
        maphi = np.arcsinh(10)
        data3 = np.arcsinh(data2 * 10)
    elif scaletype == "sinh":
        maplo = 0
        maphi = np.sinh(3)
        data3 = np.sinh(data2 * 3)
    else:
        raise NameError('Unknown scaletype function applied.')
    # finish up
    out = (data3 - maplo) / (maphi - maplo)
    np.seterr(**old_settings)
    return out


def imprep(red, green=None, blue=None, 
           xcen=None, ycen=None, xdim=None, ydim=None,
           scaletype="lin", scalepow=0.5, zlo=None, zhi=None,
           scalemode=100, scalelo=None, scalehi=None, sigmaclip=0,
           colmap="rgb", colinvert=False, alpha=1,
           smoothfwhm=0, block=1):
    """Prepare image for plotting."""
    # imports
    import numpy as np
    from scipy.ndimage import gaussian_filter as gauss
    import matplotlib
    import copy
    import astropy.stats as stats
    # setup
    if blue is None:
        blue = copy.deepcopy(red)
    if green is None:
        green = (copy.deepcopy(red) + copy.deepcopy(blue)) / 2
    if xcen is None:
        xcen = red.shape[0] / 2
    if ycen is None:
        ycen = red.shape[1] / 2
    if xdim is None:
        xdim = red.shape[0]
    if ydim is None:
        ydim = red.shape[1]
    xcen = np.tile(xcen, 3)[:3]
    ycen = np.tile(ycen, 3)[:3]
    xdim = np.tile(xdim, 3)[:3]
    ydim = np.tile(ydim, 3)[:3]
    zlo = np.tile(zlo, 3)[:3]
    zhi = np.tile(zhi, 3)[:3]
    scalemode = np.tile(scalemode, 3)[:3]
    scalelo = np.tile(scalelo, 3)[:3]
    scalehi = np.tile(scalehi, 3)[:3]
    # trim image to dimensions
    dats = [np.array([]), np.array([]), np.array([])]
    for i, dat in enumerate([red, green, blue]):
        xlo = int(xcen[i] - (xdim[i] / 2))
        xhi = int(xcen[i] + (xdim[i] / 2))
        ylo = int(ycen[i] - (ydim[i] / 2))
        yhi = int(ycen[i] + (ydim[i] / 2))
        dat = dat[xlo:xhi, ylo:yhi]
        dats[i] = dat
    # zscales, smoothing and blocking
    for i in range(len(dats)):
        if block > 1:
            new_shape = tuple(np.array(dats[i].shape) / block / block)
            dats[i] = rebin(dats[i], shape=new_shape)
        if smoothfwhm > 0:
            sigma = smoothfwhm / (2 * np.sqrt(2 * np.log(2)))
            dats[i] = gauss(input=dats[i], sigma=sigma)
        scaledat = copy.deepcopy(dats[i])[~np.isnan(dats[i])]
        if sigmaclip > 0:
            scaledat = stats.sigma_clip(scaledat, sigma=sigmaclip, masked=False)
        if zlo[i] is None:
            if scalelo[i] is None:
                scalelo[i] = (50 - (scalemode[i] / 2))
            zlo[i] = np.nanquantile(scaledat, scalelo[i]/100)
        if zhi[i] is None:
            if scalehi[i] is None:
                scalehi[i] = (50 + (scalemode[i] / 2))
            zhi[i] = np.nanquantile(scaledat, scalehi[i]/100)
    # generate average map
    avgs = [np.array([]), np.array([]), np.array([])]
    notnans = [np.array([]), np.array([]), np.array([])]
    for i, dat in enumerate(dats):
        tempavg = (dat - zlo[i]) / (zhi[i] - zlo[i])
        notnans[i] = ~np.isnan(tempavg)
        tempavg[np.isnan(tempavg)] = 0
        avgs[i] = tempavg
    avg = sum(avgs) / np.clip(sum(notnans), 1, 3)
    avg[avg == 0] = np.nan
    # apply scaling function
    scaled = tonemap(data=avg, lo=0, hi=1, scaletype=scaletype, scalepow=scalepow)
    # rescaled input images
    for i, dat in enumerate(dats):
        dats[i] = np.clip((scaled * (((dats[i] - zlo[i]) / (zhi[i] - zlo[i])) / avg)), 0, 1)
        dats[i][np.isnan(dats[i])] = 0
        if colinvert:
            dats[i] = 1 - dats[i]
    # generate colour map
    if colmap == "rgb":
        alphas = np.full_like(dats[0], alpha)
        out = np.stack((dats[0], dats[1], dats[2], alphas), axis=2)
    elif colmap == "grey":
        cmap = matplotlib.cm.get_cmap("Greys_r")
        out = cmap(dats[0], alpha=alpha)
    elif colmap == "sls":
        out = sls(dats[0], alpha=alpha)
    else:
        cmap = matplotlib.cm.get_cmap(colmap)
        out = cmap(dats[0], alpha=alpha)
    return out, zlo, zhi


# sls colormap
vred = np.array((0, 0.04314, 0.08627, 0.12941, 0.17255, 0.21569, 0.25882,
                 0.30588, 0.34902, 0.39216, 0.43529, 0.47843, 0.52157, 0.50588,
                 0.49412, 0.47843, 0.46275, 0.44706, 0.43529, 0.41961, 0.40392,
                 0.39216, 0.37647, 0.36078, 0.3451, 0.31765, 0.2902, 0.25882,
                 0.23137, 0.20392, 0.17255, 0.1451, 0.11373, 0.08627, 0.05882,
                 0.02745, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0.03137, 0.06667, 0.09804, 0.13333, 0.16471, 0.2, 0.23137,
                 0.26667, 0.29804, 0.33333, 0.36471, 0.39608, 0.43137, 0.46275,
                 0.49804, 0.52941, 0.56471, 0.59608, 0.62745, 0.66275, 0.69412,
                 0.72941, 0.76078, 0.79216, 0.81176, 0.82745, 0.84314, 0.86275,
                 0.87843, 0.89804, 0.91373, 0.92941, 0.94902, 0.96471, 0.98431,
                 1, 0.99608, 0.99216, 0.98824, 0.98431, 0.98039, 0.97255,
                 0.96863, 0.96471, 0.96078, 0.95686, 0.95294, 0.94902, 0.95294,
                 0.95686, 0.96078, 0.96471, 0.96863, 0.97255, 0.98039, 0.98431,
                 0.98824, 0.99216, 0.99608, 1, 1, 0.99608, 0.99608, 0.99216,
                 0.99216, 0.98824, 0.98824, 0.98824, 0.98431, 0.98431, 0.98039,
                 0.98039, 0.95686, 0.93333, 0.90588, 0.88235, 0.85882, 0.83529,
                 0.81176, 0.78824, 0.76078, 0.73725, 0.71373, 0.6902, 0.71765,
                 0.74118, 0.76863, 0.79216, 0.81961, 0.84314, 0.87059, 0.89804,
                 0.92157, 0.94902, 0.97255, 1, 1, 1, 1, 1, 1, 1, 1))
vgreen = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01961,
                   0.03922, 0.05882, 0.07843, 0.09804, 0.11765, 0.13725,
                   0.15686, 0.17647, 0.2, 0.21961, 0.23922, 0.26667, 0.29804,
                   0.32941, 0.36078, 0.39216, 0.42353, 0.45098, 0.48235,
                   0.51373, 0.5451, 0.57647, 0.60784, 0.63137, 0.65882,
                   0.68235, 0.70588, 0.73333, 0.75686, 0.78431, 0.80784,
                   0.83137, 0.85882, 0.88235, 0.9098, 0.90196, 0.89412,
                   0.8902, 0.88235, 0.87843, 0.87059, 0.86275, 0.85882,
                   0.85098, 0.84706, 0.83922, 0.83137, 0.83137, 0.83137,
                   0.83137, 0.83137, 0.83137, 0.83137, 0.83137, 0.83137,
                   0.82745, 0.82745, 0.82745, 0.82745, 0.83529, 0.83922,
                   0.84706, 0.85098, 0.85882, 0.86667, 0.87059, 0.87843,
                   0.88235, 0.8902, 0.89412, 0.90196, 0.90196, 0.90196,
                   0.90196, 0.90196, 0.90196, 0.90196, 0.90196, 0.90196,
                   0.90196, 0.90196, 0.90196, 0.90196, 0.89804, 0.89412,
                   0.88627, 0.88235, 0.87843, 0.87451, 0.86667, 0.86275,
                   0.85882, 0.8549, 0.84706, 0.84314, 0.82353, 0.80784,
                   0.78824, 0.76863, 0.74902, 0.72941, 0.71373, 0.69412,
                   0.67451, 0.6549, 0.63922, 0.61961, 0.6, 0.58039, 0.56471,
                   0.5451, 0.52549, 0.50588, 0.4902, 0.47059, 0.45098,
                   0.43137, 0.41176, 0.39608, 0.36078, 0.32941, 0.29804,
                   0.26275, 0.23137, 0.19608, 0.16471, 0.13333, 0.09804,
                   0.06667, 0.03137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0.08235, 0.16863, 0.25098, 0.33333, 0.41569, 0.50196,
                   0.58431, 0.66667, 0.74902, 0.83137, 0.91765, 1, 1, 1, 1, 1,
                   1, 1, 1))
vblue = np.array((0, 0.05098, 0.10588, 0.15686, 0.21176, 0.26275, 0.31765,
                  0.36863, 0.42353, 0.47451, 0.52941, 0.58039, 0.63529,
                  0.63922, 0.64706, 0.65098, 0.65882, 0.66275, 0.66667,
                  0.67451, 0.67843, 0.68627, 0.6902, 0.69804, 0.70196,
                  0.71373, 0.72157, 0.73333, 0.74118, 0.75294, 0.76471,
                  0.77255, 0.78431, 0.79216, 0.80392, 0.81569, 0.82353,
                  0.83922, 0.8549, 0.86667, 0.88235, 0.89804, 0.91373,
                  0.92549, 0.94118, 0.95686, 0.97255, 0.98431, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 0.97647, 0.94902, 0.92549, 0.90196,
                  0.87843, 0.85098, 0.82745, 0.80392, 0.77647, 0.75294,
                  0.72941, 0.70196, 0.67843, 0.65098, 0.62353, 0.59608,
                  0.56863, 0.54118, 0.51373, 0.48627, 0.46275, 0.43529,
                  0.40784, 0.38039, 0.35294, 0.32941, 0.30196, 0.27843,
                  0.25098, 0.22745, 0.2, 0.17647, 0.14902, 0.12549, 0.09804,
                  0.07451, 0.06667, 0.05882, 0.0549, 0.04706, 0.04314, 0.03529,
                  0.03137, 0.02353, 0.01961, 0.01176, 0.00784, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.00784, 0.01176, 0.01569,
                  0.01961, 0.02353, 0.02745, 0.03137, 0.03529, 0.03922,
                  0.04314, 0.04706, 0.05098, 0.0549, 0.05882, 0.06275, 0.06667,
                  0.07059, 0.07451, 0.07843, 0.08235, 0.08627, 0.0902, 0.09412,
                  0.08627, 0.07843, 0.07059, 0.06275, 0.0549, 0.04706, 0.03922,
                  0.03137, 0.02353, 0.01569, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.08235,
                  0.16863, 0.25098, 0.33333, 0.41569, 0.50196, 0.58431,
                  0.66667, 0.74902, 0.83137, 0.91765, 1, 1, 1, 1, 1, 1, 1, 1))
vsls = np.stack((vred, vgreen, vblue), axis=1)
sls = matplotlib.colors.ListedColormap(vsls)
