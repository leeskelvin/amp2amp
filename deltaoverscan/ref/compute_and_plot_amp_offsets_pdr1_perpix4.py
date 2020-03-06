# lsst
# setup lsst_distrib -t w_2019_23
# setup -j -r /home/erykoff/tickets/obs_subaru

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import esutil

import lsst.afw.math as afwMath
import lsst.daf.persistence as dafPersist

ccd = 43

butlerVisit = dafPersist.Butler('/datasets/hsc/repo/rerun/private/erykoff/fgcm_pdr1_run1wd/wide_deep2')
obsTable = butlerVisit.get('fgcmVisitCatalog')

butler = dafPersist.Butler('/datasets/hsc/repo/rerun/private/erykoff/hscproc/runIsrPDR1')

camera = butler.get('camera')
det = camera[ccd]
amp1, amp2, amp3, amp4 = det.getAmpInfoCatalog()
obox1 = amp1.getRawHorizontalOverscanBBox()
obox2 = amp2.getRawHorizontalOverscanBBox()
obox3 = amp3.getRawHorizontalOverscanBBox()
obox4 = amp4.getRawHorizontalOverscanBBox()

abox1 = amp1.getRawDataBBox()
abox2 = amp2.getRawDataBBox()
abox3 = amp3.getRawDataBBox()
abox4 = amp4.getRawDataBBox()

visits = obsTable['visit']

cat = np.zeros(visits.size, dtype=[('amp1b', 'i4', (abox1.getHeight(), 5)),
                                   ('amp2a', 'i4', (abox2.getHeight(), 5)),
                                   ('amp3b', 'i4', (abox3.getHeight(), 5)),
                                   ('amp4a', 'i4', (abox4.getHeight(), 5)),
                                   ('overscan1', 'i4', (obox1.getHeight(), 2)),
                                   ('overscan2', 'i4', (obox1.getHeight(), 2)),
                                   ('overscan3', 'i4', (obox1.getHeight(), 2)),
                                   ('overscan4', 'i4', (obox1.getHeight(), 2)),
                                   ('visit', 'i4')])

cat['visit'][:] = visits

for i, v in enumerate(visits):
#for i, v in enumerate(visits[:200]):
    if (i % 50) == 0:
        print('On %d of %d' % (i, visits.size))

    try:
        raw = butler.get('raw', visit=int(visits[i]), ccd=ccd)
    except:
        continue

    overscan1 = raw.maskedImage[obox1].getArrays()[0].astype(np.int64)
    overscan2 = raw.maskedImage[obox2].getArrays()[0].astype(np.int64)
    overscan3 = raw.maskedImage[obox3].getArrays()[0].astype(np.int64)
    overscan4 = raw.maskedImage[obox4].getArrays()[0].astype(np.int64)

    data1 = raw.maskedImage[abox1].getArrays()[0].astype(np.int64)
    data2 = raw.maskedImage[abox2].getArrays()[0].astype(np.int64)
    data3 = raw.maskedImage[abox3].getArrays()[0].astype(np.int64)
    data4 = raw.maskedImage[abox4].getArrays()[0].astype(np.int64)

    cat['amp1b'][i, :] = data1[:, -5: ]
    cat['amp2a'][i, :] = data2[:, 0: 5]
    cat['amp3b'][i, :] = data3[:, -5: ]
    cat['amp4a'][i, :] = data4[:, 0: 5]

    cat['overscan1'][i, :, 0] = overscan1[:, 0]
    cat['overscan1'][i, :, 1] = overscan1[:, -1]
    cat['overscan2'][i, :, 0] = overscan2[:, -1]
    cat['overscan2'][i, :, 1] = overscan2[:, 0]
    cat['overscan3'][i, :, 0] = overscan3[:, 0]
    cat['overscan3'][i, :, 1] = overscan3[:, -1]
    cat['overscan4'][i, :, 0] = overscan4[:, -1]
    cat['overscan4'][i, :, 1] = overscan4[:, 0]

fits.writeto('test_last5pix_ccd%03d.fits' % (ccd), cat, overwrite=True)

# make 80 plots...
# 4 amps on the y axis ...
# 4 amps on the x axis...
# +0, +1, +2, +3, +4 on the x axis...

ctr = 0
for ampXAxis in [1, 2, 3, 4]:
    if ampXAxis == 1:
        ampXName = 'amp1b'
        lastPix = -1
    elif ampXAxis == 2:
        ampXName = 'amp2a'
        lastPix = 0
    elif ampXAxis == 3:
        ampXName = 'amp3b'
        lastPix = -1
    elif ampXAxis == 4:
        ampXName = 'amp4a'
        lastPix = 0

    for offset in [0, 1, 2, 3, 4]:

        if lastPix == 0:
            ampFlat = cat[ampXName][:, :, offset].flatten()
        else:
            ampFlat = cat[ampXName][:, :, -1 - offset].flatten()

        for ampOver in [1, 2, 3, 4]:
            if ampOver == 1:
                overname = 'overscan1'
            elif ampOver == 2:
                overname = 'overscan2'
            elif ampOver == 3:
                overname = 'overscan3'
            elif ampOver == 4:
                overname = 'overscan4'

            overFirst = cat[overname][:, :, 0].flatten()
            overLast = cat[overname][:, :, 1].flatten()

            clean, = np.where((ampFlat > 0) & (ampFlat < 30000) &
                              (np.abs(overFirst - overLast) < 100))

            h, rev = esutil.stat.histogram(ampFlat[clean], min=1000, max=30000, binsize=20, rev=True)

            values = np.zeros(h.size)
            deltas = np.zeros(h.size)

            ok, = np.where(h >= 50)

            for ind in ok:
                i1a = rev[rev[ind]: rev[ind + 1]]

                values[ind] = afwMath.makeStatistics(ampFlat[clean[i1a]], afwMath.MEDIAN).getValue()
                deltas[ind] = afwMath.makeStatistics(overFirst[clean[i1a]] - overLast[clean[i1a]], afwMath.MEDIAN).getValue()

            if ctr == 0:
                outCat = np.zeros(4 * 5 * 4, dtype=[('ampx', 'i4'),
                                                    ('ampy', 'i4'),
                                                    ('offset', 'i4'),
                                                    ('values', 'f4', h.size),
                                                    ('deltas', 'f4', h.size),
                                                    ('npix', 'i4', h.size)])

            outCat['ampx'][ctr] = ampXAxis
            outCat['ampy'][ctr] = ampOver
            outCat['offset'][ctr] = offset
            outCat['values'][ctr, :] = values
            outCat['deltas'][ctr, :] = deltas
            outCat['npix'][ctr, ok] = h[ok]

            use, = np.where(outCat['npix'][ctr, :] > 0)

            plt.clf()
            plt.plot(outCat['values'][ctr, use], outCat['deltas'][ctr, use], 'r.')
            plt.ylim(-6, 10)
            plt.xlim(1000, 17000)
            plt.xlabel('Amp %d raw flux value, last pixel - %d (ADU)' % (outCat['ampx'][ctr], outCat['offset'][ctr]))
            plt.ylabel('First - Last Overscan for amp %d (ADU)' % (outCat['ampy'][ctr]))
            plt.title('CCD %d, Amp %d vs Amp %d + %d' % (ccd, outCat['ampy'][ctr], outCat['ampx'][ctr], outCat['offset'][ctr]))
            plt.savefig('pdr1_ccd%03d_amp%d_vs_amp%d+%d.png' % (ccd, outCat['ampy'][ctr], outCat['ampx'][ctr], outCat['offset'][ctr]))

            ctr += 1

fits.writeto('test_last5pix_ccd%03d_summary.fits' % (ccd), outCat)

## look at various correlations

