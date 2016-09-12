from __future__ import division, print_function

import numpy
from numpy.fft import fft2, ifft2
from numpy import log

import scipy.ndimage.interpolation as ndii

import scipy.ndimage.filters as scifil

#__version__ = '2013.01.18'
#__docformat__ = 'restructuredtext en'
#__all__ = ['translationSimple', 'similarity']

import matplotlib.pyplot as plt

def showTwoImages(img1, shift1, img2, shift2, txt):
#  fig = plt.figure(); plt.imshow(img1, origin='lower')
#  fig = plt.figure(); plt.imshow(img2, origin='lower')
  fig = plt.figure()
  fig.suptitle(txt)
  l1 = shift1[1]
  r1 = img1.shape[1]+shift1[1]
  b1 = shift1[0]
  t1 = img1.shape[0]+shift1[0]
  l2 = shift2[1]
  r2 = img2.shape[1]+shift2[1]
  b2 = shift2[0]
  t2 = img2.shape[0]+shift2[0]
  plt.imshow(img1, extent=(l1, r1, b1, t1), origin='lower')
  plt.imshow(img2, extent=(l2, r2, b2, t2), origin='lower', alpha=0.7)
  ax = fig.gca()
  ax.set_xlim([min(l1, l2), max(r1, r2)])
  ax.set_ylim([min(b1, b2), max(t1, t2)])

def zeropad2(x, shap):
    m, n = x.shape
    p, q = shap
    assert p > m
    assert q > n
    tb = numpy.zeros(((p - m) / 2, n))
    lr = numpy.zeros((p, (q - n) / 2))
    x = numpy.append(tb, x, axis = 0)
    x = numpy.append(x, tb, axis = 0)
    x = numpy.append(lr, x, axis = 1)
    x = numpy.append(x, lr, axis = 1)
    return x

#using a function to find peaks with the same parameters as in Stephan Preibisch's
#findPeaks() function in Stitching2D.java (stitiching plugin for imageJ/Fiji,
# https://github.com/fiji/Stitching )
def findPeaks(matrix, numPeaks):
  #computer maxima over the 8-neighborhood, wraping for edges (our matrices are fourier transforms, so that's the thing to do)
  maxbool    = matrix==scifil.maximum_filter(matrix, size=(3,3), mode='wrap')
  values     = matrix[maxbool]
  rows, cols = numpy.nonzero(maxbool)
  #order the peaks
  indexes    = numpy.argsort(values)
#  z=numpy.column_stack((rows[indexes], cols[indexes]))
  #get the $numPeaks highest peaks
  indexes    = indexes[-min(numPeaks, values.size):]
  #put the highest peaks in decreasing order
  indexes    = indexes[::-1]
  rows       = rows[indexes]
  cols       = cols[indexes]
  values     = values[indexes]
  return rows, cols, values

#shift is applied to img2 w.r.t. img1
def getAlignedSubmatrices(img1, img2, shft):
  if shft[0]>=0:
    selrowinit1 = shft[0]
    selrowinit2 = 0
    selrowend1  = img1.shape[0]
    selrowend2  = img2.shape[0]-shft[0]
  else:
    selrowinit1 = 0
    selrowinit2 = -shft[0]
    selrowend1  = img1.shape[0]+shft[0]
    selrowend2  = img2.shape[0]
  if shft[1]>=0:
    selcolinit1=shft[1]
    selcolinit2 = 0
    selcolend1  = img1.shape[1]
    selcolend2  = img2.shape[1]-shft[1]
  else:
    selcolinit1 = 0
    selcolinit2 = -shft[1]
    selcolend1  = img1.shape[1]+shft[1]
    selcolend2  = img2.shape[1]
  return img1[selrowinit1:selrowend1, selcolinit1:selcolend1], img2[selrowinit2:selrowend2, selcolinit2:selcolend2]

#adapted from openPIV: https://github.com/OpenPIV/openpiv-python/blob/master/openpiv/pyprocess.py
#but, instead of refining over the naive algorithm used in openPIV, use the position
#we have computed previously
def find_subpixel_peak_position( img, default_peak_position, subpixel_method = 'gaussian'):
    # the peak locations
    peak1_i, peak1_j = default_peak_position
    
    try:
        # the peak and its neighbours: left, right, down, up
#        c = img[peak1_i,   peak1_j]
#        cl = img[peak1_i-1, peak1_j]
#        cr = img[peak1_i+1, peak1_j]
#        cd = img[peak1_i,   peak1_j-1] 
#        cu = img[peak1_i,   peak1_j+1]
        
        c = img[peak1_i,   peak1_j]
        cl = img[(peak1_i-1)%img.shape[0], peak1_j]
        cr = img[(peak1_i+1)%img.shape[0], peak1_j]
        cd = img[peak1_i,   (peak1_j-1)%img.shape[1]] 
        cu = img[peak1_i,   (peak1_j+1)%img.shape[1]]
        
        # gaussian fit
        if numpy.any ( numpy.array([c,cl,cr,cd,cu]) < 0 ) and subpixel_method == 'gaussian':
            subpixel_method = 'centroid'
        
        try: 
            if subpixel_method == 'centroid':
                subp_peak_position = (((peak1_i-1)*cl+peak1_i*c+(peak1_i+1)*cr)/(cl+c+cr),
                                    ((peak1_j-1)*cd+peak1_j*c+(peak1_j+1)*cu)/(cd+c+cu))
        
            elif subpixel_method == 'gaussian':
                subp_peak_position = (peak1_i + ( (log(cl)-log(cr) )/( 2*log(cl) - 4*log(c) + 2*log(cr) )),
                                    peak1_j + ( (log(cd)-log(cu) )/( 2*log(cd) - 4*log(c) + 2*log(cu) ))) 
        
            elif subpixel_method == 'parabolic':
                subp_peak_position = (peak1_i +  (cl-cr)/(2*cl-4*c+2*cr),
                                        peak1_j +  (cd-cu)/(2*cd-4*c+2*cu)) 
    
        except: 
            subp_peak_position = default_peak_position
            
    except IndexError:
            subp_peak_position = default_peak_position
            
    return subp_peak_position

#test the cross-correlation (adapted from testCrossCorrelation() in
#Stitching2D.java (stitiching plugin for imageJ/Fiji, https://github.com/fiji/Stitching )
def testCrossCorrelation(img1, img2, shft, minratio):
  sub1, sub2 = getAlignedSubmatrices(img1, img2, shft)
  if sub1.size==0: #non-overlapping
    return -numpy.inf
  if sub1.size/float(img1.size)<minratio: #not enough overlap
    return -numpy.inf
#  if shft[1]<-200:
#    showTwoImages(sub1, [0,0], sub2, [0,0], '')
  dist1      = sub1-sub1.mean()
  dist2      = sub2-sub2.mean()
  covar      = (dist1*dist2).mean()
  std1       = numpy.sqrt((dist1**2).mean())
  std2       = numpy.sqrt((dist2**2).mean())
  if (std1 == 0) or (std2 == 0):
    corrcoef = 0
    #sqdiff   = n.inf
  else:
    corrcoef = covar / (std1*std2)
#  print ('testCrossCorrelation '+str(shft)+': '+str(corrcoef))
  if numpy.isnan(corrcoef):
    covar=covar
    #sqdiff   = ((sub1-sub2)**2).mean()
  return corrcoef#, sqdiff  
  
def bestShift(img1, img2, shifts, minratio):
  corrcoefs = [testCrossCorrelation(img1, img2, shft, minratio) for shft in shifts]
#  for s, c in zip(shifts, corrcoefs):
#    if (s[1]<-450) and (c>0):#if c>0.6:
#      showTwoImages(img1, [0,0], img2, s, str(s)+": "+str(c))
#  x=numpy.column_stack((corrcoefs, numpy.array(shifts), shifts[:,1]<-2400))
#  indexes    = numpy.argsort(corrcoefs)
#  indexes    = indexes[::-1]
#  xx=numpy.nonzero(numpy.logical_and(shifts[1]<-2500, shifts[1]>-2700))
  if len(shifts)==0:
    raise ValueError('Very strange, no peaks detected!')
  if len(corrcoefs)==0:
    raise ValueError('Very strange, no peaks detected (bis)!')
  idx = numpy.argmax(corrcoefs)
  return idx, corrcoefs


def translationSimple(im0, im1, subpixel=False):
    """Return translation vector to register images."""
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    #ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    lens0 = abs(f0)
    lens1 = abs(f0)
    ff0=f0/lens0
    ff1=f1/lens1
    ir = ifft2((ff0 * ff1.conjugate()))
    ir = abs(ifft2)
    zz= (abs(ff0) * abs(ff1))
    ir = ir / zz
    t0, t1 = numpy.unravel_index(numpy.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    result = (t0, t1)
    if subpixel:
      result = find_subpixel_peak_position(ir, result)
    return numpy.array(result)

import register_images as imr

def translationTestPeaks(im0, im1, numPeaks=20, refinement=True,subpixel=False, scaleSubPixel=None, minratio=0.01):
    """Return translation vector to register images."""
#    im0 = scifil.laplace(im0)
#    im1 = scifil.laplace(im1)
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
#    lens0 = abs(f0)
#    lens1 = abs(f0)
#    ff0=f0/lens0
#    ff1=f1/lens1
#    ir = ifft2((ff0 * ff1.conjugate()))
#    ir = abs(ir)
##    zz= (abs(ff0) * abs(ff1))
##    ir = ir / zz
    rows, cols, values = findPeaks(ir, numPeaks)
    rows[rows>(shape[0] // 2)] -= shape[0]
    cols[cols>(shape[1] // 2)] -= shape[1]
    #each peak in fact is four peaks: the following is adapted from the first for loop
    # of the function verifyWithCrossCorrelation() of PhaseCorrelation.java in
    # http://trac.imagej.net/browser/ImgLib/imglib1/algorithms/src/main/java/mpicbg/imglib/algorithm/fft/PhaseCorrelation.java?rev=e010ba0694e985c69a4ade7d846bef615e4e8043
    rows2 = rows.copy()
    cols2 = cols.copy()
    below0 = rows2<0
    rows2[below0] += shape[0]
    rows2[numpy.logical_not(below0)] -= shape[0]
    below0 = cols2<0
    cols2[below0] += shape[1]
    cols2[numpy.logical_not(below0)] -= shape[1]
    allshifts = numpy.column_stack((numpy.concatenate((rows, rows, rows2, rows2)),
                                    numpy.concatenate((cols, cols2, cols, cols2))))
    idx, corrcoefs = bestShift(im0, im1, allshifts, minratio)
    corrcoef = corrcoefs[idx]
    shft = numpy.array(allshifts[idx])
#    print('raro: '+str(shft)+', '+str(corrcoef))
    peak  = values[idx % values.size]
    
#    refinement = True
#    
#    if refinement:
#      num=1
#      dsp  = numpy.arange(-num, num+1).reshape((1,-1))
#      dspr = numpy.repeat(dsp, dsp.size, axis=1)
#      dspc = numpy.repeat(dsp, dsp.size, axis=0)
#      shifts = numpy.column_stack((dspr.ravel()+shft[0], dspc.ravel()+shft[1]))
#      print('before refinement: '+str(shft)+', '+str(corrcoef))
#      idx, corrcoefs = bestShift(im0, im1, shifts, minratio)
#      corrcoef = corrcoefs[idx]
#      shft = numpy.array(shifts[idx])
#      print('after refinement: '+str(shft)+', '+str(corrcoef))
#      print('neighbourhood: ')
#      for k in xrange(shifts.shape[0]):
#        print(str(shifts[k])+': '+str(corrcoefs[k]))
    
    if subpixel:
      if (scaleSubPixel is not None) and (scaleSubPixel>=2):
        sub0, sub1 = getAlignedSubmatrices(im0, im1, shft)
        finer = numpy.array(imr.dftregistration(sub0,sub1,usfac=scaleSubPixel))
        shft = shft+finer
      else:
        shft = numpy.array(find_subpixel_peak_position(ir, shft))
        
#      finershft = numpy.array(find_subpixel_peak_position(ir, shft))
#      if (scaleSubPixel is not None) and (scaleSubPixel>=2):
#        #work only with the matching submatrices, to remove spurious peaks
#        sub0, sub1 = getAlignedSubmatrices(im0, im1, shft)
#        finer = numpy.array(imr.dftregistration(sub0,sub1,usfac=scaleSubPixel))
#        finershftIMR = shft+finer
#        discreps = finershft-finershftIMR
##        print('DISCREPANCIES A: '+str(finershft))
##        print('DISCREPANCIES B: '+str(finershftIMR))
#        if (numpy.abs(discreps)<0.5).all():
#          #we only trust register_images if the expected shift is around the same
#          #as the one computed from fitting a gaussian to the peak
#          finershft = finershftIMR
#      shft=finershft
    return [shft, corrcoef, peak]

def translationUpsamplingTestPeaks(im0, im1, scale, numPeaks, subpixel=False, minratio=0.01):
  #http://www.velocityreviews.com/threads/magic-kernel-for-image-zoom-resampling.426518/
  #http://johncostella.webs.com/magic/
  if scale>1:
    im0 = ndii.zoom(im0, scale, mode='wrap')
    im1 = ndii.zoom(im1, scale, mode='wrap')
  shft = translationTestPeaks(im0, im1, numPeaks, subpixel)
  if scale>1:
    shft[0] /= scale
  return shft



#import sys
#if sys.gettrace() is not None:
#  print('debugging')
#  import tifffile as tff
#  img0 = tff.imread('/home/josedavid/3dprint/software/pypcl/corrected.0.tif')
#  img1 = tff.imread('/home/josedavid/3dprint/software/pypcl/corrected.1.tif')
##  import image_registration as ir
##  result = ir.register_images(img0, img1, usfac=1)
#  imA = img0-img0.mean()
#  imB = img1-img1.mean()
#  res = translationTestPeaks(imA, imB, numPeaks=100, subpixel=True, scaleSubPixel=1000, minratio=0.01)
#  res=res