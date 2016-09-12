# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 10:48:01 2015

@author: josedavid
"""

import platform

ISWINDOWS = platform.system().lower()=='windows'

import time
import datetime
import cPickle as pickle

import numpy as n
import cv2
import scipy.ndimage.interpolation as intpi
import scipy.interpolate as intp
import scipy.stats
import matplotlib.pyplot as plt
#import matplotlib.colorbar as cb
import matplotlib.colors as mcl
from mpl_toolkits.mplot3d import Axes3D
import ransac
import accum
if not ISWINDOWS:
  import pcl
  import pcl.registration as reg
from matplotlib import collections  as mc 
from pylab import cm 
import matplotlib._pylab_helpers
from scipy.spatial import Delaunay#, KDTree
import os
import os.path as op
import subprocess as sub
import scipy.ndimage.filters as scifil
import scipy.misc as scim

import itertools as it
import struct
import operator as opt

import sys

import imreg


import traceback

import pluconv as p

import rotations as r

import write3d as w
from numbers import Number

import tifffile as tff

from collections import namedtuple
Heightmap      = namedtuple('Heightmap',      ['img', 'size', 'step'])
Mesh           = namedtuple('Mesh',           ['points', 'triangles'])
ResultRegister = namedtuple('ResultRegister', ['xyzs', 'nonans', 'rectangles', 'processOrder', 'previousHeightmaps'])
ResultSmooth   = namedtuple('ResultSmooth',   ['xyzS', 'stats'])
ResultAll      = namedtuple('ResultAll',      ['xyzs', 'nans', 'xyzS', 'stats', 'mesh'])

def makeSimpleProcessSpecification(numImages):
  """Use this if you only know for sure that each image overlaps with the previous one"""
  return [(x, [x-1]) for x in xrange(numImages)]

def makeGridProcessSpecification(nrows, ncols, fillmode='byrow', gridmode='snake', 
                                 totalNum=None, registerOrder='zigzag'):
  """Use this if you know for sure that the images are in a grid, AND each image overlaps with all its neighbours
  in the grid. If there is some non-overlaping neighbour, the algorithm will still work but
  it will be MUCH slower. If there is some image not overlapping with any of its grid neighbours, the algorithm will fail.
  
  The parameters are:
  
    nrows: the number of rows of the grid
    
    ncols: the number of columns of the grid
    
    fillmode: 'this can be either 'byrow' (the images were acquired row by row)
              or 'bycol' (the images were acquired column by column).
              
    gridmode: 'this can be either 'grid' (each row was acquired in the same direction)
              or 'snake' (for each row, the acquisition direction is reversed w.r.t the previous one).
          
          Example of snake sequence:
          
          0   1   2   3
          7   6   5   4
          8   9  10  11
         15  14  13  12
    
          Example of grid sequence:
          
          0   1   2   3
          4   5   6   7
          8   9  10  11
         12  13  14  15

    totalNum: if specified, it is the total number of acquired images. This parameter
              makes it possible to use grids which were not completely acquired
              (i.e., there are missing images at the end of the sequence)
    
    registerOrder: if specified, this is the order of registration. It can be
                   'straight' (the images are registered in the same order as
                   they were acquired) or 'zigzag' (the images are registered
                   traversing the matrix in zigzag (this may be better to keep 
                   the errors in a lower bound w.r.t the 'straight' mode,
                   since the widest gap between two adjacent images in the
                   sequences will be lower).

          Examples:
          
          Matrix:
            0   1   2   3
            4   5   6   7
            8   9  10  11
           12  13  14  15
          'straight' order:
            0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
          'zig zag' order:
            0  1  4  8  5  2  3  6  9 12 13 10  7 11 14 15

"""
  if fillmode=='bycol':
    return makeGridProcessSpecification(ncols, nrows, fillmode='byrow', gridmode=gridmode, totalNum=totalNum)
  elif fillmode!='byrow':
    raise Exception('The fillmode parameter can be either "byrow" or "bycol", not %s' % str(fillmode))
  if totalNum is None:
    totalNum = nrows * ncols
  spec = [None]*totalNum
  #shifts = [(-1, -1), (-1, 0), (-1, +1), (0, -1), (0, +1), (+1, -1), (+1, 0), (+1, +1)] #8 neighbours
  shifts = [(-1, 0), (0, -1), (0, +1), (+1, 0)] #4 neighbours
  if gridmode=='grid':
    modeSnake = False
  elif gridmode=='snake':
    modeSnake = True
  else:
    raise Exception('The gridmode parameter can be either "snake" or "grid", not %s' % str(gridmode))
  if registerOrder=='straight':
    orderedidxs = ((i,j) for i in xrange(nrows) for j in xrange(ncols))
    #make sure that the order follows the snake, to make for easier to understand orderings
    if modeSnake:
      orderedidxs = list(orderedidxs)
      for k in xrange(len(orderedidxs)):
        i, j = orderedidxs[k]
        if (i%2)==1: #reversed row
          orderedidxs[k] = (i, ncols-j-1)
  elif registerOrder=='zigzag':
    orderedidxs = accum.zigzagOrder(nrows, ncols)#(n.arange(nrows*ncols, dtype=n.int32).reshape((nrows, ncols)))
  else:
    raise Exception('The registerOrder parameter can be either "straight" or "zigzag", not %s' % str(gridmode))
  order=0
  for i,j in orderedidxs:
    idx = i*ncols+j
    thisshifts = [(i+shifti, j+shiftj) for shifti, shiftj in shifts]
    neighs = [jj+(ii*ncols) for ii, jj in thisshifts if ((ii>=0) and (ii<nrows) and 
                                                         (jj>=0) and (jj<ncols))]
#    neighs2 = [(ii, jj) for ii, jj in thisshifts if ((ii>=0) and (ii<nrows) and 
#                                                     (jj>=0) and (jj<ncols))]
    if modeSnake:
      pair = [grid2snake(nrows, ncols, [idx])[0], grid2snake(nrows, ncols, neighs)]#, idx, [n for n in neighs]]
    else:
      pair = [idx, neighs]#, j+(i*ncols), neighs2)
    if pair[0]<totalNum:
      pair[1] = [p for p in pair[1] if (p<totalNum)]# and (p<pair[0])] #the commented out condition only works in registerOrder=='straight'
      spec[order] = pair
      order+=1
  #prune neighbours according to the order
  idxs   = n.array([p[0] for p in spec])
  for i in xrange(totalNum):
    orderthis = i
    idx, neighs = spec[i]
    #find the order for each neighbour
    orderneighs = [n.where(idxs == ne)[0][0] for ne in neighs]
    #keep only neighbours which are in previous places in the sequence
    neighs = [ne for ne, orderne in zip(neighs, orderneighs) if orderne<orderthis]
    #update neighbours
    spec[i][1] = neighs
    
    
#      if modeSnake:
#        J = ncols-j-1
#      else:
#        J = j
#  if totalNum<(nrows * ncols):
#    spec = [p for p in spec if p[0]<totalNum]
  return spec

def grid2snake(nrows, ncols, idxs):
  """translate grid to snake coordinates"""
  ncols2 = ncols*2
  for i in xrange(len(idxs)):
    idx = idxs[i]
    if (idx % ncols2) >= ncols: #if it is in a reversed row:
      base = (idx//ncols)*ncols
      idxs[i] = base + ncols-(idx-base)-1
  return idxs
    
  

def PLU2Heightmap(axes_config, data):
  """bridge PLU data to heightmap datatype"""
  if axes_config['mppx'] != axes_config['mppy']:
    raise ValueError('This code has been programmed with heightmaps with square grids.\n' \
                     'If a rectangular grid is fed to the algorithm, errors are almost guaranteed!')
  return Heightmap(img=data, size=(axes_config['xres'], axes_config['yres']),
                   step=(axes_config['mppx'], axes_config['mppy']))

def preparePLUList(files):
  """read a set of PLU files to memory and prepare them to be registered"""
  data = [None]*len(files)
  for k in xrange(len(files)):
    axes_config, measure_config, img = p.readPLU(files[k])[:3]
    if k==0:
      base = axes_config['z_0']
    else:
      #rebase all images to be at the same height as the first one (this seems
      #critical for good results of the SIFT algorithm)
      img += (axes_config['z_0']-base)
    data[k] = PLU2Heightmap(axes_config, img)
  return data

class PreparePLUIterator(object):
  """Same as preparePLUList(), but with an API compatible with computeRegistration"""
  def __init__(self):
    self.base = None
  def loader(self, filename):  
    axes_config, measure_config, img = p.readPLU(filename)[:3]
    if self.base is None:
      self.base = axes_config['z_0']
    else:
      img += (axes_config['z_0']-self.base)
    return img, (axes_config['mppx'], axes_config['mppy'])

class SimpleUniversalImageLoader(object):
  """This loader works with any known image type, just be careful if using images with different pixel steps
     (PLU files specify the pixel step), because the code will fail if images with different pixel steps
     are used together. In general, it is not a good idea to mix different image formats"""
  def __init__(self, defaultpixelstep=None):
    self.PLUbase = None
    self.defaultpixelstep = defaultpixelstep
  def loader(self, filename):  
    ext = op.splitext(filename)[1].lower()
    if ext=='.plu':
      axes_config, measure_config, img = p.readPLU(filename)[:3]
      if self.PLUbase is None:
        self.PLUbase = axes_config['z_0']
      else:
        img += (axes_config['z_0']-self.PLUbase)
      return img, (axes_config['mppx'], axes_config['mppy'])
    elif ext in ['.tif', '.tiff']:
      img = tff.imread(filename)
      return img, [self.defaultpixelstep, self.defaultpixelstep]
    elif ext in ['.png']:
      img = scim.imread(filename)
      return img, [self.defaultpixelstep, self.defaultpixelstep]
    else:
      raise ValueError('Unrecognized image format: '+ext)
  


def unpackHeightmap(x):
  """helper to use a list of heightmaps as input to RegisterTools.computeRegistration()"""
  return x.img, x.step


def rotateJustZ(R, img, xyz):
  """rotate a heightmap represented by the heightmap image and the point cloud. The rotation is in place"""
  rotz = n.dot(R, xyz.T)[2,:]
  xyz[:,2] = rotz
  img[:,:] = rotz.reshape(img.shape)

#adapted from http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
def enum(*sequential, **named):
  """hack to create C-style enums"""
  enums = dict(zip(sequential, range(len(sequential))), **named)
  return type('Enum', (), enums)  

#C-style constants (enumerations) for options
class C(object):
  SMOOTHING    = enum('RESAMPLE', 'ADHOC')
  GROUPWITH    = enum('MEAN', 'MEDIAN')
  PLANEROT     = enum('JUSTFIRST', 'ALLBYFIRST', 'ALLINDEPENDENT', 'NOROTATION')
  FIRSTPHASE   = enum('RANSACROTATION', 'PHASECORRELATION')
  RANSACMODE   = enum('ROTATION', 'DISPLACEMENT')
  SUBPIXELMODE = enum('INTERPOLATE', 'RESTORE', 'DONOTHING')
  INTERPORDER  = enum('LINEAR', 'CUBIC')
  INTERPMETHOD = enum('MAP_COORDINATES', 'GRIDDATA')
  interpd      = [[1, 3], ['linear', 'cubic']] #hack to select argument for interpolation mode, for the different scipy.interpolation function interfaces
  STATE        = enum('END', 'LOADIMAGES', 'FIRSTPHASE', 'COMPUTEPOINTCLOUD', 'SECONDPHASE')
  COLORS       =    [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],
                    [127,0,0],[0,127,0],[0,0,127],[127,127,0],[127,0,127],[0,127,127],
                    [255,127,0],[255,0,127],[0,255,127],[127,255,0],[127,0,255],[0,127,255],
                    [255,255,127],[255,127,255],[127,255,255],[127,127,255],[127,255,127],[255,127,127]]
  COLORS01     = [[a/255.0, b/255.0, c/255.0] for a,b,c in COLORS]

  #this permutation matrix is necessary because RANSAC is performed in OpenCV
  #coordinates, whose XY axes are switched with respect to our general convention
  RANSAC_Perm  = n.identity(4)[:,[1,0,2,3]]

class ConfigurationRegister(object):
  """Holds the configuration values for RegisterTools, to avoid namespace polution in RegisterTools
  Also, much functionality from external libraries is presented as objects.
  This class' state maintains references to these clases to be reused"""
  def __init__(self):
    self.detector          = None #set in initialize()
    self.extractor         = None #set in initialize()
    self.matcher           = None #set in initialize()
    self.SIFTParams        = None #set in initialize()
    #Flag for way to use NANs in alignment:
    #   -if true, all nans are set to the max value of all maps, but this turns out pretty bad (introduces a lot of alignment error) if the measurements are not over a completely horizontal plane
    #   -if false, NANs are interpolated
    self.USENAN = False
    #if true, NANs are discarded from the sub-point clouds which are fed to the ICP algorithm
    self.disableNANsForICP = True
    #IMAGE TOOLS AND CONSTANTS
    self.SIFTParams        = [['contrastThreshold', 'setDouble', 'getDouble', None],     #defaults to 0.04
                              ['edgeThreshold',     'setDouble', 'getDouble', None],     #defaults to 10
                              ['nFeatures',         'setInt',    'getInt',    None],     #defaults to 0 (all features)
                              ['nOctaveLayers',     'setInt',    'getInt',    None],     #defaults to 3
                              ['sigma',             'setDouble', 'getDouble', None]]     #defaults to 1.6
    #if the heightmap does not specify the pixel step (to scale correctly the
    #XY coordinates relative to the Z coordinates), use this as default    
    self.defaultPixelStep = 1.0
    #This is to scale up the Z dimension in case it was compressed
    #when saving the image (for example, compressed to the range [0,1])
    self.zfac = 1.0#10
    #if this flag is false, we feed SIFT with a matrix of floating points,
    #to increase accuracy in case of ultra-low contrast (might be due to
    #a large z range across all heightmaps). However, this can be false ONLY
    #if we have recompiled opencv's SIFT to accept floating point matrices
    self.discretizeKeypointDetection = False
    # FLANN parameters (http://docs.opencv.org/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html)
    FLANN_INDEX_KDTREE     = 0
    self.index_params      = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    self.search_params     = dict(checks=50)   # or pass empty dictionary
    self.matcherThreshold  = 0.7 #this is based on Lowe's paper to detect significant matches
    #RANSAC paramenters
    self.RANSAC_k          = 1000 #max number of iterations of RANSAC
    #we take the maximum admissible error per data point as the square  of a multiple of the standard grid distance
    self.RANSAC_tfun       = 'lambda maxstep: (maxstep*5)**2'#10)**2'  
    #we take the minimum number of inlier elements as a function of the number of matched keypoints
    self.RANSAC_dfun       = 'lambda num_matches: max(n.floor(num_matches*0.25), 6)' #we require at least 6 matches
    self.RANSAC_debug      = False
    self.RANSAC_METHOD     = C.RANSACMODE.ROTATION
    #WARNING: C.RANSACMODE.DISPLACEMENT IS UNTESTED, AND ALSO IT SHOULD BE USED ONLY IF THE IMAGES ARE NOT ROTATED IN ANY WAY
    #self.RANSAC_METHOD     = C.RANSACMODE.DISPLACEMENT
    #ICP paramenters
    if ISWINDOWS:
      self.registerMethod  = None #We use a different way to invoke the ICP algorithm in Windows
    else:
      self.registerMethod  = reg.icp
      #self.registerMethod  = reg.icp_nl
      #self.registerMethod  = reg.gicp
    self.registerOverlapRatio = None#0.8 #set to None to disable the rejector
    #params of the algorithm: [setMaxCorrespondenceDistance, setTransformationEpsilon, setEuclideanFitnessEpsilon, setUseReciprocalCorrespondences]
    self.icpParamsfun      = 'lambda maxstep: [None, None, None, None]'#'lambda maxstep: [maxstep*1, None, None, None]'
    #the cutoff to consider that matching keypoints can be used for ICP
    self.ICP_cutofffun     = 'lambda maxstep: maxstep*3'
    self.ICP_Keypoints     = False
    self.ICP_maxiter       = 1000
    #SMOOTHING
#    self.smoothingMethod   = C.SMOOTHING.ADHOC
#    #self.smoothingMethod   = C.SMOOTHING.RESAMPLE
    #Neighbours' processing parameters
    #we collect 8 neighbours because we suppose that any area can be overlapped with at most 8 heightmaps
    #we add 1 because the first neighbours will be the point itself
    #we add 4 because we anticipate that some of the neighbours may be from the same heightmap
    self.KNeighs           = 8+1+4
    #max distance to consider that neighbours are equivalent points from other heightmaps
    #we have to be careful here: we are assuming that the point cloud is well
    #aligned in the XY plane. If it is too inclined, results will be schewed
    #and potentially invalid
    self.NeighDist         = 'lambda step: step*0.75'#0.95#0.75
    self.neighGroupFun     = C.GROUPWITH.MEDIAN
    #self.neighGroupFun     = C.GROUPWITH.MEAN
    self.heuristicTrimSmoothedPoints = False #True produces less vertices, but significant errors (holes) may appear
    self.returnSmoothStats = True #True to get an idea of what is doing the smoothing process, False to optimize
    self.debugSequential   = True
    self.debugXYZ          = True
    self.debugSavePath     = ''
    self.copyHeightmaps    = True
    #parameters to fit a RANSAC to the plane
    self.RANSAC_fitplane_enable = True #if false, a regression plane is computed
    self.RANSAC_fitplane_k = 200 #max number of iterations of RANSAC
    self.RANSAC_fitplane_tfun = 'lambda maxstep: (maxstep*4)' #to consider a point to fit the plane, it has to be within a multiple of the grid step
    self.RANSAC_fitplane_planeratio = 0.7 #ratio of the image at least occupied by the plane
    self.RANSAC_fitplane_saferatio = 0.9 #ratio of the plane that has to be fit in order to decide that a fit is good enough
    self.RANSAC_fitplane_debug      = False
    
    #self.rotateMode = C.PLANEROT.JUSTFIRST
    self.rotateMode = C.PLANEROT.ALLBYFIRST
    #self.rotateMode = C.PLANEROT.ALLINDEPENDENT
    #self.rotateMode = C.PLANEROT.NOROTATION
    
    #When all images have approximately the same height profile and the background dominates, it is better to center the Z values by substracting the median    
    #However, this can be catastrophic is different images in the dataset are naturally at different heights (for example, an dataset of a relief, with one of the images almost entirely within higher ground, and the others mostly imaging lower ground)
    #self.substractMedian = True 
    self.substractMedian = False
    
    self.PhaseCorrScale       = 100.0 #this can be quite high with the algorithm we are currently using
    self.PhaseCorrNumPeaks    = [100, 1000, 10000] #the highest peaks might not signal the strongest cross-correlations. We incrementally try more and more peaks, up to a point
    self.PhaseCorrRecommendableCorrCoef = 0.8 #the correlation coefficient must be quite high for the phase correlation to be OK
    self.PhaseCorrSubpixel    = True 
    self.PhaseCorrWhitening   = False
    self.PhaseCorrMinratio    = 0.01 #DO NOT TRUST CROSS CORRELATION WITH TOO LOW OVERLAPING
    self.PhaseCorrRestoreSubPixelMethod = C.SUBPIXELMODE.INTERPOLATE
    #self.PhaseCorrRestoreSubPixelMethod = C.SUBPIXELMODE.RESTORE
    #self.PhaseCorrRestoreSubPixelMethod = C.SUBPIXELMODE.DONOTHING

    #self.firstPhase = C.FIRSTPHASE.RANSACROTATION
    #WARNING: THIS ONLY SHOULD BE USED IF THE FOLLOWING CONDITIONS ARE MET:
    #          - THE IMAGES ARE NOT ROTATED IN ANY WAY
    #          - THE IMAGES ARE NOT BIASED OR THE RANSAC PLANE FITTING PERFORMS WELL
    self.firstPhase = C.FIRSTPHASE.PHASECORRELATION
    
    self.interpOrder = C.INTERPORDER.LINEAR
    #self.interpOrder = C.INTERPORDER.CUBIC

  def setSIFTParams(self):
    """helper to set SIFT params on the opencv objects"""
    for x in [self.detector, self.extractor]:
      for idx, (name, setname, getname, val) in enumerate(self.SIFTParams):
        if val is not None:
          getattr(x, setname)(name, val)
        else:
          self.SIFTParams[idx][3] = getattr(x, getname)(name)
  
  def disposeObjects(self):
    """remove references"""
    self.detector          = None
    self.extractor         = None
    self.matcher           = None
    
  def initialize(self):
    """initialize objects from third-party libraries"""
    self.detector          = cv2.FeatureDetector_create("SIFT")
    self.extractor         = cv2.DescriptorExtractor_create("SIFT")
    self.setSIFTParams()
    self.matcher           = cv2.FlannBasedMatcher(self.index_params,self.search_params)
    lambdastrs = ['RANSAC_tfun', 'RANSAC_dfun', 'icpParamsfun', 'ICP_cutofffun', 'NeighDist', 'RANSAC_fitplane_tfun']
    for lambdastr in lambdastrs:
      val = getattr(self, lambdastr)
      if isinstance(val, basestring):
        setattr(self, lambdastr, eval(val))


class RegisterTools(object):
  """Main API to do registration processes and analyze and render the output.
  It can work online (everything is in the RAM) or offline (all small datasets are in RAM,
  but image data is on disk, only loaded when needed. Slower but enables the processing of
  very large datasets"""
  def __init__(self, savePath=None, originalLogHandles=None):
    if savePath is not None:
      if (len(savePath)>0) and savePath[-1]!=os.sep:
        savePath = savePath + os.sep
    #config variables
    self.savePath = savePath
    self.inMemory = savePath is None
    
    #registration process variables
    self.conf = None
    self.num = None
    self.firstRotation = None
    self.state = None
    self.finished = None
    
    self.minmaxs = None
    self.xyzs = None
    self.imgs = None
    self.imgsteps = None
    self.imgshapes = None
    self.nans = None
    self.numnonans = None
    self.nanflags = None
    self.BBs = None
    self.usedRANSAC = None
    self.rectangles = None
    self.accumRotations = None
    self.processed = None
    self.keypointsComputed = None
    self.kpdess = None
    self.kpxyzs = None
    self.newPointss = None
    self.oldPointss = None
    self.processOrder = None
    self.previousHeightmaps = None
    
    #config variables
    self.logFile = None
    self.confFile = None
    if originalLogHandles is None:
      self.originalLogHandles = [sys.stdout]
    else:
      self.originalLogHandles = originalLogHandles
    self.logHandles = self.originalLogHandles
    #most variables are vectors or lists, with one entry for each image
    #each variable is saved to a different file (facilitates easy load/save handling),
    #also very big datasets (such as the images) are saved to a sequence of files
    self.files = {
                  'state'               : 'state.npy',
                  'finished'            : 'finished.npy',
                  'num'                 : 'num.npy',
                  'firstRotation'       : 'firstRotation.npy',
                  'usedRANSAC'          : 'usedRANSAC.npy',
                  'minmaxs'             : 'minmaxs.npy',
                  'imgsteps'            : 'imgsteps.npy',
                  'imgshapes'           : 'imgshapes.npy',
                  'nanflags'            : 'nanflags.npy',
                  'numnonans'           : 'numnonans.npy',
                  'BBs'                 : 'BBs.npy',
                  'rectangles'          : 'rectangles.npy',
                  'accumRotations'      : 'accumRotations.npy',
                  'processed'           : 'processed.npy',
                  'processOrder'        : 'processOrder.npy',
                  'previousHeightmaps'  : 'previousHeightmaps.npy',
                  'keypointsComputed'   : 'keypointsComputed.npy',
                  'xyzs'                : 'xyz.%03d.npy',
                  'imgs'                : 'img.%03d.npy',
                  'nans'                : 'nan.%03d.npy',
                  'kpdess'              : 'kpdess.%03d.npy',
                  'kpxyzs'              : 'kpxyzs.%03d.npy',
                  'newPointss'          : 'newPointss.%03d.npy',
                  'oldPointss'          : 'oldPointss.%03d.npy',
                  }
    #config variables
    self.fileIsWhole = {key: ('%' not in value) for key, value in self.files.iteritems()}
    self.wholeNames = [value for key, value in self.fileIsWhole.iteritems() if value]
    #self.isLoaded = {key: False for key in self.files.keys()}
    
    if not self.inMemory:
      self.logFile = self.savePath+"log.txt"
      self.confFile = self.savePath+"conf.p"
      self.files = {key: self.savePath + value for key, value in self.files.iteritems()}
    
    #same order as in C.STATE
    self.dispatcher = [None, #END
                       self.loadImage, 
                       self.firstPhase, 
                       self.computePointCloud,
                       self.secondPhase,
                       ]

  
  def log(self, s):
    """log messages in one or more outputs (files, standard output, log window...)"""
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S -> ')
    s = s+st
    for handle in self.logHandles:
      handle.write(s)
  
  def saveVarsToList(self):
    """This is useful to save the state, reload the module, create a fresh object, and restore the state with self.loadVarsFromList()"""
    if not self.inMemory:
      raise Exception('This method can be used only when all is in memory!!!')
    return (self.conf, [(var, self.loadVar(var, -1)) for var in self.files.keys()])
    
  def loadVarsFromList(self, tupla):
    if not self.inMemory:
      raise Exception('This method can be used only when all is in memory!!!')
    self.conf, lista = tupla
    for var, value in lista:
      self.saveVar(var, -1, value)
  
  def loadVar(self, name, idx):
    """To abstract the RAM/HD modes, variables are loaded with this method"""
    if self.inMemory or self.fileIsWhole[name]:
      data = getattr(self, name)
      if isinstance(idx, Number):
        if idx==-1:
          return data
        else:
          return data[idx]
      else:
        return [data[i] for i in idx]
    else:
      if isinstance(idx, Number):
        data = n.load(self.files[name] % idx)
        return data
      else:
        data = [n.load(self.files[name] % i) for i in idx]
        return data
  
  def saveVar(self, name, idx, value):
    """To abstract the RAM/HD modes, variables are saved with this method"""
    if self.inMemory or self.fileIsWhole[name]:
      if isinstance(idx, Number):
        if idx==-1:
          setattr(self, name, value)
        else:
          data = getattr(self, name)
          data[idx] = value
      else:
        data = getattr(self, name)
        for i,v in zip(idx, value):
          data[i] = v
      if not self.inMemory:
        data = getattr(self, name)
        n.save(self.files[name], data)
    else:
      if isinstance(idx, Number):
        n.save(self.files[name] % idx, value)
      else:
        for i,v in zip(idx, value):
          n.save(self.files[name] % i, v)
        

  def initVars(self, initState, conf, num, processOrder, previousHeightmaps):
    """initialize all non-heavy vars"""
    #convert the list of pairs to a pair of lists (easier to do some tasks)
    self.conf = conf
    if not self.inMemory:
      #remove all existing files!!!!
      for name, fil in self.files.iteritems():
        if self.fileIsWhole[name]:
          if op.isfile(fil):
            os.remove(fil)
        else:
          for k in xrange(num):
            fn = fil % k
            if op.isfile(fn):
              os.remove(fn)
      self.conf.disposeObjects()
      #save configuration and create log file
      with open(self.confFile, "wb" ) as f:
        pickle.dump(self.conf, f)
      with open(self.logFile, "w" ) as f:
        f.write('')
    self.conf.initialize()
    self.saveVar('num',                -1, num)
    self.saveVar('firstRotation',      -1, n.identity(4))
    self.saveVar('minmaxs',            -1, n.zeros((num,2)))
    self.saveVar('imgsteps',           -1, n.zeros((num,2)))
    self.saveVar('imgshapes',          -1, n.zeros((num,2)))
    self.saveVar('nanflags',           -1, n.zeros((num,), dtype=n.bool))
    self.saveVar('usedRANSAC',         -1, n.zeros((num,), dtype=n.bool))
    self.saveVar('numnonans',          -1, n.zeros((num,), dtype=n.int32))
    self.saveVar('BBs',                -1, n.zeros((num,4)))
    self.saveVar('rectangles',         -1, n.zeros((num,4,3)))
    self.saveVar('accumRotations',     -1, n.repeat(n.identity(4).reshape((1,4,4)), num, axis=0))
    self.saveVar('processed',          -1, n.zeros((num,), dtype=n.bool))
    self.saveVar('processOrder',       -1, processOrder)
    self.saveVar('previousHeightmaps', -1, previousHeightmaps)
    self.saveVar('keypointsComputed',  -1, n.zeros((num,), dtype=n.bool))
    self.saveVar('finished',           -1, False)
    self.saveVar('state',              -1, initState)
    if self.inMemory:
      self.imgs       = [None]*num
      self.xyzs       = [None]*num#[n.arange(30).reshape((10,3))]*num
      self.nans       = [None]*num#[n.concatenate((n.zeros(3,dtype=n.bool), n.ones(7,dtype=n.bool)))]*num
      self.kpdess     = [None]*num
      self.kpxyzs     = [None]*num
      self.newPointss = [None]*num
      self.oldPointss = [None]*num
    
  def loadVars(self, initState, conf=None, num=None, processSpecification=None):
    """load all non-heavy vars"""
    if (not self.inMemory) and op.isfile(self.files['state']):
      #load state variables from disk
      for key, val in self.fileIsWhole.iteritems():
        if val:
          setattr(self, key, n.load(self.files[key]))
      with open(self.confFile, "rb" ) as f:
        self.conf = pickle.load(f)
      self.conf.initialize()
    else:
      #initialize state variables
      if (not self.inMemory) or (self.state is None):
        if initState is None:
          raise Exception('Cannot resume computation, no state file found!!!!')
        if conf is None:
          raise Exception('No previusly saved state, so the configuration MUST be specified!!!')
        if num is None:
          raise Exception('No previusly saved state, so the number of heightmaps MUST be specified!!!')
        if num < 2:
          raise Exception('The number of heightmaps MUST be higher than 1!!!')
        if processSpecification is None:
          processSpecification = [(x,[x-1]) for x in xrange(num)]
        processOrder       = map(opt.itemgetter(0), processSpecification)
        previousHeightmaps = map(opt.itemgetter(1), processSpecification)
        #sanity checks for the process specification
        if (len(processOrder)!=num) or (len(previousHeightmaps)!=num):
          raise Exception('Process specification is not concordant with the number of Heightmaps!!!')
        for i in xrange(num):
          if isinstance(processOrder[i], Number):
            processOrder[i] = int(processOrder[i])
          else:
            raise Exception('all objects in the process specification must be numbers, but <%s> is not!!!' % str(processOrder[i]))
          if isinstance(previousHeightmaps[i], Number):
            previousHeightmaps[i] = [int(previousHeightmaps[i])]
          elif hasattr(previousHeightmaps[i], '__iter__'):
            previousHeightmaps[i] = list(previousHeightmaps[i])
            if (i>0) and (len(previousHeightmaps[i]) == 0):
              raise Exception('all heightmaps after the first one must have at least one overlapping previous heightmap, but in position %d, %d does not!!!' % (i, processOrder[i]))
            for j in xrange(len(previousHeightmaps[i])):
              if isinstance(previousHeightmaps[i][j], Number):
                previousHeightmaps[i][j] = int(previousHeightmaps[i][j])
              else:
                raise Exception('all objects in the process specification must be numbers, but <%s> is not!!!' % str(previousHeightmaps[i][j]))
        self.log('PROCESS ORDER: %s\n' % str(processOrder))
        self.log('PREVIOUS HEIGHTMAPS FOR EACH HEIGHTMAP: %s\n' % str(previousHeightmaps))
        self.initVars(initState, conf, num, processOrder, previousHeightmaps)
    self.logHandles = self.originalLogHandles
    if (len(self.conf.debugSavePath)>0) and self.conf.debugSavePath[-1]!=os.sep:
      self.conf.debugSavePath = self.conf.debugSavePath + os.sep
    
  def loadFinished(self):
    """load state from a directory, if it contains a finished registration (if
    some state files have been deleted from the directory, this may trigger an
    exception later in the pipeline"""
    if self.inMemory:
      raise ValueError('to use RegisterTools.loadfinished(), please specify a path in the constructor')
    if not op.isfile(self.files['state']):
      raise ValueError('to use RegisterTools.loadfinished(), a registration must have been previously computed')
    self.loadVars(None)
    if not self.finished:
      raise ValueError('to use RegisterTools.loadfinished(), a registration must have been previously computed')
    
  def resetToBlankState(self):
    """remove state from memory/HD"""
    self.state = None
    if not self.inMemory and op.isfile(self.files['state']):
      os.remove(self.files['state'])
  
  def executeStateMachine(self, initState, initArgs, callbackFun=None):
    """Boilerplate code to execute a state machine specified in self.dispatcher.
    Each dispatched method is responsible for setting the next state"""
    result = (False, 'Nothing done! Initial State: '+str(initState))
    if self.inMemory:
      self.logHandles = self.originalLogHandles
    else:
      f = open(self.logFile, 'a')
      self.logHandles = self.originalLogHandles + [f]
    testcallback = callbackFun is not None
    try:
      while self.state[0]!=C.STATE.END:
        if self.state[0]==initState[0]:
          if (initArgs is None) or (initArgs[0] is None):
            raise Exception('No initial arguments provided to read images, but we are not finished reading them!!!!')
          result = self.dispatcher[self.state[0]](*initArgs)
        else:
          result = self.dispatcher[self.state[0]]()
        if not result[0]:
          self.log('Error: '+result[1]+'\n')
          break
        if testcallback:
          abort = callbackFun()
          if abort:
            self.log('COMPUTATION ABORTED\n')
            break
    except:
      s = traceback.format_exc()
      self.log(s)
      result = (False, s)
    self.logHandles = self.originalLogHandles
    if not self.inMemory:
      f.close()
    return result

  def resumeComputation(self, heightmaps=None, loader=None, callbackFun=None):
    """Boilerplate code to resume the execution of a state machine"""
    initState = [C.STATE.LOADIMAGES, 0]
    self.loadVars(initState)
    return self.executeStateMachine([-1], [heightmaps, loader], callbackFun=callbackFun)

  def computeRegistration(self, conf=None, num=None, heightmaps=None, loader=None, processSpecification=None, forceNew=False, callbackFun=None):
    """Boilerplate code to execute/resume a registration computation as a state machine"""
    initState = [C.STATE.LOADIMAGES, 0]
    if forceNew:
      self.resetToBlankState()
    self.loadVars(initState, conf, num, processSpecification)
    return self.executeStateMachine(initState, [heightmaps, loader], callbackFun=callbackFun)
    
  def loadImage(self, heightmaps, loader):
    """state machine method: Load and pre-process each image"""
    zfac   = self.conf.zfac
    
    idx = self.state[1]
    
    self.log('preparing image %d...\n' % idx)
    
    ind = self.processOrder[idx]
    self.log('  in processOrder[%d]==%d\n' % (idx, ind))
    
    img, imgstep = loader(heightmaps[ind])
    if imgstep[0] is None:
      imgstep[0] = self.conf.defaultPixelStep
    if imgstep[1] is None:
      imgstep[1] = self.conf.defaultPixelStep
    if zfac is not None:
      img      *= zfac
      #xyz[:,2] *= zfac
    imgshape = n.array(img.shape)
    if self.inMemory and self.conf.copyHeightmaps:
      img = img.copy()
    nan          = n.isnan(img)
    nanflag      = nan.any()
    numnonan     = nan.size-n.sum(nan)
    xyz = None
    if nanflag:
      self.log('Interpolating NANs for %d\n'%ind)
      xyz  = self.interpolateNANs(img, imgstep, nan, None, 0.0)
    #THE CODE INSIDE THIS "IF" STATEMENT SHOULD BE INSIDE makePlanar(), BUT IT
    #IS EASIER AND MORE READABLE TO PUT IT HERE
    if self.conf.rotateMode!=C.PLANEROT.NOROTATION:
      if xyz is None:
        xyz  = r.image2XYZ(img, imgstep)
      maxstep = n.max(imgstep)
      if nanflag:
        xyzRANSAC = xyz[n.logical_not(nan.ravel()),:]
      else:
        xyzRANSAC = xyz
      result = self.makePlanar(maxstep, img, imgstep, xyzRANSAC, xyz, idx)
      if not result[0]: return result
    #after the plane rotation, substract the median to compensate for height drift
    #seems weird, but it happens
#    if self.conf.firstPhase==C.FIRSTPHASE.PHASECORRELATION:
#      img -= n.median(img)
    if self.conf.substractMedian:
      img -= n.median(img)
    
    minmax    = n.array([n.min(img), n.max(img)])
    BB        = r.imageBB(img, imgstep)
    rectangle = r.imageRectangle(img, imgstep)
#    normals[first] = r.normalsForMatrixXYZ(xyzs[first], imgs[first].shape)
    
    self.saveVar('imgs',         ind, img)
    self.saveVar('imgsteps',     ind, imgstep)
    self.saveVar('imgshapes',    ind, imgshape)
    self.saveVar('nans',         ind, nan)
    self.saveVar('nanflags',     ind, nanflag)
    self.saveVar('numnonans',    ind, numnonan)
    self.saveVar('minmaxs',      ind, minmax)
    self.saveVar('BBs',          ind, BB)
    self.saveVar('nanflags',     ind, nanflag)
    self.saveVar('rectangles',   ind, rectangle)
    if idx==0:
      self.saveVar('processed',  ind, True)
      self.saveVar('xyzs',       ind, r.image2XYZ(img, imgstep))

    idx += 1
    if idx==self.num:
      step = n.unique(self.imgsteps)
      if step.size!=1:
        return (False, 'This system has some infrastructure in place for varying pixel steps (across XY and across images), but it is mostly untested and likely to be buggy (swapping X and Y steps, or assuming that neighboring heightmaps have the same steps, for example). Also, most many of the used algorithms (such as phase correlation) require uniform steps. Because of that, we require all steps to be identical for all heightmaps. However, the loaded dataset has several different steps: '+str(step))
#      if self.conf.debugXYZ: w.writePLYFileColorList(self.conf.debugSavePath+'test.afterMakePlanar.ply', (r.image2XYZ(self.loadVar('imgs', idx), self.imgsteps[idx])[n.logical_not(self.loadVar('nans', idx).ravel()),:] for idx in xrange(self.num)), C.COLORS, sizes=self.numnonans)
#      #THIS WAS SENSIBLE WHEN RANSAC ROTATION AND PHASE CORRELATION WERE MUTUALLY EXCLUSIVE. BU NOW, KEYPOINTS ARE COMPUTED ON DEMAND
#      if self.conf.firstPhase==C.FIRSTPHASE.RANSACROTATION:
#        self.saveVar('state', -1, [C.STATE.COMPUTEKEYPOINTS, 0])
#      else:
#        self.saveVar('state', -1, [C.STATE.FIRSTPHASE, 1])
      self.saveVar('state', -1, [C.STATE.FIRSTPHASE, 1])
    else:
      self.saveVar('state', 1, idx)
    return (True, '')

  def lazyComputeKeypoints(self, img, ind):     
    """state machine helper method: lazy computation of keypoints of each image"""

    if self.keypointsComputed[ind]:
      
      return (self.loadVar('kpdess', ind), self.loadVar('kpxyzs', ind), img)
      
    else:

      self.log('Lazy computation of keypoints and descriptors for %d\n' % ind)      
      imgstep = self.imgsteps[ind]
      minmaxs = self.minmaxs
      if img is None:
        img = self.loadVar('imgs', ind)
      kpdes, kpxyz = self.getKeypoints(img, imgstep, minmaxs[:,0].min(), minmaxs[:,1].max())
      self.saveVar('kpdess',            ind, kpdes)
      self.saveVar('kpxyzs',            ind, kpxyz)
      self.saveVar('keypointsComputed', ind, True)
      
      return (kpdes, kpxyz, img)
  
  def firstPhase(self):
    """state machine method: Compute first approximation to the transformation to register the image"""
    #As each image i is supposed to overlap with the images i-1 and i+1,
    #we can do this as a simple loop. First image is untouched, the rest are
    #registered one by one. This function works as the first part of the body
    #of the loop to register the images one by one
    i = self.state[1]
    idx = self.processOrder[i]
    possiblePrevs = self.previousHeightmaps[i]
    
    doRANSAC = self.conf.firstPhase==C.FIRSTPHASE.RANSACROTATION
    doPC     = self.conf.firstPhase==C.FIRSTPHASE.PHASECORRELATION
    
    imgidx       = None
    imgprev      = None
    selectedPrev = None
    
    self.log('FIRST PHASE %d to %s...\n' % (idx, str(possiblePrevs)))
    if doPC:
      bestRots = [None]*len(possiblePrevs)
      corrcoefs = n.empty((len(possiblePrevs),))
      corrcoefs.fill(-n.inf)
      imgidx  = self.loadVar('imgs', idx)
      mincorrcoef = self.conf.PhaseCorrRecommendableCorrCoef
      for z in xrange(len(possiblePrevs)):
        prev = possiblePrevs[z]
        if not self.processed[prev]:
          self.log('\n#####################################\nTHIS WARNING IS LIKELY BOGUS IF YOU BUILT THE PROCESS SPECIFICATION WITH makeGridProcessSpecification\n\nWARNING: possibly wrong specification: heightmap %d is supposed to be matched to %d, but %d has not been processed\nIf more heightmaps have been specified as overlapping with %d, they will be tried. Otherwise, an error will be raised\n#####################################\n' % (idx, prev, prev, idx))
          continue
    
        self.log('trying phase correlation %d to %d (%d of %d)...\n' % (idx, prev, z, len(possiblePrevs)))
          
        imgprev = self.loadVar('imgs', prev)
      
        if imgidx.shape != imgprev.shape:
        
          self.log('The default first phase method is phase correlation, but the images %d and %d are of different sizes.\nRather than zero-padding in the Fourier transform (may be unreliable for our implementation of phase correlation).\n\nIf more heightmaps have been specified as overlapping with %d, they will be tried. Otherwise, we fall back to RANSAC Rotation\n' % (idx, prev, idx))
          continue
      
        else:
      
          for numPeaks in self.conf.PhaseCorrNumPeaks:      
            result = self.doPhaseCorrelation(numPeaks, imgidx, imgprev, i, idx, prev)
            if not result[0]: return result
            bestRot3D = result[1]
            corrcoef = result[2]
            if corrcoef>=mincorrcoef:
              corrcoefs[z] = corrcoef
              bestRots[z]  = bestRot3D
              break
          if corrcoef<mincorrcoef:
            self.log('\n#####################################\nWARNING: After trying up to %d peaks, the correlation coefficient is too low, this probably means that the registering is grossly inexact.\nIn turn, this probably means that the images were not overlapping to begin with, but phase correlation sometimes fails for overlapping images if the overlap is too small, there is too much noise, or the heightmaps are not coplanar\n#####################################\n' % numPeaks)

  #      self.log('PREVS: %s\n' % str(possiblePrevs))      
#      self.log('CORRCOEFS: %s\n' % str(corrcoefs))      
      ii = n.argmax(corrcoefs)
#      self.log('II: %s\n' % str(ii))
#      self.log('PREVS[II]: %s\n' % str(possiblePrevs[ii]))
      
      if corrcoefs[ii]<mincorrcoef:
        self.log('\n#####################################\nWARNING: It was not possible to find a rotation using phase correlation. Falling back to RANSAC rotation. This is a more robust algorithm, but it generates less precise rotations...\n#####################################\n')
        doRANSAC = True
      else:
        self.log('Best phase correlation: %d with %d\n' % (idx, possiblePrevs[ii]))
        bestRot3D    = bestRots[ii]
        selectedPrev = possiblePrevs[ii]

#      if (not doRANSAC) or (len(possiblePrevs)>1):
#        imgprev = None
    elif not doRANSAC:
      return (False, 'ERROR: first phase method not understood: '+str(self.conf.firstPhase))
          
    if doRANSAC:
      
      bestRots = [None]*len(possiblePrevs)
      numRANSACmatchess = n.zeros((len(possiblePrevs),))
      for z in xrange(len(possiblePrevs)):
        prev = possiblePrevs[z]
        if not self.processed[prev]:
          if not doPC:
            self.log('\n#####################################\nWARNING: possibly wrong specification: heightmap %d is supposed to be matched to %d, but %d has not been processed\nIf more heightmaps have been specified as overlapping with %d, they will be tried. Otherwise, an error will be raised\n#####################################\n' % (idx, prev, prev, idx))
          continue
        
        self.log('trying RANSAC rotation %d to %d (%d of %d)...\n' % (idx, prev, z, len(possiblePrevs)))
        imgprev = self.loadVar('imgs', prev) #play it safe...
#        if imgprev is None: #imgprev may remain loaded from phase correlation
#          imgprev = self.loadVar('imgs', prev)

        result = self.doRANSACRotation(i, idx, prev, imgidx, imgprev)  
        if not result[0]: return result
        if result[1] is None:
          continue
        bestRot3D, numRANSACmatches, imgidx = result[1:4]
        result                              = None
#        imgprev                             = None
        bestRots[z]                         = bestRot3D
        numRANSACmatchess[z]                = numRANSACmatches
        
      ii = n.argmax(numRANSACmatchess)
      if numRANSACmatchess[ii]==0:
        self.log('\n#####################################\nERROR: it was not possible to find a feasible RANSAC rotation.\nAs the RANSAC rotation algorithm is quite robust, this most probably means that the heightmap %d was not actually overlapping with any of the previous heightmaps %s, or the heightmaps were too noisy or not co-planar enough\n#####################################\n' % (idx, str(possiblePrevs)))
        raise Exception('ERROR: not rotation found. Are you sure that the process specification was correct?')
      else:
        self.log('Best RANSAC rotation: %d with %d\n' % (idx, possiblePrevs[ii]))
        bestRot3D    = bestRots[ii]
        selectedPrev = possiblePrevs[ii]
      
    self.log("bestRot3D:\n%s\n" % str(bestRot3D))
    #accumulate the transformation to the one from the previous heightmap
    accumRotation = n.dot(bestRot3D, self.accumRotations[selectedPrev])
    #apply the transform to the point cloud
    xyzprev = self.loadVar('xyzs', selectedPrev)
    if imgidx is None:
      imgidx = self.loadVar('imgs', idx)
    xyzidx = r.image2XYZ(imgidx, self.imgsteps[idx])
    xyzidx    = r.doTransformT(xyzidx, accumRotation)
#    imgidx    = xyzidx[:,2].reshape(self.imgshapes[idx])
    if self.conf.debugXYZ:
      nonanidx  = n.logical_not(self.loadVar('nans', idx))
      nonanprev = n.logical_not(self.loadVar('nans', selectedPrev))
      w.writePLYFileColorList(self.conf.debugSavePath+'debug.FIRSTPHASE.%03d.ply' % idx, [xyzprev[nonanprev.ravel(),:], xyzidx[nonanidx.ravel(),:]], [[255,0,0],[0,255,0]])
      nonanidx  = None
      nonanprev = None
    self.saveVar('rectangles', idx, r.doTransformT(self.rectangles[idx], accumRotation))
    self.saveVar('xyzs', idx, xyzidx)
#    self.saveVar('imgs', idx, imgidx)
    self.saveVar('accumRotations', idx, accumRotation)
    self.saveVar('BBs', idx, n.vstack((xyzidx[:,0:2].min(axis=0), xyzidx[:,0:2].max(axis=0))).flatten())
    self.saveVar('usedRANSAC', idx, doRANSAC)
    self.saveVar('state', 0, C.STATE.COMPUTEPOINTCLOUD)
#      #apply the transform to the keypoints (necessary only if trying (as an intermediate step to tune the transformation) to do a RANSAC with several previous point clouds)
#      kpxyzs[idx] = r.doTransformT(kpxyzs[idx],                    accumRotation)
    return (True, '')

  def computePointCloud(self):
    """state machine method: compute the point clouds to perform the fine registration"""
    i = self.state[1]
    idx = self.processOrder[i]
    
    xyzidx   =               self.loadVar('xyzs', idx)
    nonanidx = n.logical_not(self.loadVar('nans', idx))

    #find overlaping bounding boxes with previously registered clouds
    overlaping  = n.nonzero(r.getOverlapingBoundingBoxes(self.BBs, idx, self.processOrder[:i]))[0]
    self.log('computing points to apply ICP for heightmap %d with previous heightmaps %s\n'  % (idx, str(overlaping)))
    if ((not self.usedRANSAC[idx]) and #self.conf.firstPhase == C.FIRSTPHASE.PHASECORRELATION) and 
       self.conf.PhaseCorrSubpixel and
       (self.conf.PhaseCorrRestoreSubPixelMethod == C.SUBPIXELMODE.INTERPOLATE)):
        #IN THIS METHOD, WE DO NOT DIRECTLY USE POINTS FROM PREVIOUSLY PROCESSED HEIGHTMAPS.
        #INSTEAD, WE INTERPOLATE VALUES TO THE GRID  OF THE CURRENT HEIGHTMAP
        #THIS METHOD IS POTENTIALLY VERY FRAGILE, SO WE ONLY USE IT WITH PHASE
        #CORRELATION, WHICH GIVES A VERY GOOD SUBPIXEL REGISTRATION
        toAddNew = [None]*len(overlaping)
        toAddOld = [None]*len(overlaping)
        if self.conf.debugXYZ:
          toAddOldP = [None]*len(overlaping)
        testPoints = [None]*len(overlaping)
        for idxcurrent in xrange(len(overlaping)):
          current  = self.processOrder[overlaping[idxcurrent]]
          older    = n.array([self.processOrder[o] for o in overlaping[:idxcurrent]])
          testPoints[idxcurrent] = r.testPoinstInsidePolygon(xyzidx, self.rectangles[current])
          
          xyzcurrent   =               self.loadVar('xyzs', current)          
          nonancurrent = n.logical_not(self.loadVar('nans', current))
          
          areOK = testPoints[idxcurrent]
          currentOK = r.testPoinstInsidePolygon(xyzcurrent, self.rectangles[idx])
          if idxcurrent>0:
            inOlder  = reduce(n.logical_or, testPoints[:idxcurrent])
            currentInOlder = reduce(n.logical_or, (r.testPoinstInsidePolygon(xyzcurrent, self.rectangles[old]) for old in older))
            areOK = n.logical_and(areOK, n.logical_not(inOlder))
            currentOK = n.logical_and(currentOK, n.logical_not(currentInOlder))
          if self.conf.disableNANsForICP:
            areOK = n.logical_and(areOK, nonanidx.ravel())
          points = xyzidx[areOK,:]
          currentPoints = xyzcurrent[currentOK,:]
          if (points.size==0) or (currentPoints.size==0):
            toAddNew[idxcurrent] = toAddOld[idxcurrent] = n.empty((0,3))
            if self.conf.debugXYZ:
              toAddOldP[idxcurrent] = toAddNew[idxcurrent]
            continue
          oldZ   = intp.griddata(currentPoints[:,0:2], currentPoints[:,2], points[:,0:2],
                                 method=C.interpd[C.INTERPMETHOD.GRIDDATA][self.conf.interpOrder], fill_value=n.nan, rescale=False)
          subOK = n.logical_not(n.isnan(oldZ))
          toAddNew[idxcurrent] = points[subOK,:]
          toAddOld[idxcurrent] = toAddNew[idxcurrent].copy()
          toAddOld[idxcurrent][:,2] = oldZ[subOK]
          if self.conf.debugXYZ:
            toAddOldP[idxcurrent] = currentPoints[nonancurrent.ravel()[currentOK],:]
        oldPoints = n.vstack(toAddOld)
        newPoints = n.vstack(toAddNew)
        if self.conf.debugXYZ:
          w.writePLYFileColorList(self.conf.debugSavePath+'debug.ICP1.%03d.ply' % idx, toAddNew+toAddOld+toAddOldP, it.cycle(C.COLORS))
    else:
      #THIS MAY BE MORE EFFICIENT USING A GOOD CLIPPING LIBRARY INSTEAD OF BOUNDING BOXES,
      #BUT DOING POLYGONS CLIPPING CAN GET HAIRY VERY QUICKLY (INTERSECTIONS AND DIFFERENCES
      #CAN BECOME SETS OF POLYGONS, AND CLIPPING LIBRARIES TEND TO REFUSE TO WORK
      #AND BEGIN THROWING EXCEPTIONS IF THERE ARE TOO SMALL EDGES. CGAL LIBRARY MAY BE ABLE
      #TO GET AROUND THE LATTER ISSUE, BUT IS INCREDIBLY CUMBERSOME TO USE
      #now, we have to tread carefully, we do not want to add overlaped parts
      #more than once, because that would introduce accumulative errors in
      #the registering process (and may wreack havoc in the ICP algorithm).
      #We use the following alogirthm:
      #  - we consider already registered clouds with overlaping BB, from older
      #    to newer
      #  - for each such cloud, we consider its points inside the BB of the 
      #    newest cloud, but not inside the BB of any older cloud
      #  - we also consider all points in the newest cloud overlapping with any
      #    of the BBs of the older clouds
      #IN THIS METHOD, WE DIRECTLY USE POINTS FROM PREVIOUSLY PROCESSED HEIGHTMAPS.
      toAdd = [None]*len(overlaping)
#        toAddN = [None]*len(overlaping)
      for idxcurrent in xrange(len(overlaping)):
        current  = self.processOrder[overlaping[idxcurrent]]
        older    =  n.array([self.processOrder[o] for o in overlaping[:idxcurrent]])
        
        xyzcurrent   =               self.loadVar('xyzs', current)          
        nonancurrent = n.logical_not(self.loadVar('nans', current))
        areOK = r.testPoinstInsidePolygon(xyzcurrent, self.rectangles[idx])
        
        if idxcurrent>0:
          inOlder  = reduce(n.logical_or, (r.testPoinstInsidePolygon(xyzcurrent, self.rectangles[old]) for old in older))
          areOK    = n.logical_and(areOK, n.logical_not(inOlder))
        if self.conf.disableNANsForICP:
          areOK = n.logical_and(areOK, nonancurrent.ravel())
        #added for this point cloud: all points overlaping with the new cloud, but not with previous clouds
        toAdd[idxcurrent] = xyzcurrent[areOK]
#          toAddN[idxcurrent] = normals[current][areOK]
      #set of points in already registered clouds which overlap with the new cloud,
      #to be submitted to the ICP algorithm
      oldPoints = n.vstack(toAdd)
#        oldNormals = n.vstack(toAddN)
      #set of points in the new cloud which overlap with any of the already registered clouds,
      #to be submitted to the ICP algorithm
      newPoints = reduce(n.logical_or, (r.testPoinstInsidePolygon(xyzidx, self.rectangles[ov]) for ov in overlaping))
      if self.conf.disableNANsForICP:
        newPoints = n.logical_and(newPoints, nonanidx.ravel())
#        normals[idx] = r.normalsForMatrixXYZ(xyzs[idx], imgs[idx].shape)
#        newNormals = normals[idx][newPoints]
      newPoints = xyzidx[newPoints]
      if self.conf.debugXYZ:
        w.writePLYFileColorList(self.conf.debugSavePath+'debug.ICP1.%03d.ply' % idx, [newPoints]+toAdd, it.cycle(C.COLORS))
    
    #this code is for taking only points from PREV, instead of all previous heightmaps
#      newPoints = r.testPoinstInsidePolygon(xyzs[idx], rectangles[prev])
#      newPoints = xyzs[idx][newPoints]
#      
#      oldPoints = r.testPoinstInsidePolygon(xyzs[prev], rectangles[idx])
#      oldPoints = xyzs[prev][oldPoints]
#      toAdd = [oldPoints]
    self.saveVar('newPointss', idx, newPoints)
    self.saveVar('oldPointss', idx, oldPoints)
    self.saveVar('state', 0, C.STATE.SECONDPHASE)
    return (True, '')

  def secondPhase(self):
    """state machine method: compute the fine registration"""
    i = self.state[1]
    idx = self.processOrder[i]
    
    oldPoints     = self.loadVar('oldPointss',     idx)
    newPoints     = self.loadVar('newPointss',     idx)

    maxstep = n.max(self.imgsteps[idx])
    
    self.log('applying ICP for heightmap %d\n'  % (idx))

    #if idx==2: return (False, 'test')
    result = self.applyICP(maxstep, oldPoints, newPoints)#, oldNormals, newNormals)
    #return (maxstep, oldPoints, newPoints)
    if not result[0]: return (False, 'Failed in the ICP for heightmap %d: %s'  % (idx, result[1]))
    transfICP = result[1]
    
    if ((self.conf.firstPhase == C.FIRSTPHASE.PHASECORRELATION) and 
       self.conf.PhaseCorrSubpixel and
       (self.conf.PhaseCorrRestoreSubPixelMethod == C.SUBPIXELMODE.RESTORE)):
        #THERE MAY BE SEVERAL WAYS OF DOING THIS:
        #    -AFTER THE ROTATION, RE-TRANSLATE IN XY ALL POINTS BY A FRACTIONAL AMOUNT, COMPUTED BY TAKING THE MEAN OF ALL XY DISPLACEMENTS
        #    -HOWEVER, A SIMPLER APPROACH: BETTING THAT THE XY DISPLACEMENT ON THE ROTATION MATRIX IS JUST FOR  SNAPPING THE TWO GRIDS, nullify it
        transfICP[3,0:2] = 0

    if self.conf.debugXYZ: w.writePLYFileColorList(self.conf.debugSavePath+'debug.ICP2.%03d.ply' % idx, [r.doTransformT(newPoints, transfICP), oldPoints], [[255,0,0],[0,255,0]])
    
    #apply the transform to the point cloud (not the accumulated matrix, only the last one)
    xyzidx = self.loadVar('xyzs', idx)
    xyzidx = r.doTransformT(xyzidx,       transfICP)
    #ACCORDING TO THE REASONING EXPOSED IN THE FIRST COMMENT AFTER THE LINE
    #CONTAINING "imgprev = self.loadVar('imgs', prev)" IN def firstPhase(self),
    #THE FOLLOWING LINE SHOULD BE COMMENTED OUT. HOWEVER, THE CODE SEEMS TO
    #BREAK IF IT IS COMMENTED OUT, SO WE KEEP IT
#    self.saveVar('imgs',           idx, xyzidx[:,2].reshape(self.imgshapes[idx]))
    self.saveVar('xyzs',           idx, xyzidx)
    self.saveVar('rectangles',     idx, r.doTransformT(self.rectangles[idx], transfICP))
    self.saveVar('BBs',            idx, n.vstack((xyzidx[:,0:2].min(axis=0), xyzidx[:,0:2].max(axis=0))).flatten())
    self.saveVar('accumRotations', idx, n.dot(transfICP, self.accumRotations[idx])) #accumulate the transformation, it may be needed in the future
    self.saveVar('processed',      idx, True) #flag the heightmap as processed
#      normals[idx] = r.normalsForMatrixXYZ(xyzs[idx], imgs[idx].shape)
    i+=1
    if i==self.num:
#      #do postprocessing, and end
#      zfac = self.conf.zfac
#      if zfac is not None:
#        for i in xrange(self.num):
#          xyz = self.loadVar('xyzs', i)
#          xyz[:,2] /= zfac
#          xyz=None
#          self.saveVar('xyzs', i, xyz)
#          img = self.loadVar('imgs', i)
#          img /= zfac
#          self.saveVar('xyzs', i, img)
##          self.saveVar('imgs', i, xyz[:,2].reshape(self.imgshapes[i]))
      self.saveVar('finished',     -1,   True)
      self.saveVar('state',        -1,   [C.STATE.END, 0])
    else:
      #process next image
      self.saveVar('state',        -1,  [C.STATE.FIRSTPHASE, i])
    return (True, '')
    

  def doRANSACRotation(self, i, idx, prev, imgidx, imgprev):
    """state machine helper method: compute a rotation (plain SVD or RANSAC)"""
    
    (kpdesprev, kp1xyz, imgprev) = self.lazyComputeKeypoints(imgprev, prev)
    (kpdesidx,  kp2xyz, imgidx)  = self.lazyComputeKeypoints(imgidx,  idx)
    
    self.log('%d, first keypoint matching %d to %d...\n' % (i, idx, prev))
    #first, use RANSAC to register it approximately with the previous image
#    kpdesprev = self.loadVar('kpdess', prev)
#    kpdesidx  = self.loadVar('kpdess', idx)
    filteredMatches = self.matchKeypoints(kpdesprev, kpdesidx)
    if filteredMatches.shape[0]<3:
      return (False, 'too few matches between keypoints in heightmaps %d and %d: %d' % (idx, prev, filteredMatches.shape[0]))
#      showMatches(imgs[idx], imgs[prev], kpxyzs[idx][:,0:2]/imgsteps[idx], kpxyzs[prev][:,0:2]/imgsteps[prev])
#      return (False, 'test')
#    kp1xyz = self.loadVar('kpxyzs', prev)
#    kp2xyz = self.loadVar('kpxyzs', idx)
    kp1xyz = kp1xyz[filteredMatches[:,0]]
    kp2xyz = kp2xyz[filteredMatches[:,1]]
    maxstep = n.max([self.imgsteps[prev], self.imgsteps[idx]])
#      if self.conf.debugXYZ:
#        n.savez(self.conf.debugSavePath+'debug.FIRSTPHASE.%03d.npz' % idx, img1=self.loadVar('imgs', idx), img2=self.loadVar('imgs', prev), kp1xyz=kp1xyz, kp2xyz=kp2xyz)
    self.log('applying RANSAC %d to %d...\n' % (idx, prev))
    resultR  = self.applyRANSAC(maxstep, kp1xyz, kp2xyz)
    if not resultR[0]:
      return (False, 'Failed in the RANSAC for heightmap %d with heightmap %d: %s' % (idx, prev, resultR[1]))
    bestRot3D = resultR[1]
    if bestRot3D is None:
      return (True, None)
    ransac_matches = resultR[2]
    #take care: XY axes are switched in opencv coordinates, so we need to multiply in both sides to undo that transformation
    bestRot3D = n.dot(C.RANSAC_Perm, n.dot(bestRot3D, C.RANSAC_Perm))
    return (True, bestRot3D, len(ransac_matches), imgidx, imgprev)

  def doPhaseCorrelation(self, numPeaks, imgidx, imgprev, i, idx, prev):
    """state machine helper method: phase correlation for first-phase registration"""
    self.log('first phase %d: phase correlation %d to %d with %d peaks...\n' % (i, idx, prev, numPeaks))
    #IT SEEMS INTUITIVE TO IDENTIFY img==xyz[:,2].reshape(img), AND IT SEEMS
    #LIKE THE RIGHT THING TO DO. HOWEVER, DOING IT SEEMS TO SCREW UP THE
    #PHASE CORRELATION ALGORITHM, SOMEHOW. SO, WE KEEP img AND xyz[:,2] AS
    #DISTINCT DATASETS: img IS THE HEIGHTMAP PRE-ROTATION, xyz[:,2] IS THE
    #DATASET POST-ROTATION
#      xyzidx  = r.image2XYZ(imgidx, self.imgsteps[idx])
#      imgidx  = r.doTransformT(xyzidx,  self.accumRotations[prev])[:,2].reshape(imgidx.shape)
#      xyzidx = None
    #DO NOT SUBSTRACT THE MEAN OR THE MEDIAN:
    #    THE IMAGES ARE ALREADY WELL ALIGNED AS THEY COME OUT OF THE CONFOCAL,
    #    SUBSTRACTING ANYTHING WILL MAKE THEM OFF-BALANCE, GREATLY INCREASING
    #    NOISE IN CROSS CORRELATIONS OF NARROW OVERLAPINGS (WHICH MAKES ALL 
    #    THE PROCEDURE TO FAIL)
#        imgs[idx[ -= n.median(imgs[idx])
    resultR = self.registerPhaseCorrelation(imgprev, imgidx, self.imgsteps[idx], numPeaks)
    if not resultR[0]:
      return (False, 'Failed in the PHASE CORRELATION for heightmap %d with heightmap %d: %s' % (idx, prev, resultR[1]))
#    bestRot3D = resultR[1]
    #XYShift = bestRot3D[3,0:2]
    self.log('correlation coefficient: %s\n' % (resultR[2]))
    self.log('peak value             : %s\n' % (resultR[3]))
    return resultR
    
  def registerPhaseCorrelation(self, img1, img2, imgstep, numPeaks):
    """state machine helper method: execute phase correlation (this method should be
    integrated into doPhaseCorrelation, it is just a legacy function from a 
    previous incarnation of the registration workflow"""
    #whitening: correlation algorithms (and many data analysis algorithms 
    #for that matter) work best when data is highly uncorrelated with itself
    #i,e,, it is like white noise. Unfortunately, image or topography data is
    #usually highly correlated (and right to be so), While proper whitening 
    #is a costly operation involving the computation of the eigenvalues of the
    #image (see http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/ ),
    #simple derivatives can also do the trick. Here, we use a laplacian filter
    #as a simple whitening method.
    #see also http://dsp.stackexchange.com/questions/8875/phase-correlation-poor-performance-on-noisy-blurred-images
    if self.conf.PhaseCorrWhitening:
      img1 = scifil.laplace(img1)
      img2 = scifil.laplace(img2)
    shift,corrcoef,peakval  = imreg.translationTestPeaks(img1, img2,
                                                         numPeaks=numPeaks,
                                                         subpixel=self.conf.PhaseCorrSubpixel,
                                                         scaleSubPixel=self.conf.PhaseCorrScale,
                                                         minratio=self.conf.PhaseCorrMinratio)
#    print "After translationTestPeaks: shift:"
#    print shift
#    if corrcoef < self.conf.PhaseCorrMinCoef:
#      return (False, 'In Phase correlation: correlation coefficient is too weak')
#    if peakval < self.conf.PhaseCorrMinPeakVal:
#      return (False, 'In Phase correlation: peak value is too weak')
    translation      = n.identity(4)
    translation[3,0] = shift[1]*imgstep[0]
    translation[3,1] = shift[0]*imgstep[1]
    return (True, translation, corrcoef,peakval)
  
  def getKeypoints(self, img, imgstep, minv, maxv):
    """state machine helper method: for an image, return  a tuple with
    opencv's keypoints and descriptors, and the XYZ positions of the keypoints"""
    #scale the image in 0..1 in a global reference frame
    imgu8 = ((img-minv)/(maxv-minv)*255)
    if self.conf.discretizeKeypointDetection:
      imgu8 = imgu8.astype(n.uint8)
    #uses opencv's python API to extract keypoints from an image. NaN values
    #are supposed to be absent.
#    kp = self.conf.detector.detect(imgu8)
#    (kp, des) = self.conf.extractor.compute(imgu8,kp)
    sift = cv2.SIFT(contrastThreshold=0)
    kp, des = sift.detectAndCompute(imgu8,None)
    #positions of keypoints in XY,Z
    kpxy = n.array([k.pt[::-1] for k in kp])
    kpz  = intpi.map_coordinates(img, kpxy.T, order=C.interpd[C.INTERPMETHOD.MAP_COORDINATES][self.conf.interpOrder])
    #scale SIFT coordinates to XYZ coordinates (so the RANSAC rotation can be directly applied to the original coordinates)
    kpxy *= [imgstep]
    #IDEALLY, THIS STEP SHOULD HANDLE NAN VALUES WITH A MASK, AND IGNORE NAN
    #VALUES FOR INTERPOLATION PURPOSES. SINCE SCIPY DOES NOT SUPPORT THIS,
    #WE JUST WORK WITH OUR ADHOC APPROACH OF SETTING NANs TO A CONSTANT.
    #THIS ONLY WORKS IF THE DISTRIBUTION OF NANS IS IDENTICAL FOR BOTH IMAGES
    kpxyz = n.column_stack((kpxy, kpz))
    return (des, kpxyz)
    
  def matchKeypoints(self, des1, des2):
    """state machine helper method: Return pairs of matched keypoints between two sets of keypoints"""
    matches = self.conf.matcher.knnMatch(des1,des2,k=2)
    threshold = self.conf.matcherThreshold
    filteredMatches = n.array([[a.queryIdx, a.trainIdx] for (a,b) in matches if (a.distance < threshold*b.distance)])
    return filteredMatches

  def getRotationForRANSACPlane(self, maxstep, xyz, idx, justSVD=False):
    """helper method to compute plain SVD / RANSAC plane rotations, this is used in the
    state machine and can also be used for the registration analysis workflow"""
    #first, apply RANSAC to fit a plane
    t = self.conf.RANSAC_fitplane_tfun(maxstep)
    d = self.conf.RANSAC_fitplane_planeratio*self.conf.RANSAC_fitplane_saferatio
    fun1 = r.fitPlaneSVD
    fun2 = r.errorPointsToPlane
    #self.conf.RANSAC_fitplane_debug=True
    if justSVD:
      bestplane = r.fitPlaneSVD(xyz)
    else:
      try:
        result = ransac.ransac(xyz, fun1,
             fun2, xyz.shape[1]+1, self.conf.RANSAC_fitplane_k, t, d, 
             self.conf.RANSAC_fitplane_debug, True)
      except ValueError as ev:
        return (False, 'RANSAC fitplane error: '+ev.message)
      if result is None:
        self.log('\n######################################\nWARNING: It was not possible to fit a plane to the heightmap.\nEither the heighmap was too rugged to fit a plane, or the parameters for the RANSAC algorithm were too tight\nA plane will be fit to the whole heightmap, but this will degrade the "planarness" of the heightmaps,\nand phase correlation, if used, may not work or lead to deceptive results.\n######################################\n')
        bestplane = r.fitPlaneSVD(xyz)
      else:
        bestplane, ransac_matches = result
    
#    bestplane = r.fitPlaneSVD(xyz)
    
    if self.conf.debugXYZ and (idx>=0):
      minx,miny,maxx,maxy = n.vstack((xyz[:,0:2].min(axis=0), xyz[:,0:2].max(axis=0))).flatten()
      rectangle = n.array([[minx,miny,0], [maxx,miny,0], [maxx,maxy,0], [minx,maxy,0]])
      for i in xrange(len(rectangle)):
        rectangle[i,2] = -(rectangle[i,0]*bestplane[0] + rectangle[i,1]*bestplane[1] + bestplane[3])/bestplane[2]
      w.writePLYPointsAndPolygons(self.conf.debugSavePath+("debug.FITTEDPLANE.%03d.ply"%idx), n.vstack((rectangle, xyz)), [range(4)])
                                  
#    dists = r.distancePointsToPlane(xyz, bestplane)
#    r.showPointsAndPlane(xyz, 100, bestplane, values=dists, vmin=-10, vmax=10)
    
    #get a rotation matrix to rotate the fit plane onto the XY plane
    planeNormal = bestplane[:3]
    if planeNormal[2]<0:
      #if the plane is upside down, correct it!
      planeNormal = -planeNormal
    planeNormal /= n.linalg.norm(planeNormal)
#    self.log('best plane: %s\n     normal: %s\n' % (str(bestplane), str(bestplane[0:3]/n.linalg.norm(bestplane[0:3]))))
    self.log('planeNormal: %s\n' % str(planeNormal))
    R = r.rotateVectorToVector(planeNormal, n.array([0,0,1]))
    return (True, R)
  
  def makePlanar(self, maxstep, img, imgstep, xyzRANSAC, xyzROTATE, idx=0):
    """helper method to perform plane rotations, this is used in the
    state machine and can also be used for the registration analysis workflow"""
    if self.conf.rotateMode==C.PLANEROT.NOROTATION:
      return (True, '')
#    if xyzROTATE is None:
#      xyzROTATE = r.image2XYZ(img, imgstep)
#    if xyzRANSAC is None:
#      xyzRANSAC = xyzROTATE
    if self.conf.rotateMode==C.PLANEROT.ALLINDEPENDENT:
      self.log('Computing plane fitting for %d\n'%idx)
      result = self.getRotationForRANSACPlane(maxstep,  xyzRANSAC, idx, justSVD=(not self.conf.RANSAC_fitplane_enable))
      if not result[0]: return result
      R = result[1]    
      rotateJustZ(R, img, xyzROTATE)
      return (True, '')
    elif self.conf.rotateMode in [C.PLANEROT.JUSTFIRST, C.PLANEROT.ALLBYFIRST]:
      if idx==0:
        self.log('Computing plane fitting for %d\n'%idx)
        result = self.getRotationForRANSACPlane(maxstep,  xyzRANSAC, idx, justSVD=(not self.conf.RANSAC_fitplane_enable))
        if not result[0]: return result
        R = result[1]    
        self.firstRotation = R
        self.saveVar('firstRotation', -1, R)
        rotateJustZ(R, img, xyzROTATE)
      elif self.conf.rotateMode==C.PLANEROT.ALLBYFIRST:
        R = self.firstRotation
        rotateJustZ(R, img, xyzROTATE)
      return (True, '')
    else:
      return (False, 'Rotation mode not understood: '+str(self.conf.rotateMode))
        

    
  def applyRANSAC(self, maxstep, kp1xyz, kp2xyz):
    """state machine helper method: Find the approximate transformation
    between two sets of keypoints with the RANSAC algorithm, to rotate the
    2nd set to match the 1st set"""

    t = self.conf.RANSAC_tfun(maxstep)
    d = self.conf.RANSAC_dfun(kp1xyz.shape[0])

    kpallxyz = n.column_stack((kp2xyz, kp1xyz))
      
    if    self.conf.RANSAC_METHOD==C.RANSACMODE.ROTATION:
      fun1 = r.findTransformation_RANSAC
    elif self.conf.RANSAC_METHOD==C.RANSACMODE.DISPLACEMENT:
      fun1 = r.findTranslation_RANSAC
    fun2 = r.get_error_RANSAC
    try:
      result = ransac.ransac(kpallxyz, fun1,
           fun2, kpallxyz.shape[1]/2, self.conf.RANSAC_k, t, d, 
           self.conf.RANSAC_debug, True)
    except ValueError as ev:
      return (False, 'RANSAC error: '+ev.message)
    if result is None:
      return (True, None)
    else:
      bestRot3D, ransac_matches = result
    return (True, bestRot3D, ransac_matches)
  
  
  def applyICP(self, maxstep, kp1xyz, kp2xyz):#, oldNormals, newNormals):
    """state machine helper method: apply ICP for fine registration"""
    
    if self.conf.ICP_Keypoints and maxstep is not None:
      #we assume that RANSAC will have put the corresponding keypoints at most
      #3 pixels apart. We need a fairly low cutoff because ICP is not like RANSAC: 
      #it will take into account all keypoints, greatly distorting the results 
      #if there are errors in the keypoint matchings
      dists = n.linalg.norm(kp1xyz-kp2xyz, axis=1)
      cutoff = self.conf.ICP_cutofffun(maxstep)
      ok = dists<cutoff
      kp1xyz = kp1xyz[ok,:]
      kp2xyz = kp2xyz[ok,:]
    
    if ISWINDOWS:
      #In Windows, CPython 2.x is compiled with MVS2008, but PCL has C++11 code,
      #which cannot be compiled with MVS2008. Either I port everything to CPython
      #3.x, or I compile everything from scratch in a C++11 compiler (possibly to fail
      #to compile the PCL-python bindings). Salomonic solution: pack in a standalone
      #command-line tool the bare essentials I need from PCL. The unfortunate
      #side effect is that some additional features, such as self.adhocSmoothing()
      #methods rely on PCL functionality unimplemented in Windows, so they can
      #be used only in Linux
      #next TODO: add options to the ICPTool to be more configurable in Windows, just as in Linux
      converged, fitness, transf, timeSpent = useICPTool(self.conf.ICP_maxiter, kp1xyz, kp2xyz)
    else:
      icpParams = self.conf.icpParamsfun(maxstep)
    
      source = pcl.PointCloud(kp1xyz.astype(n.float32))
      target = pcl.PointCloud(kp2xyz.astype(n.float32))
      converged, transf, estimate, fitness = self.conf.registerMethod(source, target, self.conf.ICP_maxiter, self.conf.registerOverlapRatio, icpParams)
      
    if converged:
      return (True, n.linalg.inv(transf).T)
    else:
      return (False, 'ICP was not able to converge')
  
  def interpolateNANs(self, img, imgstep, nans, xyz=None, nanval=0.0):
    #METHOD  = 0 #fill outsiders with nanval, assuming that either (a) values will not be too different from surroundings, or (b) this will not happen
    METHOD  = 1 #fill outsiders with nans, then wipe them out interpolating with griddata/nearest
    #METHOD = 2 #use RBF, seems to be practical only for very tiny samples (constructs distance matrix of samples!!!)
    if xyz is None:
      xyz   = r.image2XYZ(img, imgstep)
    nonans  = n.logical_not(nans)
    xyzok   = xyz[nonans.ravel(),:]
    xynook  = xyz[nans.ravel(),0:2]
    if METHOD<=1:
      if   METHOD==0:
        fillval = nanval
      elif METHOD==1:
        fillval = n.nan
      result  = intp.griddata(
                                 xyzok[:,0:2], xyzok[:,2], xynook, 
                                 method=C.interpd[C.INTERPMETHOD.GRIDDATA][self.conf.interpOrder], 
                                 fill_value=fillval,
                                 rescale=False)
      xyz[nans.ravel(),2] = result
      img[nans] = result
      if METHOD==1:
        nans    = n.isnan(img)
        nonans  = n.logical_not(nans)
        xyzok   = xyz[nonans.ravel(),:]
        xynook  = xyz[nans.ravel(),0:2]
        result  = intp.griddata(
                                   xyzok[:,0:2], xyzok[:,2], xynook, 
                                   method='nearest', 
                                   rescale=False)
        xyz[nans.ravel(),2] = result
        img[nans] = result
    elif METHOD==2: #requires too much memory (RBF computes a distance matrix for all datapoints, phew)
      f = intp.Rbf(xyzok[:,0], xyzok[:,1], xyzok[:,2])
      xyz[nans.ravel(), 2] = f(xynook[:,1], xynook[:,2])
      img[nans] = xyz[nans.ravel(), 2]
#      #MAYBE A BETTER ALTERNATIVE: instead of nanval, fill with griddata set to 'nearest'
#      img[nans] = nanval
#      xyz = xyz[n.logical_not(nans),:]
    return xyz
  
  def smoothByOldest(self, BB=None, removeNANs=True, skipEmpty=True):
    """For each registered heightmap, remove the points in regions which are overlapping
    with heightmaps processed earlier in the pipeline, as specified by self.processOrder.
    If removeNANs, also removes NAN points. Returns a generator with the resulting heightmaps"""
    if not self.finished:
      raise Exception('Regitration has not been done/finished!!!!')
#    if BB is None:
#      BB        = n.array([self.BBs[:,0].min(), self.BBs[:,1].min(), self.BBs[:,2].max(), self.BBs[:,3].max()])
    for i in xrange(self.num):
      idx      = self.processOrder[i]
      if BB is not None:
        BBs = n.vstack((BB, self.BBs[idx]))
        overlaping = r.getOverlapingBoundingBoxes(BBs, 0, 1)
        if overlaping.size!=1:
          raise Exception('This means that the code in getOverlapingBoundingBoxes() has to be carefully vectorized')
        if not overlaping:
          if not skipEmpty:
            yield n.zeros((0,3))
          continue
      newxyz   = self.loadVar('xyzs', idx)
      if removeNANs and self.nanflags[idx]:
        nan    = self.loadVar('nans', idx)
        newxyz = newxyz[n.logical_not(nan.ravel()),:]
      if (i>0) and (newxyz.size>0):
        older = self.processOrder[:i]
        #find overlaping bounding boxes with previously registered clouds
        overlaping  = r.getOverlapingBoundingBoxes(self.BBs, idx, older)
        older = [old for old,ovl in zip(older, overlaping) if ovl]
        toRemove = n.zeros((newxyz.shape[0],),dtype=n.bool)
        for old in older:
          toRemove = n.logical_or(toRemove, r.testPoinstInsidePolygon(newxyz, self.rectangles[old]))
        newxyz = newxyz[n.logical_not(toRemove),:]
      if (BB is not None) and (newxyz.size>0):
        newxyz = newxyz[n.logical_and(n.logical_and(newxyz[:,0]>=BB[0], newxyz[:,1]>=BB[1]),
                                      n.logical_and(newxyz[:,0]<=BB[2], newxyz[:,1]<=BB[3]) ),
                        :]
      if skipEmpty and (newxyz.size==0):
        continue
      yield newxyz


  def adhocSmoothingAllInMemory(self, xyzs, heightmapSteps):
    """LEGACY METHOD: Use a 2-dimensional K-d tree to get each point's neighbours. Neighbours
    whose XY distance is smaller than a fraction of the step of the original
    heightmaps are supposed to correspond to the same point, so their Z values
    are averaged. This hack works only if the model is approximately well oriented,
    but that might not be the case..."""
    print 'Adhoc smoothing...'
    #BEFORE DOING ALL THIS WE SHOULD MAKE SURE THAT THE Z DIRECTION IS THE SAME AS IN THE MICROSCOPE
    #HOWEVER: IF THE COORDINATE SYSTEM IS THE SAME AS IN THE FIRST PROCESSED HEIGHTMAP, IT SHOULD BE!
    step       = n.unique(heightmapSteps)
    if step.size!=1:
      return (False, 'assertion failed: the step must be the same in X and in Y for each heightmap, and the same for all heightmaps')
    #the threshold is squared, because the algorithm returns squared distances    
    threshold  = self.conf.NeighDist(step[0])**2
    #combine all clouds into a gigantic one
    sizeheighs = [x.shape[0] for x in xyzs]
    numPoints  = n.sum(sizeheighs)
    xyzs32     = (x[:,0:2].astype(n.float32) for x in xyzs) #lazy generation
    bigcloud   = pcl.PointCloud2()
    bigcloud.from_arrays(xyzs32, numPoints)
    tree       = pcl.KdTreeFLANN2(bigcloud)
    numneighs  = self.conf.KNeighs
    k_indices, k_sqr_distances = tree.nearest_k_search_for_cloud(bigcloud, numneighs) 
    #make an array of labels: each point has a label that identifies its parent heightmap
    numheighs  = len(xyzs)
    if   numheighs<=255:
      typ      = n.uint8
    elif numheighs<=(255*255):
      typ      = n.uint16
    else:
      typ      = n.uint32
    #labels     = n.zeros((numPoints,), dtype=typ)
    labels     = n.repeat(n.arange(numheighs, dtype=typ), sizeheighs)
    zs         = n.zeros((numPoints,))
    cumsizes   = n.cumsum(sizeheighs)
    rmin       = rmin = n.concatenate((n.array([0]), cumsizes[:-1]))
    rmax       = cumsizes
    for k in xrange(numheighs):
      #labels[rmin[k]:rmax[k]] = k
      zs[rmin[k]:rmax[k]]     = xyzs[k][:,2]
    #mask to take neighbours into consideration: not in the same heightmap and nearer than the threshold
    neighmask  = n.logical_and(labels[k_indices]!=labels.reshape(numPoints,1),
                               k_sqr_distances<threshold)
    zs         = n.take(zs, k_indices) # faster than indexing
    #zs         = zs[k_indices]
    #unmask the first row because we are going to take the mean between them and their neighbours
    neighmask[:,0] = True
    #put nans in values from points which we are not going to process
    zs[n.logical_not(neighmask)] = n.nan
    if self.conf.returnSmoothStats:
      means  = scipy.stats.nanmean(zs,axis=1)
      stats  = [means, scipy.stats.nanstd(zs,axis=1), n.nanmin(zs,axis=1), n.nanmax(zs,axis=1), n.sum(n.logical_not(n.isnan(zs)), axis=1), labels]
      
    if self.conf.heuristicTrimSmoothedPoints:
      #for each point, we only keep it if its index is the minimum of its neighbours
      #This is a very quick-and-dirty solution
      toKeep = n.nanargmin(zs, axis=1)==0
    #for each candidate, group the values using the configurable function
    #(nanmean, nanmedian, something like that, the thing is that if the only
    #unmasked value is the point itself, it should return the Z value of the point itself)
    if self.conf.neighGroupFun==C.GROUPWITH.MEAN:
      if self.conf.returnSmoothStats:
        zs     = means
        means  = None
      else:
        zs     = scipy.stats.nanmean(zs,axis=1)
    elif self.conf.neighGroupFun==C.GROUPWITH.MEDIAN:
      zs       = scipy.stats.nanmedian(zs,axis=1)
    else:
      return (False, 'neighGroupFun option not recognized: '+str(self.conf.neighGroupFun))
    #replace z values for each candidate
    xyzs       = n.vstack(xyzs)
    #xyzs[candidates,2]  = zs[candidates]
    xyzs[:,2]  = zs
    if self.conf.heuristicTrimSmoothedPoints:
      xyzs = xyzs[toKeep]
      if self.conf.returnSmoothStats:
        stats = [stat[toKeep] for stat in stats]
    if not self.conf.returnSmoothStats:
      stats = None
    return (True, ResultSmooth(xyzS=xyzs, stats=stats))
      

  def testWorkflow(self, conf, heightmaps, processSpecification=None):
    "test the workflow with lots of debugging output"""
    if not self.inMemory:
      raise Exception('For this simple procedure, we must use inMemory!!!')
#    resultReg = self.registerManySamplesSequential(heightmaps, processSpecification)
#    if not resultReg[0]: return resultReg
#    heightmapsSteps = n.array([x.step for x in heightmaps])
    num = len(heightmaps)
    result = self.computeRegistration(conf=conf, num=num, heightmaps=heightmaps, loader=unpackHeightmap, processSpecification=None)
    if result[0]:
      w.writePLYFileColorList(self.conf.debugSavePath+'test.original.ply', (self.loadVar('xyzs', idx)[n.logical_not(self.loadVar('nans', idx).ravel()),:] for idx in xrange(self.num)), it.cycle(C.COLORS), sizes=self.numnonans)
      #we do not know beforehand the total number of points returned by self.smoothByOldest(), so we have to hold them in memory
      allxyzs = list(self.smoothByOldest(removeNANs=True))
      w.writePLYFileColorList(self.conf.debugSavePath+'test.older.ply', allxyzs, it.cycle(C.COLORS))
      allxyzs=None
      return self.showAdhocSmoothing(self.xyzs, self.nans)
    return result
    
  def showAdhocSmoothing(self, xyzs, nans):
    """legacy method to see if the registering algorithm is working. Really, this is part of testWorkflow()"""
    heightmapsSteps = self.imgsteps.copy()
    resultSmo = self.adhocSmoothingAllInMemory(xyzs, heightmapsSteps)
    
    w.writePLYFileColorList(self.conf.debugSavePath+'test.smooth.ply', [resultSmo[1].xyzS], [[0, 0, 0]])
    allnonans = n.logical_not(n.vstack(nans))
    showSmoothingErrorsPLY(self.conf.debugSavePath+'test.errors.ply', n.vstack(xyzs), resultSmo[1].stats, allnonans, [255, 0, 255])
    
#    showSmoothingErrorsIMG(1000, resultSmo[1].xyzS, resultSmo[1].stats, [False, True], allnonans)
    
    figures=[manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    for idx, f in enumerate(figures):
      mng = f.canvas.manager
      #mng = plt.get_current_fig_manager()
      mng.resize(*mng.window.maxsize())
      f.savefig(self.conf.debugSavePath+'test.discrepancies.0%d.png' % idx)
      plt.close(f)
    
    zsub=10
    resultMesh = createMeshFromPointCloud(resultSmo[1].xyzS, zsub)    
    if not resultMesh[0]:
      resultMesh.append(xyzs)
      resultMesh.append(nans)
      resultMesh.append(resultSmo)
      return resultMesh
    #seemingly, meshlab does not handle PLY meshes with colored vertices
#    stats = result[2]
#    labels = stats[-1]
#    colors = n.array(colors)
#    colors = colors[labels]
    w.writePLYMesh(self.conf.debugSavePath+'test.mesh.ply', resultMesh[1].points, resultMesh[1].triangles)#, colors)

    return (True, ResultAll(xyzs=xyzs, nans=nans, xyzS=resultSmo[1].xyzS, stats=resultSmo[1].stats, mesh=resultMesh[1]))

  def createMeshFromRegisteredPoints(self, BB=None, filename=None, zsub=1):
    """method for registration analysis workflows: create a mesh from the
    registered points, taking only the oldest points in each region"""
    if not self.finished:
      raise Exception('Regitration has not been done/finished!!!!')
      
    if BB is None:
      BB        = n.array([self.BBs[:,0].min(), self.BBs[:,1].min(), self.BBs[:,2].max(), self.BBs[:,3].max()])
    
    filteredPoints = (x for x in self.smoothByOldest(BB=BB, removeNANs=False))
    resultMesh = createMeshFromPointCloud(n.vstack(filteredPoints), zsub)    
    
    if filename is None:
      return resultMesh
    else:
      w.writePLYMesh(filename, resultMesh[1].points, resultMesh[1].triangles)

  def createMeshFromImage(self, image, scale=1.0, filename=None, zsub=1):
    """method for registration analysis workflows: Create a mesh from an image.
    If a filename is specified, it is save in PLY format, otherwise, it is returned"""
    if hasattr(image, '__iter__'):
      image = image[0]
      
    resultMesh = createMeshFromPointCloud(r.image2XYZ(image, [scale, scale]), zsub)    
    
    if filename is None:
      return resultMesh
    else:
      w.writePLYMesh(filename, resultMesh[1].points, resultMesh[1].triangles)

  def showImage(self, image, uselog=False, vmin=None, vmax=None, colormap=cm.jet, nancolor=[1, 0, 1]):
    """method for registration analysis workflows:  Show an image using matplotlib:
                 -uselog: flag to use log scale (default True)
                 -vmin: cap the min value of the colored range (default None)
                 -vmax: cap the max value of the colored range (default None)
                 -colormap: colormap to use (default jet)
                 -nancolor: color for NAN values
    """
    if vmin is None:
      smin = ''
    else:
      smin = ' (capped to %f)' % vmin
    if vmax is None:
      smax = ''
    else:
      smax = ' (capped to %f)' % vmax
    tit = 'in Z: min %f%s, max %f%s' % (n.nanmin(image), smin, n.nanmax(image), smax)
    colormap.set_bad(color=nancolor)
    f = plt.figure()
    if uselog:
      plt.imshow(image.T, cmap=colormap, vmin=vmin, vmax=vmax, norm=mcl.LogNorm())
    else:
      plt.imshow(image.T, cmap=colormap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(tit)
    f.show()

  def saveImage(self, image, name, scaleMin=None, scaleMax=None, scaleTiff=False, useColor=True, bitDepth=16, uselog=False, vmin=None, vmax=None, colormap=cm.jet, nancolor=[1, 0, 1], scale=1.0):
    """method for registration analysis workflows: saves images. Possible formats are:
    
        .npy: numpy array forma
        .tif, .tiff: TIFF image format. if scaleTiff is True, scales the image from 0 to 1. 
        .png: PNG image format. If useColor is False, bitDepth can be 8 or 16 for this image type
        .ply: PLY point cloud format.
        
       For images not in the NPY format, the scale range is determined by scaleMin and/or scaleMax, if they are specified
       
       For PLY and PNG file formats, if useColor is True, the image is colored with the following parameters (similar to self.showImage()):
                 -uselog: flag to use log scale (default True)
                 -vmin: cap the min value of the colored range (default None)
                 -vmax: cap the max value of the colored range (default None)
                 -colormap: colormap to use (default jet)
                 -nancolor: color for NAN values
       For PLY the file format, scale is the step between pixels

    """

    ext         = op.splitext(name)[1].lower()
    
    istiff      = ext in ['.tiff', '.tif']
    isnpy       = ext=='.npy'
    ispng       = ext=='.png'
    isply       = ext=='.ply'
    
    if not (istiff or isnpy or ispng or isply):
      raise ValueError('Unknown image type (%s)!!!!' % str(ext))
      
    doScaling   = (not isnpy) and (not isply) and ((not istiff) or scaleTiff) and (ispng and (not useColor))
    doColor     = useColor and (ispng or isply)
    
    if doScaling:
      #scale image
      smin      = scaleMin
      smax      = scaleMax
      if smin is None:
        smin    = image.min()
      if smax is None:
        smax    = image.max()
      image     = (image-smin)/(smax-smin)
      if not istiff:
        #transform to uint8
        if   bitDepth==8:
          typ   = n.uint8
          mul   = 255
        elif bitDepth==16:
          typ   = n.uint16
          mul   = 65535
        else:
          raise ValueError('Incorrect bit depth!!!')
        image  *= mul
        image   = image.astype(typ)
        
    if doColor:
      colormap.set_bad(nancolor)
      if vmin is None:
        vmin = n.nanmin(image)
      if vmax is None:
        vmax = n.nanmax(image)
      if uselog:
        norm    = mcl.LogNorm
      else:
        norm    = mcl.Normalize
      scalarMap = cm.ScalarMappable(norm=norm(clip=True,vmin=vmin,vmax=vmax), cmap=colormap)
      zvals     = image
      image     = scalarMap.to_rgba(image, bytes=True)[:,:,0:3]
      image[n.isnan(zvals),:] = n.array(nancolor).reshape(1,3)
#      nans      = n.isnan(zvals)
#      for c in xrange(len(nancolor)):
#        image[nans,c] = nancolor[c]
        
    if istiff:
      tff.imsave(name, image)
    elif isnpy:
      n.save(name, image)
    elif isply:
      xyz = r.image2XYZ(zvals, [scale, scale])
      noNAN = n.logical_not(n.isnan(xyz[:,2]))
      #xyz[n.isnan(xyz[:,2]),2] = 0
      if doColor:
        w.writePLYFileWithColor(name, xyz[noNAN], image.reshape(-1, 3)[noNAN])
      else:
        w.writePLYFile(         name, xyz)
    else:
      scim.imsave(name, image)
      

  def getImageRotation(self, img, imgstep, method='RANSAC'):
    """method for registration analysis workflows: 
    
      Rotate an image with a given pixel step, using either a simple regression plane or
      a RANSAC fitting plane algorithm with the same settings as the registration algorithm. 
      This is intended to be used with images of type 'last' and 'avg' generated by self.getImages().
      
      the possible methods are 'RANSAC' (fit plane using a RANSAC algorithm)
      and 'SVD' (simple regression plane)
      
      WARNING: If the pixel step is the same as in the original heightmaps, everything is OK,
      and this function can be used safely. This happens when neither of the parameters
      scale, XPixels or YPixels is specified in self.getImages(). However, if any scaling is done,
      and the RANSAC method is used, please be aware that the parameters for the
      RANSAC_fit_plane algorithm (in particular, RANSAC_fitplane_k, RANSAC_fitplane_tfun,
      RANSAC_fitplane_planeratio, and RANSAC_fitplane_saferatio) might have to be adjusted for the
      algorithm to work reliably
    """
    justSVD = method.lower()=='svd'
    if not justSVD and (method.lower()!='ransac'):
      self.log('NO ROTATION DONE, BECAUSE THE ROTATION METHOD WAS NOT UNDERSTOOD: '+str(method))
    nan = n.isnan(img)
    anynan = nan.any()
    if not anynan:
      nan = None
    xyz = r.image2XYZ(img, [imgstep, imgstep])
    if anynan:
      result = self.getRotationForRANSACPlane(imgstep, xyz[n.logical_not(nan).ravel(),:], -1, justSVD=justSVD)
    else:
      result = self.getRotationForRANSACPlane(imgstep, xyz, -1, justSVD=justSVD)
    if not result[0]:
      self.log('IT WAS NOT POSSIBLE TO ROTATE THE IMAGE!!!')
      return n.identity(4)
    R = result[1]
    return R
  
  def rotateImage(self, img, imgstep, R):
    """method for registration analysis workflows: apply image rotations"""
    xyz = r.image2XYZ(img, [imgstep, imgstep])
    #img[:,:] = n.dot(R, xyz.T)[2,:].reshape(img.shape)
    img[:,:] = n.dot(xyz, R.T)[:,2].reshape(img.shape)
#    if anynan:
#      img[nan] = n.nan
  
  def computeEncompassingBB(self, step=None):  
    """method for registration analysis workflows: compute the bounding box for all
    the registrated images, fixed to the grid of the first registered image (which,
    by the registering algorihtm, should be perfectly orthogonal and start at [0,0])"""
    if step is None:
      step = n.unique(self.imgsteps)
    if step.size!=1:
      raise Exception('This should never happen: the registered images have non-uniform pixel steps: '+str(step))
    BB        = n.array([self.BBs[:,0].min(), self.BBs[:,1].min(), self.BBs[:,2].max(), self.BBs[:,3].max()])
    BB        = n.round(BB/step)*step
    return BB
  
  def getImages(self, BB=None, scale=None, XPixels=None, YPixels=None, removeNANs=True, interpolate=False, outputs=['last']):
    """method for registration analysis workflows:
    
    Create images of the registered heightmaps with BB == [ minX, minY, maxX, maxY ].
        If BB is None, it is assumed to span all the images, fixed
        to the grid of the first registered image (which, by the registering
        algorihtm, should be perfectly orthogonal and start at [0,0]).
        
        The size of the output images is determined by:
          -if "scale" is not None, it should be a number between 0 and 1. The
           resulting image will have a size equal to the extent of the bounding
           box specified by BB (divided by the step of the source images), and
           scaled by this parameter.
          -otherwise, if XPixels is not None, the image will be scaled to have
           this number of pixels in the X direction
          -otherwise, if YPixels is not None, the image will be scaled to have
           this number of pixels in the Y direction
        
       If removeNANs is True, interpolated values where the input image had
       NANs are not used to construct the output
       
       If interpolate is True, values are interpolated to the image grid,
       otherwise, the nearest value is used.
       
       outputs is a list of possible image outputs:
          'min', 'max': minimum/maximum of pixels across all registered images
          'diff':       difference between maximum and minimum across all registered images
          'avg', 'std': average/standard deviation of pixels across all registered images
          'count':      number of heightmaps contributing to each pixel
          'first':      the value of the first heightmap is used
          'last':       the value of the last heightmap is used
        
       If I have time, it would be nice to add a 'blend' output: heightmaps are
       iteratively added to the final image using linear blending. To do that,
       for each heightmap, we should:
            -compute the average only of the heightmap to blend (to deal with scaling and round-off effects)
            -detect the overlapping region (which may not be a rectangle,
             and may be composed of several disconnected regions)
            -assign a weight to each pixel, going smoothly from 1 in the
             regions bordering the non-overlaping part, to 0 in the outer
             regions
            -in the overlapping region, do the weighted average between the
             heightmap and the previous image
       
       Returns two values:
          * a dictionary whose keys are the values in "outputs", and the values are the correpsonding images
          * an integer representing the pixel step
      """
    if not self.finished:
      raise Exception('Regitration has not been done/finished!!!!')
    if not hasattr(outputs, '__iter__'):
      outputs = [outputs]
    if type(outputs)!=list:
      outputs = list(outputs)
    step        = n.unique(self.imgsteps)
    if step.size!=1:
      raise Exception('This should never happen: the registered images have non-uniform pixel steps: '+str(step))
    if BB is None:
      BB        = self.computeEncompassingBB(step)
    else:
      BB        = n.array(BB)
    imgshape    = n.round(n.array([BB[2]-BB[0], BB[3]-BB[1]])/step)
    if scale is not None:
      outputstep = step*scale
      imgshape *= scale
    elif XPixels is not None:
      outputstep = step*XPixels/float(imgshape[0])
      imgshape *= XPixels/imgshape[0]
    elif YPixels is not None:
      outputstep = step*YPixels/float(imgshape[1])
      imgshape *= YPixels/imgshape[1]
    else:
      outputstep = step
    imgshape    = imgshape.astype(n.uint32)
    compDiff    = 'diff' in outputs
    compMin     = 'min' in outputs
    compMax     = 'max' in outputs
    compAvg     = 'avg' in outputs
    compStd     = 'std' in outputs
    compCount   = 'count' in outputs
    keepCount   = compAvg or compStd
    removeLasts = 0
    outputs     = [o for o in outputs] #make copy of output list
    if compDiff and not compMin:
      outputs.append('min')
      removeLasts += 1
    if compDiff and not compMax:
      outputs.append('max')
      removeLasts += 1
    if compStd and not compAvg:
      outputs.append('avg')
      removeLasts += 1
    if keepCount and not compCount:
      outputs.append('count')
      removeLasts += 1
    imgs = [None]*len(outputs)
    for idx in xrange(len(outputs)):
      o = outputs[idx]
      if o in ['avg', 'std']:
        img = n.zeros(imgshape)
      elif o=='last':
        img = n.empty(imgshape)
        img.fill(n.nan)
      elif o=='first':
        img = n.empty(imgshape)
        img.fill(n.nan)
      elif o=='count':
        img = n.zeros(imgshape, dtype=n.int16)
      elif o=='max':
        img = n.empty(imgshape)
        img.fill(-n.inf)
      elif o=='min':
        img = n.empty(imgshape)
        img.fill(n.inf)
      elif o=='diff':
        img = None
      else:
        raise Exception('Unrecognized output option: '+str(o))
      imgs[idx] = img
    for i in xrange(self.num):
      idx = self.processOrder[i]
      BBs = n.vstack((BB, self.BBs[idx]))
      overlaping = r.getOverlapingBoundingBoxes(BBs, 0, 1)
      if overlaping.size!=1:
        raise Exception('This means that the code in getOverlapingBoundingBoxes() has to be carefully vectorized')
      if not overlaping:
        continue
      xyz = self.loadVar('xyzs', idx)
      if removeNANs:
        nan = self.loadVar('nans', idx)
        xyz = xyz[n.logical_not(nan.ravel()),:]
      xy  = xyz[:,0:2]
      if self.inMemory:
        xy = xy.copy()
      xy  -= BB[0:2].reshape((1,2))
      xy  /= (BB[2:4]-BB[0:2]).reshape((1,2))
      xy  *= imgshape.reshape((1,2))-1
      xyR  = n.round(xy).astype(n.uint32)
      mask = n.logical_and(n.logical_and(xyR[:,0]>=0,          xyR[:,1]>=0),
                           n.logical_and(xyR[:,0]<imgshape[0], xyR[:,1]<imgshape[1]) )
      xyR  = xyR[mask,:]
      if interpolate:
        #This is quite tricky to get right, so for the time being I will leave it unimplemented
        raise Exception('Interpolation option not implemented!')
#        #get three points of the plane of the heightmap
#        points  = self.rectangles[idx,0:3,:]
#        #get the plane equation for the heightmap
#        plane   = r.planeEquationFrom3Points(points)
#        #project the xyR coordinates back to the plane of the heightmap
#        z       = -(xyR[:,0]*plane[0] + xyR[:,1]*plane[1] + plane[3])/plane[2]
#        xyzproj = n.column_stack((xyR, z))
#        #THIS IS UNIMPLEMENTED BECUASE IT WOULD MAKE FOR A HALF-BAKED ALGORITHM:
#        #IT DOES NOT TAKE INTO ACCOUNT THAT THE IMAGE PLANE IS NOT PARALLEL
#        #TO THE XY PLANE, SO TO FIND THE XY DISPLACED COORDINATES, WE NEED
#        #TO PROJECT THE ROUNDED COORDINATES BACK TO THAT PLANE, INSTEAD OF
#        #SIMPLY ASSUMING THAT THE XY COORDINATES ARE APPROXIMATELY CORRECT
#        #displacements in final space
#        xydiffs  = xyR-xy
#        #revert transformations to original space
#        xydiffs /= imgshape.reshape((1,2))-1
#        xydiffs *= (BB[2:4]-BB[0:2]).reshape((1,2))
#        #transform original space to matrix space. This should be done with a
#        #proper inverse transformation using the inverse of the rotation in 
#        #the associated accumRotation, but this exposes a new problem: how to 
#        #deal with the fact that the rotation makes the coordinates 
#        #non-parallel to the XY plane (with a residual Z component). Instead,
#        #we are betting that the heightmap is almost parallel to the XY plane,
#        #so a simple rescaling should do the trick without much loss of
#        #precision
#        xydiffs /= step
#        #sanity checks: inside [-1,-1] - img.shape
      else:
        z  = xyz[mask,2]
      mask = None
      if z.size==0:
        continue
      z    = z.copy() #to make sure that z is contiguous, as it is required by pcl.doAccumXXX functions
      inds = (xyR[:,0]*imgshape[1]+xyR[:,1])#.astype(n.uint32)
      for img,out in zip(imgs,outputs):
        if out=='min':
          accum.doAccumMin(img.ravel(), inds, z, inds.size)
        elif out=='max':
          accum.doAccumMax(img.ravel(), inds, z, inds.size)
        elif out=='avg':
          accum.doAccumSum(img.ravel(), inds, z, inds.size)
        elif out=='std':
          accum.doAccumSum(img.ravel(), inds, z*z, inds.size)
        elif out=='count':
          accum.doAccumCount(img.ravel(), inds, inds.size)
        elif out=='last':
          accum.doAccumLast(img.ravel(), inds, z, inds.size)
        elif out=='first':
          accum.doAccumFirst(img.ravel(), inds, z, inds.size)
#          img[xyR[:,0], xyR[:,1]] = z #this seems to not work as intended
    idxcount = idxavg = idxstd = idxmin = idxmax = idxdiff = None
    for idx,o in enumerate(outputs):
      if o=='count': idxcount = idx
      if o=='avg':   idxavg   = idx
      if o=='std':   idxstd   = idx
      if o=='min':   idxmin   = idx
      if o=='max':   idxmax   = idx
      if o=='diff':  idxdiff  = idx
    if idxmin is not None:
      imgs[idxmin][n.isinf(imgs[idxmin])] = n.nan
    if idxmax is not None:
      imgs[idxmax][n.isinf(imgs[idxmax])] = n.nan
    if idxdiff is not None:
      imgs[idxdiff] = imgs[idxmax]-imgs[idxmin]
    if idxavg  is not None:
      imgs[idxavg] /= imgs[idxcount]
    if idxstd  is not None:
      imgs[idxstd] /= imgs[idxcount]
      imgs[idxstd] -= imgs[idxavg]**2
      imgs[idxstd] = n.sqrt(imgs[idxstd])
    #keep only the images which were asked for
    imgs    =    imgs[:(len(imgs)   -removeLasts)]
    outputs = outputs[:(len(outputs)-removeLasts)]
    return dict(zip(outputs, imgs)), outputstep
  
  def saveRegisteredHeightmapstoPLY(self, filename, removeNANs=True, smoothByOldest=True):
    """method for registration analysis workflows: saves all registered images to a single PLY file.
    Options:
    
       removeNANs: remove the non-valid values
       
       smoothByOldest: in overlapping regions, put only points from the oldest heightmap"""
    if not self.finished:
      raise Exception('Regitration has not been done/finished!!!!')
    if smoothByOldest:
      #do it two times, one for the number of points, the other for putting the points
      #this is extremely inefficient, but resilient to very large datasets
      szs = [x.shape[0] for x in self.smoothByOldest(removeNANs=removeNANs)]
      sequence = self.smoothByOldest(removeNANs=removeNANs)
    else:
      if removeNANs:
        sequence = (self.loadVar('xyzs', idx)[n.logical_not(self.loadVar('nans', idx).ravel()),:] for idx in self.processOrder[0:self.num])
        szs      = self.numnonans
      else:
        sequence = (self.loadVar('xyzs', idx)                                                     for idx in self.processOrder[0:self.num])
        szs      = [x*y for x,y in self.imgshapes]
    w.writePLYFileColorList(filename, sequence, it.cycle(C.COLORS), sizes=szs)
  
#THE FOLLOWING WAS INTENDED TO MIMIC ADHOC SMOOTHING, USING THE NEW FRAMEWORK
#TO PROCESS HEIGHTMAPS OFF-LINE (NOT ALL AT ONCE IN MEMORY), BUT IT DOES NOT WORK AS
#INTENDED, BECAUSE IT IMPLICITLY ASSUMED THAT HEIGHTMAPS WERE PERFECTLY ALIGNED
#TO THEIR BOUNDING BOXES, WHICH IS CLEARLY NOT THE CASE EVEN FOR VERY CAREFULLY ACQUIRED HEIGHTMAPS, OUCH! 
#  def getDiffsByHeightmap(self, removeNANs=True):
#    if not self.finished:
#      raise Exception('Regitration has not been done/finished!!!!')
#    for k in self.processOrder:
#      diffs = self.getImages(BB=self.BBs[k], removeNANs=False, outputs=['diff'])['diff'].T.ravel()
#      xyz = self.loadVar('xyzs', k)
#      if removeNANs:
#        nonan   = n.logical_not(self.loadVar('nans', k).ravel())
#        diffs = diffs[nonan]
#        xyz   = xyz[nonan,:]
#      yield xyz, diffs
#
#  def getColoredDiffsByHeightmap(self, vmin, vmax, removeNANs=True, uselog=True, colormap=cm.jet, nancolor=[1,0,1]):
#    if not self.finished:
#      raise Exception('Regitration has not been done/finished!!!!')
#    colormap.set_bad(nancolor)
#    if uselog:
#      norm    = mcl.LogNorm
#    else:
#      norm    = mcl.Normalize
#    scalarMap = cm.ScalarMappable(norm=norm(clip=True,vmin=vmin,vmax=vmax), cmap=colormap)
#    for xyz, diffs in self.getDiffsByHeightmap(removeNANs):
#      image     = scalarMap.to_rgba(diffs.ravel(), bytes=True)[:,0:3]
#      image[n.isnan(diffs.ravel()),:] = n.array(nancolor).reshape(1,3)
#      yield xyz, image
#  
#  def saveDiffsByHeightmaptoPLY(self, filename, vmin, vmax, removeNANs=True, uselog=True, colormap=cm.jet, nancolor=[1,0,1]):
#    if not self.finished:
#      raise Exception('Regitration has not been done/finished!!!!')
#    if removeNANs:
#      total =  n.sum(self.numnonans)
#    else:
#      total = n.sum(self.imgshapes[:,0]*self.imgshapes[:,1])
#    w.writePLYFileWithColorIterators(filename, total, self.getColoredDiffsByHeightmap(vmin, vmax, removeNANs, uselog, colormap, nancolor))

def simpleComputeRegistration(heightmapFiles, originalLogHandles=None, forceNew=True, processSpecification=None, path=None, conf=None):
  """A very simple entry point to compute a registration with the RegisterTools API"""
  if conf is None:
    conf = ConfigurationRegister()
  if path is not None:
    if path[-1]!=os.sep:
      path = path+os.sep
    conf.debugSavePath = path
    conf.debugXYZ = True
  else:
    conf.debugXYZ = False
  rt = RegisterTools(savePath=path, originalLogHandles=originalLogHandles)
  result = rt.computeRegistration(conf=conf, 
                                  num=len(heightmapFiles), 
                                  heightmaps=heightmapFiles,
                                  loader=SimpleUniversalImageLoader().loader,
                                  processSpecification=processSpecification,
                                  forceNew=forceNew)
  if not rt.finished:
    raise Exception('There was a problem: '+str(result[1]))
#  imgs = rt.getImages(removeNANs=True, outputs=['last', 'diff', 'avg'])
#  if path is not None:
#    rt.saveImage(imgs['avg'], path+'final.avg.ply')
#    rt.saveImage(imgs['avg'], path+'final.avg.png')
#    rt.saveImage(imgs['last'], path+'final.last.ply')
#    rt.saveImage(imgs['last'], path+'final.last.png')
#    rt.saveImage(imgs['diff'], path+'final.diff.png', uselog=True, vmin=0.1, vmax=10)
#  else:
#    rt.showImage(imgs['avg'])
#    rt.showImage(imgs['last'])
#    rt.showImage(imgs['diff'], uselog=True, vmin=0.1, vmax=10)
  return rt


#def t(maxIter=10, nn=None):
#  cosa  = n.loadtxt('cosa.xyz')
#  cosa1 = cosa[cosa[:,3]==1,0:3]
#  cosa0 = cosa[cosa[:,3]==0,0:3]
#  print cosa1.shape
#  print cosa0.shape
#  """print 'QA z[FIRST]='+str(cosa1[0,2])
#  print 'QA x[END]  ='+str(cosa1[-1,0])
#  print 'QB z[FIRST]='+str(cosa0[0,2])
#  print 'QB x[END]  ='+str(cosa0[-1,0])"""
#  if nn is not None:
#    cosa1 = cosa1[0:nn,:]
#    cosa0 = cosa0[0:nn,:]
#  return useICPTool(maxIter, cosa1, cosa0)
#  has_converged, fitnessScore, transf = useICPTool(10, cosa1, cosa0)
#  print 'has_converged: '+str(has_converged)
#  print 'fitnessScore: ' +str(fitnessScore)
#  print 'transf: '
#  print transf

if getattr(sys, 'frozen', False):
    icpToolPath = op.join(op.dirname(op.realpath(sys.executable)), 'icptool.exe') # frozen
else:
    icpToolPath = op.join(op.dirname(op.realpath(__file__)), 'icptool.exe') # unfrozen


def useICPTool(max_iter, xyz1, xyz2):
  """helper to use the ICP tool in Windows"""
  if xyz1.size==0:
    raise Exception('CANNOT CALL ICPTOOL WITH AN EMPTY POINT CLOUD (1)!')
  if xyz2.size==0:
    raise Exception('CANNOT CALL ICPTOOL WITH AN EMPTY POINT CLOUD (2)!')
#  if xyz1.base is not None:
#    xyz1 = xyz1.copy()
#    if xyz1.base is not None:
#      raise Exception('This should never happen!')
#  if xyz2.base is not None:
#    xyz2 = xyz2.copy()
#    if xyz2.base is not None:
#      raise Exception('This should never happen!')
  max_iter = int(max_iter)
  #print "MAXITER: %d, N1: %s, N2: %s" % (max_iter, str(xyz1.shape), str(xyz2.shape))
  data = n.concatenate((
          n.array(n.array(xyz1.shape[0]).astype(n.int32)  .data, dtype=n.uint8, copy=False),
          n.array(n.array(xyz2.shape[0]).astype(n.int32)  .data, dtype=n.uint8, copy=False),
          n.array(n.array(     max_iter).astype(n.int32)  .data, dtype=n.uint8, copy=False),
          n.array(                  xyz1.astype(n.float32).data, dtype=n.uint8, copy=False),
          n.array(                  xyz2.astype(n.float32).data, dtype=n.uint8, copy=False) ))
  t1 = time.time()
  p = sub.Popen([icpToolPath], stdout=sub.PIPE, stderr=sub.PIPE, stdin=sub.PIPE)
  stdout, stderr = p.communicate(input=data.data)
  t2 = time.time()
  #print 'TIME: '+str(t2-t1)
  data = None
  #if stdout is None: stdout='None'
  outputlen      = 4+8+16*4
  returncode     = p.returncode
  if (returncode!=0) or (len(stdout)<outputlen):
    raise Exception('Error executing icptool. Return code: %d. num of output bytes: %d. stdout: <%s>. stderr: <%s>' % (returncode, len(stdout), str(stdout), str(stderr)))
  tupl = struct.unpack('=id', stdout[0:(4+8)])
  has_converged = tupl[0]
  fitnessScore  = tupl[1]
  transf = n.fromstring(stdout[(4+8):], dtype=n.float32, count=16).reshape((4,4)).T
  return has_converged, fitnessScore, transf, t2-t1


def trimXYZWithPolygon(xyz, polygon):
  """simple helper returning XYZ points inside a XY-aligned, infinite-height prism"""
  return xyz[r.testPoinstInsidePolygon(xyz, polygon)]

import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = n.asarray(axis)
    theta = n.asarray(theta)
    axis = axis/math.sqrt(n.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return n.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def smallPerturbations(xyzs):
  """apply a small random transformation to a point cloud"""
  xyzs = [xyz.copy() for xyz in xyzs]
  fivedegs = n.pi/36
  for idx, xyz in enumerate(xyzs):
    mean = n.nanmean(xyz, axis=0)
    xyz = xyz-mean
    axis = n.random.random((3,))
    axis /= n.linalg.norm(axis)
    theta = (n.random.random((1,))*fivedegs)+fivedegs
    m = rotation_matrix(axis, theta)
    xyzs[idx] = r.doTransformT(xyz, m)+mean


def splitHeightmap(heightmap):
  """split a heightmap in four overlapping parts (for testing purposes)"""
  img = heightmap.img
  size = n.array(heightmap.size)
  fac = n.array([0.8,0.8])
  sub = n.floor(size[::-1]*fac).astype(int)
  sub1 = img[:sub[0], :sub[1]]
  sub2 = img[:sub[0], -sub[1]:]
  sub3 = img[-sub[0]:,-sub[1]:]
  sub4 = img[-sub[0]:,:sub[1]]
  subs = [sub1, sub2, sub3, sub4]
  hs = [Heightmap(s, sub[::-1], heightmap.step) for s in subs]
  return hs

def splitAndInterpolateHeightmap(heightmap, debugSavePath=None):
  """split a heightmap in four overlapping, parts not aligned to the same grid (for testing purposes)"""
  heightmap = Heightmap(img=heightmap.img.copy(), size=heightmap.size, step=heightmap.step)
  rt =RegisterTools(savePath=None)
  nans = n.isnan(heightmap.img)
  nonans = n.logical_not(nans)
  xyz = rt.interpolateNANs(heightmap, nans, nanval=0.0)
  img = xyz[:,2].reshape(heightmap.img.shape)#heightmap.img.copy()
  size = n.array(heightmap.size)
  fac = n.array([0.8,0.8])
  sub = n.floor(size[::-1]*fac).astype(int)
  xs = n.arange(img.shape[0])
  ys = n.arange(img.shape[1])
  nx, ny = n.meshgrid(ys, xs)
  c1 = 0.25
  c2 = 0.5
  shift1 = n.array([ (c1+n.random.random((1,))*c2),  (c1+n.random.random((1,))*c2)])
  shift2 = n.array([ (c1+n.random.random((1,))*c2), -(c1+n.random.random((1,))*c2)])
  shift2[0]=0
  shift3 = n.array([-(c1+n.random.random((1,))*c2), -(c1+n.random.random((1,))*c2)])
  shift4 = n.array([-(c1+n.random.random((1,))*c2),  (c1+n.random.random((1,))*c2)])
  print "shift2: "+str(shift2)
  print "shift3: "+str(shift3)
  print "shift4: "+str(shift4)
  coord1x = nx[:sub[0], :sub[1]]
  coord1y = ny[:sub[0], :sub[1]]
  nans1 = nans[:sub[0], :sub[1]]
#  print 'a'+str(coord1x.shape)
#  print 'b'+str(coord1y.shape)
#  print 'c'+str(nans1.shape)
#  print 'd'+str(img.shape)
#  print 'e'+str(nans.shape)
#  print 'f'+str(nx.shape)
#  print 'g'+str(ny.shape)
#  print 'h'+str((nx[:sub[0], :sub[1]]).shape)
#  print 'i'+str((ny[:sub[0], :sub[1]]).shape)
#  print (nans[:sub[0], :sub[1]]).shape
  coord2x = nx[:sub[0], -sub[1]:]+shift2[0]
  coord2y = ny[:sub[0], -sub[1]:]+shift2[1]
  nans2 = nans[:sub[0], -sub[1]:]
  coord3x = nx[-sub[0]:,-sub[1]:]+shift3[0]
  coord3y = ny[-sub[0]:,-sub[1]:]+shift3[1]
  nans3 = nans[-sub[0]:,-sub[1]:]
  coord4x = nx[-sub[0]:,:sub[1]] +shift4[0]
  coord4y = ny[-sub[0]:,:sub[1]] +shift4[1]
  nans4 = nans[-sub[0]:,:sub[1]]
  
  coords1 = n.vstack((coord1y.ravel(), coord1x.ravel()))
  coords2 = n.vstack((coord2y.ravel(), coord2x.ravel()))
  coords3 = n.vstack((coord3y.ravel(), coord3x.ravel()))
  coords4 = n.vstack((coord4y.ravel(), coord4x.ravel()))
  
  o = 3  
  values1 = intpi.map_coordinates(img, coords1, order=o).reshape(nans1.shape)
  values2 = intpi.map_coordinates(img, coords2, order=o).reshape(nans2.shape)
  values3 = intpi.map_coordinates(img, coords3, order=o).reshape(nans3.shape)
  values4 = intpi.map_coordinates(img, coords4, order=o).reshape(nans4.shape)
  
#  vs = [values1, values2, values3, values4]
#  #map_coordinates may produce 0 strips in some cases
#  for v in vs:
#    if (v[0,:]==0.0).all():
#      v=v[1:v.shape[0],:]
#    if (v[:,0]==0.0).all():
#      v=v[:,1:v.shape[0]]
#    if (v[-1,:]==0.0).all():
#      v=v[0:-1,:]
#    if (v[:,-1]==0.0).all():
#      v=v[:,0:-1]
  
#  values1[nans1] = n.nan
#  values2[nans2] = n.nan
#  values3[nans3] = n.nan
#  values4[nans4] = n.nan
  
  values1 = values1[1:-1,1:-1]
  values2 = values2[1:-1,1:-1]
  values3 = values3[1:-1,1:-1]
  values4 = values4[1:-1,1:-1]
  
  if debugSavePath is not None:
    coord1x = coord1x.reshape(nans1.shape)[1:-1,1:-1].ravel()
    coord1y = coord1y.reshape(nans1.shape)[1:-1,1:-1].ravel()
    coord2x = coord2x.reshape(nans2.shape)[1:-1,1:-1].ravel()
    coord2y = coord2y.reshape(nans2.shape)[1:-1,1:-1].ravel()
    coord3x = coord3x.reshape(nans3.shape)[1:-1,1:-1].ravel()
    coord3y = coord3y.reshape(nans3.shape)[1:-1,1:-1].ravel()
    coord4x = coord4x.reshape(nans4.shape)[1:-1,1:-1].ravel()
    coord4y = coord4y.reshape(nans4.shape)[1:-1,1:-1].ravel()
    allxyzs = n.vstack((
         n.column_stack((coord1x, coord1y, values1.ravel())),
         n.column_stack((coord2x, coord2y, values2.ravel())),
         n.column_stack((coord3x, coord3y, values3.ravel())),
         n.column_stack((coord4x, coord4y, values4.ravel())) ))
    resultMesh = createMeshFromPointCloud(allxyzs, 10)    
    w.writePLYMesh(debugSavePath+'perfect.mesh.ply', resultMesh[1].points, resultMesh[1].triangles)#, colors)
  
  
  hs = [Heightmap(values1, sub[::-1]-2, heightmap.step),
        Heightmap(values2, sub[::-1]-2, heightmap.step),
        Heightmap(values3, sub[::-1]-2, heightmap.step),
        Heightmap(values4, sub[::-1]-2, heightmap.step)
        ]
  
  return hs
    

def copyHeightmaps(heightmaps):
  """exactly what it says on the tin"""
  return [Heightmap(img=h.img.copy(), size=h.size, step=h.step) for h in heightmaps]


def upsampleXYZ(xyz, searchRadius, density):
  """use PCL to upsample point cloud"""
  cloud = pcl.PointCloud(xyz.astype(n.float32))
  mls = cloud.make_moving_least_squares()
  mls.set_upsampling_method('RANDOM_UNIFORM_DENSITY')
  mls.set_search_radius(searchRadius)
  mls.set_point_density(density)
  mls.set_polynomial_fit(True)
  resampled = mls.process()
  resampled = resampled.to_array()
  return resampled


def showSmoothingErrorsPLY(filename, xyzs, stats, mask=None, nomaskcolor=[255,0,255], colormap=cm.jet):
    """helper for saving data generated by adhocSmoothing"""
    colormap = cm.jet    
    means, devs, mins, maxs, nums, labels = stats
    diffs = maxs-mins
    colormap.set_bad()
    scalarMap = cm.ScalarMappable(norm=mcl.LogNorm(vmin=0.1, vmax=10, clip=True), cmap=colormap)
    colors = scalarMap.to_rgba(diffs, bytes=True)[:,0:3]
    if mask is not None:
      colors[n.logical_not(mask.ravel()),:] = n.array([nomaskcolor])
    w.writePLYFileWithColor(filename, xyzs, colors)

##weave is not supported in a freezed app, and anyways this is outdated and redundant with other functions
#import scipy.weave as weave
#
#def showSmoothingErrorsIMG(pixels, xyzs, stats, logarithmic=None, mask=None, colormap=cm.jet):
#    """helper for saving data generated by adhocSmoothing"""
#    means, devs, mins, maxs, nums, labels = stats
#    diffs      = maxs-mins
#    xy         = xyzs[:,0:2].copy()
#    if mask is not None:
#      diffs    = diffs[mask.ravel()]
#      xy       = xy[mask.ravel(),:]
#    mind       = diffs.min()
#    maxd       = diffs.max()
#    mins       = xy.min(axis=0)
#    maxs       = xy.max(axis=0)
#    size       = n.array((pixels, n.round(pixels*(maxs[1]-mins[1])/(maxs[0]-mins[0]))))
#    xy        -= mins.reshape((1,2))
#    xy        /= (maxs-mins).reshape((1,2))
#    xy        *= (size.reshape((1,2))-1)
#    xy = n.floor(xy)
#    inds       = (xy[:,0]*size[1]+xy[:,1]).astype(n.uint32)
#    img        = n.zeros(size)-n.inf#n.zeros((numpixels,))
#    numdiffs   = diffs.size
#    code = """
#      for (int i=0;i<numdiffs;i++) {
#        if (diffs[i]>img[inds[i]]) {
#          img[inds[i]] = diffs[i];
#        }
#      }
#"""
#    weave.inline(code, ['numdiffs', 'img', 'diffs', 'inds'])
#    
##    mask = img==-n.inf    
##    masked = n.ma.array(img, mask)
#    #colormap.set_bad('w',1.)
#    colormap.set_under(color='k')
#
##    for i in xrange(img.size):
##      v        = diffs[inds==i]
##      if v.size:
##        img[i] = v.max()
#    #img        = img.reshape(size)
#    tit = 'Z Range: %f, max diff: %f' % (xyzs[:,2].max()-xyzs[:,2].min(), maxd)
#    if logarithmic is None:
#      logarithmic = False
#    logarithmic = n.array(logarithmic)
#    for l in logarithmic:
#      fig        = plt.figure()
#      if l:
#        #cax      = plt.imshow(img.T, cmap=colormap, norm=mcl.LogNorm())
#        cax      = plt.imshow(img.T, cmap=colormap, vmin=0.1, vmax=10, norm=mcl.LogNorm())
#        #cax      = plt.pcolor(img.T, cmap=colormap, vmin=0.01, vmax=100, norm=mcl.LogNorm())
#      else:
#        cax      = plt.imshow(img.T, cmap=colormap)
#      plt.colorbar()
#      plt.title(tit)

#adapted from MeshSamplingTools.cpp of CloudCompare: https://github.com/cloudcompare/trunk/blob/master/CC/src/MeshSamplingTools.cpp
def samplePointsOnMesh(mesh, numberOfPoints):
  """Possibly needed for a future integration in a 3d printing workflow"""
  if numberOfPoints <=0:
    return n.zeros((0,3))
  #list of arrays of triangle points
  ps   = [n.take(mesh.points, mesh.triangles[:,i], axis=0) for i in range(3)]
  vec1 = ps[1]-ps[0]
  vec2 = ps[2]-ps[0]
  #the area of a triangle is half of the vector product norm of the two
  #vectors defining the triangle. We do not bother to divide by 2
  triangleAreas        = n.linalg.norm(n.cross(vec1, vec2), axis=1)
  totalarea            = n.sum(triangleAreas)
  density              = numberOfPoints / totalarea
  numPointsByTriangleF = triangleAreas*density
  numPointsByTriangleI = n.round(numPointsByTriangleF).astype(n.uint32)
  zeroPoints = numPointsByTriangleI==0
  #for triangles without points, given them a chance to have one point, equal to their alloted fractional number of points
  numPointsByTriangleI[zeroPoints] = n.random.random((n.sum(zeroPoints),))<=numPointsByTriangleF[zeroPoints]
  #total number of points to create
  totalpoints = n.sum(numPointsByTriangleI)
  if   totalpoints<=255:
    typ      = n.uint8
  elif totalpoints<=(255*255):
    typ      = n.uint16
  else:
    typ      = n.uint32
  #for each generated point, the index of its triangle
  labels     = n.arange(mesh.triangles.shape[0], dtype=typ)
  labels     = n.repeat(labels, numPointsByTriangleI)
  #prepare array of random positions in the square [0..1,0..1]
  positions = n.random.random((totalpoints,2))
  #for those in the upper triangle, flip them to the lower triangle (we will map the lower triangle to each triangle)
  upperhalf = n.sum(positions,axis=1)>1
  positions[upperhalf,:] = 1-positions[upperhalf,:]
  #map coordinates in lower unit triangle to the corresponding triangles of the mesh
  samples = ( n.take(ps[0], labels, axis=0) +                   #base point
             (n.take(vec1, labels, axis=0) * positions[:,0:1]) +  #triangle's first  vector
             (n.take(vec2, labels, axis=0) * positions[:,1:2]) )  #triangle's second vector
  return samples

def bigResample(xyzs, searchRadius):
  """Given a list of registered point clouds, merge them into one giant point
  cloud and resample with movingleastsquares, in the hope of removing double walls
  and multi walls. Alas, in practice it is not working very good"""
  #combine all clouds into a gigantic one
  numPoints = n.sum([x.shape[0] for x in xyzs])
  xyzs32    = (x.astype(n.float32) for x in xyzs)
  bigcloud  = pcl.PointCloud()
  bigcloud.from_arrays(xyzs32, numPoints)
  mls = bigcloud.make_moving_least_squares()
  mls.set_search_radius(searchRadius)
  #mls.set_polynomial_fit(True)
  mls.set_polynomial_fit(False)
  #mls.set_polynomial_order(2)
  resampled = mls.process()
  resampled = resampled.to_array()
  return resampled

def createMeshFromPointCloud(points, zsub):#(d, zmax, zmin, zsub):
  """Creates a mesh from a point cloud, as a cylinder: top and base meshes
  connected by a ribbon"""
  usedPoints = points #legacy from meshdiff
  #create upper face
  try:
    #tessU = Delaunay(usedPoints[:,0:2], qhull_options='QJ') #This is to make sure that all points are used
    tessU = Delaunay(usedPoints[:,0:2])
  except:
    traceback.print_exc()
    return [False, 'Error trying to generate a mesh from the point cloud: Delaunay triangulation of the point cloud failed']
  tU = tessU.simplices
  
  #get border edges in border triangles
  i1, i2 = (tessU.neighbors==-1).nonzero() #indexes of vertexes not opossed to a triangle, they are not in the edge of the mesh, but the other two vertexes of the triangle are!
  i21 = (i2+1)%3  #these are the column indexes of vertexes in the edge of the mesh
  i22 = (i21+1)%3 #
  ps = n.column_stack((tU[i1,i21], tU[i1,i22])) #edges at the edge of the mesh
  #order the points in the edges (counterclockwise)
  ordered = n.empty(ps.shape[0], dtype=n.int32)
  ordered[0] = ps[0,0] #seed the sequence with the first edge
  ordered[1] = ps[0,1]
  ps[0,:] = -1
  io = 2
  while io<ordered.size:
    i1, i2 = (ps==ordered[io-1]).nonzero() #get the position of the last vertex in the ordered sequence
    if i1.size!=1: #each vertex should appear only twice in the list of edges
      return [False, "could not get ordered border for delaunay triangulation"]
    ordered[io] = ps[i1, (i2+1)%2] #add the adjacent vertex to the ordered list
    ps[i1,:] = -1 #remove the edge from the list of edges
    io += 1
  #points in the base are those at the edge, but lowered by a certain amount  
  newpoints = usedPoints[ordered,:]
  newpoints[:,2] = usedPoints[:,2].min()-zsub
  #ordered list of vertexes at the edges of the upper mesh
  nidxU = ordered
  #same, list, but shifted
  nidxUp1 = n.concatenate((ordered[ordered.shape[0]-1:ordered.shape[0]], ordered[0:-1]))
  #ordered list of vertexes at the lower mesh
  nidxL = n.arange(usedPoints.shape[0], usedPoints.shape[0]+newpoints.shape[0])
  #same, list, but shifted the other way around
  nidxLm1 = n.concatenate((nidxL[1:nidxL.size], nidxL[0:1]))
  #triangles for the connecting ribbon
  Tmed1 = n.column_stack((nidxU, nidxUp1, nidxL))
  Tmed2 = n.column_stack((nidxLm1, nidxU, nidxL))
  #get base mesh
  try:
    #tessB = Delaunay(newpoints[:,0:2], qhull_options='QJ') #This is to make sure that all points are used
    tessB = Delaunay(newpoints[:,0:2])
  except:
    traceback.print_exc()
    return [False, 'Error trying to generate a mesh from the point cloud: Delaunay triangulation of the base failed']
  #reindex the triangles of the base mesh
  tB = nidxL[tessB.simplices]
  # this is to have all triangles of the base mesh to be counterclockwise
  tB = tB[:,[0,2,1]] 
  tA = n.concatenate((tU, Tmed1, Tmed2, tB))
  #create arrays with all points
  allPoints  = n.concatenate((usedPoints, newpoints))
  return (True, Mesh(allPoints, tA))
  #return RetVal(True, (allPoints, tA, (tU.copy(), Tmed1, Tmed2, tB)))


##########################################################################3

def showPoints3D(points):
  """helper to show a point cloud"""
  fig = plt.figure()
  ax  = Axes3D(fig)
  ax.scatter(points[:,0], points[:,1], points[:,2])#, c=colors)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  fig.add_axes(ax)
  plt.show()  

def showMatches(img1, img2, kp1xy, kp2xy):
  """helper to show matched keypoints between overlapping images"""
  fig = plt.figure()
  s1 = img1.shape
  s2 = img2.shape
  print s1
  print s2
  step = 0
  plt.imshow(img1, extent=(0, s1[1], 0, s1[0]), origin='lower')
  bx = (s1[1]+step)#*0.7
  by = (s1[0]+step)#*0.7
  plt.imshow(img2, extent=(bx, bx+s2[1], by, by+s2[0]), origin='lower')
  a1 = kp1xy
  a2 = kp2xy.copy()
  a2[:,0] += by
  a2[:,1] += bx
  lines = [[p1[::-1], p2[::-1]] for p1, p2 in zip(a1, a2)]
  cmap = cm.prism
  cmap = cm.spring
  cs = cmap(n.linspace(0, 1, len(lines)))
  lc = mc.LineCollection(lines, linewidths=1, colors=cs)
  ax = fig.gca()
  ax.add_collection(lc)
  ax.set_xlim([0, s1[1]+step+s2[1]])
  ax.set_ylim([0, s1[0]+step+s2[0]])
  

def showTwoImages(img1, shift1, img2, shift2):
  """helper to show two images with custom shifts, to show registering results"""
  fig = plt.figure(); plt.imshow(img1, origin='lower')
  fig = plt.figure(); plt.imshow(img2, origin='lower')
  fig = plt.figure()
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
  
  
def showZs(data, nbins, rang=None):
  """helper to show histograms"""
  x = data.flatten()
  hist, bins = n.histogram(x, bins=nbins, range=rang)
  width = 0.7 * (bins[1] - bins[0])
  center = (bins[:-1] + bins[1:]) / 2
  plt.figure()
  plt.bar(center, hist, align='center', width=width)
  plt.show()

def show2DZs(dataX, dataY, nbinsX, nbinsY, labelX, labelY, useLog=False):
  """helper to show 2D histograms"""
  H, xedges, yedges = n.histogram2d(dataX.ravel(), dataY.ravel(), bins=(nbinsX,nbinsY))
  if useLog:
    zeros = H==0
    H = n.log10(H)
    H[zeros] = n.nan
  extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
  plt.figure()
  plt.imshow(H, extent=extent, interpolation='nearest')
  plt.colorbar()
  plt.xlabel(labelX)
  plt.ylabel(labelY)
  plt.show()
