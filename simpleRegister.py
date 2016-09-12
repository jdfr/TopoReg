####################################################################
##     SIMPLE SCRIPT TO USE THE REGISTRATOR      ##
####################################################################


import heightmaps as h
import numpy as n
import glob

####################################################################
##     CONFIGURATION      ##
####################################################################

def doConfiguration():
  conf = h.ConfigurationRegister()
  
  #if the heightmap does not specify the pixel step (to scale correctly the
  #XY coordinates relative to the Z coordinates), use this as default    
  conf.defaultPixelStep = 1.0
  #This is to scale up the Z dimension in case it was compressed
  #when saving the image (for example, compressed to the range [0,1])
  conf.zfac = 1.0
  #RANSAC Rotation paramenters
  conf.RANSAC_k          = 1000 #max number of iterations of RANSAC
  #we take the maximum admissible error per data point as the square  of a multiple of the standard grid distance
  conf.RANSAC_tfun       = 'lambda maxstep: (maxstep*5)**2'#10)**2' 
  conf.RANSAC_dfun       = 'lambda num_matches: max(n.floor(num_matches*0.25), 6)' #we require at least 6 matches
  conf.RANSAC_debug      = False
  conf.ICP_cutofffun     = 'lambda maxstep: maxstep*3'
  #lower this to increase efficiency, at the cost of a bigger chance to end up with a sub-optimal alignment
  conf.ICP_maxiter       = 1000 
  #Neighbours' processing parameters
  #we collect 8 neighbours because we suppose that any area can be overlapped with at most 8 heightmaps
  #we add 1 because the first neighbours will be the point itself
  #we add 4 because we anticipate that some of the neighbours may be from the same heightmap
  conf.debugXYZ          = True
  conf.debugSavePath     = ''
  conf.copyHeightmaps    = False
  #parameters to fit a RANSAC to the plane
  conf.RANSAC_fitplane_enable = True #if false, a regression plane is computed
  conf.RANSAC_fitplane_k = 200 #max number of iterations of RANSAC
  conf.RANSAC_fitplane_tfun = 'lambda maxstep: (maxstep*4)' #to consider a point to fit the plane, it has to be within a multiple of the grid step
  conf.RANSAC_fitplane_planeratio = 0.7 #ratio of the image at least occupied by the plane
  conf.RANSAC_fitplane_saferatio = 0.9 #ratio of the plane that has to be fit in order to decide that a fit is good enough
  conf.RANSAC_fitplane_debug      = False
  
  #conf.rotateMode = h.C.PLANEROT.JUSTFIRST
  conf.rotateMode = h.C.PLANEROT.ALLBYFIRST
  #conf.rotateMode = h.C.PLANEROT.ALLINDEPENDENT
  #conf.rotateMode = h.C.PLANEROT.NOROTATION
  
  #When all images have approximately the same height profile and the background dominates, it is better to center the Z values by substracting the median    
  #However, this can be catastrophic is different images in the dataset are naturally at different heights (for example, an dataset of a relief, with one of the images almost entirely within higher ground, and the others mostly imaging lower ground)
  #self.substractMedian = True 
  conf.substractMedian = False
    
  conf.PhaseCorrScale       = 100.0 #this can be quite high with the algorithm we are currently using
  conf.PhaseCorrNumPeaks    = [100, 1000, 10000] #the highest peaks might not signal the strongest cross-correlations. We incrementally try more and more peaks, up to a point
  conf.PhaseCorrRecommendableCorrCoef = 0.8 #the correlation coefficient must be quite high for the phase correlation to be OK
  conf.PhaseCorrSubpixel    = True 
  conf.PhaseCorrWhitening   = False
  conf.PhaseCorrMinratio    = 0.01 #DO NOT TRUST CROSS CORRELATION WITH TOO LOW OVERLAPING
  conf.PhaseCorrRestoreSubPixelMethod = h.C.SUBPIXELMODE.INTERPOLATE
  #conf.PhaseCorrRestoreSubPixelMethod = h.C.SUBPIXELMODE.RESTORE
  #conf.PhaseCorrRestoreSubPixelMethod = h.C.SUBPIXELMODE.DONOTHING
  
  #conf.firstPhase = h.C.FIRSTPHASE.RANSACROTATION
  #WARNING: THIS ONLY SHOULD BE USED IF THE FOLLOWING CONDITIONS ARE MET:
  #          - THE IMAGES ARE NOT ROTATED IN ANY WAY
  #          - THE IMAGES ARE NOT BIASED OR THE RANSAC PLANE FITTING PERFORMS WELL
  conf.firstPhase = h.C.FIRSTPHASE.PHASECORRELATION
  
  conf.interpOrder = h.C.INTERPORDER.LINEAR
  #conf.interpOrder = h.C.INTERPORDER.CUBIC
  
  return conf


if __name__ == "__main__":

  ####################################################################
  ##     DECLARATION OF IMAGE FILES      ##
  ####################################################################
  
  path           = '/home/josedavid/3dprint/software/pypcl/25/'
  startNum       = 1
  numHeightmaps  = 6
  #heightmapFiles = [path+('mina.dark.%03d.plu' % i) for i in xrange(startNum, startNum+numHeightmaps)]
  heightmapFiles = sorted(glob.glob(path+'*.plu'))
  if len(heightmapFiles)!=25:
    raise Exception('This script works for 25 files in a snake grid!')
  processSpecification=h.makeGridProcessSpecification(5, 5, gridmode='snake', registerOrder='zigzag')
#  ##this is the default process specification
#  #processSpecification = [
#  #    (0, -1), #first  heightmap
#  #    (1,  0), #second heightmap, align to first  heightmap
#  #    (2,  1), #third  heightmap, align to second heightmap
#  #    (3,  2), #fourth heightmap, align to third  heightmap
#  #    (4,  3), #fifth  heightmap, align to fourth heightmap
#  #    (6,  4), #sixth  heightmap, align to fifth  heightmap
#  #    ]
#  
#  #a more clever process specification, registering in zig-zag, and prefering alignment along long edges
#  processSpecification = [
#      (0,  [-1]), #first  heightmap
#      (1,  [0]), #second heightmap, align to first  heightmap
#      (5,  [0]), #fifth  heightmap, align to first  heightmap
#      (4,  [5, 1]), #fourth heightmap, align to fifth  heightmap
#      (2,  [1]), #second heightmap, align to first  heightmap
#      (3,  [2, 4]), #third heightmap, align to second heightmap
#      ]
#  processSpecification = h.makeGridProcessSpecification(2, 3, fillmode='byrow', gridmode='snake', totalNum=6)

  ####################################################################
  ##     EXECUTION       ##
  ####################################################################

  conf = doConfiguration()
  
  rt   = h.simpleComputeRegistration(heightmapFiles, processSpecification=processSpecification, path=path, conf=conf)

