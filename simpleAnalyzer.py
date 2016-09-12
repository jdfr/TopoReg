####################################################################
##   SIMPLE SCRIPT TO GENERATE DATA FROM A FINISHED REGISTRATION  ##
####################################################################

import matplotlib.pyplot as plt

import heightmaps as h
import numpy as n

##this is to avoid serialization problems
#import simpleRegister


path           = '/home/josedavid/3dprint/software/pypcl/25/'

rt   = h.RegisterTools(savePath=path)
rt.loadFinished()

imgs, pixelstep = rt.getImages(removeNANs=True, outputs=['last', 'diff', 'avg'])


#R = rt.getImageRotation(imgs['avg'], pixelstep, method='SVD') # VERY FAST
##R = rt.getImageRotation(imgs['avg'], pixelstep, method='RANSAC') #VERY SLOW
#rt.rotateImage(imgs['avg'], pixelstep, R)
#rt.rotateImage(imgs['last'], pixelstep, R)


##save all heightmaps in one file, removing duplicated points in overlapping regions
#rt.saveRegisteredHeightmapstoPLY(path+'final.smoothed.ply', removeNANs=True, smoothByOldest=True)

#save images of several versions of the composite heightmaps
rt.saveImage(imgs['avg'],  path+'final.avg.ply')
rt.saveImage(imgs['avg'],  path+'final.avg.png')
rt.saveImage(imgs['last'], path+'final.last.ply')
rt.saveImage(imgs['last'], path+'final.last.png')
rt.saveImage(imgs['diff'], path+'final.diff.png', uselog=True, vmin=0.1, vmax=10)

#save all heightmaps in one file
rt.saveRegisteredHeightmapstoPLY(path+'final.registered.ply', removeNANs=True, smoothByOldest=False)

##show the composite heightmaps
#rt.showImage(imgs['avg'])
#plt.xlabel('Averaged composite heightmap')
#rt.showImage(imgs['last'])
#plt.xlabel('Overwritten composite heightmap')
#rt.showImage(imgs['diff'], uselog=True, vmin=0.1, vmax=10)
#plt.xlabel('Discrepancies composite heightmap')
#plt.show()
