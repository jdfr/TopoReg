# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 10:51:11 2015

@author: josedavid
"""

import numpy as n
import matplotlib.pyplot as plt

#def fitPlaneLSQR(XYZ):
#  A = n.column_stack((XYZ[:,0:2], n.ones((XYZ.shape[0],))))
#  coefs, residuals, rank, s = n.linalg.lstsq(A, XYZ[:,2])
#  plane = n.array([coefs[1], coefs[2], -1, coefs[0]])
#  return plane, coefs, residuals, rank, s, A, XYZ[:,2]

#fit plane to points. Points are rows in the matrix data
#taken from http://stackoverflow.com/questions/15959411/fit-points-to-a-plane-algorithms-how-to-iterpret-results
def fitPlaneSVD(xyz):
    [rows,cols] = xyz.shape
    # Set up constraint equations of the form  AB = 0,
    # where B is a column vector of the plane coefficients
    # in the form b(1)*X + b(2)*Y +b(3)*Z + b(4) = 0.
    p = (n.ones((rows,1)))
    AB = n.hstack([xyz,p])
    [u, d, v] = n.linalg.svd(AB,0)
    B = v[3,:]                    # Solution is last column of v.
    #showPointsAndPlane(XYZ[::100,:], B)
#    nn = n.linalg.norm(B[0:3])
#    B = B / nn
#    return B[0:3] #return just the normal vector
    return B #returns [A,B,C,D] where the plane is Ax+By+Cz+D=0

from pylab import cm 
import write3d as w

#plane is plane[0]*X+plane[1]*Y+plane[2]*Z+plane[3]=0
def showPointsAndPlane(xyz, fac, plane, values=None, vmin=None, vmax=None):
  showF=True
  if showF:
    xx, yy = n.meshgrid([n.nanmin(xyz[:,0]),n.nanmax(xyz[:,0])], [n.nanmin(xyz[:,1]),n.nanmax(xyz[:,1])])
    zz=-(xx*plane[0]+yy*plane[1]+plane[3])/plane[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
  if values is None:
    if showF:
      ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
  else:
    vals = values.copy()
    vals-=vmin#vals.min()
    vals/=vmax#vals.max()
    colors = cm.jet(values)
    if showF:
      ax.scatter(xyz[::fac,0], xyz[::fac,1], xyz[::fac,2], c=colors[::fac])
    w.writePLYFileWithColor('/home/josedavid/3dprint/software/pypcl/strawlab/cent/cosa.ply', xyz, colors*255)
  if showF:
    ax.plot_surface(xx, yy, zz, alpha=0.7, color=[0,1,0])
    ax.set_xlim(n.nanmin(xyz[:,0]),n.nanmax(xyz[:,0]))
    ax.set_ylim(n.nanmin(xyz[:,1]),n.nanmax(xyz[:,1]))
    ax.set_zlim(n.nanmin(xyz[:,2]),n.nanmax(xyz[:,2]))
    plt.show()  
  #return colors, vals

#points are rows in first matrix
#plane is specified by [A,B,C,D] where the plane is Ax+By+Cz+D=0
#taken from http://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
def distancePointsToPlane(points, plane):
  return ((points*(plane[0:3].reshape(1,3))).sum(axis=1) + plane[3]) / n.linalg.norm(plane[0:3])

def errorPointsToPlane(points, plane):
  return n.abs(distancePointsToPlane(points, plane))

def planeEquationFrom3Points(points):
  det = n.linalg.det(points)
  ABCD = n.array([0.0,0.0,0.0,1.0]) #the following is parametric in D. we choose D=1.0
  for k in xrange(3):
    mat      = points.copy()
    mat[:,k] = 1
    ABCD[k]  = -n.linalg.det(mat)*ABCD[3]/det
  return ABCD

#get rotation vector A onto onto vector B
#from http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
def rotateVectorToVector(A,B):
  a = A.reshape((3,))/n.linalg.norm(A)
  b = B.reshape((3,))/n.linalg.norm(B)
  v = n.cross(a, b)
  s = n.linalg.norm(v) #sin of angle
  c = n.dot(a, b) #cos of angle
  V = n.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) #skew-symmetric cross-product matrix of v
  R = n.identity(3)+V+n.dot(V, V)*((1-c)/(s*s))
  return R


#return rigid transformation between two sets of points
#http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
#see also http://robokitchen.tumblr.com/post/67060392720/finding-a-rotation-quaternion-from-two-pairs-of
#see also http://nghiaho.com/?page_id=671
#see also http://igl.ethz.ch/projects/ARAP/svd_rot.pdf
#see also http://en.wikipedia.org/wiki/Kabsch_algorithm
def findTransformation(v0, v1):
  """rigid transformation to convert v0 to v1. This allows any number of
  vectors higher than v0.shape[1], doing a best fit"""
  #local copy
  v0 = n.array(v0, dtype=n.float64, copy=True)
  v1 = n.array(v1, dtype=n.float64, copy=True)
  ndims = v0.shape[0]
  # move centroids to origin
  t0 = -n.mean(v0, axis=1)
  M0 = n.identity(ndims+1)
  M0[:ndims, ndims] = t0
  v0 += t0.reshape(ndims, 1)
  t1 = -n.mean(v1, axis=1)
  M1 = n.identity(ndims+1)
  M1[:ndims, ndims] = t1
  v1 += t1.reshape(ndims, 1)
  # Rigid transformation via SVD of covariance matrix
  u, s, vh = n.linalg.svd(n.dot(v1, v0.T))
  # rotation matrix from SVD orthonormal bases
  R = n.dot(u, vh)
  if n.linalg.det(R) < 0.0:
      # R does not constitute right handed system
      R -= n.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
      s[-1] *= -1.0
  # homogeneous transformation matrix
  M = n.identity(ndims+1)
  M[:ndims, :ndims] = R
  # move centroids back
  M = n.dot(n.linalg.inv(M1), n.dot(M, M0))
  M /= M[ndims, ndims]
  return M

def inverseTransform(M):
  P = M[0:3,0:3]
  T = M[0:3,3]
  P1 = n.linalg.inv(P)
  M1 = n.vstack((n.column_stack((P1, n.dot(-P1, T))), n.array([0.0, 0.0, 0.0, 1])))
  return M1
  
def inverseTransformT(M):
  return inverseTransform(M.T).T

def doTransform(M, xyz):
  """transform in the usual way as done in textbooks (coordinate vectors are vertical)"""
  xyz1 = n.vstack((xyz, n.ones((1, xyz.shape[1]))))
  return n.dot(M, xyz1)[0:-1,:]

def doTransformT(xyzT, MT):
  """transform for coordinate vectors as used by here (coordinate vectors are rows)"""
  xyz1 = n.column_stack((xyzT, n.ones((xyzT.shape[0],))))
  return n.dot(xyz1, MT)[:,0:-1]

def findTransformation_RANSAC(data):
  """Calls findTransformation inside RANSAC (a transformation is a model)"""
  s = data.shape[1]/2
#  if data.shape[0]!=3:
#    print 'Mira findTransformation_RANSAC: '+str(data.shape)
#  return findTransformation(data[:,:s].T, data[:,s:].T)
  return findTransformation(data[:,:s].T, data[:,s:].T).T
  
def get_error_RANSAC(data, model):
  """Gets error of a model (a transformation) inside RANSAC"""
  s = data.shape[1]/2
#  set1 = data[:,:s].T
#  set2 = data[:,s:].T
#  transf = doTransform(model, set1)
#  err = n.sum((transf-set2)**2, axis=0)
  set1 = data[:,:s]
  set2 = data[:,s:]
  transf = doTransformT(set1, model)
  err = n.sum((transf-set2)**2, axis=1)
  return err

def findTranslation_RANSAC(data):
  """Calls findTransformation inside RANSAC (a transformation is a model)"""
  raise Exception("This is untested!!!")
  s = data.shape[1]/2
  rot = n.identity(4)
  #rot[0:3,3] = -n.mean(data[:,s:]-data[:,:s], axis=0)
  rot[3,0:3] = -n.mean(data[:,s:]-data[:,:s], axis=0)
  return rot
  
#def get_errordisp_RANSAC(data, model):
#  """Gets error of a model (a transformation) inside RANSAC"""
#  s = data.shape[1]/2
##  set1 = data[:,:s].T
##  set2 = data[:,s:].T
##  transf = doTransform(model, set1)
##  err = n.sum((transf-set2)**2, axis=0)
#  set1 = data[:,:s]
#  set2 = data[:,s:]
#  err = n.sum(((set1+model)-set2)**2, axis=1)
#  return err

def testPointsInsideBoundingBox(points, BB):
  """points: coordinates in row format
     BB: bounding box"""
#
#     Both arguments have to be two dimensional matrices, vectors will break
#     the code. Returns a boolean matrix whose element [i,j] is the test of
#     the point points[j,:] inside the BB BBs[i,:]"""
#  if len(points.shape)==1:
#    points = n.reshape(points, (1, points.size))
#  if len(BBs.shape)==1:
#    BBs = n.reshape(BBs, (1, BBs.size))
#  return reduce(n.logical_and, [points[:,0:1]>=BBs[:,0:1].T, points[:,0:1]<=BBs[:,2:3].T,
#                                points[:,1:2]>=BBs[:,1:2].T, points[:,1:2]<=BBs[:,3:4].T])
  return reduce(n.logical_and, [points[:,0]>=BB[0], points[:,0]<=BB[2],
                                points[:,1]>=BB[1], points[:,1]<=BB[3]])

def getOverlapingBoundingBoxes(BBs, A, Bs):
  """returns a bool array mask to cover only bounding boxes in Bs that overlap with bounding box A"""
#bool DoBoxesIntersect(Box a, Box b) {
#  return (abs(a.x - b.x) * 2 < (a.width + b.width)) &&
#         (abs(a.y - b.y) * 2 < (a.height + b.height));
#}
  minxA    = BBs[A,0]
  minyA    = BBs[A,1]
  widthA   = BBs[A,2]-minxA
  heightA  = BBs[A,3]-minyA
  minxBs   = BBs[Bs,0]
  minyBs   = BBs[Bs,1]
  widthBs  = BBs[Bs,2]-minxBs
  heightBs = BBs[Bs,3]-minyBs
  overlaps = n.logical_and(((n.abs(minxA-minxBs)*2) < (widthA+widthBs)),
                           ((n.abs(minyA-minyBs)*2) < (heightA+heightBs)))
  return overlaps

def imageRectangle(img, imgstep):
  """Return the rectangle defining the bounding box of an (axis-aligned) heightmap"""
  minx,miny,maxx,maxy = imageBB(img, imgstep)
  rectangle = n.array([[minx,miny,0], [maxx,miny,0], [maxx,maxy,0], [minx,maxy,0]])
  return rectangle

def heightmapRectangle(heightmap):
  """Return the rectangle defining the bounding box of an (axis-aligned) heightmap"""
  minx,miny,maxx,maxy = heightmapBB(heightmap)
  rectangle = n.array([[minx,miny,0], [maxx,miny,0], [maxx,maxy,0], [minx,maxy,0]])
  return rectangle



def testPoinstInsidePolygon(points, pol):#, _PointInPolygon(pt, outPt): 
  """adapted & vectorized from Clipper._PointInPolygon. It assumes that there
  are way more points to test than polygon vertices"""
#def _PointInPolygon(pt, outPt): 
#    outPt2 = outPt
#    result = False
#    while True:
#        if ((((outPt2.pt.y <= pt.y) and (pt.y < outPt2.prevOp.pt.y)) or \
#            ((outPt2.prevOp.pt.y <= pt.y) and (pt.y < outPt2.pt.y))) and \
#            (pt.x < (outPt2.prevOp.pt.x - outPt2.pt.x) * (pt.y - outPt2.pt.y) / \
#            (outPt2.prevOp.pt.y - outPt2.pt.y) + outPt2.pt.x)): result = not result
#        outPt2 = outPt2.nextOp
#        if outPt2 == outPt: break
  count = n.zeros((points.shape[0],), dtype=int) #shape as a vector instead as a column vector to avoid broadcast errors
  for kthis in xrange(pol.shape[0]):
    kprev = (kthis-1)%pol.shape[0]
    count += n.logical_and( n.logical_or(
             n.logical_and( pol[kthis,1] <= points[:,1], points[:,1] < pol[kprev,1]), 
             n.logical_and( pol[kprev,1] <= points[:,1], points[:,1] < pol[kthis,1]) ), 
             points[:,0] < ((pol[kprev,0]-pol[kthis,0])*(points[:,1]-pol[kthis,1])/(pol[kprev,1]-pol[kthis,1])+pol[kthis,0]) )
  inside = (count%2)==1
  return inside

def imageBB(img, imgstep):
  """Return the bounding box of an (axis-aligned) heightmap"""
  return n.array([[0.0,0.0], [img.shape[1]*imgstep[0], img.shape[0]*imgstep[1]]]).flatten()

def heightmapBB(heightmap):
  """Return the bounding box of an (axis-aligned) heightmap"""
  return n.array([[0.0,0.0], [heightmap.size[0]*heightmap.step[0], heightmap.size[1]*heightmap.step[1]]]).flatten()

def heightmap2XYZ(heightmap):
  """convert a heightmap to a point cloud, pcl-style (a npointsXndimensions matrix)"""
#  #rebase heightmap to cancel z_0 away
#  data -= axes_config['z_0']
  #create XY grid
  xs = n.linspace(0, heightmap.size[0]*heightmap.step[0], heightmap.size[0])
  ys = n.linspace(0, heightmap.size[1]*heightmap.step[1], heightmap.size[1])
  
  grid = n.meshgrid(xs, ys)
  
  xyz = n.column_stack((grid[0].ravel(), grid[1].ravel(), heightmap.img.ravel()))
  return xyz

def image2XYZ(img, step):
  """convert a heightmap to a point cloud, pcl-style (a npointsXndimensions matrix)"""
#  #rebase heightmap to cancel z_0 away
#  data -= axes_config['z_0']
  #create XY grid
  xs = n.linspace(0, img.shape[1]*step[0], img.shape[1])
  ys = n.linspace(0, img.shape[0]*step[1], img.shape[0])
  
  grid = n.meshgrid(xs, ys)
  
  xyz = n.column_stack((grid[0].ravel(), grid[1].ravel(), img.ravel()))
  return xyz

#from https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy 
#see also this for some subtle details out of this simple procedure: http://stackoverflow.com/questions/6656358/calculating-normals-in-a-triangle-mesh
def normalsFromMesh(points, triangles):
  #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
  norm = n.zeros( points.shape, dtype=points.dtype )
  #Create an indexed view into the vertex array using the array of three indices for triangles
  tris = points[triangles]
  #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
  nrm = n.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
  # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
  # we need to normalize these, so that our next step weights each normal equally.
  normalize_v3(nrm)
  # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
  # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
  # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
  # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
  norm[ triangles[:,0] ] += nrm
  norm[ triangles[:,1] ] += nrm
  norm[ triangles[:,2] ] += nrm
  normalize_v3(norm)
  return norm
  
def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = n.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def meshForGrid(shape):
  """create a mesh for a grid in C-style matrix order"""
  idxs = n.arange(n.prod(shape)).reshape(shape)
  #v1, v2, v3, v4: vertices of each square in the grid, clockwise
  v1 = idxs[:-1,:-1]
  v2 = idxs[:-1,1:]
  v3 = idxs[1:,1:]
  v4 = idxs[1:,:-1]
  faces = n.vstack((
      n.column_stack((v1.ravel(), v2.ravel(), v4.ravel())),    #triangles type 1
      n.column_stack((v2.ravel(), v3.ravel(), v4.ravel())) ))  #triangles type 2
  return faces

def normalsForMatrixXYZ(xyz, shape):
  """assuming that xyz comes from a grid, compute the normals"""
  triangles = meshForGrid(shape)
  return normalsFromMesh(xyz, triangles)
  

#############################################################################
#NOT USED

#adapted from the python port of the clipper library
def testPointInsidePolygon(point, polygonPoints):#, _PointInPolygon(pt, outPt): 
  """adapted & vectorized from Clipper._PointInPolygon"""
  this = polygonPoints
  prev = n.roll(polygonPoints, 1, 0)
  inside = n.zeros((point.shape[0],1), dtype=bool)
  tests = n.logical_and( n.logical_or(
            n.logical_and( this[:,1] <= point[1], point[1] < prev[:,1]), 
            n.logical_and( prev[:,1] <= point[1], point[1] < this[:,1]) ),
            point[0] < ((prev[:,0]-this[:,0])*(point[1]-this[:,1])/(prev[:,1]-this[:,1])+this[:,0]))
  inside = (n.sum(tests)%2)==1
  return inside
#def _PointInPolygon(pt, outPt): 
#    outPt2 = outPt
#    result = False
#    while True:
#        if ((((outPt2.pt.y <= pt.y) and (pt.y < outPt2.prevOp.pt.y)) or \
#            ((outPt2.prevOp.pt.y <= pt.y) and (pt.y < outPt2.pt.y))) and \
#            (pt.x < (outPt2.prevOp.pt.x - outPt2.pt.x) * (pt.y - outPt2.pt.y) / \
#            (outPt2.prevOp.pt.y - outPt2.pt.y) + outPt2.pt.x)): result = not result
#        outPt2 = outPt2.nextOp
#        if outPt2 == outPt: break


