# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 13:05:40 2015

@author: josedavid
"""

import numpy as n
from itertools import izip

def writeXYZ(filename, xyz):
  """write a point cloud as a .xyz file"""
  n.savetxt(filename, xyz, delimiter='\t', newline='\n')


def writePLYFileColorList(filename, xyzss, colors, nanss=None, sizes=None):
  """given a list of points clouds and a matching list of colours, write the
  coloured point clouds to a PLY file. A list of masks for the point clouds
  can also be specified """
  if nanss is None:
    if sizes is None:
      totals = [x.shape[0] for x in xyzss]
    else:
      totals = [int(x) for x in sizes]
    nanss  = [None]*len(totals)
  else:
    totals = n.zeros((len(xyzss),))
    for xyzs, nans, idx in zip(xyzss, nanss, xrange(len(xyzss))):
      if nans is not None:
        totals[idx] += n.sum(nans)
  with open(filename, 'w') as f:
    f.write('ply\nformat ascii 1.0\nelement vertex ')
    f.write(str(n.sum(totals)))
    f.write('\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    for idx,(nans, color, xyzs) in enumerate(izip(nanss, colors, xyzss)):
#      print 'printing %d (%d points) with color %s' % (idx, totals[idx], str(color))
      if nans is None:
        for xyz in xyzs:
          f.write('%f %f %f %d %d %d\n' % (xyz[0], xyz[1], xyz[2], color[0], color[1], color[2]))
      else:
        for xyz,nan in zip(xyzs,nans.ravel()):
          if not nan:
            f.write('%f %f %f %d %d %d\n' % (xyz[0], xyz[1], xyz[2], color[0], color[1], color[2]))

def writePLYFile(filename, xyzs):
  with open(filename, 'w') as f:
    f.write('ply\nformat ascii 1.0\nelement vertex ')
    f.write(str(xyzs.shape[0]))
    f.write('\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
    for xyz in xyzs:
      f.write('%f %f %f\n' % (xyz[0], xyz[1], xyz[2]))

def writePLYFileWithColorIterators(filename, totalpoints, xyzsAndcolorsIterator):
  with open(filename, 'w') as f:
    f.write('ply\nformat ascii 1.0\nelement vertex ')
    f.write(str(totalpoints))
    f.write('\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    for xyzs, colors in xyzsAndcolorsIterator:
      for xyz,color in izip(xyzs, colors):
        f.write('%f %f %f %d %d %d\n' % (xyz[0], xyz[1], xyz[2], color[0], color[1], color[2]))

def writePLYFileWithColor(filename, xyzs, colors):
  with open(filename, 'w') as f:
    f.write('ply\nformat ascii 1.0\nelement vertex ')
    f.write(str(xyzs.shape[0]))
    f.write('\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    for xyz,color in izip(xyzs, colors):
      f.write('%f %f %f %d %d %d\n' % (xyz[0], xyz[1], xyz[2], color[0], color[1], color[2]))
#def writePLYFileWithColor(filename, xyzs, values, colormap):
#  maxv    = values.max()
#  minv    = values.min()
#  values -= minv
#  values /= (maxv-minv)
#  #colors = colormap(values)
#  with open(filename, 'w') as f:
#    f.write('ply\nformat ascii 1.0\nelement vertex ')
#    f.write(str(values.size))
#    f.write('\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
#    for xyz,color in zip(xyzs, colormap(values)):
#      f.write('%f %f %f %d %d %d\n' % (xyz[0], xyz[1], xyz[2], color[0], color[1], color[2]))

def writePLYFileWithColor2D(filename, xys, colors):
  with open(filename, 'w') as f:
    f.write('ply\nformat ascii 1.0\nelement vertex ')
    f.write(str(xys.shape[0]))
    f.write('\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    for xy,color in izip(xys, colors):
      f.write('%f %f 0 %d %d %d\n' % (xy[0], xy[1], color[0], color[1], color[2]))
#def writePLYFileWithColor2D(filename, xys, values, colormap):
#  maxv    = values.max()
#  minv    = values.min()
#  values -= minv
#  values /= (maxv-minv)
#  #colors = colormap(values)
#  with open(filename, 'w') as f:
#    f.write('ply\nformat ascii 1.0\nelement vertex ')
#    f.write(str(values.size))
#    f.write('\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
#    for xy,color in zip(xys, colormap(values)):
#      f.write('%f %f 0 %d %d %d\n' % (xy[0], xy[1], color[0], color[1], color[2]))


def showBothRegs(filename, xyz1, xyz2, nan1, nan2):
  """write Show two points clouds to two PLY files, one with and one without masks"""
  writePLYFileColorList(filename+'1.ply', [[xyz1, [255, 0, 0]], [xyz2, [0, 255, 0]]])
  xyz1 = xyz1.copy()
  xyz2 = xyz2.copy()
  xyz1[nan1.ravel(),2] = n.nan
  xyz2[nan2.ravel(),2] = n.nan
  writePLYFileColorList(filename+'2.ply', [[xyz1, [255, 0, 0]], [xyz2, [0, 255, 0]]])

def writePLYPolygons(filename, dim, polygons, colors):
  """given a list of polygons and a matching list of colours, write the
  coloured polygons to a PLY file"""
  totals = [pol.shape[0] for pol in polygons]
  count = 0
  with open(filename, 'w') as f:
    f.write('ply\nformat ascii 1.0\nelement vertex ')
    f.write(str(n.sum(totals)))
    f.write('\nproperty float x\nproperty float y\n')
    if dim==3:
      f.write('property float z\n')
    f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nelement face ')
    f.write(str(len(polygons)))
    f.write('\nproperty list uchar int vertex_index\nend_header\n')
    for xyzs,color,idx in izip(polygons, colors, xrange(len(polygons))):
      #print 'printing %d (%d points) with color %s' % (idx, totals[idx], str(color))
      if dim==2:
        for xyz in xyzs:
          f.write('%f %f %d %d %d\n' % (xyz[0], xyz[1], color[0], color[1], color[2]))
      else:
        for xyz in xyzs:
          f.write('%f %f %f %d %d %d\n' % (xyz[0], xyz[1], xyz[2], color[0], color[1], color[2]))
    for pol in polygons:
      f.write(str(xyzs.shape[0]))
      basecount = count
      for k in xrange(pol.shape[0]):
        f.write(' '+str(count))
        count +=1
      f.write(' '+str(basecount)+'\n')

def writePLYPointsAndPolygons(filename, xyzs, polygons):
  """given a list of polygons and a matching list of colours, write the
  coloured polygons to a PLY file"""
  dim = xyzs.shape[1]
  with open(filename, 'w') as f:
    f.write('ply\nformat ascii 1.0\nelement vertex ')
    f.write(str(xyzs.shape[0]))
    f.write('\nproperty float x\nproperty float y\n')
    if dim==3:
      f.write('property float z\n')
    f.write('element face ')
    f.write(str(len(polygons)))
    f.write('\nproperty list uchar int vertex_index\nend_header\n')
    for xyz,color,idx in zip(xyzs, polygons, xrange(len(polygons))):
      #print 'printing %d (%d points) with color %s' % (idx, totals[idx], str(color))
      if dim==2:
        for xyz in xyzs:
          f.write('%f %f\n' % (xyz[0], xyz[1]))
      else:
        for xyz in xyzs:
          f.write('%f %f %f\n' % (xyz[0], xyz[1], xyz[2]))
    for pol in polygons:
      f.write(str(len(pol)))
      for p in pol:
        f.write(' '+str(int(p)))
      f.write('\n')

def writePLYMesh(filename, xyzs, triangles, colors=None):
  """given a list of polygons and a matching list of colours, write the
  coloured polygons to a PLY file"""
  dim = xyzs.shape[1]
  with open(filename, 'w') as f:
    f.write('ply\nformat ascii 1.0\nelement vertex ')
    f.write(str(xyzs.shape[0]))
    f.write('\nproperty float x\nproperty float y\n')
    if dim==3:
      f.write('property float z\n')
    if colors is not None:
      f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
    f.write('element face ')
    f.write(str(triangles.shape[0]))
    f.write('\nproperty list uchar int vertex_index\nend_header\n')
    if colors is not None:
      if dim==2:
        for xyz, color in izip(xyzs, colors):
            f.write('%f %f %d %d %d\n' % (xyz[0], xyz[1], color[0], color[1], color[2]))
      else:
        for xyz, color in izip(xyzs, colors):
          f.write('%f %f %f %d %d %d\n' % (xyz[0], xyz[1], xyz[2], color[0], color[1], color[2]))
    else:
      if dim==2:
        for xyz in xyzs:
            f.write('%f %f\n' % (xyz[0], xyz[1]))
      else:
        for xyz in xyzs:
          f.write('%f %f %f\n' % (xyz[0], xyz[1], xyz[2]))
    for triangle in triangles:
      f.write('3 %d %d %d\n' % (triangle[0], triangle[1], triangle[2]))
