#cython: embedsignature=True
#
# Copyright 2014 Netherlands eScience Center

from libcpp cimport bool
from cpython cimport bool as boolp

cimport numpy as np
import numpy as np

cimport _pcl
cimport pcl_defs as cpp
from shared_ptr cimport sp_assign

np.import_array()

#cdef extern from "pcl/registration/trimmed_icp.h" namespace "pcl::registration" nogil:
#  cdef cppclass TrimmedICP[Point, Scalar]:
#        cppclass Matrix4:
#            float *data()
#        TrimmedICP() except +
#        void setNewToOldEnergyRatio (float)
#        void init (cpp.PointCloudPtr_t) except +
#        void align(cpp.PointCloud[Point] &, int, Matrix4 &) except +
        
        
cdef extern from "pcl/registration/registration.h" namespace "pcl" nogil:
    cdef cppclass Registration[Source, Target]:
        cppclass Matrix4:
            float *data()
        void align(cpp.PointCloud[Source] &) except +
        void align(cpp.PointCloud[Source] &, Matrix4 &) except +
        Matrix4 getFinalTransformation() except +
        double getFitnessScore() except +
        bool hasConverged() except +
        void setInputSource(cpp.PointCloudPtr_t) except +
        void setInputTarget(cpp.PointCloudPtr_t) except +
        void setMaximumIterations(int) except +
        void setMaxCorrespondenceDistance (double)
        double getMaxCorrespondenceDistance ()
        void setTransformationEpsilon (double )
        double  getTransformationEpsilon ()
        void setEuclideanFitnessEpsilon (double )
        double getEuclideanFitnessEpsilon ()
        void addCorrespondenceRejector(cpp.CorrespondenceRejectorTrimmedPtr_t)
        cpp.CorrespondenceRejectorTrimmedVector_t getCorrespondenceRejectors()

cdef extern from "pcl/registration/icp.h" namespace "pcl" nogil:
    cdef cppclass IterativeClosestPoint[Source, Target](Registration[Source, Target]):
        IterativeClosestPoint() except +
        void setUseReciprocalCorrespondences (bool)
        bool getUseReciprocalCorrespondences ()

cdef extern from "pcl/registration/gicp.h" namespace "pcl" nogil:
    cdef cppclass GeneralizedIterativeClosestPoint[Source, Target](IterativeClosestPoint[Source, Target]):
        GeneralizedIterativeClosestPoint() except +

cdef extern from "pcl/registration/icp_nl.h" namespace "pcl" nogil:
    cdef cppclass IterativeClosestPointNonLinear[Source, Target](IterativeClosestPoint[Source, Target]):
        IterativeClosestPointNonLinear() except +


cdef extern from "pcl/registration/ndt.h" namespace "pcl" nogil:
    cdef cppclass NormalDistributionsTransform[Source, Target](Registration[Source, Target]):
        NormalDistributionsTransform() except +
        void setResolution (float) except +
        void setStepSize (double)
        void setOulierRatio (double)
        double getTransformationProbability ()
        int getFinalNumIteration ()

cdef extern from "myndt.h":
  double executeNDT(cpp.PointCloudPtr_t, cpp.PointCloudPtr_t, #cpp.PointCloud_t *,
                bool, double, bool, double, bool, double, bool, double, double *)
  bool registerWithNormals(int npoints1, double *xyz1, double *normals1, int npoints2, double *xyz2, double *normals2, double *vec,
                         bool usemaxiter, int maxiter, 
                         bool usemaxcorrdist, double maxcorrdist, 
                         bool usetepsilon, double tepsilon, 
                         bool usefepsilon, double fepsilon)

def registerWithNormalsFun(xyz1, norm1, xyz2, norm2, maxiter=None, maxcorrdist=None, tepsilon=None, fepsilon=None):
  cdef bool usemaxiter     = maxiter is not None
  cdef bool usemaxcorrdist = maxcorrdist is not None
  cdef bool usetepsilon    = tepsilon is not None
  cdef bool usefepsilon    = fepsilon is not None
 
  cdef int maxiterc   = 0
  cdef double maxcorrdistc = 0.0
  cdef double tepsilonc   = 0.0
  cdef double fepsilonc    = 0.0
  
  cdef bool hasConverged
  cdef double *vec
  cdef double *x1
  cdef double *n1
  cdef double *x2
  cdef double *n2
  cdef np.ndarray[dtype=np.float64_t, ndim=2, mode='fortran'] transf

  if usemaxiter:
    maxiterc = usemaxiter
  if usemaxcorrdist:
    maxcorrdistc = maxcorrdist
  if usetepsilon:
    tepsilonc = tepsilon
  if usetepsilon:
    fepsilonc = fepsilon
    
  transf = np.empty((4, 4), dtype=np.float64, order='fortran')
  vec = <double *>np.PyArray_DATA(transf)
  x1 = <double *>np.PyArray_DATA(xyz1)
  x2 = <double *>np.PyArray_DATA(xyz2)
  n1 = <double *>np.PyArray_DATA(norm1)
  n2 = <double *>np.PyArray_DATA(norm2)
  
  hasConverged = registerWithNormals(xyz1.shape[0], x1, n1, xyz2.shape[0], x2, n2, vec,
                                     usemaxiter, maxiterc,
                                     usemaxcorrdist, maxcorrdistc,
                                     usetepsilon, tepsilonc,
                                     usefepsilon, fepsilonc)
  return transf, hasConverged  
  
#void executeNDT(pcl::PointCloud<pcl::PointXYZ> &source, pcl::PointCloud<pcl::PointXYZ> &target, pcl::PointCloud<pcl::PointXYZ> &output,
#                bool useepsilon, double epsilon, 
#                bool usestep, double step, 
#                bool useresolution, double resolution,
#                bool useiters, double maxiters
#                double *vec);
def myNDT(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None, resolution=None, stepSize=None, epsilon=None):
  #cdef _pcl.PointCloud result = _pcl.PointCloud()
  cdef np.ndarray[dtype=np.float64_t, ndim=2, mode='fortran'] transf
 
  cdef bool useepsilon    = epsilon is not None
  cdef bool usestep       = stepSize is not None
  cdef bool useresolution = resolution is not None
  cdef bool useiters      = max_iter is not None
 
  cdef double max_iterc   = 0.0
  cdef double resolutionc = 0.0
  cdef double stepSizec   = 0.0
  cdef double epsilonc    = 0.0
  
  cdef double prob
  
  cdef double *vec
 
  transf = np.empty((4, 4), dtype=np.float64, order='fortran')
  vec = <double *>np.PyArray_DATA(transf)
  
  if useepsilon:
    epsilonc = epsilon
  if usestep:
    stepSizec = stepSize
  if useresolution:
    resolutionc = resolution
  if useiters:
    max_iterc = max_iter
  
  prob = executeNDT(source.thisptr_shared, target.thisptr_shared, #result.thisptr(),
                    useepsilon, epsilonc,
                    usestep, stepSizec, 
                    useresolution, resolutionc,
                    useiters, max_iterc,
                    vec)

  return transf, prob

cdef runNDT(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None, resolution=None, stepSize=None, icpParams=None):
    
    cdef NormalDistributionsTransform[cpp.PointXYZ, cpp.PointXYZ] reg      
    
#    cdef Registration[cpp.PointXYZ, cpp.PointXYZ].Matrix4 guess
#    cdef np.float32_t *guess_data
#    guess_data = <np.float32_t *>guess.data()
#    
#    for i in range(16):
#      guess_data[i] = 0.0
#    for i in range(0, 16, 5):
#      guess_data[i] = 1.0
    
    if max_iter is not None:
        reg.setMaximumIterations(max_iter)

    cdef _pcl.PointCloud result = _pcl.PointCloud()
    cdef double valf

    if icpParams is not None:
      if icpParams[0] is not None and type(icpParams[0])==float:
        valf = icpParams[0]
        reg.setMaxCorrespondenceDistance(valf)
      if icpParams[1] is not None and type(icpParams[1])==float:
        valf = icpParams[1]
        reg.setTransformationEpsilon(valf)
      if icpParams[2] is not None and type(icpParams[2])==float:
        valf = icpParams[2]
        reg.setEuclideanFitnessEpsilon(valf)

    if stepSize is not None:
      reg.setStepSize(stepSize)

    if resolution is not None: #just before of calling align() because this does some initilization work on the target (voxel filtering)
      reg.setResolution(resolution)

    #print 'AAA'    
    reg.setInputSource(source.thisptr_shared)
    reg.setInputTarget(target.thisptr_shared)
    #print 'AA'    
    with nogil:
        reg.align(result.thisptr()[0])
    #print 'B'    
    
    # Get transformation matrix and convert from Eigen to NumPy format.
    cdef Registration[cpp.PointXYZ, cpp.PointXYZ].Matrix4 mat
    mat = reg.getFinalTransformation()
    cdef np.ndarray[dtype=np.float32_t, ndim=2, mode='fortran'] transf
    cdef np.float32_t *transf_data

    transf = np.empty((4, 4), dtype=np.float32, order='fortran')
    transf_data = <np.float32_t *>np.PyArray_DATA(transf)

    for i in range(16):
        transf_data[i] = mat.data()[i]

    return reg.hasConverged(), transf, result, reg.getFitnessScore(), reg.getTransformationProbability(), reg.getFinalNumIteration()

def ndt(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None, resolution=None, stepSize=None, icpParams=None):
  runNDT(source, target, max_iter, resolution, stepSize, icpParams)

#cdef object runWithDisplacement(Registration[cpp.PointXYZ, cpp.PointXYZ] &reg,
#                _pcl.PointCloud source, _pcl.PointCloud target, max_iter,
#                np.ndarray[np.float32_t, ndim=1] guessDisp):
#    reg.setInputSource(source.thisptr_shared)
#    reg.setInputTarget(target.thisptr_shared)
#
#    if max_iter is not None:
#        reg.setMaximumIterations(max_iter)
#
#    cdef _pcl.PointCloud result = _pcl.PointCloud()
#    
#    cdef Registration[cpp.PointXYZ, cpp.PointXYZ].Matrix4 guess
#    cdef np.float32_t *guess_data
#    cdef np.float32_t *guess_source
#    cdef size_t i
#    if guessDisp is not None:
#      guess_data = <np.float32_t *>guess.data()
#      guess_source = <np.float32_t *>np.PyArray_DATA(guessDisp)
#      for i in range(1,5):
#        guess_data[i] = 0.0;
#      for i in range(6,10):
#        guess_data[i] = 0.0;
#      guess_data[11] = 0.0
#      for i in range(0, 16, 5):
#        guess_data[i] = 1.0;
#      for i in range(0,3):
#        guess_data[i+12] = guess_source[i]
#      with nogil:
#        reg.align(result.thisptr()[0], guess)
#    else:
#      with nogil:
#        reg.align(result.thisptr()[0])
#
#    # Get transformation matrix and convert from Eigen to NumPy format.
#    cdef Registration[cpp.PointXYZ, cpp.PointXYZ].Matrix4 mat
#    mat = reg.getFinalTransformation()
#    cdef np.ndarray[dtype=np.float32_t, ndim=2, mode='fortran'] transf
#    cdef np.float32_t *transf_data
#
#    transf = np.empty((4, 4), dtype=np.float32, order='fortran')
#    transf_data = <np.float32_t *>np.PyArray_DATA(transf)
#
#    for i in range(16):
#        transf_data[i] = mat.data()[i]
#
#    return reg.hasConverged(), transf, result, reg.getFitnessScore()

cdef object run(IterativeClosestPoint[cpp.PointXYZ, cpp.PointXYZ] &reg,
                _pcl.PointCloud source, _pcl.PointCloud target, max_iter=None, overlapRatio=None, icpParams=None):
    
    reg.setInputSource(source.thisptr_shared)
    reg.setInputTarget(target.thisptr_shared)

    
    cdef cpp.CorrespondenceRejectorTrimmedPtr_t rej
    if overlapRatio is not None:
      sp_assign(rej, new cpp.CorrespondenceRejectorTrimmed())
      rej.get()[0].setOverlapRatio(overlapRatio)
      reg.addCorrespondenceRejector(rej)
      #print 'doing it with a CorrespondenceRejectorTrimmed set to:'
      #print '    1 : '+str(rej.get()[0].getOverlapRatio())
      #print '    2 : '+str((<cpp.CorrespondenceRejectorTrimmedPtr_t>(reg.getCorrespondenceRejectors()[0])).get()[0].getOverlapRatio())
    
    if max_iter is not None:
        reg.setMaximumIterations(max_iter)

    cdef _pcl.PointCloud result = _pcl.PointCloud()
    cdef double valf
    cdef bool valb

    if icpParams is not None:
      if icpParams[0] is not None and type(icpParams[0])==float:
        valf = icpParams[0]
        reg.setMaxCorrespondenceDistance(valf)
      if icpParams[1] is not None and type(icpParams[1])==float:
        valf = icpParams[1]
        reg.setTransformationEpsilon(valf)
      if icpParams[2] is not None and type(icpParams[2])==float:
        valf = icpParams[2]
        reg.setEuclideanFitnessEpsilon(valf)
      if icpParams[3] is not None and type(icpParams[2])==boolp:
        valb = icpParams[3]
        reg.setUseReciprocalCorrespondences(valb)

    with nogil:
        reg.align(result.thisptr()[0])
    
    # Get transformation matrix and convert from Eigen to NumPy format.
    cdef Registration[cpp.PointXYZ, cpp.PointXYZ].Matrix4 mat
    mat = reg.getFinalTransformation()
    cdef np.ndarray[dtype=np.float32_t, ndim=2, mode='fortran'] transf
    cdef np.float32_t *transf_data

    transf = np.empty((4, 4), dtype=np.float32, order='fortran')
    transf_data = <np.float32_t *>np.PyArray_DATA(transf)

    for i in range(16):
        transf_data[i] = mat.data()[i]

    return reg.hasConverged(), transf, result, reg.getFitnessScore()

#cdef trimmedICP(_pcl.PointCloud source, _pcl.PointCloud target, int numPoints):
#    cdef TrimmedICP[cpp.PointXYZ, float] ticp
#    cdef TrimmedICP[cpp.PointXYZ, float].Matrix4 mat
#    icp.init(target.thisptr_shared)
#    for i in range(16):
#        mat.data()[i] = 0.0
#    for i in range(0,16,5):
#      mat.data()[i] = 1.0
#    with nogil:
#        ticp.align(source.thisptr()[0], numPoints, mat)
#    cdef np.ndarray[dtype=np.float32_t, ndim=2, mode='fortran'] transf
#    cdef np.float32_t *transf_data
#
#    transf = np.empty((4, 4), dtype=np.float32, order='fortran')
#    transf_data = <np.float32_t *>np.PyArray_DATA(transf)
#
#    for i in range(16):
#        transf_data[i] = mat.data()[i]
#    
#    return transf

def icp(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None, overlapRatio=None, icpParams=None):
    """Align source to target using iterative closest point (ICP).

    Parameters
    ----------
    source : PointCloud
        Source point cloud.
    target : PointCloud
        Target point cloud.
    max_iter : integer, optional
        Maximum number of iterations. If not given, uses the default number
        hardwired into PCL.

    Returns
    -------
    converged : bool
        Whether the ICP algorithm converged in at most max_iter steps.
    transf : np.ndarray, shape = [4, 4]
        Transformation matrix.
    estimate : PointCloud
        Transformed version of source.
    fitness : float
        Sum of squares error in the estimated transformation.
    """
    cdef IterativeClosestPoint[cpp.PointXYZ, cpp.PointXYZ] icp
    return run(icp, source, target, max_iter, overlapRatio, icpParams)


def gicp(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None, overlapRatio=None, icpParams=None):
    """Align source to target using generalized iterative closest point (GICP).

    Parameters
    ----------
    source : PointCloud
        Source point cloud.
    target : PointCloud
        Target point cloud.
    max_iter : integer, optional
        Maximum number of iterations. If not given, uses the default number
        hardwired into PCL.

    Returns
    -------
    converged : bool
        Whether the ICP algorithm converged in at most max_iter steps.
    transf : np.ndarray, shape = [4, 4]
        Transformation matrix.
    estimate : PointCloud
        Transformed version of source.
    fitness : float
        Sum of squares error in the estimated transformation.
    """
    cdef GeneralizedIterativeClosestPoint[cpp.PointXYZ, cpp.PointXYZ] gicp
    return run(gicp, source, target, max_iter, overlapRatio, icpParams)


def icp_nl(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None, overlapRatio=None, icpParams=None):
    """Align source to target using generalized non-linear ICP (ICP-NL).

    Parameters
    ----------
    source : PointCloud
        Source point cloud.
    target : PointCloud
        Target point cloud.

    max_iter : integer, optional
        Maximum number of iterations. If not given, uses the default number
        hardwired into PCL.

    Returns
    -------
    converged : bool
        Whether the ICP algorithm converged in at most max_iter steps.
    transf : np.ndarray, shape = [4, 4]
        Transformation matrix.
    estimate : PointCloud
        Transformed version of source.
    fitness : float
        Sum of squares error in the estimated transformation.
    """
    cdef IterativeClosestPointNonLinear[cpp.PointXYZ, cpp.PointXYZ] icp_nl
    return run(icp_nl, source, target, max_iter, overlapRatio, icpParams)


#def icp_guess(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None, guessDisp=None):
#    """Align source to target using iterative closest point (ICP).
#
#    Parameters
#    ----------
#    source : PointCloud
#        Source point cloud.
#    target : PointCloud
#        Target point cloud.
#    max_iter : integer, optional
#        Maximum number of iterations. If not given, uses the default number
#        hardwired into PCL.
#
#    Returns
#    -------
#    converged : bool
#        Whether the ICP algorithm converged in at most max_iter steps.
#    transf : np.ndarray, shape = [4, 4]
#        Transformation matrix.
#    estimate : PointCloud
#        Transformed version of source.
#    fitness : float
#        Sum of squares error in the estimated transformation.
#    """
#    cdef IterativeClosestPoint[cpp.PointXYZ, cpp.PointXYZ] icp
#    return runWithDisplacement(icp, source, target, max_iter, guessDisp)
#
#
#def gicp_guess(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None, guessDisp=None):
#    """Align source to target using generalized iterative closest point (GICP).
#
#    Parameters
#    ----------
#    source : PointCloud
#        Source point cloud.
#    target : PointCloud
#        Target point cloud.
#    max_iter : integer, optional
#        Maximum number of iterations. If not given, uses the default number
#        hardwired into PCL.
#
#    Returns
#    -------
#    converged : bool
#        Whether the ICP algorithm converged in at most max_iter steps.
#    transf : np.ndarray, shape = [4, 4]
#        Transformation matrix.
#    estimate : PointCloud
#        Transformed version of source.
#    fitness : float
#        Sum of squares error in the estimated transformation.
#    """
#    cdef GeneralizedIterativeClosestPoint[cpp.PointXYZ, cpp.PointXYZ] gicp
#    return runWithDisplacement(gicp, source, target, max_iter, guessDisp)
#
#
#def icp_nl_guess(_pcl.PointCloud source, _pcl.PointCloud target, max_iter=None, guessDisp=None):
#    """Align source to target using generalized non-linear ICP (ICP-NL).
#
#    Parameters
#    ----------
#    source : PointCloud
#        Source point cloud.
#    target : PointCloud
#        Target point cloud.
#
#    max_iter : integer, optional
#        Maximum number of iterations. If not given, uses the default number
#        hardwired into PCL.
#
#    Returns
#    -------
#    converged : bool
#        Whether the ICP algorithm converged in at most max_iter steps.
#    transf : np.ndarray, shape = [4, 4]
#        Transformation matrix.
#    estimate : PointCloud
#        Transformed version of source.
#    fitness : float
#        Sum of squares error in the estimated transformation.
#    """
#    cdef IterativeClosestPointNonLinear[cpp.PointXYZ, cpp.PointXYZ] icp_nl
#    return runWithDisplacement(icp_nl, source, target, max_iter, guessDisp)
