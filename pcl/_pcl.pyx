#cython: embedsignature=True

from collections import Sequence
import numbers
import numpy as np

cimport numpy as cnp

cimport pcl_defs as cpp

cimport cython
from cython.operator import dereference as deref
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

from shared_ptr cimport sp_assign

cdef extern from "minipcl.h":
#    void setUpSamplingMethod(cpp.MovingLeastSquares_t &, int)
    void mpcl_compute_normals(cpp.PointCloud_t, int ksearch,
                              double searchRadius,
                              cpp.PointNormalCloud_t) except +
    void mpcl_sacnormal_set_axis(cpp.SACSegmentationNormal_t,
                                 double ax, double ay, double az) except +
    void mpcl_extract(cpp.PointCloudPtr_t, cpp.PointCloud_t *,
                      cpp.PointIndices_t *, bool) except +
                      
cdef extern from "adhoc.h":
  void detection_and_correspondence(cpp.PointCloudPtr_t,
                                    cpp.PointCloudPtr_t,
                                    int, int, float*, float*,
                                    int, float,
                                    int , float ,
                                    float , int , int , bool , float ,
                                    bool ,
                                    cpp.vector[int] , cpp.vector[float], cpp.vector[float])



@cython.boundscheck(False)
def doAccumMin(cnp.ndarray[dtype=cnp.float64_t, ndim=1] img, cnp.ndarray[dtype=cnp.uint32_t, ndim=1] inds,  cnp.ndarray[dtype=cnp.float64_t, ndim=1] values, int num):
  cdef cnp.float64_t *arr
  cdef cnp.float64_t *val
  cdef cnp.uint32_t  *ins
  cdef cnp.int32_t   i
  cdef cnp.uint32_t   ix
  cdef cnp.float64_t  v
  cdef cnp.int32_t    n
  n = num
  arr = <cnp.float64_t *>img.data
  val = <cnp.float64_t *>values.data
  ins =  <cnp.uint32_t *>inds.data
  for i in range(n):
    ix = ins[i]
    v  = val[i]
    if arr[ix]>v:
      arr[ix] = v
#    ix = inds[i]
#    v  = values[i]
#    if img[ix]>v:
#      img[ix] = v
  
@cython.boundscheck(False)
def doAccumMax(cnp.ndarray[dtype=cnp.float64_t, ndim=1] img, cnp.ndarray[dtype=cnp.uint32_t, ndim=1] inds,  cnp.ndarray[dtype=cnp.float64_t, ndim=1] values, int num):
  cdef cnp.float64_t *arr
  cdef cnp.float64_t *val
  cdef cnp.uint32_t  *ins
  cdef cnp.int32_t   i
  cdef cnp.uint32_t   ix
  cdef cnp.float64_t  v
  cdef cnp.int32_t    n
  n = num
  arr = <cnp.float64_t *>img.data
  val = <cnp.float64_t *>values.data
  ins =  <cnp.uint32_t *>inds.data
  for i in range(n):
    ix = ins[i]
    v  = val[i]
    if arr[ix]<v:
      arr[ix] = v
#    ix = inds[i]
#    v  = values[i]
#    if img[ix]<v:
#      img[ix] = v
  
@cython.boundscheck(False)
def doAccumSum(cnp.ndarray[dtype=cnp.float64_t, ndim=1] img, cnp.ndarray[dtype=cnp.uint32_t, ndim=1] inds,  cnp.ndarray[dtype=cnp.float64_t, ndim=1] values, int num):
  cdef cnp.float64_t *arr
  cdef cnp.float64_t *val
  cdef cnp.uint32_t  *ins
  cdef cnp.int32_t   i
  cdef cnp.uint32_t   ix
  cdef cnp.float64_t  v
  cdef cnp.int32_t    n
  n = num
  arr = <cnp.float64_t *>img.data
  val = <cnp.float64_t *>values.data
  ins =  <cnp.uint32_t *>inds.data
  for i in range(n):
    arr[ins[i]] += val[i]
#    img[inds[i]] += values[i]
  
@cython.boundscheck(False)
def doAccumCount(cnp.ndarray[dtype=cnp.int16_t, ndim=1] img, cnp.ndarray[dtype=cnp.uint32_t, ndim=1] inds, int num):
  cdef cnp.int16_t *arr
  cdef cnp.uint32_t  *ins
  cdef cnp.int32_t   i
  cdef cnp.uint32_t   ix
  cdef cnp.int32_t    n
  n = num
  arr = <cnp.int16_t *>img.data
  ins =  <cnp.uint32_t *>inds.data
  for i in range(n):
    arr[ins[i]] += 1
#    img[inds[i]] += 1

@cython.boundscheck(False)
def doAccumLast(cnp.ndarray[dtype=cnp.float64_t, ndim=1] img, cnp.ndarray[dtype=cnp.uint32_t, ndim=1] inds,  cnp.ndarray[dtype=cnp.float64_t, ndim=1] values, int num):
  cdef cnp.float64_t *arr
  cdef cnp.float64_t *val
  cdef cnp.uint32_t  *ins
  cdef cnp.int32_t   i
  cdef cnp.uint32_t   ix
  cdef cnp.float64_t  v
  cdef cnp.int32_t    n
  n = num
  arr = <cnp.float64_t *>img.data
  val = <cnp.float64_t *>values.data
  ins =  <cnp.uint32_t *>inds.data
  for i in range(n):
    arr[ins[i]] = val[i]


#cdef extern from "cpd_impl.h":
#    double cpd_comp_P(double* x,double* y, double sigma2,double outlier,double* P1,double* Pt1,double* Px,int N,int M,int D,double *P,double *temp_x)
#    void cpd_comp_correspondence(double* x,double* y, double sigma2,double outlier,double* Pc,int N,int M,int D, double*P, double*P1)




##/* Input arguments */
###define IN_x		prhs[0]
###define IN_y		prhs[1]
###define IN_sigma2	prhs[2]
###define IN_outlier	prhs[3]
##
##
##/* Output arguments */
###define OUT_P1		plhs[0]
###define OUT_Pt1		plhs[1]
###define OUT_Px		plhs[2]
###define OUT_E		plhs[3]
#@cython.boundscheck(False)
#def cpd_comp_P_gateway(cnp.ndarray[cnp.float64_t, ndim=2, mode="fortran"] IN_x, 
#                       cnp.ndarray[cnp.float64_t, ndim=2, mode="fortran"]IN_y, 
#                       double IN_sigma2, double IN_outlier):
#  cdef int N = IN_x.shape[0]
#  cdef int M = IN_y.shape[0]
#  cdef int D = IN_x.shape[1]
#  cdef cnp.ndarray[cnp.float64_t, ndim=1] OUT_P1
#  cdef cnp.ndarray[cnp.float64_t, ndim=1] OUT_Pt1
#  cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="fortran"] OUT_Px
#  cdef cnp.ndarray[cnp.float64_t, ndim=1] P
#  cdef cnp.ndarray[cnp.float64_t, ndim=1] temp_x
#  
#  cdef double E, sigma2, outlier
#  
#  sigma2  = IN_sigma2
#  outlier = IN_outlier
#  
#  OUT_P1  = np.zeros((M,), dtype=np.float64)
#  OUT_Pt1 = np.zeros((N,), dtype=np.float64)
#  OUT_Px  = np.zeros((M, D), dtype=np.float64, order='fortran')
#  P       = np.zeros((M,), dtype=np.float64)
#  temp_x  = np.zeros((D,), dtype=np.float64)
#  
#  E = cpd_comp_P(<double*>IN_x.data, <double*>IN_y.data, sigma2, outlier, 
#                 <double*>OUT_P1.data, <double*>OUT_Pt1.data, 
#                 <double*>OUT_Px.data, N, M, D,
#                 <double*>P.data, <double*>temp_x.data)
#  return (OUT_P1, OUT_Pt1, OUT_Px, E)
##/* Gateway routine */
##void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
##{
##  double *x, *y, *sigma2, *outlier, *P1, *Pt1, *Px, *E;
##  int     N, M, D;
##  
##  /* Get the sizes of each input argument */
##  N = mxGetM(IN_x);
##  M = mxGetM(IN_y);
##  D = mxGetN(IN_x);
##  
##  /* Create the new arrays and set the output pointers to them */
##  OUT_P1     = mxCreateDoubleMatrix(M, 1, mxREAL);
##  OUT_Pt1    = mxCreateDoubleMatrix(N, 1, mxREAL);
##  OUT_Px    = mxCreateDoubleMatrix(M, D, mxREAL);
##  OUT_E       = mxCreateDoubleMatrix(1, 1, mxREAL);
##
##    /* Assign pointers to the input arguments */
##  x      = mxGetPr(IN_x);
##  y       = mxGetPr(IN_y);
##  sigma2       = mxGetPr(IN_sigma2);
##  outlier    = mxGetPr(IN_outlier);
##
## 
##  
##  /* Assign pointers to the output arguments */
##  P1      = mxGetPr(OUT_P1);
##  Pt1      = mxGetPr(OUT_Pt1);
##  Px      = mxGetPr(OUT_Px);
##  E     = mxGetPr(OUT_E);
##   
##  /* Do the actual computations in a subroutine */
##  cpd_comp(x, y, sigma2, outlier, P1, Pt1, Px, E, N, M, D);
##  
##  return;
##}
#
#@cython.boundscheck(False)
#def cpd_comp_correspondence_gateway(
#      cnp.ndarray[cnp.float64_t, ndim=2, mode="fortran"] IN_x, 
#      cnp.ndarray[cnp.float64_t, ndim=2, mode="fortran"]IN_y, 
#      double IN_sigma2, double IN_outlier):
#  cdef int N = IN_x.shape[0]
#  cdef int M = IN_y.shape[0]
#  cdef int D = IN_x.shape[1]
#  cdef cnp.ndarray[cnp.float64_t, ndim=1] OUT_Pc
#  cdef cnp.ndarray[cnp.float64_t, ndim=1] P
#  cdef cnp.ndarray[cnp.float64_t, ndim=1] P1
#
#  cdef double sigma2, outlier
#  
#  sigma2  = IN_sigma2
#  outlier = IN_outlier
#  
#  OUT_Pc  = np.zeros((M,), dtype=np.float64)
#  P       = np.zeros((M,), dtype=np.float64)
#  P1      = np.zeros((M,), dtype=np.float64)
#  
#  cpd_comp_correspondence(<double*>IN_x.data, <double*>IN_y.data, sigma2,
#                          outlier, <double*>OUT_Pc.data, N, M, D, 
#                          <double*>P.data, <double*>P1.data)
#  return OUT_Pc
#
##/* Input arguments */
###define IN_x		prhs[0]
###define IN_y		prhs[1]
###define IN_sigma2	prhs[2]
###define IN_outlier	prhs[3]
##
##
##/* Output arguments */
###define OUT_Pc		plhs[0]
##
##
##
##/* Gateway routine */
##void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
##{
##  double *x, *y, *sigma2, *outlier, *Pc;
##  int     N, M, D;
##  
##  /* Get the sizes of each input argument */
##  N = mxGetM(IN_x);
##  M = mxGetM(IN_y);
##  D = mxGetN(IN_x);
##  
##  /* Create the new arrays and set the output pointers to them */
##  OUT_Pc     = mxCreateDoubleMatrix(M, 1, mxREAL);
##
##    /* Assign pointers to the input arguments */
##  x      = mxGetPr(IN_x);
##  y       = mxGetPr(IN_y);
##  sigma2       = mxGetPr(IN_sigma2);
##  outlier    = mxGetPr(IN_outlier);
##
## 
##  
##  /* Assign pointers to the output arguments */
##  Pc      = mxGetPr(OUT_Pc);
##   
##  /* Do the actual computations in a subroutine */
##  cpd_comp(x, y, sigma2, outlier, Pc, N, M, D);
##  
##  return;
##}



#cdef extern from "adhoc.h":
#    void SimpleResample(cpp.PointCloud_t a, cpp.PointCloud_t b, double searchRadius, bool computeNormals) except +

SAC_RANSAC = cpp.SAC_RANSAC
SAC_LMEDS = cpp.SAC_LMEDS
SAC_MSAC = cpp.SAC_MSAC
SAC_RRANSAC = cpp.SAC_RRANSAC
SAC_RMSAC = cpp.SAC_RMSAC
SAC_MLESAC = cpp.SAC_MLESAC
SAC_PROSAC = cpp.SAC_PROSAC

SACMODEL_PLANE = cpp.SACMODEL_PLANE
SACMODEL_LINE = cpp.SACMODEL_LINE
SACMODEL_CIRCLE2D = cpp.SACMODEL_CIRCLE2D
SACMODEL_CIRCLE3D = cpp.SACMODEL_CIRCLE3D
SACMODEL_SPHERE = cpp.SACMODEL_SPHERE
SACMODEL_CYLINDER = cpp.SACMODEL_CYLINDER
SACMODEL_CONE = cpp.SACMODEL_CONE
SACMODEL_TORUS = cpp.SACMODEL_TORUS
SACMODEL_PARALLEL_LINE = cpp.SACMODEL_PARALLEL_LINE
SACMODEL_PERPENDICULAR_PLANE = cpp.SACMODEL_PERPENDICULAR_PLANE
SACMODEL_PARALLEL_LINES = cpp.SACMODEL_PARALLEL_LINES
SACMODEL_NORMAL_PLANE = cpp.SACMODEL_NORMAL_PLANE 
#SACMODEL_NORMAL_SPHERE = cpp.SACMODEL_NORMAL_SPHERE
SACMODEL_REGISTRATION = cpp.SACMODEL_REGISTRATION
SACMODEL_PARALLEL_PLANE = cpp.SACMODEL_PARALLEL_PLANE
SACMODEL_NORMAL_PARALLEL_PLANE = cpp.SACMODEL_NORMAL_PARALLEL_PLANE
SACMODEL_STICK = cpp.SACMODEL_STICK


cnp.import_array()


cdef class Segmentation:
    """
    Segmentation class for Sample Consensus methods and models
    """
    cdef cpp.SACSegmentation_t *me
    def __cinit__(self):
        self.me = new cpp.SACSegmentation_t()
    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())],\
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)
    def set_model_type(self, cpp.SacModel m):
        self.me.setModelType(m)
    def set_method_type(self, int m):
        self.me.setMethodType (m)
    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)

#yeah, I can't be bothered making this inherit from SACSegmentation, I forget the rules
#for how this works in cython templated extension types anyway
cdef class SegmentationNormal:
    """
    Segmentation class for Sample Consensus methods and models that require the
    use of surface normals for estimation.

    Due to Cython limitations this should derive from pcl.Segmentation, but
    is currently unable to do so.
    """
    cdef cpp.SACSegmentationNormal_t *me
    def __cinit__(self):
        self.me = new cpp.SACSegmentationNormal_t()
    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())],\
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)
    def set_model_type(self, cpp.SacModel m):
        self.me.setModelType(m)
    def set_method_type(self, int m):
        self.me.setMethodType (m)
    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)
    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients (b)
    def set_normal_distance_weight(self, float f):
        self.me.setNormalDistanceWeight (f)
    def set_max_iterations(self, int i):
        self.me.setMaxIterations (i)
    def set_radius_limits(self, float f1, float f2):
        self.me.setRadiusLimits (f1, f2)
    def set_eps_angle(self, double ea):
        self.me.setEpsAngle (ea)
    def set_axis(self, double ax, double ay, double az):
        mpcl_sacnormal_set_axis(deref(self.me),ax,ay,az)

cdef class PointCloud2:
    """Represents a cloud of points in 2-d space."""
    def __cinit__(self, init=None):
        cdef PointCloud2 other

        sp_assign(self.thisptr_shared, new cpp.PointCloud[cpp.PointXY]())

        if init is None:
            return
        elif isinstance(init, (numbers.Integral, np.integer)):
            self.resize(init)
        elif isinstance(init, np.ndarray):
            self.from_array(init)
        elif isinstance(init, type(self)):
            other = init
            self.thisptr()[0] = other.thisptr()[0]
        else:
            raise TypeError("Can't initialize a PointCloud from a %s"
                            % type(init))

    property width:
        """ property containing the width of the point cloud """
        def __get__(self): return self.thisptr().width
    property height:
        """ property containing the height of the point cloud """
        def __get__(self): return self.thisptr().height
    property size:
        """ property containing the number of points in the point cloud """
        def __get__(self): return self.thisptr().size()
    property is_dense:
        """ property containing whether the cloud is dense or not """
        def __get__(self): return self.thisptr().is_dense

    def __repr__(self):
        return "<PointCloud of %d points>" % self.size

    # Pickle support. XXX this copies the entire pointcloud; it would be nice
    # to have an asarray member that returns a view, or even better, implement
    # the buffer protocol (https://docs.python.org/c-api/buffer.html).
    def __reduce__(self):
        return type(self), (self.to_array(),)

    @cython.boundscheck(False)
    def from_array(self, cnp.ndarray[cnp.float32_t, ndim=2] arr not None):
        """
        Fill this object from a 2D numpy array (float32)
        """
        assert arr.shape[1] >= 2

        cdef cnp.npy_intp npts = arr.shape[0]
        self.resize(npts)
        self.thisptr().width = npts
        self.thisptr().height = 1

        cdef cpp.PointXY *p
        for i in range(npts):
            p = cpp.getptr(self.thisptr(), i)
            p.x, p.y = arr[i, 0], arr[i, 1]

    @cython.boundscheck(False)
    def from_arrays(self, lazyarrays, int total):
        """
        Fill this object from a list of 2D numpy array (float32)
        """
        
        cdef cnp.ndarray[cnp.float32_t, ndim=2] arr
        cdef int i, j, k, siz
        cdef cpp.PointXY *p

        self.resize(total)
        self.thisptr().width = total
        self.thisptr().height = 1

        k = 0
        for x in lazyarrays:
          arr = x
          siz = arr.shape[0]
          for j in range(siz):
            p = cpp.getptr(self.thisptr(), k)
            k += 1
            p.x, p.y = arr[j, 0], arr[j, 1]

    @cython.boundscheck(False)
    def to_array(self):
        """
        Return this object as a 2D numpy array (float32)
        """
        cdef float x,y,z
        cdef cnp.npy_intp n = self.thisptr().size()
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] result
        cdef cpp.PointXY *p

        result = np.empty((n, 2), dtype=np.float32)

        for i in range(n):
            p = cpp.getptr(self.thisptr(), i)
            result[i, 0] = p.x
            result[i, 1] = p.y
        return result

    def resize(self, cnp.npy_intp x):
        self.thisptr().resize(x)

    def get_point(self, cnp.npy_intp row, cnp.npy_intp col):
        """
        Return a point (3-tuple) at the given row/column
        """
        cdef cpp.PointXY *p = cpp.getptr_at(self.thisptr(), row, col)
        return p.x, p.y

    def __getitem__(self, cnp.npy_intp idx):
        cdef cpp.PointXY *p = cpp.getptr_at(self.thisptr(), idx)
        return p.x, p.y




cdef class PointCloud:
    """Represents a cloud of points in 3-d space.

    A point cloud can be initialized from either a NumPy ndarray of shape
    (n_points, 3), from a list of triples, or from an integer n to create an
    "empty" cloud of n points.

    To load a point cloud from disk, use pcl.load.
    """
    def __cinit__(self, init=None):
        cdef PointCloud other

        sp_assign(self.thisptr_shared, new cpp.PointCloud[cpp.PointXYZ]())

        if init is None:
            return
        elif isinstance(init, (numbers.Integral, np.integer)):
            self.resize(init)
        elif isinstance(init, np.ndarray):
            self.from_array(init)
        elif isinstance(init, Sequence):
            self.from_list(init)
        elif isinstance(init, type(self)):
            other = init
            self.thisptr()[0] = other.thisptr()[0]
        else:
            raise TypeError("Can't initialize a PointCloud from a %s"
                            % type(init))

    property width:
        """ property containing the width of the point cloud """
        def __get__(self): return self.thisptr().width
    property height:
        """ property containing the height of the point cloud """
        def __get__(self): return self.thisptr().height
    property size:
        """ property containing the number of points in the point cloud """
        def __get__(self): return self.thisptr().size()
    property is_dense:
        """ property containing whether the cloud is dense or not """
        def __get__(self): return self.thisptr().is_dense

    def __repr__(self):
        return "<PointCloud of %d points>" % self.size

    # Pickle support. XXX this copies the entire pointcloud; it would be nice
    # to have an asarray member that returns a view, or even better, implement
    # the buffer protocol (https://docs.python.org/c-api/buffer.html).
    def __reduce__(self):
        return type(self), (self.to_array(),)

    property sensor_origin:
        def __get__(self):
            cdef cpp.Vector4f origin = self.thisptr().sensor_origin_
            cdef float *data = origin.data()
            return np.array([data[0], data[1], data[2], data[3]],
                            dtype=np.float32)

    property sensor_orientation:
        def __get__(self):
            # NumPy doesn't have a quaternion type, so we return a 4-vector.
            cdef cpp.Quaternionf o = self.thisptr().sensor_orientation_
            return np.array([o.w(), o.x(), o.y(), o.z()])

    @cython.boundscheck(False)
    def from_array(self, cnp.ndarray[cnp.float32_t, ndim=2] arr not None):
        """
        Fill this object from a 2D numpy array (float32)
        """
        assert arr.shape[1] == 3

        cdef cnp.npy_intp npts = arr.shape[0]
        self.resize(npts)
        self.thisptr().width = npts
        self.thisptr().height = 1

        cdef cpp.PointXYZ *p
        for i in range(npts):
            p = cpp.getptr(self.thisptr(), i)
            p.x, p.y, p.z = arr[i, 0], arr[i, 1], arr[i, 2]

    @cython.boundscheck(False)
    def from_arrays(self, lazyarrays, int total):
        """
        Fill this object from a list of 2D numpy array (float32)
        """
        
        cdef cnp.ndarray[cnp.float32_t, ndim=2] arr
        cdef int i, j, k, siz
        cdef cpp.PointXYZ *p

        self.resize(total)
        self.thisptr().width = total
        self.thisptr().height = 1

        k = 0
        for x in lazyarrays:
          arr = x
          siz = arr.shape[0]
          for j in range(siz):
            p = cpp.getptr(self.thisptr(), k)
            k += 1
            p.x, p.y, p.z = arr[j, 0], arr[j, 1], arr[j, 2]

    @cython.boundscheck(False)
    def to_array(self):
        """
        Return this object as a 2D numpy array (float32)
        """
        cdef float x,y,z
        cdef cnp.npy_intp n = self.thisptr().size()
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] result
        cdef cpp.PointXYZ *p

        result = np.empty((n, 3), dtype=np.float32)

        for i in range(n):
            p = cpp.getptr(self.thisptr(), i)
            result[i, 0] = p.x
            result[i, 1] = p.y
            result[i, 2] = p.z
        return result

    def from_list(self, _list):
        """
        Fill this pointcloud from a list of 3-tuples
        """
        cdef Py_ssize_t npts = len(_list)
        cdef cpp.PointXYZ *p
        cdef cnp.npy_intp n

        self.resize(npts)
        self.thisptr().width = npts
        self.thisptr().height = 1
        for i, l in enumerate(_list):
            n = i
            p = cpp.getptr(self.thisptr(), n)
            p.x, p.y, p.z = l

    def to_list(self):
        """
        Return this object as a list of 3-tuples
        """
        return self.to_array().tolist()

    def resize(self, cnp.npy_intp x):
        self.thisptr().resize(x)

    def get_point(self, cnp.npy_intp row, cnp.npy_intp col):
        """
        Return a point (3-tuple) at the given row/column
        """
        cdef cpp.PointXYZ *p = cpp.getptr_at(self.thisptr(), row, col)
        return p.x, p.y, p.z

    def __getitem__(self, cnp.npy_intp idx):
        cdef cpp.PointXYZ *p = cpp.getptr_at(self.thisptr(), idx)
        return p.x, p.y, p.z

    def from_file(self, char *f):
        """
        Fill this pointcloud from a file (a local path).
        Only pcd files supported currently.

        Deprecated; use pcl.load instead.
        """
        return self._from_pcd_file(f)

    def _from_pcd_file(self, const char *s):
        cdef int error = 0
        with nogil:
            ok = cpp.loadPCDFile(string(s), deref(self.thisptr()))
        return error

#    def _from_ply_file(self, const char *s):
#        cdef int ok = 0
#        with nogil:
#            error = cpp.loadPLYFile(string(s), deref(self.thisptr()))
#        return error

    def to_file(self, const char *fname, bool ascii=True):
        """Save pointcloud to a file in PCD format.

        Deprecated: use pcl.save instead.
        """
        return self._to_pcd_file(fname, not ascii)

    def _to_pcd_file(self, const char *f, bool binary=False):
        cdef int error = 0
        cdef string s = string(f)
        with nogil:
            error = cpp.savePCDFile(s, deref(self.thisptr()), binary)
        return error

#    def _to_ply_file(self, const char *f, bool binary=False):
#        cdef int error = 0
#        cdef string s = string(f)
#        with nogil:
#            error = cpp.savePLYFile(s, deref(self.thisptr()), binary)
#        return error

    def make_segmenter(self):
        """
        Return a pcl.Segmentation object with this object set as the input-cloud
        """
        seg = Segmentation()
        cdef cpp.SACSegmentation_t *cseg = <cpp.SACSegmentation_t *>seg.me
        cseg.setInputCloud(self.thisptr_shared)
        return seg

    def make_segmenter_normals(self, int ksearch=-1, double searchRadius=-1.0):
        """
        Return a pcl.SegmentationNormal object with this object set as the input-cloud
        """
        cdef cpp.PointNormalCloud_t normals
        mpcl_compute_normals(deref(self.thisptr()), ksearch, searchRadius,
                             normals)

        seg = SegmentationNormal()
        cdef cpp.SACSegmentationNormal_t *cseg = <cpp.SACSegmentationNormal_t *>seg.me
        cseg.setInputCloud(self.thisptr_shared)
        cseg.setInputNormals (normals.makeShared());

        return seg

    def make_statistical_outlier_filter(self):
        """
        Return a pcl.StatisticalOutlierRemovalFilter object with this object set as the input-cloud
        """
        fil = StatisticalOutlierRemovalFilter()
        cdef cpp.StatisticalOutlierRemoval_t *cfil = <cpp.StatisticalOutlierRemoval_t *>fil.me
        cfil.setInputCloud(self.thisptr_shared)
        return fil

    def make_voxel_grid_filter(self):
        """
        Return a pcl.VoxelGridFilter object with this object set as the input-cloud
        """
        fil = VoxelGridFilter()
        cdef cpp.VoxelGrid_t *cfil = <cpp.VoxelGrid_t *>fil.me
        cfil.setInputCloud(self.thisptr_shared)
        return fil

    def make_passthrough_filter(self):
        """
        Return a pcl.PassThroughFilter object with this object set as the input-cloud
        """
        fil = PassThroughFilter()
        cdef cpp.PassThrough_t *cfil = <cpp.PassThrough_t *>fil.me
        cfil.setInputCloud(self.thisptr_shared)
        return fil

#    def make_moving_least_squares(self):
#        """
#        Return a pcl.MovingLeastSquares object with this object as input cloud.
#        """
#        mls = MovingLeastSquares()
#        cdef cpp.MovingLeastSquares_t *cmls = <cpp.MovingLeastSquares_t *>mls.me
#        cmls.setInputCloud(self.thisptr_shared)
#        return mls

    def make_kdtree_flann(self):
        """
        Return a pcl.kdTreeFLANN object with this object set as the input-cloud

        Deprecated: use the pcl.KdTreeFLANN constructor on this cloud.
        """
        return KdTreeFLANN(self)

    def make_octree(self, double resolution):
        """
        Return a pcl.octree object with this object set as the input-cloud
        """
        octree = OctreePointCloud(resolution)
        octree.set_input_cloud(self)
        return octree

    def extract(self, pyindices, bool negative=False):
        """
        Given a list of indices of points in the pointcloud, return a 
        new pointcloud containing only those points.
        """
        cdef PointCloud result
        cdef cpp.PointIndices_t *ind = new cpp.PointIndices_t()

        for i in pyindices:
            ind.indices.push_back(i)

        result = PointCloud()
        mpcl_extract(self.thisptr_shared, result.thisptr(), ind, negative)
        # XXX are we leaking memory here? del ind causes a double free...

        return result

cdef class StatisticalOutlierRemovalFilter:
    """
    Filter class uses point neighborhood statistics to filter outlier data.
    """
    cdef cpp.StatisticalOutlierRemoval_t *me
    def __cinit__(self):
        self.me = new cpp.StatisticalOutlierRemoval_t()
    def __dealloc__(self):
        del self.me

    def set_mean_k(self, int k):
        """
        Set the number of points (k) to use for mean distance estimation. 
        """
        self.me.setMeanK(k)

    def set_std_dev_mul_thresh(self, double std_mul):
        """
        Set the standard deviation multiplier threshold.
        """
        self.me.setStddevMulThresh(std_mul)

    def set_negative(self, bool negative):
        """
        Set whether the indices should be returned, or all points except the indices. 
        """
        self.me.setNegative(negative)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc

#cdef class MovingLeastSquares:
#    """
#    Smoothing class which is an implementation of the MLS (Moving Least Squares)
#    algorithm for data smoothing and improved normal estimation.
#    """
#    cdef cpp.MovingLeastSquares_t *me
#    def __cinit__(self):
#        self.me = new cpp.MovingLeastSquares_t()
#    def __dealloc__(self):
#        del self.me
#
#    def set_search_radius(self, double radius):
#        """
#        Set the sphere radius that is to be used for determining the k-nearest neighbors used for fitting. 
#        """
#        self.me.setSearchRadius (radius)
#
#    def set_polynomial_order(self, bool order):
#        """
#        Set the order of the polynomial to be fit. 
#        """
#        self.me.setPolynomialOrder(order)
#
#    def set_polynomial_fit(self, bint fit):
#        """
#        Sets whether the surface and normal are approximated using a polynomial,
#        or only via tangent estimation.
#        """
#        self.me.setPolynomialFit(fit)
#
#    def set_point_density(self, density):
#      self.me.setPointDensity(density)
#    
#    def set_upsampling_method(self, method):
#      cdef int val
#      if   method=='NONE':
#        val=0
#      elif method=='DISTINCT_CLOUD':
#        val=1
#      elif method=='SAMPLE_LOCAL_PLANE':
#        val=2
#      elif method=='RANDOM_UNIFORM_DENSITY':
#        val=3
#      elif method=='VOXEL_GRID_DILATION':
#        val=4
#      else:
#        raise ValueError('upsampling method not understood: '+str(method))
#      setUpSamplingMethod(self.me[0], val)
#
#    def process(self):
#        """
#        Apply the smoothing according to the previously set values and return
#        a new pointcloud
#        """
#        cdef PointCloud pc = PointCloud()
#        self.me.process(pc.thisptr()[0])
#        return pc

cdef class VoxelGridFilter:
    """
    Assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    """
    cdef cpp.VoxelGrid_t *me
    def __cinit__(self):
        self.me = new cpp.VoxelGrid_t()
    def __dealloc__(self):
        del self.me

    def set_leaf_size (self, float x, float y, float z):
        """
        Set the voxel grid leaf size.
        """
        self.me.setLeafSize(x,y,z)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc

cdef class PassThroughFilter:
    """
    Passes points in a cloud based on constraints for one particular field of the point type
    """
    cdef cpp.PassThrough_t *me
    def __cinit__(self):
        self.me = new cpp.PassThrough_t()
    def __dealloc__(self):
        del self.me

    def set_filter_field_name(self, field_name):
        cdef bytes fname_ascii
        if isinstance(field_name, unicode):
            fname_ascii = field_name.encode("ascii")
        elif not isinstance(field_name, bytes):
            raise TypeError("field_name should be a string, got %r"
                            % field_name)
        else:
            fname_ascii = field_name
        self.me.setFilterFieldName(string(fname_ascii))

    def set_filter_limits(self, float filter_min, float filter_max):
        self.me.setFilterLimits (filter_min, filter_max)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc


cdef class KdTreeFLANN2:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTreeFLANN(pc).
    """
    cdef cpp.KdTreeFLANN2_t *me

    def __cinit__(self, PointCloud2 pc not None):
        self.me = new cpp.KdTreeFLANN2_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    def nearest_k_search_for_cloud(self, PointCloud2 pc not None, int k=1):
        """
        Find the k nearest neighbours and squared distances for all points
        in the pointcloud. Results are in ndarrays, size (pc.size, k)
        Returns: (k_indices, k_sqr_distances)
        """
        cdef cnp.npy_intp n_points = pc.size
        cdef cnp.ndarray[float, ndim=2] sqdist = np.zeros((n_points, k),
                                                          dtype=np.float32)
        cdef cnp.ndarray[int, ndim=2] ind = np.zeros((n_points, k),
                                                     dtype=np.int32)

        for i in range(n_points):
            self._nearest_k(pc, i, k, ind[i], sqdist[i])
        return ind, sqdist

    def nearest_k_search_for_point(self, PointCloud2 pc not None, int index,
                                   int k=1):
        """
        Find the k nearest neighbours and squared distances for the point
        at pc[index]. Results are in ndarrays, size (k)
        Returns: (k_indices, k_sqr_distances)
        """
        cdef cnp.ndarray[float] sqdist = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] ind = np.zeros(k, dtype=np.int32)

        self._nearest_k(pc, index, k, ind, sqdist)
        return ind, sqdist

    @cython.boundscheck(False)
    cdef void _nearest_k(self, PointCloud2 pc, int index, int k,
                         cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
                         cnp.ndarray[ndim=1, dtype=float, mode='c'] sqdist
                        ) except +:
        # k nearest neighbors query for a single point.
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        k_indices.resize(k)
        k_sqr_distances.resize(k)
        self.me.nearestKSearch(pc.thisptr()[0], index, k, k_indices,
                               k_sqr_distances)

        for i in range(k):
            sqdist[i] = k_sqr_distances[i]
            ind[i] = k_indices[i]

cdef class KdTreeFLANN:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTreeFLANN(pc).
    """
    cdef cpp.KdTreeFLANN_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new cpp.KdTreeFLANN_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    def nearest_k_search_for_cloud(self, PointCloud pc not None, int k=1):
        """
        Find the k nearest neighbours and squared distances for all points
        in the pointcloud. Results are in ndarrays, size (pc.size, k)
        Returns: (k_indices, k_sqr_distances)
        """
        cdef cnp.npy_intp n_points = pc.size
        cdef cnp.ndarray[float, ndim=2] sqdist = np.zeros((n_points, k),
                                                          dtype=np.float32)
        cdef cnp.ndarray[int, ndim=2] ind = np.zeros((n_points, k),
                                                     dtype=np.int32)

        for i in range(n_points):
            self._nearest_k(pc, i, k, ind[i], sqdist[i])
        return ind, sqdist

    def nearest_k_search_for_point(self, PointCloud pc not None, int index,
                                   int k=1):
        """
        Find the k nearest neighbours and squared distances for the point
        at pc[index]. Results are in ndarrays, size (k)
        Returns: (k_indices, k_sqr_distances)
        """
        cdef cnp.ndarray[float] sqdist = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] ind = np.zeros(k, dtype=np.int32)

        self._nearest_k(pc, index, k, ind, sqdist)
        return ind, sqdist

    @cython.boundscheck(False)
    cdef void _nearest_k(self, PointCloud pc, int index, int k,
                         cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
                         cnp.ndarray[ndim=1, dtype=float, mode='c'] sqdist
                        ) except +:
        # k nearest neighbors query for a single point.
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        k_indices.resize(k)
        k_sqr_distances.resize(k)
        self.me.nearestKSearch(pc.thisptr()[0], index, k, k_indices,
                               k_sqr_distances)

        for i in range(k):
            sqdist[i] = k_sqr_distances[i]
            ind[i] = k_indices[i]

cdef cpp.PointXYZ to_point_t(point):
    cdef cpp.PointXYZ p
    p.x = point[0]
    p.y = point[1]
    p.z = point[2]
    return p

cdef class OctreePointCloud:
    """
    Octree pointcloud
    """
    cdef cpp.OctreePointCloud_t *me

    def __cinit__(self, double resolution):
        self.me = NULL
        if resolution <= 0.:
            raise ValueError("Expected resolution > 0., got %r" % resolution)

    def __init__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = new cpp.OctreePointCloud_t(resolution)

    def __dealloc__(self):
        del self.me
        self.me = NULL      # just to be sure

    def set_input_cloud(self, PointCloud pc):
        """
        Provide a pointer to the input data set.
        """
        self.me.setInputCloud(pc.thisptr_shared)

    def define_bounding_box(self):
        """
        Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
        """
        self.me.defineBoundingBox()
        
    def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
        """
        Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
        """
        self.me.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    def add_points_from_input_cloud(self):
        """
        Add points from input point cloud to octree.
        """
        self.me.addPointsFromInputCloud()

    def delete_tree(self):
        """
        Delete the octree structure and its leaf nodes.
        """
        self.me.deleteTree()

    def is_voxel_occupied_at_point(self, point):
        """
        Check if voxel at given point coordinates exist.
        """
        return self.me.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    def get_occupied_voxel_centers(self):
        """
        Get list of centers of all occupied voxels.
        """
        cdef cpp.AlignedPointTVector_t points_v
        cdef int num = self.me.getOccupiedVoxelCenters (points_v)
        return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    def delete_voxel_at_point(self, point):
        """
        Delete leaf node / voxel at given point.
        """
        self.me.deleteVoxelAtPoint(to_point_t(point))

cdef class OctreePointCloudSearch(OctreePointCloud):
    """
    Octree pointcloud search
    """
    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = <cpp.OctreePointCloud_t*> new cpp.OctreePointCloudSearch_t(resolution)
 
    def radius_search (self, point, double radius, unsigned int max_nn = 0):
        """
        Search for all neighbors of query point that are within a given radius.

        Returns: (k_indices, k_sqr_distances)
        """
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        if max_nn > 0:
            k_indices.resize(max_nn)
            k_sqr_distances.resize(max_nn)
        cdef int k = (<cpp.OctreePointCloudSearch_t*>self.me).radiusSearch(to_point_t(point), radius, k_indices, k_sqr_distances, max_nn)
        cdef cnp.ndarray[float] np_k_sqr_distances = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] np_k_indices = np.zeros(k, dtype=np.int32)
        for i in range(k):
            np_k_sqr_distances[i] = k_sqr_distances[i]
            np_k_indices[i] = k_indices[i]
        return np_k_indices, np_k_sqr_distances


cdef detection_and_correspondenceSIFT(PointCloud cloud1, PointCloud cloud2, 
                                      int num1, int num2,
                                      cnp.ndarray[ndim=1, dtype=float] intensities1,
                                      cnp.ndarray[ndim=1, dtype=float] intensities2,
                                      int normal_num_neighs, float normal_radius,
                                      int feature_num_neighs, float feature_radius,
                                      float min_scale, int nr_octaves, int nr_scales_per_octave, bool usecontrast, float min_contrast,
                                      bool useOldMethod):

  cdef vector[int] correspondences
  cdef vector[float] scores1
  cdef vector[float] scores2
  cdef cnp.ndarray[ndim=1, dtype=int, mode='c'] corrs
  cdef cnp.ndarray[ndim=1, dtype=float, mode='c'] scors1
  cdef cnp.ndarray[ndim=1, dtype=float, mode='c'] scors2

  detection_and_correspondence(cloud1.thisptr_shared, cloud2.thisptr_shared,
                                      num1, num2,
                                      <float*>intensities1.data,
                                      <float*>intensities2.data,
                                      normal_num_neighs, normal_radius,
                                      feature_num_neighs, feature_radius,
                                      min_scale, nr_octaves, nr_scales_per_octave, usecontrast, min_contrast,
                                      useOldMethod, correspondences, scores1, scores2)
  corrs = np.empty((correspondences.size(),),dtype=int)
  scors1 = np.empty((scores1.size(),),dtype=int)
  scors2 = np.empty((scores2.size(),),dtype=float)
  for k in range(correspondences.size()):
    corrs[k] = correspondences[k]
  for k in range(scores1.size()):
    scors1[k] = scores1[k]
  for k in range(scores2.size()):
    scors2[k] = scores2[k]
  return (corrs, scors1, scors2)
  
