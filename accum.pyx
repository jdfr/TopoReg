#cython: embedsignature=True

cimport numpy as cnp
import numpy as np

cimport cython


ctypedef union ieee754:
  cnp.float64_t f
  cnp.uint64_t u

#portable isnan definition
cdef cnp.uint64_t isnanc(cnp.float64_t val):
  cdef ieee754 uni
  uni.f = val
  return ( <unsigned>(uni.u >> 32) & 0x7fffffff ) + ( <unsigned>uni.u != 0 ) > 0x7ff00000

def isnan(cnp.float64_t val):
  return isnanc(val)

@cython.boundscheck(False)
def doAccumMin(cnp.ndarray[dtype=cnp.float64_t, ndim=1] img, cnp.ndarray[dtype=cnp.uint32_t, ndim=1] inds,  cnp.ndarray[dtype=cnp.float64_t, ndim=1] values, int num):
  cdef cnp.float64_t *arr
  cdef cnp.float64_t *val
  cdef cnp.uint32_t  *ins
  cdef cnp.int32_t    i
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
  cdef cnp.int32_t    i
  cdef cnp.uint32_t   ix
  cdef cnp.float64_t  v
  cdef cnp.int32_t    n
  n   = num
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
  cdef cnp.int32_t   n
  n   = num
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
  cdef cnp.int32_t   n
  n   = num
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
  cdef cnp.int32_t   n
  n   = num
  arr = <cnp.float64_t *>img.data
  val = <cnp.float64_t *>values.data
  ins =  <cnp.uint32_t *>inds.data
  for i in range(n):
    arr[ins[i]] = val[i]

@cython.boundscheck(False)
def doAccumFirst(cnp.ndarray[dtype=cnp.float64_t, ndim=1] img, cnp.ndarray[dtype=cnp.uint32_t, ndim=1] inds,  cnp.ndarray[dtype=cnp.float64_t, ndim=1] values, int num):
  cdef cnp.float64_t *arr
  cdef cnp.float64_t *val
  cdef cnp.uint32_t  *ins
  cdef cnp.int32_t    i
  cdef cnp.uint32_t   ind
  cdef cnp.int32_t    n
  cdef ieee754        uni
  
  n    = num
  arr  = <cnp.float64_t *>img.data
  val  = <cnp.float64_t *>values.data
  ins  =  <cnp.uint32_t *>inds.data
  for i in range(n):
    ind = ins[i]
    #manual inlining of isnan
    uni.f = arr[ind]
    if ( <unsigned>(uni.u >> 32) & 0x7fffffff ) + ( <unsigned>uni.u != 0 ) > 0x7ff00000:
      arr[ind] = val[i]
    #if isnan(arr[ind]):
    #  arr[ind] = val[i]


cdef enum:
  U, L, D, R

#from http://www.mathworks.com/matlabcentral/fileexchange/37671-matlabsimulink-for-digital-signal-processing/content/zigzag.m
#function v = zigzag(u)
#% returns a vector v with the elements of a matrix u in zigzag order.
#% Copyleft: Won Y. Yang, wyyang53@hanmail.net, CAU for academic use 
#[M,N] = size(u);  m=1; n=1; v(1)=u(m,n);  d='r';
#for i=2:M*N
#   switch d
#     case 'u',  m=m-(m>1); n=n+(n<N); v(i) = u(m,n);  
#                if n==N,  d='d';  elseif m==1, d='r'; end
#     case 'l',  m=m+(m<M); n=n-(n>1); v(i) = u(m,n);  
#                if m==M, d='r'; elseif n==1, d='d'; end  
#     case 'd',  m=m+(m<M); v(i) = u(m,n);  
#                if n==1,  d='u';  else  d='l';  end
#     case 'r',  n=n+(n<N); v(i) = u(m,n);  
#                if m==1,  d='l';  else d='u';  end
#   end
#end
@cython.boundscheck(False)
def zigzagOrder(cnp.ndarray[dtype=cnp.int32_t, ndim=2] mat):
    cdef cnp.int32_t n, m, N, M, N1, M1, total, i, j, mode, err
    cdef cnp.ndarray[dtype=cnp.int32_t, ndim=1] vals
    M       = mat.shape[0]
    N       = mat.shape[1]
    N1      = N-1
    M1      = M-1
    m = n   = 0
    total   = M*N
    mode    = R
    vals    = np.empty((total,), dtype=np.int32)
    vals[0] = mat[m,n]
    for i in range(1, total):
      if mode==U:
          m      -= m>0
          n      += n<N1
          if n==N1:
            mode  = D
          elif m==0:
            mode  = R
      elif mode==L:
          m      += m<M1
          n      -= n>0
          if m==M1:
            mode  = R
          elif n==0:
            mode  = D
      elif mode==D:
          m      += m<M1
          if n==0:
            mode  = U
          else:
            mode  = L
      elif mode==R:
          n      += n<N1
          if m==0:
            mode  = L
          else:
            mode  = U
      vals[i] = mat[m,n]
    return vals
        
@cython.boundscheck(False)
def zigzagOrder(cnp.int32_t M, cnp.int32_t N): #M=nrows, N=ncols
    cdef cnp.int32_t n, m, N1, M1, total, i, j, mode, err
    cdef cnp.ndarray[dtype=cnp.int32_t, ndim=2] idxs
    N1        = N-1
    M1        = M-1
    m = n     = 0
    total     = M*N
    mode      = R
    idxs      = np.empty((total,2), dtype=np.int32)
    idxs[0,0] = m
    idxs[0,1] = n
    for i in range(1, total):
      if mode==U:
          m      -= m>0
          n      += n<N1
          if n==N1:
            mode  = D
          elif m==0:
            mode  = R
      elif mode==L:
          m      += m<M1
          n      -= n>0
          if m==M1:
            mode  = R
          elif n==0:
            mode  = D
      elif mode==D:
          m      += m<M1
          if n==0:
            mode  = U
          else:
            mode  = L
      elif mode==R:
          n      += n<N1
          if m==0:
            mode  = L
          else:
            mode  = U
      idxs[i,0] = m
      idxs[i,1] = n
    return idxs
      


