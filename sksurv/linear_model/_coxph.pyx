import numpy as np
cimport numpy as np
cimport cython

np.import_array()

cdef extern from "coxph.cpp" nogil:
    np.npy_uint64 coxph_fit[T](
        T* X_ptr, np.npy_uint8* event_ptr, T* time_ptr, T* w_ptr, T* alpha_ptr,
        np.npy_int64 n_samples, np.npy_int64 n_features, T tol, np.npy_uint64 n_iter, bint breslow
    ) except +

@cython.boundscheck(False)
@cython.wraparound(False)
def fit(np.ndarray[np.float64_t, ndim=2, mode="fortran"] X,
        np.ndarray[np.uint8_t, ndim=1] event,
        np.ndarray[np.float64_t, ndim=1] time,
        np.ndarray[np.float64_t, ndim=1] w,
        np.ndarray[np.float64_t, ndim=1] alpha,
        np.npy_float64 tol,
        int n_iter,
        bint breslow):
    """
    Fit Cox PH model using C++ implementation.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Data matrix

    event : array-like, shape = (n_samples,)
        A structured array containing the binary event indicator

    time : array-like, shape = (n_samples,)
        Time of event or time of censoring

    w : array-like, shape = (n_features,)
        Initial values for the coefficients

    alpha : array-like, shape = (n_features,)
        Regularization parameters

    tol : float
        Convergence tolerance

    n_iter : int
        Maximum number of iterations

    breslow : bool
        Whether to use Breslow's method for tie handling
    """
    iter_opt = coxph_fit(
        <np.npy_float64*>X.data,
        <np.npy_uint8*>event.data,
        <np.npy_float64*>time.data,
        <np.npy_float64*>w.data,
        <np.npy_float64*>alpha.data,
        X.shape[0],
        X.shape[1],
        tol,
        n_iter,
        breslow
    )
    return iter_opt, w
