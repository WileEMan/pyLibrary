import numpy as np
import copy

# Diagnostics
import matplotlib.pyplot as plot

pi = np.pi

## Calculation Helpers

# calc_axes_delta(): calculate dx as x[1] - x[0], except allow the 0 position to be passed in such that 
# it could be calculated from the middle or end of the array as well.  And dx can be n-dimensional.

# for 2-D where g is setup as g[X,Y]:
#   dx = axes[0, 1, 0] - axes[0, 0, 0]
#   dy = axes[1, 0, 1] - axes[1, 0, 0]

# for 3-D where g is setup as g[X,Y,Z]:
#   dx = axes[0, 1, 0, 0] - axes[0, 0, 0, 0]
#   dy = axes[1, 0, 1, 0] - axes[1, 0, 0, 0]
#   dz = axes[2, 0, 0, 1] - axes[2, 0, 0, 0]

def calc_axes_delta(axes, from_indices = None):
    if len(axes.shape) == 1:
        # 1-d case, much easier.        
        if from_indices is None: return axes[1] - axes[0]
        return axes[from_indices+1] - axes[from_indices]

    # If axes.shape has more than one element, then the first entry is the number of dimensions.
    # This can still be a 1D case if the first dimension has length of 1.

    # In the 2+-d case, axes not only contains a length for each individual axis, but the first
    # shape element specifies the number of dimensions.  The dimensionality of axes is one higher 
    # than its corresponding g(x) function because each x value has multiple ordinates whereas
    # the g(x) function only has a single output value for each x.
    tdims = axes.shape[0]
    
    if from_indices is None: from_indices = np.zeros(axes.shape[0], dtype='int').tolist()
    if isinstance(from_indices, np.ndarray): from_indices = from_indices.tolist()            
    dr = np.zeros(axes.shape[0])
    for iDim in range(tdims):
        indices0 = [iDim] + from_indices
        indices1 = copy.deepcopy(indices0)            
        indices1[1 + iDim] += 1
        dr[iDim] = axes[tuple(indices1)] - axes[tuple(indices0)] 
    return dr

##  Aliasing helpers (described inside Forward()).

def ShiftBy(x,dx):        
    if isinstance(dx,float) or isinstance(dx,int):     
        # 1-d case, much easier.
        ErrorOnIndex0 = x[0] / dx
        sb = int((round(ErrorOnIndex0) % len(x)) - len(x))
        return 0 if sb == -len(x) else sb
    
    # 2+-D case
    if isinstance(x,list): x = np.asarray(x)
    if isinstance(dx,list): dx = np.asarray(dx)
    tdims = x.shape[0]
    zero_index = (slice(None),) + (0,) * tdims
    ErrorOnIndex0 = x[zero_index] / dx           # Equivalent to x[:, 0, 0, ..., 0] / dx.
    dim_shapes = x.shape[1:]
    sb = ((np.round(ErrorOnIndex0) % dim_shapes) - dim_shapes).astype('int')
    for iDim in range(sb.size): 
        if sb[iDim] == -dim_shapes[iDim]: sb[iDim] = 0    
    return sb

def IndexOfZeroAlias(x,dx): return -ShiftBy(x,dx)

def PositionOfZeroAlias(x,dx): return x[(slice(None),) + tuple(IndexOfZeroAlias(x,dx))]

def NumberPeriodsToZero(x,dx): 
    if isinstance(dx,float) or isinstance(dx,int):
        # 1-d case:
        return np.ceil((x[0] - dx/2) / (len(x) * dx))
        
    # 2+-D case
    if isinstance(x,list): x = np.asarray(x)
    if isinstance(dx,list): dx = np.asarray(dx)
    tdims = x.shape[0]
    zero_index = (slice(None),) + (0,) * tdims
    dim_shapes = x.shape[1:]
    return np.ceil((x[zero_index] - dx/2) / (dim_shapes * dx))

def ZeroAliasError(x,dx): 
    if isinstance(dx,float) or isinstance(dx,int):
        # 1-d case:
        return x[IndexOfZeroAlias(x,dx)] - (NumberPeriodsToZero(x,dx) * (len(x) * dx))
        
    # 2+-D case
    if isinstance(x,list): x = np.asarray(x)
    if isinstance(dx,list): dx = np.asarray(dx)
    dim_shapes = x.shape[1:]
    return x[(slice(None),) + tuple(IndexOfZeroAlias(x,dx))] - (NumberPeriodsToZero(x,dx) * (dim_shapes * dx))
    
## N-D construction helpers

# mgrid_linspace2d() provides numpy.mgrid but using semantics similar to linspace.  The
# arguments to linspace() are the same except each argument is now a 2-D list, tuple, or array.
# Example:
#   mgrid_linspace2d([0,0], [3,3], [4,4]) produces a 2-D meshed version of np.linspace(0, 3, 4).
def mgrid_linspace2d(start, stop, N, endpoint = True):
    if endpoint:
        return np.mgrid[start[0]:stop[0]:(1j*N[0]), start[1]:stop[1]:(1j*N[1])]
    else:        
        ret = np.mgrid[start[0]:stop[0]:(1j*(N[0] + 1)), start[1]:stop[1]:(1j*(N[1] + 1))]
        return ret[:,0:-1,0:-1]

# mgrid_arange2d() provides a numpy.mgrid but using semantics similar to arange.  The 
# arguments to arange() are the same except each argument is now a 2-D list, tuple or array.
# Example:
#   mgrid_arange([0,0], [4,4], [1,1]) produces the same result as the above example for 
#   mgrid_linspace2d().  Note that the stop position has to be higher because np.linspace and 
#   mgrid_linspace2d() default to endpoint = True.
def mgrid_arange2d(start, stop, dr):
    N_x = np.floor((stop[0] - start[0]) / dr[0]) + 1
    N_y = np.floor((stop[1] - start[1]) / dr[1]) + 1
    ret = np.mgrid[start[0]:stop[0]:(1j*N_x), start[1]:stop[1]:(1j*N_y)]
    return ret[:,0:-1,0:-1]
    
# mgrid_linspace() provides numpy.mgrid but using semantics similar to linspace.  The
# arguments to linspace() are the same except each argument is now a N-D list, tuple, or array.
# Example:
#   mgrid_linspace([0], [3], [4]) produces a 1-D meshed version of np.linspace(0, 3, 4).
# In the 1-D case, a first dimension of length 1 is prepended.  The above example would have 
# shape of (1,4).
def mgrid_linspace(start, stop, N, endpoint = True):    
    if isinstance(start, int): start = np.asarray([start])
    if isinstance(stop, int): stop = np.asarray([stop])
    if isinstance(N, int): N = np.asarray([N])
    
    axes = []    
    for iDim in range(len(start)):
        axes.append(np.linspace(start[iDim], stop[iDim], N[iDim], endpoint = endpoint))    
    return np.asarray(np.meshgrid(*axes, indexing="ij"))

# mgrid_arange() provides a numpy.mgrid but using semantics similar to arange.  The 
# arguments to arange() are the same except each argument is now a N-D list, tuple or array.
# Example:
#   mgrid_arange([0], [4], [1]) produces the same result as the above example for 
#   mgrid_linspace().  Note that the stop position has to be higher because np.linspace and 
#   mgrid_linspace() default to endpoint = True.
def mgrid_arange(start, stop, dr):
    if isinstance(start, int): start = np.asarray([start])
    if isinstance(stop, int): stop = np.asarray([stop])
    if isinstance(dr, int): dr = np.asarray([dr])
    
    axes = []
    for iDim in range(len(start)):
        axes.append(np.arange(start[iDim], stop[iDim], dr[iDim]))
    return np.asarray(np.meshgrid(*axes, indexing="ij"))
    
## Entry Points
    
def Forward(axes, g, Periodicity='Periodic', fft_axes=None):
#
# (G,Faxes) = wbFFT.Forward2d(axes, g, Periodicity='Periodic', fft_axes=(-2,-1))
#
#   Takes the n-dimensional discrete fourier transform of the function 
#   (vector) g and provides the frequency-domain axes corresponding to 
#   the provided sampling axes (xy).
#
#   axes, g, Faxes, and G are array_like.  
#
#   1-D configuration: axes can be specified as a 1-D list or array.  The returned
#   Faxes will follow the meshgrid convention and have shape (1,N) where N is the 
#   length of the transformed result.  To retrieve the single axis as a 1-D array,
#   use Faxes[0].
#
#   N-D configuration: Most commonly g and G will be 2-D matrices.  The axes input 
#   can be either in the numpy.mgrid or numpy.meshgrid formats.  The difference in 
#   these formats is that the first dimension of numpy.meshgrid is given as a list 
#   with length equal to the number of dimensions whereas numpy.mgrid uses a 
#   numpy.ndarray to include the first dimension.  For example:
#
#   x = [0, 1, 2, 3]
#   y = [0, 1, 2, 3]
#   axes = numpy.meshgrid(x,y)    
#   -> [array([[0, 1, 2, 3],
#              [0, 1, 2, 3],
#              [0, 1, 2, 3],
#              [0, 1, 2, 3]]), 
#       array([[0, 0, 0, 0],
#              [1, 1, 1, 1],
#              [2, 2, 2, 2],
#              [3, 3, 3, 3]])]
#   shapes are [(4,4), (4,4)]
#
#   axes = numpy.mgrid[0:4, 0:4]
#   -> array([[[0, 0, 0, 0],
#              [1, 1, 1, 1],
#              [2, 2, 2, 2],
#              [3, 3, 3, 3]],
#
#             [[0, 1, 2, 3],
#              [0, 1, 2, 3],
#              [0, 1, 2, 3],
#              [0, 1, 2, 3]]])
#   shape is (2, 4, 4).
#
#   In both of the above examples, g must have shape (4,4) and the dimensions 
#   correspond to (x,y) or x being axes[0] and y being axes[1].
#
#   The returned Faxes will use the mgrid format and in the above example
#   would have shape (2, 4, 4).
#
#   Axes-only mode:
#
#       If None is specified in place of g, then G will be returned as None and 
#       only the frequency axes associated with the input axes will be calculated.
#
#   fft_axes argument:
#
#       This argument is passed into the fftn() routine to tell it what ax(es) should 
#       be transformed.  If not specified for 1-D or 2-D inputs, all axes are transformed.
#       For 3-D and above, the default is to transform the last 2 axes.
#
#       In the event that fft_axes does not cover all input axes (i.e. a 2-D transform
#       of a 3-D g(r) function), then the untransformed ax(es) will be returned in Faxes 
#       unaltered from axes.
#
#   Periodicity argument (case insensitive):
#       'Periodic' (default):       Assume the input function is periodic.
#       'FiniteSupport' or 'fs':    Take the input function as zero
#       anywhere outside the sampled region.  
#
# Remarks:
#   To see higher frequencies, the input function (vector) must be sampled 
#   at a higher frequency.  In other words, increase dx (X_sampling).
#
#   To emphasize lower frequencies, decrease the sampling rate dx on the 
#   input vector.
#
# Optimization:
#   The execution time for fft depends on the length of the transform.  It 
#   is fastest for powers of two. It is almost as fast for lengths that 
#   have only small prime factors. It is typically several times slower 
#   for lengths that are prime or which have large prime factors.
#
#   For optimization, use a function g(x) which includes a sample at x=0 or
#   an alias thereof.  If a sample is not found very near to an x=0 alias,
#   Forward() will apply a phase-shift at the end which adds an extra vector
#   multiplication.
#
#   If the provided g(x) function does not have x=0 (or the closest match)
#   as index 0, then a np.roll() is applied to make it so.  For an
#   optimization, ensure that x=0 (or the closest match) happens at index
#   0.
#
# Requirements:
#   The axes must be provided on a uniformly spaced grid and must have the
#   same number of samples as g.
#
# Validation:
#   See wbFFT_test.py and wbFFT2D_test.py.
#
# Known Issues:
#   The phase correction to compensate for lack of an exact zero alias
#   seems to wreak havoc on some test cases.  It has been disabled 
#   for now.
#
# Author(s):
#   Wiley Black, N-D Python implementation, April 2021
#   Wiley Black, derived from wbFFT.m (MATLAB implementation) in June 2018
#   Wiley Black, June 2013

    if isinstance(axes, list):
        axes = np.asarray(axes)
        # axes is now in the mgrid format even if it came in from the meshgrid format.

    if len(axes.shape) == 1:
        # The input is in 1-D format where just a single array has been passed in for axes.  That is 
        # fine, but we need to put it into a consistent format where axes.shape = (1,N).
        axes = axes[np.newaxis, :]      # See numpy.expand_dims().   

    if axes.shape[0] != len(axes.shape[1:]): raise Exception("axes was not provided in a valid format.")

    tdims = axes.shape[0]                # The total # of dimensions in our input function (matrix), g.

    if g is not None:
        if len(g.shape) != tdims: raise Exception("Axes must provide the same number of ordinates as dimensions of the input value matrix.")

        for iDim in range(tdims):
            if g.shape[iDim] < 2: raise Exception("Requires a function (matrix) with at least 2 elements in each dimension.")
            if g.shape[iDim] != axes.shape[iDim+1]: raise Exception("axes shape (except 1st dimension) must match shape of g input.")

    if fft_axes is None: 
        fft_axes = (-1,) if tdims == 1 else (-2,-1)    
    
    # In order to define the output axis, we must determine the sampling 
    # frequencies, Fs.  We find this by examination of the provided axes.          
    
    ri0 = np.zeros(axes.shape[0], dtype='int').tolist()
    dr = calc_axes_delta(axes, ri0)

    # Verify our requirement about uniform spacing by testing at the mid index and last index:
    
    ri_end = [axes.shape[iDim]-1 for iDim in range(1,len(axes.shape))]          # Find last index in each dimension    
    dr_end = calc_axes_delta(axes, ri_end - np.asarray([1 for _ in range(1,len(axes.shape))]))
    
    ri_mid = np.floor(np.asarray([(ri_end[iDim] + ri0[iDim]) / 2 for iDim in range(len(ri0))])).astype('int')        # Find middle index in each dimension    
    dr_mid = calc_axes_delta(axes, ri_mid)
    
    for iDim in range(len(dr)):
        if np.abs(dr[iDim] - dr_end[iDim]) > 1.0e-6 or np.abs(dr[iDim] - dr_mid[iDim] > 1.0e-6):
            raise Exception("Only uniformly spaced sampling grids are supported.")

    Fs = 1/dr    
    
    ## Reorder x-axis to start at 0...
    
    # Consider the following sinusoidal function:
    #   g = cos(2*pi*60*X);         % 60 Hz Sinusoid    
    # In case A, the function is sampled at x=zero and at x=1/120.
    # In case B, the function is sampled at x=-1/120 and at x=zero.
    
    # In both cases, we have the same sinusoidal function, a 60 Hz sinusoid
    # with no phase shift from the origin.  The fft of case A will yield a
    # correct kronecker delta at -60 Hz and 60 Hz.  The fft of case B will
    # yield a kronecker delta at -60 Hz and 60 Hz, but with amplitude -1.
    # In order to prevent this, we want to shift any samples prior to zero
    # to the end of the periodic signal.  Since the input function is
    # assumed periodic, this is valid.
    
    # Now consider case C, where we sample at x=1/120 and x=2/120.
    # In this case, our sample nearest to zero is greater than or equal to
    # dx away from zero.  That means that we can still circular shift the
    # function around to get case A.                   
            
    # If x = [10, 20, 30, _40_]:
    # A periodic copy of the function samples occurs at x = [50, 60, 70, 80].
    # A periodic copy of the function samples occurs at x = [-30, -20, -10, 0].
    # Therefore, x=40 (index 3) is an alias of 0 and can be utilized as such.  
    # The goal in this example would be to use x_indices = [3, 0, 1, 2].  We find
    # this programmatically as:
    #   ErrorOnIndex0 = x[0] / dx    
    # Need to roll each by:
    #       mod(round(ErrorOnIndex0), len(x)) - len(x)
    
    # For lots of helpful examples and more comments explaining this logic, see 
    # wbFFT_helper_test.py.
    
    # Although we use the x-axis to figure out how much to circshift (roll) by,
    # what we actually want is to roll the function g.  After that, our fft will
    # always have the precondition that the first value corresponds to X=0, and
    # we can calculate a brand new Fx axis from that precondition.
    
    # In the periodic case, we can assume our roll comes at no expense.  In either case, 
    # we could compensate for any imperfect delta to zero or the zero alias (i.e. closest 
    # sampled point is at 0.1 instead of 0).  Attempts to do this so far don't help much 
    # (see ZeroAliasError below).
    
    # We proceed to shift x=0 into the 0th index position in both periodic and finite support 
    # cases.  In the FiniteSupport case, maybe this isn't necessary, but it makes for consistent 
    # shifting around of indices and reassembly regardless of mode.  
        
    # 1d version: ShiftBy = int((round(x[0] / dx) % len(x)) - len(x))
    r = axes
    N = axes.shape[1:]
    if g is None:
        G = None
    else:
        RollBy = ShiftBy(r,dr)    
        if all(RollBy == 0):
            g_copy = g
        else:
            g_copy = np.roll(g, RollBy, axis=list(range(tdims)))

        ## Compute fast-fourier transform
        #s = g_copy.shape
        L = np.asarray(g_copy.shape)        
        
        if Periodicity.lower() == "periodic":
            G = np.fft.fft2(g_copy,s=None,axes=fft_axes)/L.prod()       # By MATLAB fft() example.  Correct for a periodic signal such as mixed cosines with DC bias.        
        elif Periodicity.lower() == "finitesupport" or Periodicity.lower() == "fs":
            # By Scott's method.  Correct for finite support functions such as a tri().        
            # Note: Scott's method was in 1D and used multiplication by dr.  I'm only 
            # guessing that it is correct here.
            G = np.fft.fft2(g_copy,s=None,axes=fft_axes)*np.prod(dr)
        else:
            raise Exception("Unsupported option for periodicity parameter.")
        G = np.fft.fftshift(G, axes=fft_axes)
    
    ## Compute output axes
    
    # Convert any negative indexing to 0..tdims-1 indexing... 
    tmp_fft_axes = []
    for fft_axis in fft_axes:
        tmp_fft_axes.append(fft_axis if fft_axis >= 0 else tdims+fft_axis)
    fft_axes = tmp_fft_axes
        
    start = []
    stop = []    
    for iDim in range(tdims):
        if iDim in fft_axes: 
            start.append(-0.5 * Fs[iDim])
            stop.append(0.5 * Fs[iDim])
        else:
            start.append(axes[iDim][0])
            stop.append(axes[iDim][-1])
    Fr = mgrid_linspace(start, stop, N, endpoint=False)
    
    # Alternate technique in Python: use fftfreq.  Didn't get it working right though.
    #new_axes = []
    #for iDim in range(tdims):
        #if iDim in fft_axes:
            #na = np.fft.fftfreq(N[iDim], dr[iDim])            
            #new_axes.append(na)
        #else:
            #na = np.linspace(axes[iDim][0], axes[iDim][-1], N[iDim])            
            #new_axes.append(na)
    #Fr2 = np.asarray(np.meshgrid(*new_axes, indexing="ij"))
    #Fr2 = np.fft.fftshift(Fr2, axes=fft_axes)
    #Fr_error = np.abs(Fr - Fr2)
    #print("max error = " + str(np.max(Fr_error)))
    #Fr = Fr2
    
    # If all axes are being FFT'd, then the following could be used instead:
    #Fr = mgrid_linspace((-0.5,) * tdims, (0.5,) * tdims, N, endpoint=False)
    #for iDim in range(tdims): Fr[iDim] *= Fs[iDim]

    ## Apply phase-shift to compensate for sample delay    

    if Periodicity.lower() == "finitesupport" or Periodicity.lower() == "fs":
        # Check for a finite support function that does not include zero.
        ZA = PositionOfZeroAlias(r, dr)
        if any(np.abs(ZA) >= np.abs(dr)):
            raise Exception(
                "Cannot perform FiniteSupport FFT where the zero position is outside the provided axes- no\n" +
                "amount of phase shift can capture the axis being shifted in the transformed result, and\n" +
                "an Inverse() of the result would have lost this axis shift away from zero position."
                )
        
    # Correct for zero alias error if any.
    # I believe this is equivalent to fft interpolation.
    # In the wbFFT2D_test() 2nd test, applying this reduces the imaginary component as we iterate in offsets.  It never gets to 
    # zero, but is closer to zero with the correction applied than without.
    #ZAE = ZeroAliasError(r, dr)
    #if any(ZAE != np.repeat(0, tdims)):  # A speed optimization might check for 1.0e-6 or know in advance.  Going for accuracy here.                                    
        #print("Warning: origin (or an alias of) was not sampled.  This may not work great.  ZAE = " + str(ZAE))
        #ZAE = np.expand_dims(ZAE, axis = tuple([1+x for x in range(tdims)]))                        
        #G = np.exp(np.sum(-1j*2*pi*ZAE*Fr, axis=0)) * G

    return (G,Fr)

def Inverse(Faxes, G, Periodicity='Periodic', fft_axes=None):
#   
# (g,r) = wbFFT.Inverse(Faxes, G, Periodicity='Periodic', fft_axes=(-2,-1)))
#
#   Takes the inverse discrete fourier transform of the function (vector/matrix) 
#   G and provides the time or spatial-domain axis corresponding to the 
#   provided frequency axis (Faxes).
#
#   Periodicity argument (case insensitive):
#       'Periodic' (default):       Assume the input function is periodic.
#       'FiniteSupport' or 'fs':    Take the input function as zero
#       anywhere outside the sampled region.
#
#   If G is provided as None, then only the axes will be calculated and g will be 
#   returned as None.
#
# Requirements:
#   The F-axes must be provided on a uniformly spaced grid, and must match
#   the function G in size.  See accepted formats in Forward().
#
# Optimization:
#   The execution time for fft depends on the length of the transform.  It 
#   is fastest for powers of two. It is almost as fast for lengths that 
#   have only small prime factors. It is typically several times slower 
#   for lengths that are prime or which have large prime factors.
#
#   For optimization, use a function G() which includes a sample at Fr=0 or
#   an alias thereof.  If a sample is not found very near to an Fr=0 alias,
#   Inverse() will apply a phase-shift at the end which adds an extra vector
#   multiplication.
#
#   If the provided G(Fr) function does not have Fr=0 (or the closest match)
#   as index 0, then a np.roll() is applied to make it so.  For an
#   optimization, ensure that Fr=0 (or the closest match) happens at index
#   0.
#
# Validation:
#   See wbIFFT_test.py and wbFFT2D_test.py.
#
# Known Issues:
#   Same as Forward().
#
# Author(s):
#   2-D Python port in April 2021
#   Python port in March 2019
#   Wiley Black, June 2013

    if isinstance(Faxes, list):
        Faxes = np.asarray(Faxes)
        # Faxes is now in the mgrid format even if it came in from the meshgrid format.

    if len(Faxes.shape) == 1:
        # The input is in 1-D format where just a single array has been passed in for Faxes.  That is 
        # fine, but we need to put it into a consistent format where Faxes.shape = (1,N).
        Faxes = Faxes[np.newaxis, :]    # See numpy.expand_dims().   

    if Faxes.shape[0] != len(Faxes.shape[1:]): raise Exception("Faxes was not provided in a valid format.")

    tdims = Faxes.shape[0]                  # The total # of dimensions in our input function (matrix), G.       
    
    if G is not None:
        if len(G.shape) != tdims: raise Exception("Faxes must provide the same number of ordinates as dimensions of the input value matrix.")   

        for iDim in range(tdims):
            if G.shape[iDim] < 2: raise Exception("Requires a function (matrix) with at least 2 elements in each dimension.")
            if G.shape[iDim] != Faxes.shape[iDim+1]: raise Exception("Faxes shape (except 1st dimension) must match shape of G input.")

    if fft_axes is None: 
        fft_axes = (-1,) if tdims == 1 else (-2,-1)
    
    # In order to define the output axis, we must determine the sampling 
    # frequencies, Fs.  We find this by examination of the provided axes.          
    
    Fi0 = np.zeros(Faxes.shape[0], dtype='int').tolist()
    Fs = calc_axes_delta(Faxes, Fi0)

    # Verify our requirement about uniform spacing by testing at the mid index and last index:
    
    Fi_end = [Faxes.shape[iDim]-1 for iDim in range(1,len(Faxes.shape))]          # Find last index in each dimension    
    Fs_end = calc_axes_delta(Faxes, Fi_end - np.asarray([1 for _ in range(1,len(Faxes.shape))]))
    
    Fi_mid = np.floor(np.asarray([(Fi_end[iDim] + Fi0[iDim]) / 2 for iDim in range(len(Fi0))])).astype('int')        # Find middle index in each dimension    
    Fs_mid = calc_axes_delta(Faxes, Fi_mid)
    
    for iDim in range(len(Fs)):
        if np.abs(Fs[iDim] - Fs_end[iDim]) > 1.0e-6 or np.abs(Fs[iDim] - Fs_mid[iDim] > 1.0e-6):
            raise Exception("Only uniformly spaced frequency sampling grids are supported.")
   
    ## Reorder Fx-axis to start at 0...    
    #  See Forward() for reasoning.    
        
    if G is None:
        g = None
    else:
        RollBy = ShiftBy(Faxes,Fs)
        if all(RollBy == 0):
            G_copy = G
        else:
            G_copy = np.roll(G, RollBy, axis=list(range(tdims)))    
        
        ## Compute fast-fourier transform and frequency axis        
        if Periodicity.lower() == "periodic":
            L = np.asarray(G_copy.shape).prod()        
            g = np.fft.ifft2(G_copy * L, s=None, axes=fft_axes)        
        elif Periodicity.lower() == "finitesupport" or Periodicity.lower() == "fs":
    #        g = ifft(G_copy,N)*df         % By Scott's method.

            # Starting with Scott's method as listed above, I added in the *L and made it N-D.
            L = np.asarray(G_copy.shape).prod()
            g = np.fft.ifft2(G_copy * L, s=None, axes=fft_axes) * Fs.prod()

            #Range = np.ptp(Faxes) + Fs        
            #g = np.fft.ifft2(G_copy, s=None, axes=fft_axes)*Range*N/L         # Empirical method from old wbIFFT implementation.    
        else:
            raise Exception("Unsupported option for periodicity parameter.")
            
        g = np.fft.ifftshift(g)    

    ## Compute output axes
    
    dr = 1/Fs
    N = Faxes.shape[1:]
    
    # Convert any negative indexing to 0..tdims-1 indexing... 
    tmp_fft_axes = []
    for fft_axis in fft_axes:
        tmp_fft_axes.append(fft_axis if fft_axis >= 0 else tdims+fft_axis)
    fft_axes = tmp_fft_axes
        
    start = []
    stop = []    
    for iDim in range(tdims):
        if iDim in fft_axes: 
            start.append(-0.5 * dr[iDim])
            stop.append(0.5 * dr[iDim])
        else:
            start.append(Faxes[iDim][0])
            stop.append(Faxes[iDim][-1])
    r = mgrid_linspace(start, stop, N, endpoint=False)
    
    # If all axes are being FFT'd, then the following could be used instead:
    #r = mgrid_linspace((-0.5,) * tdims, (0.5,) * tdims, N, endpoint=False)
    #for iDim in range(tdims): r[iDim] *= (1/Fs[iDim])

    ## TODO: Apply phase-shift for sample delay.    
    
    if Periodicity.lower() == "finitesupport" or Periodicity.lower() == "fs":
        # Check for a finite support function that does not include zero.
        ZA = PositionOfZeroAlias(Faxes, Fs)
        if any(np.abs(ZA) >= np.abs(Fs)):
            raise Exception(
                "Cannot perform FiniteSupport IFFT where the zero position is outside the provided axes- no\n" +
                "amount of phase shift can capture the axis being shifted in the transformed result, and\n" +
                "a Forward() of the result would have lost this axis shift away from zero position."
                )
    
    #if Periodicity.lower() == "periodic":
        # To see this correction, use the wbFFT2D_test.py 2nd test and choose a Fs such that y=0 is not part of the 
        # sampled axis.  It doesn't really fix it completely though- the imaginary component is still present in gp.        
        #ZAE = ZeroAliasError(Faxes, Fs)
        #if any(ZAE != np.repeat(0, tdims)):  # A speed optimization might check for 1.0e-6 or know in advance.  Going for accuracy here.                                    
            #print("Warning: origin (or an alias of) was not sampled.  This may not work great.  ZAE = " + str(ZAE))
            #ZAE = np.expand_dims(ZAE, axis = tuple([1+x for x in range(tdims)]))                        
            #g = np.exp(np.sum(1j*2*pi*ZAE*r, axis=0)) * g   
    
    return (g,r)
