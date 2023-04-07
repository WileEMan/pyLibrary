import numpy as np

pi = np.pi

def is_1D(r):
    if isinstance(r,list): r = np.asarray(r)
    return len(r.shape) == 1

def to_mgrid_shape(x,y=None):
    if isinstance(x,list): x = np.asarray(x)
    if y:
        return np.asarray(np.meshgrid([x, y], indexing="ij"))
    else:
        r = x
        if len(r.shape) == 1: raise Exception("to_mgrid_shape() should only be called after is_1D has been verified.")
        if r.shape[0] != len(r.shape)-1: raise Exception("Expected mgrid format axes to be provided for matrix implementation.")
        return r

def tri(x,y=None):
# z = tri(x)
# z = tri(x,y)
# z = tri(r)
#
#   Provides the triangle function in one or two dimensions according to 
#   the definition of Gaskill, chapter 3, page 45, which is:
#
#       tri((x - x0)/b) = { 0,                          |(x - x0)/b| >= 1
#                         { 1 - |(x - x0)/b|,           |(x - x0)/b| < 1
#
#   And, if a y axis is provided,
#
#       tri((x - x0)/b, (y - y0)/d) = tri((x - x0)/b) tri((y - y0)/d)
#
#   In r format (where r is a numpy array of dimension > 1), multiple 
#   axes are assumed to be embedded in the single matrix and the dimensionality
#   is given by the first shape entry.  This is verified when applicable.
#
#Author(s):
#   Wiley Black
#   J. Scott Tyo
#   College of Optical Sciences
#   University of Arizona
#   2008, 2013, 2018, 2021

    if y == None and is_1D(x):
        # One-dimensional tri function
        
        z = np.zeros(len(x))                    # Outside of (-1,1) z = 0.
        
        LeftRamp = (x > -1) & (x < 0)           # For (-1,0) z = 1+x.
        z[LeftRamp] = 1 + x[LeftRamp]
        RightRamp = (x >= 0) & (x < 1)          # For [0,1) z = 1-x.
        z[RightRamp] = 1 - x[RightRamp]
    else:
        # Two-dimensional tri function
        
        r = to_mgrid_shape(x,y)
        
        # First, calculate a version in only the x dimension...
        zx = np.zeros(r.shape[1:])
        x = r[0]
                
        LeftRamp = (x > -1) & (x < 0)
        zx[LeftRamp] = 1 + x[LeftRamp]
        RightRamp = (x >= 0) & (x < 1)
        zx[RightRamp] = 1 - x[RightRamp]
        
        # Now, multiply by the y dimension tri(), producing the final
        # output.
        z = np.zeros(r.shape[1:])
        y = r[1]
        
        TopRamp = (y > -1) & (y < 0)
        z[TopRamp] = zx[TopRamp] * (1 + y[TopRamp])
        BtmRamp = (y >= 0) & (y < 1)
        z[BtmRamp] = zx[BtmRamp] * (1 - y[BtmRamp])

    return z

def sinc(x,y=None):
#z = sinc(x)
#z = sinc(x,y)
#z = sinc(r)
#
#   Provides the scalar, 1-D, or 2-D sinc function defined by Gaskill.
#
#   The returned z matrix will be indexed as [iy,ix].
#
#   Author(s):
#       Wiley Black, 2018, 2021
#       Prof. J Scott Tyo, 2008

    if y != None or not(is_1D(x)):
        # Two-dimensional sinc()
        r = to_mgrid_shape(x,y)

        x = r[0]
        y = r[1]
        z = np.zeros(r.shape[1:])
        
        #XZero = (r[0] == 0)
        #YZero = (r[1] == 0)        
        #XYValid = ~(XZero | YZero)
        XValid = (r[0] != 0)
        YValid = (r[1] != 0)
        XYValid = XValid & YValid
        Invalid = (r[0] == 0) & (r[1] == 0)
        
        z[Invalid]  = 1
        z[YValid]   =                                                      (np.sin(pi * y[YValid]) / (pi * y[YValid]))
        z[XValid]   = (np.sin(pi * x[XValid]) / (pi * x[XValid]))        
        z[XYValid]  = (np.sin(pi * x[XYValid]) / (pi * x[XYValid])) *    (np.sin(pi * y[XYValid]) / (pi * y[XYValid]))
        
        #for iy in range(len(y)):
            #for ix in range(len(x)):
                #if x[ix] == 0:
                    #if y[iy] == 0:
                        #z[iy,ix] = 1
                    #else:
                        #z[iy,ix] = np.sin(pi * y[iy]) / (pi * y[iy])
                #elif y[iy] == 0:
                    #z[iy,ix] = np.sin(pi * x[ix]) / (pi * x[ix])
                #else:
                    #z[iy,ix] = np.sin(pi * x[ix]) / (pi * x[ix]) * np.sin(pi * y[iy]) / (pi * y[iy])
        
    else:
        if isinstance(x, list): x = numpy.asarray(x)
        
        z = np.zeros(x.size)
        
        for ix in range(x.size):
            if x[ix] == 0:
                z[ix] = 1
            else:
                z[ix] = np.sin(pi * x[ix]) / (pi * x[ix])

    return z
    

def gaus(x,y=None,Center=None,FWHM=None):
#GAUS
#   This function provides the gaussian function as defined by Gaskill,
#   chapter 3, page 47.  Note the factor of pi in the exponent.
#
#   gaus(x) will return the 1-D gaussian function for the axis x.
#   gaus(x,y) will return the 2-D gaussian function.  The returned function
#   in 2-D will be indexed as z[xi,yi].  For example, to retrieve the 2nd
#   x-axis position and the 1st y-axis position, use z[1,0].
#
#   The Center or FWHM parameters can be used to adjust the center position
#   or FWHM of the Gaussian.  Alternatively, the same adjustments can be made 
#   to the x and y axis before passing them in (i.e. gaus((x-x0)/b)).  When
#   the y axis is provided, the Center and/or FWHM may be specified as a tuple
#   with the (x,y) values or may be a scalar applied for both axes.
#
#   The gaussian function, in this definition, has an area of |b| in the
#   form Gaus(X/b).  The function starts at an amplitude of 1.0, and
#   reaches 0.5 by x=|0.47|, and 0.043 by x=|1.0|.  By x=|1.48|, the
#   function reaches 0.001 amplitude.
#
#   The FWHM of a function that peaks at amplitude 1, as gaus() does, can 
#   be found by solving gaus((x-x0)/b) = 1/2.  In Gaskill (Bracewell)'s definition,
#   this leads to ((x-x0)/b) = +/- sqrt(ln(1/2)/(-pi)) or approximately
#   ((x-x0)/b) = +/- 0.46971864.  The FWHM can be adjusted to a new x by solving
#   (x-x0) = +/- b * sqrt(ln(1/2)/(-pi)) such that b provides the new desired (x-x0).
#   For example, to provide a Gaussian centered at 0 with FWHM located at +/- 3, 
#   3 = +/- b * sqrt(ln(1/2)/(-pi)) yields b = +/- 3 / sqrt(ln(1/2)/(-pi)).
#
#   J. Scott Tyo
#   College of Optical Sciences
#   University of Arizona
#   2008
#
#   W. Black
#    - Extended to 2-D and ported to Python

    xp = x
    yp = y
    
    if Center is not None:
        if isinstance(Center, tuple):
            xp = xp - Center[0]
            if yp is not None: yp = yp - Center[1]
        else:
            xp = xp - Center
            if yp is not None: yp = yp - Center
            
    if FWHM is not None:
        # First, convert from FWHM to b as discussed above.        
        if isinstance(FWHM, tuple):
            b = (FWHM[0] / np.sqrt(np.log(1/2) / (-np.pi)), FWHM[1] / np.sqrt(np.log(1/2) / (-np.pi)))
            xp = xp / b[0]
            if yp is not None: yp = yp / b[1]
        else:
            b = FWHM / np.sqrt(np.log(1/2) / (-np.pi))
            xp = xp / b
            if yp is not None: yp = yp / b

    if yp is None:
        return np.exp(-pi*(xp**2))
    else:
        return np.outer(np.transpose(np.exp(-pi*(xp**2))), np.exp(-pi*(yp**2)))

