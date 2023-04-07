# Call setup_cuda() to retrieve a "backend" object that can be used for CUDA or CPU operations.  The argument to setup_cuda()
# is a boolean that selects whether to use CUDA or the GPU.  The backend can then be used in various operations and will
# automatically alias to either the CPU or GPU, allowing code that supports either with a single switch.

import numpy
import scipy
import cupy
import cupyx
import cupyx.scipy.signal

def setup_cuda(use_cuda: bool = True):
    """
    Returns a "backend" (or "be") object that can be conveniently used for either CUDA/GPU or CPU operations without any
    code changes.  The "be" object aliases to either numpy or cupy.  The "be" object also includes be.scipy, be.to_cpu, and
    be.to_gpu aliases.  The to_cpu(arr) and to_gpu(arr) functions accept a numpy or cupy array and returns either a numpy
    or cupy array.  When operating in CPU mode, the to_cpu() and to_gpu() functions return the same input array and have
    no effect.
    """

    if use_cuda:
        be = cupy
        be.scipy = cupyx.scipy
        def to_cpu(arr): return cupy.asnumpy(arr)
    else:
        be = numpy
        be.scipy = scipy
        def to_cpu(arr): return arr
    def to_gpu(arr): return be.asarray(arr)
    be.to_cpu = to_cpu
    be.to_gpu = to_gpu
    be.using_gpu = use_cuda
    return be

def is_gpu(arr): return isinstance(arr, cupy.ndarray)
def is_cpu(arr): return isinstance(arr, numpy.ndarray)

def to_gpu(arr): return cupy.asarray(arr)

def to_cpu(arr):
    if arr is None: return None
    if isinstance(arr, cupy.ndarray):
        return cupy.asnumpy(arr)
    elif isinstance(arr, numpy.ndarray):
        return arr
    else:
        raise Exception("Expected arr argument to be an ndarray from either numpy or cupy.")