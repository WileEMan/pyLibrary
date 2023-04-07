import numpy as np
import cupy as cp
import gpu_util
import xml.etree.ElementTree as xmlt

# Polynomial fit approach.

# Inspiration taken from:  https://stackoverflow.com/a/57923405/1609125
# My version is written to mask out certain pixels from the regression and I needed an offset approach to
# give the polynomial fit freedom to position the terms off-center.

class fit2d:
    def __init__(self, img, kx: int, ky: int, include_mask = None, n_spatial_offsets: int = 0, scale_factor = 1, _internal: bool = False):
        """
        Performs a 2d least-squares fit of the given image.  If include_mask is provided, then it must have the same
        shape as img and the least-squares fit uses only points where include_mask is True in the
        estimation.

        Once the fit is performed, the reconstruct() function can be used to generate an image from the coefficients,
        which are retained within the fit2d object.

        kx, ky: specify the order of the polynomial fit.  For example, kx and ky of 1 request a bilinear fit.
                kx and ky of 2 would request a quadratic fit in each dimension.

        n_spatial_offsets: specify the number of offset polynomials to incorporate into the fit.  For example,
                if using a quadratic fit, it might be desirable to include a quadratic that is centered
                in the center of the image as well as quadratics that are off-center.  Default uses only
                a single polynomial.  If 2 or more is specified, then the polynomials are offset by
                1/n_spatial_offsets of each dimension of the image, for each dimension.

        scale_factor: if specified, the input images are downsampled before performing the fit, but the reconstruction
                still applies at the original image dimensions.  This can be used to reduce the memory and computational
                footprint with little loss of accuracy, since the polynomial 2d fit is a smoothing function and a
                downsampling operation on the input image (and mask) is also a smoothing function.  By default,
                the scale_factor is 1 meaning to retain the dimensions.  A scale factor of 5 would resample the
                input image (and mask if provided) to 1/5th of its original dimensions.

        The img argument can be a numpy.ndarray or a cupy.ndarray.  If a cupy.ndarray is used, then the fit2d is performed
        in the GPU and results are saved there.  The to_cpu() or to_gpu() functions can move the results appropriately,
        and the reconstruct() routine will use whichever device the fit2d is currently stored on.
        """
        if _internal: return

        if isinstance(img, cp.ndarray): be = gpu_util.setup_cuda(True)
        elif isinstance(img, np.ndarray): be = gpu_util.setup_cuda(False)
        else: raise TypeError("Expected img argument to be a numpy or cupy ndarray.")
        if include_mask is not None: include_mask = be.to_gpu(include_mask)

        width = img.shape[1]
        height = img.shape[0]

        # If a scaling factor is requested, apply it now
        downfactor = 1/scale_factor
        if scale_factor != 1:
            img = be.scipy.ndimage.zoom(img, downfactor, output=img.dtype, order=1)
            if include_mask is not None:
                include_mask = be.scipy.ndimage.zoom(include_mask.astype(be.int), downfactor, output=be.float32, order=1)
                include_mask = (include_mask >= 0.95)

        xlin = be.arange(0, width, scale_factor)
        ylin = be.arange(0, height, scale_factor)
        X, Y = be.meshgrid(xlin, ylin)
        if include_mask is not None:
            X = X[include_mask]
            Y = Y[include_mask]
            Z = img[include_mask]
        else:
            Z = img

        # I get notably different results when using cupy (GPU) if I use be.float32 than
        # using float64.  With float64, results are consistent on both CPU and GPU.  A bug?
        # Or a quirk of GPU arithmetic?  I'm not sure, but this seems to resolve it.  If
        # needed, could make this an 'if' based on whether we're using cupy and GPU or not.
        """
        X = X.astype(be.float32).flatten()
        Y = Y.astype(be.float32).flatten()
        Z = Z.astype(be.float32).flatten()
        """
        X = X.astype(be.float64).flatten()
        Y = Y.astype(be.float64).flatten()
        Z = Z.astype(be.float64).flatten()

        if n_spatial_offsets > 0:
            x_offsets = be.linspace(0, width, n_spatial_offsets)
            y_offsets = be.linspace(0, height, n_spatial_offsets)
        else:
            x_offsets = [0]
            y_offsets = [0]

        A = []
        for j in range(ky+1):
            for i in range(kx+1):
                for y_offset in y_offsets:
                    for x_offset in x_offsets:
                        A.append(be.ones_like(X) * (X - float(x_offset))**float(i) * (Y - float(y_offset))**float(j))
        A = be.array(A).T

        coeff, r, rank, s = be.linalg.lstsq(A, Z, rcond=None)

        # Save the coefficients and the values that enable reconstruction from the coefficients
        self.coeff = coeff
        self.width = width
        self.height = height
        self.kx = kx
        self.ky = ky
        self.n_spatial_offsets = n_spatial_offsets

    def to_cpu(self): self.coeff = gpu_util.to_cpu(self.coeff)
    def to_gpu(self): self.coeff = gpu_util.to_gpu(self.coeff)

    def reconstruct(self):
        be = gpu_util.setup_cuda(isinstance(self.coeff, cp.ndarray))

        if self.n_spatial_offsets > 0:
            x_offsets = be.linspace(0, self.width, self.n_spatial_offsets)
            y_offsets = be.linspace(0, self.height, self.n_spatial_offsets)
        else:
            x_offsets = [0]
            y_offsets = [0]

        X, Y = be.meshgrid(be.arange(self.width, dtype=be.float32), be.arange(self.height, dtype=np.float32), copy=False)
        ret = be.zeros(X.shape, dtype=be.float32)
        index = 0
        for j in range(self.ky+1):
            for i in range(self.kx+1):
                for y_offset in y_offsets:
                    for x_offset in x_offsets:
                        ret += self.coeff[index] * (X - x_offset)**i * (Y - y_offset)**j
                        index += 1
        return ret

    def to_xml(self) -> xmlt.Element:
        xml = xmlt.Element("Polynomial-2d-fit")
        xml.attrib["width"] = str(self.width)
        xml.attrib["height"] = str(self.height)
        xml.attrib["kx"] = str(self.kx)
        xml.attrib["ky"] = str(self.ky)
        xml.attrib["n-spatial-offsets"] = str(self.n_spatial_offsets)
        xc = xmlt.SubElement(xml, "Coefficients")
        xc.attrib["count"] = str(len(self.coeff))
        xc.text = ','.join([str(entry) for entry in gpu_util.to_cpu(self.coeff)])
        return xml

    @staticmethod
    def from_xml(xml_element: xmlt.Element):
        if xml_element.tag != "Polynomial-2d-fit":
            raise Exception("Expected <Polynomial-2d-fit> as element containing fit2d object content.")
        self = fit2d(None, 0, 0, _internal = True)
        self.width = int(xml_element.attrib["width"])
        self.height = int(xml_element.attrib["height"])
        self.kx = int(xml_element.attrib["kx"])
        self.ky = int(xml_element.attrib["ky"])
        self.n_spatial_offsets = int(xml_element.attrib["n-spatial-offsets"])
        xc = xml_element.find("Coefficients")
        if xc is None: raise Exception("<Coefficients> not found within <Polynomial-2d-fit> element.")
        self.coeff = xc.text.split(',')
        self.coeff = [float(entry) for entry in self.coeff]
        if len(self.coeff) != int(xc.attrib["count"]):
            raise Exception(f"Coefficients found ({len(self.coeff)}) did not match count listed ({xc.attrib['count']}).")
        return self
