"""
This module provides classes to be used as containers of matrix operations.
It is deviced to keep efficient implementations of the different methods, so
the idea is to explicitly state here hoy left and right matrix multiplications
are calculated, eigenvectors and other useful matrix operations.

Classes
-------

ConvMarix           A Class containing typical convolution matrix operations.

Python version of the phasepack module by Juan M. Bujjamer, University of
Buenos Aires, 2018. Based on MATLAB implementation by Rohan Chandra,
Ziyuan Zhong, Justin Hontz, Val McCulloch, Christoph Studer,
& Tom Goldstein.
Copyright (c) University of Maryland, 2017
"""
__version__ = "1.0.0"
__author__ = 'Juan M. Bujjamer'

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import LinearOperator, eigs, lsqr
from numpy.random import multivariate_normal as mvnrnd

class ConvolutionMatrix(object):
    """ Convolution matrix container.
    """
    def __init__(self, A=None, mv=None, rmv=None, shape=None):
        self.A = A
        if A is not None:
            self.shape = A.shape
            def mv(v):
                return A@v
            def rmv(v):
                return A.conjugate().T@v
        elif any([mv is not None, rmv is not None]):
            if shape:
                self.shape = shape
            else:
                print('If A is not given, its shape must be provided.')
                raise Exception(RuntimeError)
            if not callable(mv):
                print('Input mv was not a function. Both mv and rmv shoud be functions, or both empty.')
                raise Exception(RuntimeError)
            elif not callable(rmv):
                print('Input rmv was not a function. Both mv and rmv shoud be functions, or both empty.')
                raise Exception(RuntimeError)
        else:
            # One of both inputs are needed for ConvolutionMatrix creation
            print('A was not an ndarray, and both multiplication functions A(x) and At(x) were not provided.')
            raise Exception(RuntimeError)
        self.m = self.shape[0]
        self.n = self.shape[1]
        self.matrix = LinearOperator(self.shape, matvec=mv, rmatvec=rmv)
        self.check_adjoint()

    def validate_input(self, b0, opts):
        assert (np.abs(b0) == b0).all, 'b must be real-valued and non-negative'

        if opts.customx0:
            assert np.shape(opts.customx0) == (n, 1), 'customx0 must be a column vector of length n'

    def check_adjoint(self):
        """ Check that A and At are indeed ajoints of one another
        """
        y = np.random.randn(self.m)
        Aty = self.matrix.rmatvec(y)
        x = np.random.randn(self.n)
        Ax = self.matrix.matvec(x)
        inner_product1 = y.conjugate().T@Ax
        inner_product2 = Aty.conjugate().T@x
        error = np.abs(inner_product1-inner_product2)/np.abs(inner_product1)
        print('Both matrices were adjoints', error)

    def hermitic(self):
        return

    def lsqr(self, b, tol, maxit, x0):
        """ Solution of the least squares problem for ConvolutionMatrix.
        """
        if b.shape[1]>0:
            b = b.reshape(-1)
        if x0.shape[1]>0:
            x0 = x0.reshape(-1)
        ret = lsqr(self.matrix, b, damp=0.01, atol=tol/100, btol=tol/100, iter_lim=maxit, x0=x0)
        x = ret[0]
        return x

    def hmul(self, x):
        """ Hermitic mutliplication
        returns At*x
        """
        return self.matrix.rmatvec(x)

    def __mul__(self, x):
        return self.matrix.matvec(x)

    def __matmul__(self, x):
        """Implementation of left ConvolutionMatrix multiplication, i.e. A@x"""
        return self.matrix.dot(x)

    def __rmatmul__(self, x):
        """Implementation of right ConvolutionMatrix multiplication, i.e. x@A"""
        return

    def __rmul__(self, x):
        if type(x) is float:
            lvec = np.ones(self.shape[1])*x
        else:
            lvec = x
        return x*self.A # This is not optimal

    def calc_yeigs(self, m, b0, idx, squared=True ):
        if squared:
            v = (idx*b0**2).reshape(-1)
        else:
            v = (idx*b0).reshape(-1)
        def ymatvec(x):
            return 1/m*self.matrix.rmatvec(v*self.matrix.matvec(x))
        yfun = LinearOperator((self.n, self.n), matvec=ymatvec)
        [eval, x0] = eigs(yfun, k=1, which='LR',tol=1E-5)
        return x0

class FourierOperator(object):
    """ Linear operator creator from problem data.

        Create a measurement operator that maps a vector of pixels into Fourier
        measurements using a collection of PSF's.
    """
    def __init__(self, psf_collection):
        npsf, nrows, ncols = psf_collection.shape
        self.psf_collection = psf_collection
        self.npsf = npsf
        self.nrows = nrows
        self.ncols = ncols

    # def mv(self, xvec):
    #     """The fourier mask matrix operator
    #     As the reconstruction method stores iterates as vectors, this
    #     function needs to accept a vector as input.
    #     """
    #     xvec2d = xvec.reshape(self.nrows, self.ncols)
    #     bvec2d = np.fft.fft2(self.psf_collection*xvec2d)
    #     bvec_array = bvec2d.reshape(self.npsf*self.nrows*self.ncols, 1)
    #     return bvec_array

    # def rmv(self, bvec):
    #     # The adjoint/transpose of the measurement operator
    #     # The reconstruction method stores measurements as vectors, so we need
    #     # to accept a vector input, and convert it back into a 3D array of
    #     # Fourier measurements.  bvec2d_array = bvec.reshape(self.npsf, self.nrows, self.ncols)
    #     bvec2d_array = bvec.reshape(self.npsf, self.nrows, self.ncols)
    #     conv_images = np.fft.ifft2(bvec2d_array)
    #     conv_images = conv_images*self.psf_collection.conjugate()*self.nrows*self.ncols
    #     imagesvec = np.sum(conv_images,axis=0).reshape(-1, 1)
    #     return imagesvec

    def mv(self, xvec):
        """The fourier mask matrix operator
        As the reconstruction method stores iterates as vectors, this
        function needs to accept a vector as input.
        """
        xvec2d = xvec.reshape(self.nrows, self.ncols)
        xvec2d_fft = fft2(xvec2d)
        bvec2d = ifft2(self.psf_collection*xvec2d_fft)
        bvec_array = bvec2d.reshape(self.npsf*self.nrows*self.ncols, 1)
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(np.abs(bvec2d[0,:,:]))
        # ax2.imshow(np.abs(self.psf_collection[0,:,:]))
        # plt.show()
        return bvec_array

    def rmv(self, bvec):
        # The adjoint/transpose of the measurement operator
        # The reconstruction method stores measurements as vectors, so we need
        # to accept a vector input, and convert it back into a 3D array of
        # Fourier measurements.
        bvec2d_array = bvec.reshape(self.npsf, self.nrows, self.ncols)
        conv_images = fft2(bvec2d_array)
        conv_images = conv_images*self.psf_collection.conjugate()
        conv_images = ifft2(conv_images)
        imagesvec = np.sum(conv_images, axis=0).reshape(-1, 1)
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(np.abs(conv_images[0,:,:]))
        # ax2.imshow(np.abs(bvec2d_array[0,:,:]))
        # plt.show()
        return imagesvec

class FPMOperator(object):
    """ Linear operator creator from problem data.

        Create a measurement operator that maps a vector of pixels into Fourier
        measurements using a collection of PSF's.
    """
    def __init__(self, psf_collection, m, n):
        npsf, nrows, ncols = psf_collection.shape
        self.psf_collection = psf_collection
        self.npsf = npsf
        self.nrows = nrows
        self.ncols = ncols
        self.m = m
        self.n = n
        self.ratio = n//m

    def mv(self, xvec):
        """The fourier mask matrix operator.

        It aplies an FT and then rescales the result.

        As the reconstruction method stores iterates as vectors, this
        function needs to accept a vector as input.
        """
        xvec2d = xvec.reshape(self.nrows, self.ncols)
        xvec2d_fft = fft2(xvec2d)
        bvec2d = ifft2(self.psf_collection*xvec2d_fft)
        bvec_rescaled = bvec2d[:,0::self.ratio,0::self.ratio]
        # bvec_array = bvec_rescaled.reshape(self.npsf*self.m*self.m, 1)
        bvec_array = bvec_rescaled.ravel()
        return bvec_array

    def rmv(self, bvec):
        # The adjoint/transpose of the measurement operator
        # The reconstruction method stores measurements as vectors, so we need
        # to accept a vector input, and convert it back into a 3D array of
        # Fourier measurements.
        padded_bvec = np.zeros((self.npsf,self.n,self.n), dtype=complex)
        padded_bvec[:,::self.ratio,::self.ratio] = bvec.reshape(self.npsf, self.m, self.m)
        bvec2d_array = padded_bvec.reshape(self.npsf, self.n, self.n)
        conv_images = fft2(bvec2d_array)
        conv_images = conv_images*self.psf_collection.conjugate()
        conv_images = ifft2(conv_images)
        imagesvec = np.sum(conv_images, axis=0).reshape(-1, 1)
        return imagesvec
