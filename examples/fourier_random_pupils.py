"""
This script will create phaseless measurements from a test image, and
then recover the image using phase retrieval methods.  We now describe
the details of the simple recovery problem that this script implements.

                        Recovery Problem
This script loads a test image, and converts it to grayscale.
Measurements of the image are then obtained by applying a linear operator
to the image, and computing the magnitude (i.e., removing the phase) of
the linear measurements.

                      Measurement Operator
Measurement are obtained using a linear operator, called 'A', that
obtains masked Fourier measurements from an image.  Measurements are
created by multiplying the image (coordinate-wise) by a 'mask,' and then
computing the Fourier transform of the product.  There are 8 masks,
each of which is an array of binary (+1/-1) variables. The output of
the linear measurement operator contains the Fourier modes produced by
all 8 masks.

                        Data structures
PhasePack assumes that unknowns take the form of vectors (rather than 2d
images), and so we will represent our unknowns and measurements as a
1D vector rather than 2D images.

                     The Recovery Algorithm
The image is recovered by calling the method 'solve_phase_retrieval', and
handing the measurement operator and linear measurements in as arguments.
A struct containing options is also handed to 'solve_phase_retrieval'.
The entries in this struct specify which recovery algorithm is used.

For more details, see the Phasepack user guide.

Based on MATLAB implementation by Rohan Chandra, Ziyuan Zhong, Justin Hontz,
Val McCulloch, Christoph Studer & Tom Goldstein.
Copyright (c) University of Maryland, 2017.
Python version of the phasepack module by Juan M. Bujjamer.
University of Buenos Aires, 2018.
"""

# import cProfile
import ppack
from imageio import imread
from skimage import color
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy.random import rand
import time
import scipy

from ppack.containers import Options
from ppack.matops import ConvolutionMatrix, FourierOperator
from ppack.retrieval import Retrieval
from ppack.math import sign, hermitic

## Build a test problem
# Specify the target image and number of measurements/masks
image = imread('ppack/data/shapes.png')  # Load the image from the 'data' folder.
image = color.rgb2gray(image)    # Convert image to grayscale.
nummasks = 32           # Select the number of Fourier masks.
numrows, numcols = image.shape # Image dimensions
ximg = image.reshape(-1, 1)
# Create masks consisting of circular pupil apertures
# The idea is to iterate over led matrix in rectangular coordinates, starting
# by the center and going through the square in spirals.

## Build a test problem
# Specify the target image and number of measurements/masks
image = imread('ppack/data/shapes.png')  # Load the image from the 'data' folder.
image = color.rgb2gray(image)    # Convert image to grayscale.
nummasks = 32           # Select the number of Fourier masks.
numrows, numcols = image.shape # Image dimensions
ximg = image.reshape(-1, 1)
# Create masks consisting of circular pupil apertures
# The idea is to iterate over led matrix in rectangular coordinates, starting
# by the center and going through the square in spirals.
def random_iterator(n, m):
    """ Defines a rectangular iterator for the pupil construction.
    """
    yc, xc =  [m//2, n//2] # image center
    for i in range(100):
        x = np.random.randint(10,n-10)
        y = np.random.randint(10,m-10)
        yield x, y

iterator = random_iterator(numcols, numrows)
xx, yy = np.meshgrid(range(numcols), range(numrows))
image_gray = np.zeros_like(image)
masks = np.zeros((nummasks, numrows, numcols))
fig, ax = plt.subplots(1, 1)
fig.show()
for j in range(nummasks):
    ny, nx = next(iterator)
    # Create a circular array
    c = (xx-nx)**2+(yy-ny)**2
    image_gray = [c < 5**2][0]
    psf = fftshift(image_gray)
    masks[j,:,:] =  psf
    ax.cla()
    ax.imshow(masks[j,:,:])
    fig.canvas.draw()
nummasks, numrows, numcols = masks.shape
fo = FourierOperator(masks)
b = np.abs(fo.mv(ximg))
A = ConvolutionMatrix(mv=fo.mv, rmv=fo.rmv, shape=(numrows*numcols*nummasks,
                                                   numrows*numcols))
# ## Run the Phase retrieval Algorithm
# Set options for PhasePack - this is where we choose the recovery algorithm.
opts = Options(algorithm = 'fienup', init_method = 'truncated_spectral', tol =
               5E-4, verbose = 2, max_iters=60, alpha_ub=5, alpha_lb=.03, alpha_h=6)
# Create an instance of the phase retrieval class, which manages initializers
# and selection of solvers acording to the options provided.
retrieval = Retrieval(A, b, opts)
# Call the solver using the measurement operator all the information provided
# by the ConvolutionMatrix 'A'.
x, outs, opts = retrieval.solve_phase_retrieval()
# Convert the vector output back into a 2D image
recovered_image = x.reshape(numrows, numcols)
# Phase retrieval can only recover images up to a phase ambiguity.
# Let's apply a phase rotation to align the recovered image with the original
# so it looks nice when we display it.
rotation = sign(hermitic(recovered_image.ravel())@image.ravel())
recovered_image = np.real(rotation*recovered_image)

## Print and plot results
print('Image recovery required %d iterations (%f secs)\n' %
      (outs.iteration_count, outs.solve_times[-1]))
fig, axes = plt.subplots(1, 4)
# Plot the original image
axes[0].imshow(image)
axes[0].set_title('Original Image')
# Plot the recovered image
axes[1].imshow(np.real(recovered_image))
axes[1].set_title('Recovered Image')
axes[2].imshow(np.imag(recovered_image))
axes[2].set_title('Recovered Image')
axes[3].semilogy(outs.solve_times, outs.residuals)
axes[3].set_title('Convergence curve')

plt.show()
