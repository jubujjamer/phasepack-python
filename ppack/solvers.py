"""
This module provides the implementation of the different solvers for the phase
retrieval problem.

Functions:
----------

solve_fienup      Solver using the Fienup algorithm.

solve_twf         Truncated Wirtinger Flow method.

Based on MATLAB implementation by Rohan Chandra, Ziyuan Zhong, Justin Hontz,
Val McCulloch, Christoph Studer & Tom Goldstein.
Copyright (c) University of Maryland, 2017.
Python version of the phasepack module by Juan M. Bujjamer.
University of Buenos Aires, 2018.
"""
import time
import warnings
import numpy as np
from numpy.linalg import norm

from .containers import display_verbose_output, stop_now, ResultsContainer
from .math import gd_options, gradient_descent_solver

def solve_fienup(A, At, b0, x0, opts):
    """ Solver for Fienup algorithm.

    Parameters:
    -----------
    A:    m x n matrix or a function handle to a method that
          returns A*x.
    At:   The adjoint (transpose) of 'A'. If 'A' is a function handle,
          'At' must be provided.
    b0:   m x 1 real,non-negative vector consists of all the measurements.
    x0:   n x 1 vector. It is the initial guess of the unknown signal x.
    opts: A struct consists of the options for the algorithm. For details,
          see header in solve_phase_retrieval.m or the User Guide.

    Note: When a funphasepackction handle is used, the
    value of 'At' (a function handle for the adjoint of 'A') must be
    supplied.

    Outputs:
    --------
    sol:  n x 1 vector. It is the estimated signal.
    outs: A struct consists of the convergence info. For details,
          see header in solve_phase_retrieval.m or the User Guide.

    See the script 'test_fienup.m' for an example of proper usage of this
    function.

    Notations:
    ----------
    The notations mainly follow those used in Section 2 of the Fienup paper.
    gk:    g_k   the guess to the signal before the k th round
    gkp:   g_k'  the approximation to the signal after the k th round of
            iteration
    gknew: g_k+1 the guess to the signal before the k+1 th round
    Gkp:   G_k'  the approximation to fourier transfor of the signal after
                    satisfying constraints on fourier-domain
    beta:  \beta the Tuning parameter for object-domain update

    Algorithm Description:
    ----------------------
    Fienup Algorithm is the same as Gerchberg-Saxton Algorithm except when
    the signal is real and non-negative (or has constraint in general). When
    this happens, the update on the object domain is different.

    Like Gerchberg-Saxton, Fienup transforms back and forth between the two
    domains, satisfying the constraints in one before returning to the
    other. The method has four steps (1) Left multipy the current estimation
    x by the measurement matrix A and get Ax. (2) Keep phase, update the
    magnitude using the measurements b0, z = b0.*sign(Ax). (3) Solve the
    least-squares problem
                sol = \argmin ||Ax-z||^2
    to get our new estimation x. We use Matlab built-in solver lsqr()
    for this least square problem.
    (4) impose_results_container constraints on x(This step is ignored when there is
    no constraints)

    For a detailed explanation, see the Fienup paper referenced below.

    References:phasepack
    -----------
    Paper Title:   Phase retrieval algorithms: a comparison
    Place:         Section II for notation and Section V for the
                    Input-Output Algorithm
    Authors:       J. R. Fienup
    Address: https://www.osapublishing.org/ao/abstract.cfm?uri=ao-21-15-2758

    PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
    Christoph Studer, & Tom Goldstein
    Copyright (c) University of Maryland, 2017
    """

    def validate_options(opts):
        try:
            float(opts.fienup_tuning)
        except:
            raise Exception("%s should be a number" % opts.fienup_tuning)
    # Initialization
    gk = x0                      # Initial Guess, corresponds to g_k in the paper
    gkp = x0                     # corresponds to g_k' in the paper
    gknew = x0                   # corresponds to g_k+1 in the paper
    beta = opts.fienup_tuning     # GS tuning parameter
    # Initialize vectors for recording convergence information
    container = ResultsContainer(opts)
    start_time = time.monotonic()
    for iter in range(opts.max_iters):
        current_time = time.monotonic()-start_time
        Ax = A*gk # Intermediate value to save repetitive computation
        Gkp = b0*Ax/np.abs(Ax) # This is MATLAB's definition of complex sign
        # Record convergence information and check stopping condition.
        # If xt is provided, reconstruction error will be computed and used for stopping condition, otherwise residual will be computed and used for
        # stopping condition.
        if opts.xt:
            x = gk
            xt = opts.xt
            # Compute optimal rotation
            alpha = (x.T@xt)/(x.T@x)
            x = alpha*x
            current_recon_error = norm(x-xt)/norm(xt);
            if opts.record_recon_errors:
                container.append_recon_error(current_recon_error)

        if not opts.xt or opts.record_residuals:
            current_resid = norm(A.hmul((Ax-Gkp)))/norm(Gkp)
        if opts.record_residuals:
            container.append_residual(current_resid)
        if opts.record_times:
            container.append_record_time(current_time)
        if opts.record_measurement_errors:
            current_measurement_error = norm(np.abs(A*gk)-b0)/norm(b0)
            container.append_measurement_error(current_measurement_error)
        # Display verbose output if specified
        if opts.verbose == 2:
            display_verbose_output(iter, container.last_time(),
                               container.last_residual(),
                               container.last_recon_error(),
                               container.last_meas_error())
        # Test stopping criteria.
        if stop_now(opts, container.last_time(),
                         container.last_residual(),
                         container.last_recon_error()):
            break
        # Solve the least-squares problem
        #                    gkp = \argmin ||Ax-Gkp||^2.
        # If A is a matrix
        #                    gkp = inv(A)*Gkp.
        # If A is a fourier transform( and measurements are not oversampled
        # i.e. m==n)
        #                    gkp = inverse fourier transform of Gkp
        gkp = A.lsqr(Gkp, opts.tol, opts.max_inner_iters, gk)
        # gkp = np.fft.ifft(Gkp)
        # If the signal is real and non-negative, Fienup updates object domain
        # following the constraint
        if not opts.is_complex and opts.is_non_negative_only:
            inds = gkp < 0  # Get indices that are outside the non-negative constraints. May also need to check if isreal
            inds2 = not inds # Get the complementary indices
            # hybrid input-output (see Section V, Equation (44))
            gknew[inds] = gk[inds] - beta*gkp[inds]
            gknew[inds2] = gkp[inds2]
        else: # Otherwise, its update is the same as the GerchBerg-Saxton algorithm
            gknew = gkp.reshape(-1,1)
        gk = gknew # update gk
    sol = gk
    # Create output according to the options chosen by user
    container.iteration_count = iter
    return sol, container


def solve_twf(A, At, b0, x0, opts):
    """
    Implementation of the truncated Wirtinger Flow (TWF) algorithm.
    The code below is adapted from implementation of the
    Wirtinger Flow algorithm designed and implemented by E. Candes, X. Li,
    and M. Soltanolkotabi.

    Inputs:
        A:    m x n matrix or a function handle to a method that
              returns A*x.
        At:   The adjoint (transpose) of 'A'. If 'A' is a function handle,
              'At' must be provided.
        b0:   m x 1 real,non-negative vector consists of all the measurements.
        x0:   n x 1 vector. It is the initial guess of the unknown signal x.
        opts: A struct consists of the options for the algorithm. For details,
              see header in solve_phase_retrieval.m or the User Guide.

        Note: When a function handle is used, the value of 'At' (a function
        handle for the adjoint of 'A') must be supplied.

    Outputs:
        sol:  n x 1 vector. It is the estimated signal.
        outs: A struct containing convergence info. For details,
              see header in solve_phase_retrieval.m or the User Guide.


     See the script 'test_tWF.m' for an example of proper usage of this
     function.

    Notations:
        x is the estimation of the signal. y is the vector of measurements such
        that yi = |<ai,x>|^2 for i = 1,...,m Most of our notations are
        consistent with the notations used in the TWF paper referenced below.

    Algorithm Description:
     Similar to WF, TWF successively refines the estimate via a gradient
     descent scheme.  The loss function is the negative log of the Poisson
     likelihood.

     Unlike WF, TWF regularizes the gradient flow in a data-dependent fashion
     by operating only upon some iteration-varying index subsets that
     correspond to those data yi whose resulting gradient components are in
     some sense not excessively large.

     This gives us a more stable search directions and avoids the overshoot
     problem of the Wirtinger Flow Algorithm.

     We also add a feature: when opts.is_complex==false and
     opts.is_non_negative_only==true i.e. when the signal is real and
     non_negative signal, then at each iteration, negative values in the
     latest solution vector will be set to 0. This helps to speed up the
     convergence of errors.

     For a detailed explanation, see the TWF paper referenced below.

     References
     Paper Title:   Solving Random Quadratic Systems of Equations Is Nearly
                    as Easy as Solving Linear Systems
     Place:         Algorithm 1
     Authors:       Yuxin Chen, Emmanuel J. Candes
     arXiv Address: https://arxiv.org/abs/1505.05114

    PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
    Christoph Studer, & Tom Goldstein
    Copyright (c) University of Maryland, 2017

    """
    gd_opts = gd_options(opts)

    def update_objective(x, Ax):
        y = b0**2 # The TWF formulation uses y as the measurements rather than
                  # b0
        m = y.size # number of Measurements
        Kt = 1/m*norm(y-np.abs(Ax)**2, ord=1) # 1/m * sum(|y-|a'z|^2|)
        # Truncation rules
        # Unlike what specified in the TWF paper Algorithm1, the
        # term sqrt(n)/abs(x) does not appear in the following equations
        Axabs = np.abs(Ax)
        normx = norm(x)
        Eub =  Axabs/normx <= opts.alpha_ub
        Elb =  Axabs/normx >= opts.alpha_lb
        Eh  =  np.abs(y-Axabs**2) <= opts.alpha_h*Kt*Axabs/normx
        mask = Eub*Elb*Eh
        s = np.sum(mask)
        print('limits', Axabs/normx, opts.alpha_lb)
        # print(Eub, Elb, Eh)

        def f(z):
            absz = np.abs(z)**2
            argument = absz-y*np.log(absz)
            return (0.5/s)*np.sum(argument*mask)

        def gradf(z):
            argument = np.abs(z)**2-y
            return (1.0/s)*mask*argument/z.conjugate()

        return f, gradf

    sol, outs = gradient_descent_solver(A, At, x0, b0, update_objective, gd_opts)

    return sol, outs

def solve_gerchberg_saxton(A, At, b0, x0, opts):
    return

def solve_kaczmarz_simple(A, At, b0, x0, opts):
    return

def solve_phase_max(A, At, b0, x0, opts):
    return

def solve_phase_lamp(A, At, b0, x0, opts):
    return

def solve_phase_lift(A, At, b0, x0, opts):
    return

def solve_raf(A, At, b0, x0, opts):
    return

def solve_rwf(A, At, b0, x0, opts):
    return

def solve_sketchy_cgm(A, At, b0, x0, opts):
    return

def solve_taf(A, At, b0, x0, opts):
    return

def opts_custom_algorithm(A, At, b0, x0, opts):
    return

def solve_amplitude_flow(A, At, b0, x0, opts):
    return

def solve_coordinate_descent(A, At, b0, x0, opts):
    return



def solve_wirt_flow(A, At, b0, x0, opts):

    return
