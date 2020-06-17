"""
This module provides classes to be used as containers of options and partial
results for the phase reconstruction algorithms.

Classes
-------

Options             A Class with options for all the reconstruction algorithms.

ResultsContainer    A Class containing results and time progress of each
                    iteration

Based on MATLAB implementation by Rohan Chandra, Ziyuan Zhong, Justin Hontz,
Val McCulloch, Christoph Studer & Tom Goldstein.
Copyright (c) University of Maryland, 2017.
Python version of the phasepack module by Juan M. Bujjamer.
University of Buenos Aires, 2018.
"""
__version__ = "1.0.0"
__author__ = 'Juan M. Bujjamer'

import logging

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, lsqr
from numpy.random import multivariate_normal as mvnrnd
import matplotlib.pylab as plt


class Options(object):
    """ A Class to manage options  for the reconstruction algorithms.

    It has two dictionaries as attributes, one for the general options,
    used in all the algorithms and one with parameters only valid for
    specific methods.
    """
    def __init__(self, **kwargs):

        # Following options are relevant to every algorithm
        general_opts_default = {'algorithm': 'gerchbergsaxton',
                          'init_method': 'optimal',
                          'is_complex': True,
                          'is_non_negative_only': False,
                          'max_iters':10000,
                          'max_time':300, # The maximum time the solver can run
                                         # (unit: second).
                                         # Note: since elapsed time will be
                                         # checked at the end of each
                                         # iteration,the real time the solver
                                         # takes is the time of the iteration
                                         # it goes beyond this max_time.
                          'tol': 1E-4,
                          'verbose': 0,  # Choose from [0, 1, 2]. If 0, printid
                                         # out nothing. If 1, print out status
                                         # information in the end. If 2, print
                                         # out status information every round.
                          'record_times': True, # If the solver record time at
                                         # each iteration.
                          'record_measurement_errors': False, # If the solver
                                         # compute and record measurement errors
                                         # i.e. norm(abs(A*x-b0))/norm(b0) at
                                         # each iteration.
                          'record_recon_errors':False, # If the solver record
                                         # reconstruction errors i.e.
                                         # norm(xt-x)/norm(x) at each iteration.
                          'record_residuals': True, # If the solver record
                                         # residuals (metric varies acOBross
                                         # solvers) at each iteration.
                          'label': None, # Can be used to choose a label for
                                         # the algorithm to show up in the
                                         # legend of a plot.  This is used when
                                         # plotting results of benchmarks. The
                                         # algorithm name is used by default if
                                         # no label is specified.
                          # Following three options are unused by default
                          'xt': None,      # The true signal. If it is provided,
                                         # reconstruction error will be computed
                                         # and used for stopping condition.
                          'custom_algorithm': None, # Custom algorithm provided
                                         # by user.
                          'customx0': None, # Custom initializer provided by
                                         # user.
                          'init_angle': None # When the angle initializer is used
                                         # you must specify the angle between
                                         # the true signal and the initializer.
                          }

        spec_defaults = {'custom': {},
                        'amplitudeflow': {
                            'search_method': 'steepest_descent', # Specifies how
                                         # search direction for line search is
                                         # chosen upon each iteration
                            'beta_choice': [] # Specifies how beta value is chosen (only used when search method is NCG)
                            },
                        'coordinatedescent': {
                            'index_choice': 'greedy' #the rule for picking up index, choose from 'cyclic','random' ,'greedy'].
                            },
                        'fienup': {
                            'fienup_tuning': 0.5, # Tunning parameter for Gerchberg-Saxton algorithm. It influences the update of the fourier domain value at each iteration.
                            'max_inner_iters': 10 # The max number of iterations the inner-loop solver  will have.
                            },

                        'gerchbergsaxton': {
                            'max_inner_iters': 10 # The max number of iterations  the inner-loop solver will have.
                            },
                        'kaczmarz': {
                            'index_choice': 'cyclic' # the rule for picking up index, choose from ['cyclic','random']
                            },
                        'phasemax': {},
                        'phaselamp': {},
                        'phaselift': {
                            'regularization_para' : 0.1 # This controls the weight of trace(X), where X=xx' in the objective function (see phaseLift paper for details)
                            },
                        'raf': {
                            'reweight_period': 20, # The maximum number of iterations that are allowed to occurr between reweights of objective function
                            'search_method': 'steepest_descent', # Specifies how search direction for line search is chosen upon each iteration
                            'beta_choice': 'HS' #  Specifies how beta value is chosen (only used when search
                            # method is NCG). Used only for NCG solver.
                        },
                        'rwf': {
                            'eta': 0.9, # Constant used to reweight objective function (see RWF paper for details)
                            'reweight_period': 20, # The maximum number of iterations that are allowed to occurr between reweights of objective function
                            'search_method': 'steepest_descent', # Specifies how search direction for line search is chosen upon each iteration
                            'beta_choice': 'HS' # Specifies how beta value is chosen (only used when search method is NCG) Used only for NCG solver
                        },
                        'sketchycgm': {
                            'rank': 1, # rank parameter. For details see Algorithm1 in the sketchyCGM paper.
                            'eta': 1 # stepsize parameter
                            },
                        'taf': {
                            'gamma': 0.7, # Constant used to truncate objective function (see paper for details). The maximum number of iterations that are allowed to occurr between truncations of objective function
                            'truncation_period': 20, # Specifies how search direction for line search is chosen upon each iteration
                            'search_method': 'steepest_descent',
                            'beta_choice': 'HS' # Specifies how beta value is chosen (only used when search method is NCG)Used only for NCG solver
                            },
                        'twf': {
                            'truncation_period': 20, # The maximum number of iterations that are allowed to occur between truncations of objective function
                            'search_method': 'steepest_descent', # Specifies how search direction for line search is chosen upon each iteration
                            'beta_choice': 'HS', # Specifies how beta value is chosen (only used when search method is NCG). Used only for NCG solver.
                            # Truncation parameters. These default values are defined as in the proposing paper for the case where line search is used.
                            'alpha_lb': 0.1,
                            'alpha_ub': 5,
                            'alpha_h': 6
                            },
                        'wirtflow': {
                            'search_method': 'steepest_descent', # Specifies how search direction for line search is chosen upon each iteration
                            'beta_choice': 'HS' # Specifies how beta value is chosen (only used when search method is NCG). Used only for NCG solver.
                            }
                        }
        for key, val in general_opts_default.items():
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, val)

        for key, val in spec_defaults[self.algorithm.lower()].items():
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, val)

class ResultsContainer(object):
    """
    Container for results and timing information of the algorithms.

    This Class contains and initializes outputs containers for convergence info
    according to user's choice. Its inputs is the Options Class wich selects
    which parameter are stored.

    solve_times: array
                contains the time required for each iteration.
    measurement_errors: array
                       actual errors.
    recon_errors: array
                 record_recon_errors.
    residuals: array
               record_residuals.

    """
    def __init__(self, opts):
        self.solve_times = list()
        self.measurement_errors = list()
        self.recon_errors = list()
        self.residuals = list()
        self.iteration_count = list()

        if opts.record_times:
            self.solve_times = np.zeros([])
        if opts.record_measurement_errors:
            self.measurement_errors = np.zeros([])
        if opts.record_recon_errors:
            self.recon_errors = np.zeros([])
        if opts.record_residuals:
            self.residuals = np.zeros([])

    def last_time(self):
        if len(self.solve_times) > 0:
            return self.solve_times[-1]
        else:
            return None

    def last_meas_error(self):
        if len(self.measurement_errors) > 0:
            return self.measurement_errors[-1]
        else:
            return None

    def last_recon_error(self):
        if len(self.recon_errors) > 0:
            return self.recon_errors[-1]
        else:
            return None

    def last_residual(self):
        if len(self.residuals) > 0:
            return self.residuals[-1]
        else:
            return None

    def last_iteration_count(self):
        if len(self.iteration_count) > 0:
            return self.iteration_count[-1]
        else:
            return None

    def append_record_time(self, record_time):
        self.solve_times = np.append(self.solve_times, record_time)

    def append_measurement_error(self, measurement_error):
        self.measurement_errors = np.append(self.measurement_errors, measurement_error)

    def append_recon_error(self, recon_error):
        self.recon_errors = np.append(self.recon_errors, recon_error)

    def append_residual(self, residual):
        self.residuals = np.append(self.residuals, residual)

def stop_now(opts, current_time, current_resid, current_recon_error):
    """
    Used in the main loop of many solvers (i.e.solve*.m) to
    check if the stopping condition(time, residual and reconstruction error)
    has been met and thus loop should be breaked.


    Note:
    This function does not check for max iterations since the for-loop
    in the solver already gurantee it.

    Inputs:
    opts(struct)                   :  consists of options. It is as
                  defined in solver_phase_retrieval.
                  See its header or User Guide
                  for details.
    current_resid(real number)      :  Definition depends on the
                  specific algorithm used see the
                  specific algorithm's file's
                  header for details.
    current_recon_error(real number) :  norm(xt-x)/norm(xt), where xt
                  is the m x 1 true signal,
                  x is the n x 1 estimated signal
                  at current iteration.
    Outputs:
    if_stop(boolean)                :  If the stopping condition has
                  been met.



    PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
    Christoph Studer, & Tom Goldstein
    Copyright (c) University of Maryland, 2017
    """
    if_stop = False
    if current_time >= opts.max_time:
        print('Timeout reached.')
        return True
    elif opts.xt:
        assert current_recon_error, 'If xt is provided, current_recon_error must be provided.'
        print('Error limit reached.')
        if_stop = current_recon_error < opts.tol
    if current_resid < opts.tol:
        assert current_resid, 'If xt is not provided, current_resid must be provided.'
        print('Residual limit reached.')
        print(current_resid, opts.tol)
        if_stop = current_resid < opts.tol
    return if_stop

def  display_verbose_output(iter, current_time, current_resid=None, current_recon_error=None, current_measurement_error=None):
    """ Prints out the convergence information at the current
    iteration. It will be invoked inside solve*.m if opts.verbose is set
    to be >=1.

    Inputs:
      iter(integer)                        : Current iteration number.
      current_time(real number)             : Elapsed time so far(clock starts
                                             when the algorithm main loop
                                             started).
      current_resid(real number)            : Definition depends on the
                                             specific algorithm used see the
                                             specific algorithm's file's
                                             header for details.
      current_recon_error(real number)       : relative reconstruction error.
                                             norm(xt-x)/norm(xt), where xt
                                             is the m x 1 true signal, x is
                                             the n x 1 estimated signal.

      current_measurement_error(real number) : norm(abs(Ax)-b0)/norm(b0), where
                                             A is the m x n measurement
                                             matrix or function handle
                                             x is the n x 1 estimated signal
                                             and b0 is the m x 1
                                             measurements.

    PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
    Christoph Studer, & Tom Goldstein
    Copyright (c) University of Maryland, 2017
    """
    print('Iteration = %d' % iter, end=' |')
    print('iteration_time = %f' % current_time, end=' |')
    if current_resid:
        print('Residual = %.1e' % current_resid, end=' |')
    if current_recon_error:
        print('current_recon_error = %.3f' %current_recon_error, end=' |')
    if current_measurement_error:
        print('measurement_error = %.1e' %current_measurement_error, end=' |')
    print()

def plot_error_convergence(outs, opts):
    """
    This function plots some convergence curve according to the values of
    options in opts specified by user. It is used in all the test*.m scripts.
    Specifically,
    If opts.record_recon_errors is true, it plots the convergence curve of
    reconstruction error versus the number of iterations.
    If opts.record_residuals is true, it plots the convergence curve of
    residuals versus the number of iterations.
    The definition of residuals is algorithm specific. For details, see the
    specific algorithm's solve*.m file.
    If opts.record_measurement_errors is true, it plots the convergence curve
    of measurement errors.

    Inputs are as defined in the header of solve_phase_retrieval.m.
    See it for details.


    PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
    Christoph Studer, & Tom Goldstein
    Copyright (c) University of Maryland, 2017

    """

    # Plot the error convergence curve
    if opts.record_recon_errors:
        plt.figure()
        plt.semilogy(outs.recon_errors)
        plt.xlabel('Iterations')
        plt.ylabel('recon_errors')
        plt.title('Convergence curve: %s' % opts.algorithm)
    if opts.record_residuals:
        plt.figure()
        plt.semilogy(outs.residuals)
        plt.xlabel('Iterations')
        plt.ylabel('Residuals')
        plt.title('Convergence curve: %s' % opts.algorithm)
    if opts.record_measurement_errors:
        plt.figure()
        plt.semilogy(outs.measurement_errors);
        plt.xlabel('Iterations');
        plt.ylabel('measurement_erros');
        plt.title('Convergence curve: %s' % opts.algorithm)
    plt.show()

def plot_recovered_vs_original(x,xt):
    """Plots the real part of the recovered signal against
    the real part of the original signal.
    It is used in all the test*.m scripts.

    Inputs:
          x:  a n x 1 vector. Recovered signal.
          xt: a n x 1 vector. Original signal.
    """
    plt.figure()
    plt.scatter(np.real(x), np.real(xt))
    plt.plot([-3, 3], [-3, 3], 'r')
    plt.title('Visual Correlation of Recovered signal with True Signal')
    plt.xlabel('Recovered Signal')
    plt.ylabel('True Signal')
    plt.show()
