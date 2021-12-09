"""
This is companion code to Project 4 for CSEP576au21
(https://courses.cs.washington.edu/courses/csep576/21au/)

Instructor: Vitaly Ablavsky
"""

# ======================================================================
# Copyright 2021 Vitaly Ablavsky https://corvidim.net/ablavsky/
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ======================================================================


from pdb import set_trace as keyboard
import argparse, sys, os, io


from scipy.stats import vonmises
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import matplotlib.colors
import torch

fontsize = 18


def rmserror(o1, o2):
    error = o1-o2
    error_normalized = np.fmod(np.fmod(error+np.pi, 2*np.pi)+2*np.pi, 2*np.pi)-np.pi
    rmse = torch.sqrt(torch.mean(torch.square(error_normalized)))
    return rmse.item()


#########################################################################
#                        angular_diff_1m_cos(o1, o2)  
#########################################################################
def angular_diff_1m_cos(o1, o2):
    """
    Follows 
      [R1] https://arxiv.org/abs/1612.00496 (3D deepbox paper)
      [R2] https://cs.gmu.edu/~amousavi/papers/3D-Deepbox-Supplementary.pdf


    Returns value in [0, 1]


    NOTE that during network training, we typically predict not the angle o_1 or o_2
    but sin(o), cos(o) with a constraint that sin(o)^2 + cos(o)^2 = 1

    In [R2] it is shown that 4* angular_diff_1m_cos(...) is equivalent to 
    Euclidean distance between vectors defined by [sin(o_1, cos(o_1] and [sin(o_2, cos(o_2))]


    """
    
    return 0.5 * (1 - np.cos(o1 - o2))



#########################################################################
#                        angular_histogram_radians
#########################################################################
def angular_histogram_radians(omegas, omega_min, omega_max, n_bins):
    """
    example use: 
        angular_histogram_radians(omegas, -np.pi/2, np.pi/2, 8)
        angular_histogram_radians(np.abs(omegas, 0, 

    Note that the bin that contains angle=0 will not in general correspond to bin_idx=0
    """

    # counts and bin-edges
    h_counts , h_bins_e = np.histogram(omegas, n_bins, (omega_min, omega_max))
    h_bins_c = h_bins_e[0:-1] + 0.5 * (h_bins_e[1:]-h_bins_e[0:-1]) # bin centers
    return (h_counts, h_bins_e, h_bins_c)



#########################################################################
#          viz_histogram()
#########################################################################
def viz_histogram(o_err, par):
    fontsize = 18
    n_bins = 8
    #n_bins = 24 # I updated n_bins to 24 for reducing the error margin of each bin
    # :NOTE: h_bins_e partition the [0,1] range of the error metric 1m_cos
    # They should not be interpreted as angles!
    (h_counts, h_bins_e, h_bins_c) = angular_histogram_radians(o_err, 0, 1, n_bins)
    hcn = h_counts.astype('float32') / np.sum(h_counts)

    color_faces = matplotlib.colors.CSS4_COLORS['lightsteelblue']
    color_edges = 'navy'

    fig, ax = plt.subplots()
    ax.hist(h_bins_e[:-1], h_bins_e, weights=hcn, color=color_faces, edgecolor=color_edges)
    ax.set_ylim(0,1)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_ylabel('Frequency',fontsize=fontsize)
    ax.set_title(r'Histogram of $0.5 (1-\cos(\omega_{\mathrm{true}} - \omega_{\mathrm{est}}))$',fontsize=fontsize)
    if 'viz_notebook' in par and par['viz_notebook']:
        plt.ion()
        plt.show()
    elif 'viz_hist' in par and par['viz_hist']:
        if 'viz_filename_suffix' in par:
            viz_filename_suffix = par['viz_filename_suffix']
        else:
            viz_filename_suffix = ''
        viz_filename_base = 'hist_angular_error_1mc'
        viz_filename = '{}{}.png'.format(viz_filename_base,viz_filename_suffix)
        viz_filepath = os.path.join(par['outdir'],viz_filename)
        plt.savefig(viz_filepath)
        plt.close()
        print('viz saved to {}'.format(viz_filepath))

    return (h_counts, h_bins_e)
        
#########################################################################
#          viz_kde()
#########################################################################
def viz_kde(o_err, par):

    if 'viz_filename_suffix' in par:
        viz_filename_suffix = par['viz_filename_suffix']
    else:
        viz_filename_suffix = ''
    viz_filename_base = 'kde_error'
    viz_filename = '{}{}.png'.format(viz_filename_base,viz_filename_suffix)
    viz_filepath = os.path.join(par['outdir'],viz_filename)

    k_type_opt = ['gaussian', 'epanechnikov']
    k_type_idx = 0
    k_bandwidth = 0.1
    # KDE estimate of the array o_err
    kde_mdl = KernelDensity(kernel=k_type_opt[k_type_idx], bandwidth=k_bandwidth)
    t_ = np.clip(o_err, -3/4*np.pi, 3/4*np.pi)
    kde_mdl.fit(t_[:, np.newaxis])

    x_viz = np.linspace(-np.pi, np.pi, 100)[:,np.newaxis]
    y_viz = np.exp(kde_mdl.score_samples(x_viz)) # score_samples() compute log(p(x))

    
    fig, ax = plt.subplots()
    lw = 2
    ax.plot(x_viz, y_viz, color='darkorange',
            label='kernel: {}'.format(k_type_opt[k_type_idx]))
    ax.set_xticks([-np.pi,0,np.pi])
    ax.set_xticklabels([r'$-\pi$',0,r'$\pi$'],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    ax.legend(loc='upper left',fontsize=fontsize-2)
    ax.plot(t_, -0.05 - 0.5 * par['numpy_randgen'].random_sample(len(t_)), 'ok')
    #ax.plot(t_, -0.005 - 0.01 * par['numpy_randgen'].rand(len(t_)), '+b')
    #ax.text(np.pi/4,max(y_viz), "K={0} points".format(K), fontsize=fontsize-2)
    K = len(t_)
    ax.text(0.6, 0.94, "K={0} points".format(K), fontsize=fontsize-2, verticalalignment='top', transform=ax.transAxes)

    ax.set_title(r'KDE of $\omega_{\mathrm{true}} - \omega_{\mathrm{est}}$',fontsize=fontsize)

    plt.savefig(viz_filepath)
    print('viz saved to {}'.format(viz_filepath))

#########################################################################
#                        unit_test
#########################################################################
def unit_test(par):
    """
    ./run.sh metrics.py --mode=unit_test

    We denote any variable that represents angle as $\omega_{something}$ 
    abbreviated in this code as o_{something} e.g., o_true, o_est, o_delta, o_err
    
    """


    # simulate predictions
    K = 100
    kappa = 2 # 200
    o_true = np.linspace(0, 2*np.pi, K)
    # for o_est sample o_err from vonMises around omega=0 then add to o_true
    o_err = vonmises.rvs(kappa, size=K)
    o_est = o_true + o_err

    """
    
    Discussion:

    The array (set) o_err (omega_delta) contains signed scalar values; we want to compute
    various error metrics on these (signed) values, e.g.. mean aboslute error, etc.
    and also capture the shape of the error distribution. Possibilities include:


    (a) fit a von Mises distribution to this array of angles

    (b) fit a mixuture of von Mises distributions to this array of angles

    (c) estimate a histogram of this array of angles (absolute values of the arrays)
        Note that it's trivial to constrauct a histogram on S^1

    (d) fit a KDE model to this array of angles. Note that fitting a "vanilla" 
    KDE (e.g., mixture of Gaussians) would not be possible using a single chart on 
    the S^1 manifold. However, We could create two charts, e.g., (-pi,0), (0, pi)


    Given (c) we can answer a question, What's the fraction of errors that are less than X-degrees


    In the general case of 3d rotation, angular error can be reprsented as
    a triplet of the three Euler angles and represented by a point on S^3
    (can't visualize "easily" since it would need to be embedded in R^4)

    """


    # compute a histogram of o_err using a chart that covers a portion of S^1
    n_bins = 8 # (-3/4*pi, 3/4*pi) n_bins = 9 -> each bin spans 30 degrees, and omega=0 falls in the middle of a bin
    h_counts, h_bins_e, h_bins_c = angular_histogram_radians(o_err, -3./4*np.pi, 3/4.*np.pi, n_bins)

    # compute mean and variance
    hcn = h_counts.astype('float32') / np.sum(h_counts)
    h_mean = np.sum(hcn * h_bins_c)
    # h_var = ...

    viz_hist_o_err = True
    if viz_hist_o_err:
        viz_file_name = 'hist_o_err.png'
        """
        20210115 :TODO: plot the normalized histogram "hcn", Title=Histogram of $\omega_{textrm{true}} - \omega_{\textrm{est}}$
        use this color-scheme/style (except initially we won't compute/plot the smooth curve, unless that capability is part of Seaborn, i.e., we get it for free)
        https://i0.wp.com/datavizpyr.com/wp-content/uploads/2020/01/Histogram_with_edge_color_Seaborn.jpg?resize=597%2C432&ssl=1
        In our case, the Y-axis label should be Frequency (not Density)

        20210115 note to ourselves: 
                 https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.hist.html
                 "See the documentation of the weights parameter to draw a histogram of already-binned data."
        """

        #[CVD] if we want to keep colors constant across 'grams, add to to viz_par?
        color_faces = matplotlib.colors.CSS4_COLORS['lightsteelblue']
        color_edges = 'navy'

        fig, ax = plt.subplots()
        #ax.hist(o_err, bins=n_bins, color=color_faces, edgecolor=color_edges, density=True)
        ax.hist(h_bins_e[:-1], h_bins_e, weights=hcn, color=color_faces, edgecolor=color_edges)
        ax.set_xticks([-np.pi,0,np.pi])
        ax.set_xticklabels([r'$-\pi$',0,r'$\pi$'],fontsize=fontsize)
        ax.set_ylim(0,1)

        plt.yticks(fontsize=fontsize)
        ax.set_ylabel('Frequency',fontsize=fontsize)
        ax.set_title(r'Histogram of $\omega_{\mathrm{true}} - \omega_{\mathrm{est}}$',fontsize=fontsize)
        if False:
            plt.ion()
            plt.show()
            keyboard()
        viz_filepath = os.path.join(par['outdir'],viz_file_name)
        plt.savefig(viz_filepath)
        print('viz saved to {}'.format(viz_filepath))

    # compute a histogram of || o_err || using a chart covering [0, pi)

    n_bins = 5
    # In order for h_abs_bins_c to be at zero (or near zero) we have to make the left bound to
    # be less than zero. This is a hack, and not a very good one.
    h_abs_counts, h_abs_bins_e, h_abs_bins_c = angular_histogram_radians(np.abs(o_err), -1/16*np.pi, 3/4.*np.pi, n_bins)



    #
    # 1m_cos metric
    #
    # test 1m_cos
    
    # first a quick test
    o1 = np.array([-np.pi, -1/2*np.pi, 0, 1/2*np.pi, np.pi])
    diff = np.array([angular_diff_1m_cos(0.0, o_) for o_ in o1])
    assert np.min(diff) >= 0 and np.max(diff) <= 1.0

    # apply angular_diff_1m_cos to o_true vs o_est (same as if we used 0.0 vs o_err)
    err1mc = np.array([angular_diff_1m_cos(o_[0], o_[1]) for o_ in zip(o_true, o_est)])

    """
    #
    #   histogram of err1mc
    #
    n_bins = 8
    # :NOTE: h_bins_e partition the [0,1] range of the error metric 1m_cos
    # They should not be interpreted as angles!
    h_counts , h_bins_e = np.histogram(err1mc, n_bins, (0, 1))
    hcn = h_counts.astype('float32') / np.sum(h_counts)

    viz_hist_err1mc = True
    if viz_hist_err1mc:
        viz_file_name = 'hist_err1mc.png'
        # 20210115 :TODO: plot the normalized histogram "hcn", Title=Histogram of $0.5 (1-cos(\omega_{textrm{true}} - \omega_{\textrm{est}}))$
        #[CVD] see above re viz_par
        color_faces = matplotlib.colors.CSS4_COLORS['lightsteelblue']
        color_edges = 'navy'

        fig, ax = plt.subplots()
        #ax.hist(err1mc, bins=n_bins, color=color_faces, edgecolor=color_edges)
        ax.hist(h_bins_e[:-1], h_bins_e, weights=hcn, color=color_faces, edgecolor=color_edges)
        ax.set_ylim(0,1)

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        ax.set_ylabel('Frequency',fontsize=fontsize)
        ax.set_title(r'Histogram of $0.5 (1-\cos(\omega_{\mathrm{true}} - \omega_{\mathrm{est}}))$',fontsize=fontsize)
        if False:
            plt.ion()
            plt.show()
            keyboard()
        viz_filepath = os.path.join(par['out_dir'],viz_file_name)
        plt.savefig(viz_filepath)
        print('viz saved to {}'.format(viz_filepath))
    """

    

    #
    #   KDE of o_err
    #
    viz_kde(o_err, par)

    keyboard()


######################################################################################
#                  get_valid_modes()
######################################################################################
def get_valid_modes():
    valid_modes = ['unit_test']
    return valid_modes

######################################################################################
#                  main()
######################################################################################
def main(par):
    mode=par['mode']
    numpy_randgen = np.random.RandomState(par['rng_seed']) if par['rng_seed'] is not None else np.random.RandomState()
    par.update({'numpy_randgen':numpy_randgen})
    if mode == 'unit_test':
        unit_test(par)
    else:
        print('no-op: unknown mode {}'.format(mode))

######################################################################################
#                  __main__
######################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=get_valid_modes())
    parser.add_argument('--rng_seed', type=int, required=False)
    parser.add_argument('--outdir', default='/z/vm_shared/exp/scene3d/eval')

    par__main=vars(parser.parse_args(sys.argv[1:]))
    main(par__main)
