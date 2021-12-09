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


import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format},
                    linewidth=120) 


from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

import h5py

import data_io.data_loaders
import perf_eval.metrics
import geometry as geometry

import matplotlib.pyplot as plt
import matplotlib.colors

import pose_regress_mdls

import train_util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))




######################################################################################
#                  get_default_par()
######################################################################################
"""
We control the behavior of our system via a dictionary that we call "par"
We set key/value pairs in "par" either programmatically in the code or via
command-line flags. 
"""
def get_default_par():
    par = ({
        'instructor_version' : False,
        'h5_filename' : '',
        'outdir' : '',
        'split_name' : 'cvd_split_every5',
        'payload_format' : 'dict',
        'verbose' : False,
        'dloader_shuffle' : False,
        'Dataset.getitem_returns_image_viz' : True,
        'batch_sz' : 10
    })
    return par


"""
We define a set of pairs in the form of (helper_function, ut_helper_function). 
A helper function performs some small task related to setting up training/visualization; 
the corresponding ut_ function verifies programmatically that the result makes sense 
(e.g., the return data type is correct or the shape is correct or the numerical values are correct)
"""


######################################################################################
#                  setup_xforms_and_data_loaders()
######################################################################################
def setup_xforms_and_data_loaders(par):
    print('-'*5 + 'setup_xforms_and_data_loaders()'+ '-'*5)

    xform = transforms.Compose([transforms.Resize((64, 64)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,),
                                     (1.0,))])

    par.update({'transform' : xform})

    if par['instructor_version']:
        #instantiate data loader
        pass
    else:
        """
        this is a one-liner:
        call setup_data_loaders(par) defined in data_io.data_loaders
        It returns a tuple (train loader, test loader)
        """
        train_loader, test_loader = data_io.data_loaders.setup_data_loaders(par)
        #return (None, None) # for now

    return (train_loader, test_loader)



######################################################################################
#                  ut_setup_xforms_and_data_loaders()
######################################################################################
def ut_setup_xforms_and_data_loaders(par):
    print('-'*5 + 'ut_setup_xforms_and_data_loaders()'+ '-'*5)

    # par = get_default_par()

    if 'h5_filename' not in par or par['h5_filename'] is None:
        raise IOError('--h5_filename must be specified')

    if 'outdir' not in par or par['outdir'] is None:
        raise IOError('--outdir must be specified')

    dloaders = setup_xforms_and_data_loaders(par)

    status = (
        (len(dloaders) == 2) and
        hasattr(dloaders[0], "__iter__") and
        hasattr(dloaders[1], "__iter__")
        )
    return status


######################################################################################
#                 load_batch_of_data()
######################################################################################
def load_batch_of_data(data_loader, par):
    #load a batch of data
    if par['instructor_version']:
        pass
    else:
        """
        get a batch of data by
        "wrapping" data_loader in an iterator, i.e., iter(...)
        and calling next(...) on the iterator
        """
        batch_data = next(iter(data_loader))
        #return None  # for now

    return batch_data


######################################################################################
#                 ut_load_batch_of_data()
######################################################################################
def ut_load_batch_of_data(data_loader, par_runtime):
    print('-'*5 + 'ut_load_batch_of_data()'+ '-'*5)    

    par = get_default_par()
    par.update(par_runtime)

    """
    We could get this information by peeking into our dataset (the HDF5 file)
    but for now we hard-code those values here
    """
    im_w = 1280
    im_h = 720
    
    batch_data = load_batch_of_data(data_loader, par)
    if batch_data is None or 'image_viz' not in batch_data:
        return False

    img_viz = batch_data['image_viz']  # (H, W, 3)
    img_sample = img_viz[0].cpu().numpy()
    status = (
        (len(img_viz) == par['batch_sz']) and
        (img_sample.shape[0] == im_h) and
        (img_sample.shape[1] == im_w)   
    )
    return status


######################################################################################
#                 ut_project_3axis_pattern
######################################################################################
def ut_project_3axis_pattern(par):
    print('-'*5 + 'ut_project_3axis_pattern()'+ '-'*5)

    im_w = 1280
    im_h = 720
    gray_img = np.ones((im_h,im_w,3))*0.75

    camera_z = 6.5
    camera_orbit_r = 20.0

    omega = np.pi/4

    X_c_expected = np.array([[ 640.00,  696.76,  583.24,  640.00],
                             [ 360.00,  342.46,  342.46,  285.00]])
    
    X_c = geometry.project_3axis_pattern(omega, camera_z, gray_img,  camera_orbit_r, par['instructor_version'])

    if 'viz_pose3d' in par and par['viz_pose3d']:
        fig_handle = plt.figure()
        plt.imshow(gray_img)
        geometry.draw_3axis_on_image(X_c, fig_handle)
        if 'outdir' not in par or par['outdir'] is None:
            raise IOError("par['outdir'] must be specified")
        outdir = par['outdir']
        if not os.path.exists(outdir):
            raise IOError('specified outdir not found: {}'.format(outdir))
        viz_filename = 'test_3axis.png'
        viz_filepath = os.path.join(outdir,viz_filename)
        role_str = 'instructor' if par['instructor_version'] else 'student'
        plt.title('ut_project_3axis_pattern() [{} version]'.format(role_str))
        plt.show()
        plt.savefig(viz_filepath)
        plt.close()
        print('projected 3-axis pattern vsualized in {}'.format(viz_filepath))

    status = np.allclose(X_c_expected, X_c)
        
    return status


######################################################################################
#                 ut_compute_and_viz_angular_error_metrics
######################################################################################
def ut_compute_and_viz_angular_error_metrics(par):
    print('-'*5 + 'ut_compute_and_viz_angular_error_metrics()'+ '-'*5)

    if 'viz_hist' in par and par['viz_hist']:
        if 'outdir' not in par or par['outdir'] is None:
            raise IOError("par['outdir'] must be specified")
        outdir = par['outdir']
        if not os.path.exists(outdir):
            raise IOError('specified outdir not found: {}'.format(outdir))

    K = 100
    if par['instructor_version']:
        pass
    else:
        """
        :TODO:
        Study the concentration parameter $\kappa$ of the Von Mises distribution
        https://en.wikipedia.org/wiki/Von_Mises_distribution

        Then study the formula for "Orientation Score" as described in
        https://arxiv.org/pdf/1612.00496.pdf
        Sec. 5.2 top of the right column on p. 6

        and our implementation perf_eval.metrics.angular_diff_1m_cos(...)

        Finally, pick a value of $\kappa$ that makes the unit test return True
        and explain in your writeup why it makes sense.
        """
        kappa = 2.0 # <-- updated the kappa value here for which the unit test returns True

    o_true = np.linspace(0, 2*np.pi, K)
    # for o_est sample o_err from vonMises around omega=0 then add to o_true
    from scipy.stats import vonmises
    o_err = vonmises.rvs(kappa, size=K)
    o_est = o_true + o_err

    (h_counts, h_bins_e) = train_util.compute_and_viz_angular_error_metrics(o_true, o_est, par)

    status = (
        (np.sum(h_counts) == o_est.shape[0]) and
        (np.sum(h_counts[0:2])/np.sum(h_counts[2:]) > 2)
        )
    
    return status




######################################################################################
#                  run_all_unit_tests()
######################################################################################
def run_all_unit_tests(par_runtime):
    """
    python pose_regress.py --mode=unit_test
    """
    print('-'*5 + 'run_all_unit_tests()'+ '-'*5)


    par = get_default_par()
    par.update(par_runtime)


    status = ut_setup_xforms_and_data_loaders(par)
    print(status)

    (train_loader, test_loader) = setup_xforms_and_data_loaders(par)

    status = ut_load_batch_of_data(test_loader, par)
    print(status)

    par.update({'viz_pose3d' : True})
    status = ut_project_3axis_pattern(par)
    print(status)

    par.update({'viz_hist' : True})    
    status = ut_compute_and_viz_angular_error_metrics(par)
    print(status)


######################################################################################
#                  train_and_test()
######################################################################################
def train_and_test(par):
    """
    A note on variable names: traditionally, X is used to denote the input (e.g., a vector of values)
    and Y is used to denote the output (e.g., the so-called "logits" for classification or estimated function values
    for a regression. It is also common to use a variable named "omega" to refer to angles.
    In this function we do both, so "y" and "omega" are used as our regression (targets)

    """

    print('-'*5 + 'train_and_test'+ '-'*5)

    """
    Before we start training we "peek" into our dataset (an HDF5 file)
    to extract meta-data that we need for visualizing our pose predictions
    """
    h5_filename = par['h5_filename']
    db = h5py.File(h5_filename,mode='r')
    ds_eye = db['eye_world']
    """
    The camera orbits an object at a constant height and a constant distance in the XY plane
    Therefore we can take the camera location associated with any data sample (e.g., sample [0])
    to obtain the height "z" and the distance "r"
    If our network learns to predict pose ("omega") correctly then we could recover the correct camera
    viewpont (extrinisc parameters) given "z" "r" and "omega"
    """
    camera_eye_h5_xzy = ds_eye[0] 
    camera_z = camera_eye_h5_xzy[1]  # the height of the camera orbiting our object
    camera_orbit_r = np.linalg.norm(camera_eye_h5_xzy[[0,2]]) # distance in the XY plane


    par.update({'viz_hist': True,
                'n_out' : 2 if par['mode'] == 'train_and_test_2_out' else 1})
    img_shape_train = (64,64)
    par.update({'img_h':img_shape_train[0],'img_w':img_shape_train[1]})

    mdl_reg = pose_regress_mdls.PoseRegression_v1(par)
    mdl_reg = mdl_reg.to(device)    


    assert 'split_name' in par
    par.update({'payload_format':'dict',
                'dloader_shuffle': True})

    split_name = par['split_name']
    payload_format_ = par['payload_format']

    xform_0 = None
    xform_1 = transforms.Compose([transforms.ToTensor()])
    xform_2 = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,),
                                                     (1.0,))])

    xform_3 = transforms.Compose([transforms.Resize(img_shape_train),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,),
                                                     (1.0,))])

    xform_ = xform_3

    par.update({'transform' : xform_ ,
                'batch_sz' : par['train_batch_sz']})

    train_loader, test_loader = data_io.data_loaders.setup_data_loaders(par)

    """
    To verify that all the tensor shapes are compabible, you can run a forward pass
    with our network on a tensor that contains arbitrary values e.g., all zeros.
    (We don't expect a semantically meaningful result at this point, since the weights are "random", i.e., 
    contain values sampled from a weights-prior distribution).
    """
    if False: 
        batch_sz = 1
        n_channel=3
        X_test = torch.tensor(np.zeros((batch_sz,n_channel,*img_shape_train),dtype=np.float32))
        X_test = X_test.to(device)
        with torch.no_grad():
            Y_test = mdl_reg(X_test)
        keyboard() # pause here to inspect Y_test.shape


    net1 = mdl_reg
    net1.to(device)

    """
    Predicting an object pose is a challenging problem (even in a toy setting).
    One challenge stems from the fact the even a 1D pose (omega takes values
    in [0, 2pi) we have to deal with the discontinuity, e.g., at 0 (or 2pi)
    Another challenge stems from the data itself: depending on the object
    whose pose we want to predict, the prediction problem may be "multi-modal".
    One commonly-used approach is to design a network that comprises a set of
    "experts" specialized to subset of the input space (e.g., a car viewed between -pi/4 and pi/4, etc.)
    and a separate branch of the network that predicts which expert to "trust"
    
    In this example code we are not designing such a network, but we have mechanism to
    explore our network's ability to learn if we restrict the input space (input images)
    to a subset of [-pi, pi]. Try it!

    The omega-values in our HDF5 dataset are in [0, 2*np.pi), and are then transformed by the
    data loader into the range [-np.pi, np.pi) via omega = np.pi - omega

    omega_range is a tuple that defines the range of samples used during training/testing
    e.g., (-np.pi, np.pi) will accept all dataset samples
          (0, 0.5*np.pi) restricts to the range [0, pi/2] 
    """
    omega_range = (-np.pi, np.pi) 
    omega_mask = train_util.Xform_select_in_y_range(omega_range)


    """
    We'll be trying several different loss functions.
    Some loss functions are compatible with network that predicts a single scalar,
    while other loss function are compatible with network that predicts more
    than one value, e.g., [cos(omega), sin(omega)]
    """

    if par['instructor_version']:
        pass
    else:
        class DummyLossValue(object):
            def __init__(self):
                super().__init__()
            def detach(self):
                return torch.tensor(0)
        """
        We recommend you begin by trying
        nn.MSELoss with reduction set to 'sum'
        or
        train_util.diff_1m_cos_loss()
        """
        #loss_func = None
        #optimizer = None  # we suggest starting with "Adam", learning rate 1e-3
        loss_func = nn.MSELoss(reduction='sum')
        #loss_func = train_util.diff_1m_cos_loss()
        optimizer = optim.Adam(net1.parameters(), lr=5e-5, weight_decay=1e-5)

    n_epochs = par['n_epochs']

    """
    We could be using Tensorboard to monitor progress, or we can simply accumulate values of our loss function
    on the train/test partition so we can plot it at a later time.
    """
    epoch_losses = []

    train_losses = []
    test_losses = []
    train_errors = []
    test_errors = []
    epoch_numbers = []

    timestamp_train_begin = train_util.get_timestamp()
    print('starting training: {}'.format(timestamp_train_begin))
    for epoch in np.arange(n_epochs):
        net1.train()

        epoch_loss = 0

        # capture true and estimated values for subsequent analysis
        y_gt_all = []
        y_est_all = []

        for batch_idx, batch_data in enumerate(train_loader):
            inst_id = batch_data['instance_id']
            img = batch_data['image']
            omega = batch_data['omega']
            
            #print('batch_idx: {} inst_id: {}'.format(batch_idx, inst_id))
            #print('img.shape -> {} omega.shape -> {}'.format(img.shape, omega.shape))

            if par['mode'] == 'train_and_test_2_out':
                y_gt = torch.cat((torch.cos(omega), torch.sin(omega)), 1)
                y_gt = y_gt.to(device)
                x  = img.to(device)
                y_est = net1(x)
                y_gt_all.append(y_gt.detach().numpy())
                y_est_all.append(y_est.detach().numpy())
            else:
                y_gt = omega
                y_gt = y_gt.to(device)
                x  = img.to(device)
                y_est = net1(x)
                """
                We could have defined a data loader that skips data/samples that 
                are outside of our chosen omega-range. Here, we are not too concerned
                with computational efficienty so we evaluate net1() on the entire bath
                than rely on PyTorch "mask" functionality to create a view of the tensor
                that excludes the values outside the omega-range. This PyTorch functionaly
                will correctly compute the loss function, even if the *_m tensors are empty
                (in which case the loss would be zero and no gradients would be backpropagated
                """
                y_gt_m, y_est_m, x_m = omega_mask(y_gt, y_est, x)
                    
                for t_ in [(y_gt_all, y_gt_m), (y_est_all, y_est_m)]:
                    t_[0].append(t_[1].cpu().detach().numpy())


            if par['instructor_version']:
                pass
            else:
                #loss = DummyLossValue()
                if par['mode'] == 'train_and_test_2_out':
                    loss = loss_func(y_est, y_gt)
                else:
                    loss = loss_func(y_est_m, y_gt_m)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.detach().item()
                


        y_est_all = np.concatenate(y_est_all)
        y_gt_all = np.concatenate(y_gt_all)

        epoch_loss /= y_gt_all.size # :NOTE: y_gt_all comprises vectors y_gt_m (i.e., after masking)

        final_epoch = (epoch == n_epochs-1)

        verbose = True #par['verbose_perform_training']

        # if verbose:
        #     print('Epoch {}/{}, loss {:.4f}'.format(epoch, n_epochs, epoch_loss))
        # else:
        #     print('.',end='')
        #     sys.stdout.flush()


        par.update({'regression_problem' : True, 'omega_range': omega_range})

        if final_epoch or (epoch % par['eval_every_n_epochs'] == 0):
            if verbose: print('Performing testing (epoch {})'.format(epoch))
            epoch_numbers.append(epoch)
            for dloader_,S_ in [(train_loader,'train'), (test_loader,'test')]:
                par.update({'viz_filename_suffix' : '__{}__epoch_{:04d}'.format(S_,epoch)})
                eval_result = train_util.perform_testing(par, net1, loss_func, device, dloader_, S_)
                (loss, acc, y_est_all, y_gt_all, rmse_error) = eval_result
                if S_ == "train":
                    train_losses.append(loss)
                    train_errors.append(rmse_error)
                if S_ == "test":
                    test_losses.append(loss)
                    test_errors.append(rmse_error)
                if y_est_all is not None:
                    train_util.compute_and_viz_angular_error_metrics(y_gt_all.cpu().detach().numpy(),
                                                                     y_est_all.cpu().detach().numpy(),
                                                                     par)
                else:
                    print('skipping visualization (since y_est_all is None)')
                print('epoch {}/{}, {} mean loss {:.4f} rmse error {:.4f}'.format(epoch, n_epochs, S_, loss, rmse_error))

        epoch_losses.append(epoch_loss)

    timestamp_train_end = train_util.get_timestamp()
    train_util.plotLossEpochs([train_losses, test_losses], [train_errors, test_errors], epoch_numbers)
    print('train begin:end {} : {}'.format(timestamp_train_begin,timestamp_train_end))
    
    if par['viz_pose3d']:
        print('Generating visualization: pose-consistent 3-axis pattern projected on the source images')
        """
        Update our data loader so that images are fetched sequentially, in the
        order of the camera orbiting the object
        In addition, we request that the data loader returns the source images (full-size)
        in addition to the images converted to tensors via one of our transforms
        """
        par.update({'dloader_shuffle': False,
                    'Dataset.getitem_returns_image_viz' : True})
        viz_train_loader, viz_test_loader = data_io.data_loaders.setup_data_loaders(par)
        viz_dloader, viz_dloader_name = (viz_test_loader, 'test')
        net1.eval()
        omega_gt_all = [] # debugging: to confirm we iterated over the intended instances
        omega_est_all = []
        viz_img_idx = 0
        for batch_idx, batch_data in enumerate(viz_dloader):
            inst_id = batch_data['instance_id']
            img = batch_data['image'].to(device)
            img_viz = batch_data['image_viz'] # (batch, H, W, 3) i.e., numpy convention
            omega_gt = batch_data['omega']

            omega_gt_all.append(omega_gt.cpu().numpy())
            with torch.no_grad():
                omega_est = net1(img)
            omega_est_all.append(omega_est.cpu().numpy())
            batch_length = len(batch_data['instance_id'])  # to catch short final batch, possibly < batch_sz
            for idx_in_batch in range(batch_length):
                viz_img_idx += 1
                omega_true = omega_gt[idx_in_batch].item()
                omega_est_ = omega_est[idx_in_batch].item()

                viz_title_str = r'$\omega^\mathrm{}={:0.3f}, \^\omega={:0.3f}$'.format('{true}',omega_true,omega_est_)
                par.update({'polar_r' : camera_orbit_r,
                            'viz_pose_title': viz_title_str,
                            'viz_pose_filename' : 'img_augmented_3axis__{}_{:05d}.jpg'.format(viz_dloader_name,viz_img_idx)})
                geometry.viz_3axis_pattern(omega_est_,
                                           camera_z,
                                           img_viz[idx_in_batch].cpu().numpy(),
                                           par)
            # print('in the viz_dloader loop')

        omega_gt_all = np.concatenate(omega_gt_all)
        omega_est_all = np.concatenate(omega_est_all)

    keyboard()
    print('Finita la comedia')



######################################################################################
#                  main()
######################################################################################
def main(par):
    numpy_randgen = np.random.RandomState(par['rng_seed']) if par['rng_seed'] is not None else np.random.RandomState()
    par.update({'numpy_randgen':numpy_randgen})
    mode=par['mode']
    if "outdir" in par and par["outdir"]:
        try:
            os.mkdir(par["outdir"])
        except Exception as e:
            #print("Error in creating out directory", e)
            pass
    if mode == 'unit_test':
        run_all_unit_tests(par)
    elif mode == 'train_and_test' or mode == 'train_and_test_2_out':
        if 'split_name' not in par or par['split_name'] is None:
            par.update({'split_name':'cvd_split_entire'})
        train_and_test(par)
    else:
        print('no-op: unknown mode {}'.format(mode))

######################################################################################
#                  __main__
######################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices={'unit_test', 'train_and_test', 'train_and_test_2_out'})
    parser.add_argument('--h5_filename', default=None)
    parser.add_argument('--split_name', required=False)
    parser.add_argument('--rng_seed', type=int, required=False)
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--n_pose_classes', type=int, default=8)
    parser.add_argument('--eval_every_n_epochs', type=int, default=1)
    parser.add_argument('--train_batch_sz', type=int, default=10)
    parser.add_argument('--viz_pose3d', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--instructor_version', action='store_true', default=False)

    par__main=vars(parser.parse_args(sys.argv[1:]))
    main(par__main)
