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
import torch
import perf_eval.metrics

import time

#########################################################################
#          get_timestamp()                
#########################################################################
def get_timestamp():
    t = time.localtime()
    return '{0}-{1:02}-{2:02}_{3:02}-{4:02}-{5:02}'.format(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)


#########################################################################
#           compute_and_viz_angular_error_metrics
#########################################################################
def compute_and_viz_angular_error_metrics(y_gt, y_est, par):
    o_err = perf_eval.metrics.angular_diff_1m_cos(y_gt, y_est)
    h_counts, h_bins_e = perf_eval.metrics.viz_histogram(o_err, par)

    return (h_counts, h_bins_e)
        
#########################################################################
#          diff_1m_cos_loss
#########################################################################
class diff_1m_cos_loss(object):
    def __init__(self, reduction=None):
        self.reduction = reduction
    def __call__(self, y_est, y_gt):
        o_diff = 0.5 * (1 - torch.cos(y_est - y_gt))
        loss = torch.sum(o_diff)
        return loss

#########################################################################
#          y_within_range
#########################################################################
def y_within_range(y, o_range):
    mask_h = torch.lt(y, o_range[1])
    mask_l = torch.gt(y, o_range[0])
    mask = torch.logical_and(mask_h, mask_l)
    return mask


#########################################################################
#          Xform_select_in_y_range
#########################################################################
class Xform_select_in_y_range(torch.nn.Module):
    """
    This class can be extended to be used with torchvision.transforms
    However its use in data loader is slightly complicated by the fact that
    the transform is applied to both the labels and the image. So in Dataset.__getitem__
    Therefore, we'd need to handle both 'transform' and 'transform_label'

    """
    def __init__(self, omega_range):
        super().__init__()
        self.omega_range = omega_range

    def forward(self, y_gt, y_est, x=None):
        y_mask = y_within_range(y_gt, self.omega_range)
        y_est_m = None
        x_m = None
        if x is not None:
            x_m = torch.index_select(x,0, torch.where(y_mask)[0])

        if len(y_gt.shape) == 1: 
            y_gt_m = torch.masked_select(y_gt,y_mask)
        else:
            y_gt_m = torch.index_select(y_gt, 0, torch.where(y_mask)[0])

        if y_est is not None:
            # :TOREVIEW: can be simplified?
            if len(y_est.shape) == 1: # regression for 1D pose
                y_est_m = torch.masked_select(y_est,y_mask)
            else: # shape-> (batch_dim, n_class)
                y_est_m = torch.index_select(y_est, 0, torch.where(y_mask)[0])

        return (y_gt_m, y_est_m, x_m)


    def __repr__(self):
        return self.__class__.__name__ + '(omega_range = {})'.format(self.omega_range)



#########################################################################
#          perform_testing()
#########################################################################
def perform_testing(par, model, loss_func, device, loader, name):
    """
    This function can be used inside of the trianing loop to monitor progress
    """

    if not(par['instructor_version']) and loss_func is None:
        print('perform_testing() returning early (since loss_func is None)')
        return (0,*3*(None,))

    omega_mask = Xform_select_in_y_range(par['omega_range'])


    model.eval()
    epoch_loss = 0
    correct = 0
    y_est_all = []
    y_gt_all = []
    with torch.no_grad():
        n_samples = 0
        for batch_idx, batch_data in enumerate(loader):
            inst_id = batch_data['instance_id']
            img = batch_data['image']
            omega = batch_data['omega']
            
            x = img
            if not hasattr(model,'conv_feats'): # model is an MLP
                x  = x.flatten(1)
            y_omega_gt = omega
            x  = x.to(device)
            y_omega_gt = y_omega_gt.to(device)

            if par['regression_problem']:
                y_gt = y_omega_gt
            else: # testing for pose-class classification
                y_gt = pose1D_to_pose_class_v2(y_omega_gt, par['class_proto'])
            
            y_est = model(x)
            y_gt_m, y_est_m, x_m = omega_mask(y_gt, y_est, x)

            n_samples += y_gt_m.shape[0]

            loss = loss_func(y_est_m, y_gt_m)
            
            if False:
                if par['regression_problem']:
                    loss = loss_func(y_est_m, y_omega_gt_m)
                else: # testing for pose-class classification
                    loss = loss_func(y_est_m, y_class_gt)

            epoch_loss += loss.detach().item()

            if not par['regression_problem']: # i.e., classification
                y_est_m = y_est_m.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                correct += y_est_m.eq(y_gt_m.view_as(y_est_m)).sum().item()
                


            for t_ in [(y_gt_all, y_gt_m), (y_est_all, y_est_m)]:
                #t_[0].append(t_[1].cpu().detach().numpy())
                t_[0].append(t_[1])


    epoch_loss /= n_samples 
    acc = correct / n_samples 

    verbose = False # par['verbose_perform_testing']
    if verbose:
        print('{}: Average loss: {:.4f}'.format(
            name, epoch_loss))

    y_est_all = torch.cat(y_est_all)
    y_gt_all = torch.cat(y_gt_all).reshape(-1,1) # column vector
    return (epoch_loss, acc, y_est_all, y_gt_all)

