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
import os

import numpy as np

np.set_printoptions(formatter={'float': '{: 0.3f}'.format},
                    linewidth=120) # default is 75

import matplotlib.pyplot as plt

######################################################################################
#                  camera_xform_from_lookat
######################################################################################
def camera_xform_from_lookat(eye_coords, look_at_coords, instructor_version_flag):
    """
    You can view nice figures related to the "Look At" transformation  in [R1]
    but keep in mind that our construction will be different (our positive Z-axis will point toward look-at)

    

    [R1] https://learnopengl.com/Getting-started/Camera
    """
    
    eye_world = eye_coords # not homogeneous
    at_world  = look_at_coords # not homogeneous
    
    eye_world_4d = np.append(eye_world,1)
    at_world_4d = np.append(at_world,1)

    P1 =np.eye(4)

    if instructor_version_flag:
        pass
    else:
        """
        :TODO:
        Compute R and T such that they "move" (transform) a camera located
        at the origin of the world coordinate system (camera's center is at (0,0,0)
        and the camera's Z-axis is aligned with the world coordinate system's Z-axis) 
        to a pose where the camera's center is at "eye_world" and the camera's z-axis is
        pointing at "at_world". The camera should be horizontal (i.e., no roll) so the horizon line
        would appear along the camera's rows.
        

        Try to work out the math on your own. We will post detailed
        hints on Ed; please use them only after you've given this a
        good try on your own.

        """
        R = np.eye(3)
        z_axis = (at_world - eye_world)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # cross product z_axis with vertical axis to get x_axis, because camera has no roll
        vertical_axis_unit_vector = np.array([0, 0, 1])
        x_axis = np.cross(vertical_axis_unit_vector, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        R[:, 0] = x_axis
        R[:, 1] = y_axis
        R[:, 2] = z_axis
        T = eye_world


    P1[0:3,0:3] = R
    P1[0:3,-1] = T
    #print("P1", P1)
    """
    At this point, we have defined a transformation in the homogeneous coordinates that
    "moves" the camera's coordinate system (3-axis) from its old location, coincident with
    the world coordinate sytem to a new location, consistent with "eye" and "look at."
    In particular (by construction) np.matmul(P1, look_at_4d) -> eye_world_4d

    However, we are interested in how the world looks in the camera coordinate system,
    i.e., transofrming world coordinates into camera coordinates,
    thus:
    """
    P1 = np.linalg.pinv(P1) # pseudo-inverse via SVD
    #print("P1_inv", P1)
    """
    Taking an inverse of P1 via pseudeo-inverse is wasteful because
    we are not taking advantage of the special structure of P1

    Let's construct a different matrix, call it "P" such that
    when we do a P.T (transpose), we'll recover P1


    Note that we're constructing both P1 and P as an exercise; in practice you'd
    construct P, and then use P.T in the client code (i.e., would not bother with P1)
    """
    P = np.eye(4)

    if instructor_version_flag:
        pass
    else:
        """
          added my code to form P
          then call
          P = P.T

        to check that P is correct added a assert
        """
        P[0:3, 0:3] = R
        V = -np.dot(R.T, T)
        P[-1, 0:3] = V
        #print("P", P)
        P = P.T
        #print("P_T", P)
        assert np.allclose(P, P1)
        pass
    
    return P1


######################################################################################
#                 project_world_to_camera()
######################################################################################
def project_world_to_camera(P,X):
    """
    P: projection matrix
    X: 3D-world points as 3D vectors (not homegeneous)

    output: 2D camera/image points
    """
    X_w = np.concatenate((X,np.ones((1,X.shape[1]))),axis=0) # homogeneous coords

    X_c = np.dot(P, X_w)
    X_c /= X_c[2,:] # divided by the third row
    X_c = X_c[0:2,:] # drop the last row

    return X_c



######################################################################################
#                  project_3axis_pattern()
######################################################################################
def project_3axis_pattern(omega, z_height, img, r, instructor_version_flag):
    """
    This function returns a 3-axis geometric pattern consistent with
    projective geometry specified by the input arguments.

    X.shape -> (2,4) where X[:,0] is the 'origin', and the three segments are
    (X[:,0], X[:,1]) <-- toward us (eventually drawn red)
    (X[:,0], X[:,2]) <-- right (eventually drawn green)
    (X[:,0], X[:,3]) <-- Up (eventually drawn blue)
    """

    x = r * np.cos(omega)
    y = r * np.sin(omega)
    at_world = np.zeros(3)
    eye_world = np.array([x,y,z_height])

    P = camera_xform_from_lookat(eye_world, at_world, instructor_version_flag)

    """
    project a 3axis pattern
    * origin at center of image
    * longest segment 75 pixels
    """
    X = np.array([0,0,0,1,0,0,0,1,0,0,0,1]).reshape(4,3).T    # unit-length base
    X_c = project_world_to_camera(P,X)
    X_c_len = 75
    scale_ = X_c_len/np.max(X_c)
    X_c *= scale_
    origin_h = img.shape[0]//2
    origin_w = img.shape[1]//2
    X_c[0] += origin_w
    X_c[1,0:3] += origin_h
    X_c[1,3] = origin_h - X_c[1,3]

    return X_c

######################################################################################
#                 draw_3axis_on_image()
######################################################################################
def draw_3axis_on_image(X_c, fig_handle):
    plt.figure(fig_handle.number)
    for seg_idx,seg_color in enumerate(['r','g','b']):
        segment = np.array([X_c[:,0], X_c[:,seg_idx+1]])
        plt.plot(*segment.T,'w',linewidth=8) # white background behind segment
        plt.plot(*segment.T,seg_color,linewidth=4)


######################################################################################
#                  viz_3axis_pattern()
#####################################################################################
def viz_3axis_pattern(omega, z_height, img, par):    
    verbose = par['verbose']
    if 'viz_interactive' in par and par['viz_interactive']:
        viz_interactive = True
        plt.ion()
    else:
        viz_interactive = False

    if 'outdir' not in par or par['outdir'] is None:
        raise ValueError('Please provide directory for output/temp files via --outdir')
    elif not os.path.exists(par['outdir']):
        raise IOError('outdir does not exist: {}'.format(par['outdir']))
    else:
        outdir = par['outdir']

    fig_handle = plt.figure()
    plt.imshow(img)

    if not 'viz_pose_title' in par:
        plt.title(r'$\omega:{:0.3f}$'.format(omega))
    else:
        plt.title(par['viz_pose_title'])

    if 'rot_str_for_plt' in par:
        plt.suptitle(par['rot_str_for_plt'])

    r = par['polar_r']
    # X_c will contain image coordinates of three line segments
    # that form an object-centric coordinate system viewed from the
    # Look-At direction specified by the input arguments
    X_c = project_3axis_pattern(omega, z_height, img, r, par['instructor_version'])

    draw_3axis_on_image(X_c, fig_handle)

    if 'viz_interactive' in par and par['viz_interactive']:
        plt.ion()
        plt.show()

    if not viz_interactive:
        outfile_base = par['viz_pose_filename']
        outfile = os.path.join(outdir, outfile_base)
        plt.savefig(outfile)
        plt.close()
        if verbose: print('rendered image saved to {}'.format(outfile))
