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


from __future__ import print_function

from pdb import set_trace as keyboard

import sys, traceback, os, subprocess
import glob
import argparse

import numpy as np
np.set_printoptions(suppress=True) # suppress printing small values in scientific notation

import time
import pickle
import gzip
import collections
from collections import OrderedDict
import itertools

import io
import PIL.Image
import skimage.transform
import skimage.io
import h5py

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LogicError(Exception): pass

####################################################################
#                  img_bytearray2npy()
####################################################################
def img_bytearray2npy(img_buff,size_only=False):
    m_ =  skimage.io.imread(io.BytesIO(img_buff))
    if size_only:
        sz = m_.shape
        return sz if len(sz) == 3 else (sz[0],sz[1],1)
    else:
        return m_


#########################################
#          Pontiac_H5_Dataset
#########################################
class Pontiac_H5_Dataset(Dataset):
    def __init__(self, par):
        """

        payload_format: 'dict' or 'tuple'

        (Pdb) 
        type(x) -> <class 'torch.Tensor'>
        x.shape -> torch.Size([256, 784])
        torch.min(x) -> tensor(0., device='cuda:0')
        torch.max(x) ->tensor(1., device='cuda:0')

        """
        assert 'h5_filename' in par
        self.h5db = h5py.File(par['h5_filename'],'r')
        assert 'split_name' in par
        split_name = par['split_name']
        assert 'partition' in par
        partition = par['partition']
        payload_format = par['payload_format'] if 'payload_format' in par else 'dict'
        if partition not in {'train','val','test'}:
            raise ValueError('invalid partition {}'.format(partition))
        if not payload_format in ['dict','tuple']:
            raise ValueError('invalid payload_format {}'.format(payload_format))
        self.payload_format = payload_format

        self.transform = par['transform'] if 'transform' in par else None

        if 'Dataset.getitem_returns_image_viz' not in par:
            self.getitem_returns_image_viz = False
        else:
            self.getitem_returns_image_viz = par['Dataset.getitem_returns_image_viz']

        num_datasets = len(self.h5db)
        assert(len(self.h5db.keys()) == num_datasets)

        fqpn = '{}/{}'.format(split_name,partition)
        if fqpn in self.h5db.attrs:
            self.indices_ref = self.h5db.attrs[fqpn][0][4]
            self.idx_list = self.h5db[self.indices_ref][()].tolist()
            self.idx_range = [-1, -1]
            num_samples = len(self.idx_list)
        else:
            raise ValueError('invalid split/partition spec: "{}" not found in db.attrs'.format(fqpn))

    def __len__(self):
        if bool(self.indices_ref):
            return len(self.h5db[self.indices_ref][()])
        else:
            return self.idx_range[1] - self.idx_range[0]

    def __getitem__(self, idx):
        # :TOREVIEW:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # H5 constructed with splits via reference; see fqpn above
        if self.indices_ref is not None:
            idx = self.idx_list[idx]

        instance_id = idx
        t_ = self.h5db['img_as_bytearray'][idx]

        img_np = img_bytearray2npy(t_)
        # must conver to PIL.Image to use torchvision.transforms.Resize()
        img = PIL.Image.fromarray(img_np)
        # :NOTE: np.array and torch.tensor -> shape  but PIL.Image -> size
        # print('in __getitem__ img.shape={}'.format(img.size))
        omega = self.h5db['rotation_radians'][idx]
        # reconciling omega range: see notes_exp.txt 
        omega = np.pi - omega # maps [0, 2pi) -> [pi, -pi)

        if self.transform is not None:
            img_xf = self.transform(img)
        # print('in __getitem__ after transform img.shape={}'.format(img.shape))

        sample = {'instance_id':instance_id,
                  'image': img_xf,
                  'omega':omega}

        # for visualizion we need the original image
        # doing so will increase the memory footprint, so
        # must launch docker container with shared ipc memroy (--ipc=host)
        if self.getitem_returns_image_viz:
            sample.update({'image_viz' : torch.tensor(img_np, dtype=torch.uint8)})

        return sample




#########################################################################
#                        setup_data_loaders()
#########################################################################
def setup_data_loaders(par):
    assert 'payload_format' in par
    if 'split_name' not in par:
        raise ValueError("missing par['split_name']")
    if 'dloader_shuffle' not in par:
        par.update({'dloader_shuffle': True})

    print(5*'-','setup_data_loaders()')
    
    verbose = par['verbose']
    split_name = par['split_name']
    payload_format_ = par['payload_format']

    batch_sz = par['batch_sz'] if 'batch_sz' in par else 4

    dloaders = []

    partitions = ['train','test']

    for partition in partitions:
        par.update({'partition' : partition})

        if verbose:
            print('setting up dloader for {} partition'.format(partition))
        pontiac_dset = Pontiac_H5_Dataset(par)
        dloader = DataLoader(pontiac_dset, batch_size=batch_sz,
                             shuffle=par['dloader_shuffle'], 
                             num_workers=0)

        dloaders.append(dloader)

    return dloaders

#########################################################################
#                        get_partition_names()
#########################################################################
def get_partition_names(h5_filename):
    h5db = h5py.File(h5_filename,'r')
    contains_entire = False
    partitions = []
    for split_ in h5db.attrs['split']:
        #[CVD] decode dtype('S6') aka type np.bytes_ -> str (e.g.: b'train' -> 'train')
        partition_name = split_[0].decode('UTF-8')
        if partition_name == 'entire':
            contains_entire = True
        elif partition_name not in partitions:
            partitions.append(partition_name)
    return (partitions, contains_entire)
            
    


#########################################################################
#                        verify_dloaders()
#########################################################################
def verify_dloaders(par):
    """
    ./run.sh data_loaders.py --mode=verify_dloaders --h5_filename=/path/to/pontiac360.h5
       [--tb_logdir...]
    """
    print(5*'-','verify_dloaders()')
    xform_0 = None
    batch_sz = 4

    par.update({'payload_format':'dict',
                'dloader_shuffle': True})

    assert 'split_name' in par
    split_name = par['split_name']
    payload_format_ = par['payload_format']


    """
    :NOTE: see note about normalization in Pontiac_H5_Dataset()
    """
    xform_0 = None
    xform_1 = transforms.Compose([transforms.ToTensor()])
    xform_2 = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,),
                                                     (1.0,))])

    xform_3 = transforms.Compose([transforms.Resize((64, 64)),
                                  transforms.ToTensor()])
    
    xform_ = xform_3

    par.update({'batch_sz':batch_sz,
                'transform' : xform_ })

    dloaders = setup_data_loaders(par)

    for dloader_idx, dloader in enumerate(dloaders):
        print('dloader {} (of {})'.format(dloader_idx,len(dloaders)))
        for batch_idx, batch_data in enumerate(dloader):
            if payload_format_ == 'dict':
                inst_id = batch_data['instance_id']
                img = batch_data['image']
                omega = batch_data['omega']
                # knob_radial = batch_data['knob_radial']
                print('batch_idx: {} inst_id: {}'.format(batch_idx, inst_id))
                print('img.shape -> {} omega.shape -> {}'.format(img.shape, omega.shape))
                # keyboard()
                if batch_idx == 0:
                    break
    keyboard()
    print('Finita la comedia')

#########################################################################
#                        unit_test()
#########################################################################
def unit_test(par):
    h5db = h5py.File(par['h5_filename'],'r')
    print('h5db open for inspection')
    keyboard()


######################################################################################
#                  get_valid_modes()
######################################################################################
def get_valid_modes():
    valid_modes = ['unit_test', 'verify_dloaders']
    return valid_modes
    
#########################################################################
#                        main()
#########################################################################
def main(par):
    numpy_randgen = np.random.RandomState(par['rng_seed']) if 'rng_seed' in par and par['rng_seed'] is not None else np.random.RandomState()

    par.update(
        {'numpy_randgen':numpy_randgen,
         'hdf5_ds_initial_size':10, # not clear if 0 would be allowed by split_dict
         'H5PYDataset_split_name_all_data':'entire'
        })
    mode = par['mode']
    if mode=='unit_test':
        unit_test(par)
    elif mode=='verify_dloaders':
        verify_dloaders(par)
    else:
        print('unknown mode {0}. no-op'.format(mode))


#########################################################################
#                        __main__
#########################################################################
"""

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=get_valid_modes())
    parser.add_argument('--h5_filename',required=True)
    parser.add_argument('--split_name',required=True)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    par__main=vars(parser.parse_args(sys.argv[1:]))
    main(par__main)

