This is companion code for [CSEP576au21](https://courses.cs.washington.edu/courses/csep576/21au/)

Instructor: Vitaly Ablavsky

Copyright (c) 2021 Vitaly Ablavsky https://corvidim.net/ablavsky/


# Project 4 Quick-Start Guide

## Assignment

Begin by reading the PDF for Project 4: Pose Estimation from the [course webpage](https://courses.cs.washington.edu/courses/csep576/21au/).

## Dataset

Data for this assignment can be found in two formats:

  * as an HDF5 file (`pontiac_360.h5`) that contains (synthetic) images and camera/pose information and defines train/test splits. The .h5 file can be used with a custom PyTorch Dataset class we are providing for you. It is defined in `data_io.data_loaders`

     * download from https://courses.cs.washington.edu/courses/csep576/21au/projects/project4/pontiac_360.h5

  * as a ZIP archive (`pontiac_360.zip`) that contains individual frames as .jpg files and for each abc.jpg file you will find a corresponding abc.json that contains camera/pose information.
For the .ZIP file you'll need to define your own train/test split e.g., test on every 5th image (i.e., every 5 degrees). You would need to write your own PyTorch Dataset class.


     * download from https://courses.cs.washington.edu/courses/csep576/21au/projects/project4/pontiac_360.zip

The Pontiac GTO 67 model was adapted from Benedikt Bitterli https://benedikt-bitterli.me/resources/


## Code

Portions of the code that need to be modified or studied as part of this assignment are marked as `:TODO:` (for context, see the PDF for P4 on the [course webpage](https://courses.cs.washington.edu/courses/csep576/21au/))

The code has been tested on Linux platforms. Your student CSE account should provide you with access to Linux compute cycles. Alternatively, you may run this code on the platform of your choice, possibly in a virtual machine.


The main prerequisities/dependencies [code has been tested on versions in brackets]:

  * Python [v. 3.7.11]
  * PyTorch [v. 1.9.0] (you should have it installed from P3)
  * h5py [v. 2.8.0] (to handle HDF5 files), available from Anaconda and other Python distributions: https://docs.h5py.org/

**[a]** Examples below assume you're running this code at a Linux command prompt, with `p4/` as the working directory.
You should not need to manually configure `PYTHONPATH` and can just invoke the code as `python pose_regress.py --...`
with command-line flags in any order. If you use an IDE you will need to pass those flags via the appropriate GUI.

**[b]** Begin by running the unit tests:

```
python pose_regress.py --mode=unit_test --outdir=/PATH/TO/OUTDIR/ --h5_filename=/PATH/TO/pontiac_360.h5
```

  * the unit tests will all return False initially
  * `--h5_filename` refers to the HDF5 file containing the train/test data (see above for download link)
 
**[c]** To perform training and evaluation invoke this command:

```
python pose_regress.py --mode=train_and_test --n_epochs=1 --eval_every_n_epochs=1 --h5_filename=/PATH/TO/pontiac_360.h5 --outdir=/PATH/TO/OUTDIR/ --split_name=cvd_split_every5  --viz_pose3d
```

  * `--viz_pose3d`  <-- (optional) produces 72 JPG files showing the estimated pose projected onto source images
  * `--eval_every_n_epochs`  <-- we recommend setting it to `1` initially

**[d]** Some noteworthy runtime arguments that you can specify on the command line or via your preferred development environment:

* `--n_epochs_train=15` is a good number to try first






