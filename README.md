# Image Restoration Model Zoo
**Status: Not actively maintained, no support provided.**

A number of image restoration models with the glue code to run each easily.
Hopefully, it can be of use to anyone needing to run baselines on deblurring, denoising, image translation, etc.


#### Network modifications
- A lot of networks have been modified so they can take a variable number of input channels `in_ch`.
- The output is often differentiably clamped between [0, 1] to help with training. 

#### Datasets
The repo is set up to use data from either the GoPro or Need for Speed (NFS) dataset. 
Please look into `dataprep` for more details. 

#### Save structure
1. logs 
    1. 21-09-30
        1. 21-09-30-unet
            1. individual experiment
            2. individual experiment
        2. 21-09-30-dncnn
            1. individual experiment
            2. individual experiment
    2. 21-10-01
        1. 21-10-01-uform
            1. individual experiment