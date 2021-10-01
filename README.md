# Image Restoration Model Zoo
**Status: Not actively maintained, no support provided. :no_good:**

A number of image restoration models with the glue code to run each easily.
Hopefully, it can be of use to anyone needing to run baselines on deblurring, denoising, image translation, etc.

This repo is set up to use the 240fps video data from GoPro or Need for Speed (NFS).

### Network modifications
- A lot of networks have been modified so they can take a variable number of input channels `in_ch`.
- The output is often differentiably clamped between [0, 1] to help with training. 

### Getting started
The repo is set up to use data from either the GoPro or NFS dataset. 

#### 0) Install conda environment
```
conda env create -f environment.yml
conda activate model_zoo
```

#### 1) Download the dataset
I set up the repo so that the dataset can be saved in a `data` folder outside of the repo so it can be used 
in other projects, but you can change that in the following scripts. :llama:	

##### GoPro
Download [GoPro_Large_All](http://data.cv.snu.ac.kr:8008/webdav/dataset/GOPRO/GOPRO_Large_all.zip) and use
`dataprep/split_gopro.sh` to split into their respective folders.

##### NFS
Use `dataprep/process_nfs_zip.sh` to download the NFS dataset. 

#### 2) Preprocess the dataset.
Run `python dataprep/initialize_data.py --dataset=gopro` for GoPro or `--dateset=nfs` for NFS. 
Use the `--remote` flag as needed if your dataset is saved locally versus remotely.
The data is processed into a `.pt` file that is a dictionary in which each key is a video number,
and each value is a dictionary where the key is the ID of the clip generated and the value is the video
(yes, a dictionary within a dictionary :eyes:).

#### 3) Train
Run `python train.py --dataset=nfs --decoder=unet` to run the NFS dataset on a U-Net. The network is set up 
to take an 8 frame grayscale average as input and output a single frame restored image.

#### 4) Test
Run `python test.py --folder_name=21-09-30` to evaluate metrics on a model/set of models that were trained
on 21-09-30.


### Save structure
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