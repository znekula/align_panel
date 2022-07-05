# Created by Zdenek Nekula 2022
# This example shows how to display data which are saved in h5 file
# Using the Imageset class representing one particlular imageset

from imgsetlib import Imgset
import matplotlib.pyplot as plt
from skimage import transform as sktransform
import numpy as np
import h5py
import ast
import yaml
# set datapaht to the h5 data file
datapath = 'test_data/mytestfile.h5'

# create an object of Imgset from the h5 file '-4-H':
imgset1 = Imgset(datapath,'-4-H')

# Print all atributes ot the imgset1 object:
print(">>> Those are all atributes in the imageset:")
print(imgset1.__dict__.keys())

# Print metadata of image
print(">>> image metadata:")
metadata_str = imgset1.img_metadata #metadata are stored as string
metadata_dict = ast.literal_eval(metadata_str) # convert them into dictionary
metadata_prettyprint = yaml.dump(metadata_dict, default_flow_style=False) # make indentation to apear more beautifull
print(metadata_prettyprint) # print in terminal

# plot some results:
fig, ax = plt.subplots(1, 5, figsize=(20, 9))

im0 = ax[0].imshow(imgset1.unwrapped_phase_stat, cmap='gist_rainbow')
ax[0].set_title('unwrapped_phase_stat')
ax[0].axis('off')
fig.colorbar(im0, ax=ax[0])

im1 = ax[1].imshow(imgset1.unwrapped_phase, cmap='gist_rainbow')
ax[1].set_title('unwrapped_phase')
ax[1].axis('off')
fig.colorbar(im1, ax=ax[1])

# We chack if there is a transformation matrix already
try:
    tochange = imgset1.unwrapped_phase # image which should be aligned
    tmat = imgset1.tmat # trans. matrix 
    # apply transformation matrix to certain image:
    img_aligned = sktransform.warp(tochange, tmat) 

    
    im2 = ax[2].imshow(img_aligned, cmap='gist_rainbow')
    ax[2].set_title('img_aligned')
    ax[2].axis('off')
    fig.colorbar(im2, ax=ax[2])

    im3 = ax[3].imshow((imgset1.unwrapped_phase_stat + img_aligned)/2, cmap='gist_rainbow')
    ax[3].set_title('sum')
    ax[3].axis('off')
    fig.colorbar(im3, ax=ax[3])

    im4 = ax[4].imshow((imgset1.unwrapped_phase_stat - img_aligned)*2, cmap='gist_rainbow')
    ax[4].set_title('dif')
    ax[4].axis('off')
    fig.colorbar(im4, ax=ax[4])
except:
    print(">>> Probably no transformation matrix present. First, do the alignment.")
    ax[2].set_title('img_aligned - tr. matrix required')
    ax[3].set_title('sum - tr. matrix required')
    ax[4].set_title('dif - tr. matrix required')

plt.show()

