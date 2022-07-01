# Created by Zdenek Nekula 2022
# This example shows how to display data which are saved in h5 file

from imgsetlib import Imgset
import matplotlib.pyplot as plt
from skimage import transform as sktransform
import numpy as np

# create an object of Imgset from the h5 file with order 1:
imgset1 = Imgset("test_data/testpath.h5",1)

# Print all atributes ot the imgset1 object:
print(">>> Those are all atributes in the imageset:")
print(imgset1.__dict__.keys())

# Print metadata of image
print(">>> image metadata:")
print(imgset1.img_metadata)

# apply transformation matrix to certain image:
tochange = imgset1.unwrapped_phase # image which should be aligned
tmat = imgset1.tmat # trans. matrix 
img_aligned = sktransform.warp(tochange, tmat) 

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
plt.show()

