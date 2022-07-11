# Created by Zdenek Nekula 2022
# this script shows how to reshape images to have a same shape (add or delete pixels)
# if you want later to sum or subtract images, they have to have the same shape

# import libraries
from align_panel.imgsetlib import Imgset

# create instance of Imgset
imgset1 = Imgset("test_data/holography.h5", '-4-H')

# load static images as reference and moving images to be resized:
img_list = ['amplitude', 'phase', 'unwrapped_phase']

for img in img_list:
    img_stat = imgset1.get_data(img, stat=True)
    img_move = imgset1.get_data(img, stat=False)

    print(img)
    print(img_stat.shape)
    print(img_move.shape)

    # check if images have the same shape (number of pixels):
    if img_stat.shape != img_move.shape:
        print(">>> changing shape\n")
        img_move_resized = imgset1.make_same_size(img_stat, img_move)
        img_move = img_move_resized
        imgset1.savedata([img],[img_move])
    else:
        print(">>> shape is the same\n")

    