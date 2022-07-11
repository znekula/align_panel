# Created by Zdenek Nekula 2022
# this script show basic operations done during image alignment 
# from the raw data to aligned phase images

# import libraries
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from imgsetlib import Imgset

# create instance of Imgset
imgset1 = Imgset("test_data/holography.h5", '-4-H')

# load static image as reference and moving image to be aligned:
img_stat = imgset1.get_data('unwrapped_phase', stat=True)
img_move = imgset1.get_data('unwrapped_phase', stat=False)

# do autoalignment
if True:
    tmat = imgset1.autoalign(img_stat, img_move,'RIGID_BODY', 1000,del_back=True)
    print(">>> This is the tmat from autoalignment:"); print(tmat)

# manual point alignmetn
if False:
    tmat = imgset1.manual_point(img_stat_manual, img_move_manual)

# manual fine alignment, uses transformation matrix in memory for prealignment
if True:
    tmat = imgset1.manual_fine(img_stat, img_move, tmat)

# save transformation matrix into the file
if True:
    imgset1.savedata(['tmat'],[tmat])
    print("tmat saved")


