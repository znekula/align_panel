# Created by Zdenek Nekula 2022
# this script show basic operations done during image alignment 
# from the raw data to aligned phase images

# import libraries
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from imgsetlib import Imgset

# create object of Imgset
imgset1 = Imgset("test_data/mytestfile.h5", '-4-H')

# do autoalignment
if True:
    imgset1.autoalign(1000,del_back=True, img_stat = imgset1.unwrapped_phase_stat, img_move = imgset1.unwrapped_phase, keeporiginalsize = False)
    print(">>> This is the tmat from autoalignment:"); print(imgset1.tmat)

# set images for manual alignment
img_stat_manual = imgset1.unwrapped_phase_stat
img_move_manual = imgset1.unwrapped_phase

# manual point alignmetn
if False:
    imgset1.manual_point(img_stat_manual, img_move_manual)

# manual fine alignment, uses transformation matrix in memory for prealignment
if True:
    imgset1.manual_fine(img_stat_manual, img_move_manual, imgset1.tmat)

print("done all, this is the end")


