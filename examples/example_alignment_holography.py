# Created by Zdenek Nekula 2022
# this script show basic operations done during image alignment 
# from the raw data to aligned phase images

# import libraries
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from align_panel.imgsetlib import Imgset
from align_panel.notebook_panels import align_cell

# create instance of Imgset
imgset_move = Imgset("test_data/holography.h5", '-4-H')
imgset_stat = Imgset("test_data/holography.h5", '+4-H')

# load static image as reference and moving image to be aligned:
img_stat = imgset_move.get_data('unwrapped_phase', stat=True)
img_move = imgset_move.get_data('unwrapped_phase', stat=False)

# do autoalignment
if True:
    tmat = imgset_move.autoalign(img_stat, img_move,'RIGID_BODY', 62, del_back=True)
    print(">>> This is the tmat from autoalignment:"); print(tmat)

# save transformation matrix into the file
if True:
    imgset_move.savedata(['tmat'],[tmat])
    print("tmat saved")

# manual alignment
if True:
    GUI = align_cell(imgset_stat, imgset_move)
    GUI.show()




