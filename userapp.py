
#%% import
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import numpy as np
import panel as pn
from bokeh.plotting import figure, show
pn.config.inline = True
# confing.inline tells panel to load the necessary javascript files locally and not from the internet
# This is necessary when runing on the GRE649 server as it has an incomplete internet connection
pn.extension()
from libertem_ui.utils.notebook_tools import notebook_fullwidth, stop_nb
from align_panel import point_registration, fine_adjust, array_format
notebook_fullwidth()


#%% make phase
######################################################################################################
if False:
    from phase import Imgset_new
    """define data paths"""
    img1_path = 'test_data/Tilt20_OL+4%_H2.dm3'
    ref1_path = 'test_data/Tilt20_OL+4%_R2.dm3'
    img0_path = 'test_data/Tilt20_OL-4%_H2.dm3'
    ref0_path = 'test_data/Tilt20_OL-4%_R2.dm3'

    # img0_path = 'test_data/-4-H.dm3'
    # ref0_path = 'test_data/-4-R2.dm3'
    # img1_path = 'test_data/+4-H.dm3'
    # ref1_path = 'test_data/+4-R2.dm3'

    """create new imagesets"""
    imgset_new0 = Imgset_new(img0_path,ref0_path); print("<<< imgset_new0 done")
    imgset_new1 = Imgset_new(img1_path,ref1_path); print("<<< imgset_new1 done")

    """do phase reconstruction """
    imgset_new0.phase_reconstruction()
    imgset_new1.phase_reconstruction()

    """save data"""
    filename = "test_data/mytestfile5.h5"
    imgset_new0.save(filename,0)
    imgset_new1.save(filename,1)
#######################################################################################################


#%% auto aling
#######################################################################################################
from imgsetlib import Imgset
imgset1 = Imgset("test_data/mytestfile5.h5",1)

#imgset1.autoalign(1000,del_back=False, img_stat = imgset1.amplitude_stat, img_move = imgset1.amplitude)
print(">>> This is the tmat from autoalignment:"); print(imgset1.tmat)


img_stat_manual = imgset1.unwrapped_phase_stat
img_move_manual = imgset1.unwrapped_phase

if False:
    imgset1.manual_point(img_stat_manual, img_move_manual)
if True:
    imgset1.manual_fine(img_stat_manual, img_move_manual, imgset1.tmat)

print("done all, this is the end")


