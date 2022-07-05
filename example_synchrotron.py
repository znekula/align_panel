# Created by Zdenek Nekula 2022
# This is an example showing the functionality of "phase" library


# import library
from phase import Imgset_new_synchrotron

# define data paths
img0_path = 'test_data/20201218_013_PtychographyRun01_Object_Amplitude.tif'
img1_path = 'test_data/20201218_014_-Ptychography-Run01_Object_Amplitude.tif'

# create new imagesets
imgset_new0 = Imgset_new_synchrotron(img0_path)
imgset_new1 = Imgset_new_synchrotron(img1_path)

# save data into one h5 file
filename = "test_data/synchrotron.h5"
imgset_new0.save(filename,'plus', imgset_ref=True)
imgset_new1.save(filename,'minus', imgset_ref=False)



# Inner structure of h5 datafile of synchrotron:
# 
# (f) datafile.h5
# |—— (g) ref_imageset_name
# |     |—— (d) img
# |     |—— (d) img_matadata
# |
# |—— (g) ord_imageset_name1
# |     |—— (d) img
# |     |—— (d) img_matadata
# |     |—— (d) tmat              ...transformation matrix created by imgsetlib by alignment to the ref_imageset
# |
# |—— (g) ord_imageset_name2
# |     |—— (d) img
# |     |—— (d) img_matadata
# |     |—— (d) tmat        

