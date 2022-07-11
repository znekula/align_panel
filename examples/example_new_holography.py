# Created by Zdenek Nekula 2022
# This is an example showing the functionality of "phase" library


# import library
from imgsetnew import Imgset_new_holography

# define data paths
img0_path = 'test_data/+4-H.dm3'
ref0_path = 'test_data/+4-R2.dm3'

img1_path = 'test_data/-4-H.dm3'
ref1_path = 'test_data/-4-R2.dm3'

# create new imagesets
imgset_new0 = Imgset_new_holography(img0_path,ref0_path)
imgset_new1 = Imgset_new_holography(img1_path,ref1_path)

# do phase reconstruction
imgset_new0.phase_reconstruction()
imgset_new1.phase_reconstruction()

# save data into one h5 file
filename = "test_data/holography.h5"
imgset_new0.save(filename,'+4-H', imgset_ref=True)
imgset_new1.save(filename,'-4-H', imgset_ref=False)




# Inner structure of h5 datafile:
# 
# (f) datafile.h5
# |—— (g) ref_imageset_name
# |     |—— (d) img
# |     |—— (d) ref
# |     |—— (d) img_matadata
# |     |—— (d) ref_metadata
# |     |—— (d) amplitude
# |     |—— (d) phase
# |     |—— (d) unwrapped_phase
# |
# |—— (g) ord_imageset_name1
# |     |—— (d) img
# |     |—— (d) ref
# |     |—— (d) img_matadata
# |     |—— (d) ref_metadata
# |     |—— (d) amplitude
# |     |—— (d) phase
# |     |—— (d) unwrapped_phase
# |     |—— (d) tmat              ...transformation matrix created by imgsetlib by alignment to the ref_imageset
# |
# |—— (g) ord_imageset_name2
# |     |—— (d) img
# |     |—— (d) ref
# |     |—— (d) img_matadata
# |     |—— (d) ref_metadata
# |     |—— (d) amplitude
# |     |—— (d) phase
# |     |—— (d) unwrapped_phase
# |     |—— (d) tmat 

