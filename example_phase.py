# Created by Zdenek Nekula 2022
# This is an example showing the functionality of "phase" library


# import library
from phase import Imgset_new

# define data paths
img0_path = 'test_data/+4-H.dm3'
ref0_path = 'test_data/+4-R2.dm3'

img1_path = 'test_data/-4-H.dm3'
ref1_path = 'test_data/-4-R2.dm3'

# create new imagesets
imgset_new0 = Imgset_new(img0_path,ref0_path)
imgset_new1 = Imgset_new(img1_path,ref1_path)

# do phase reconstruction
imgset_new0.phase_reconstruction()
imgset_new1.phase_reconstruction()

# save data into one h5 file
filename = "test_data/mytestfile.h5"
imgset_new0.save(filename,'+4-H', imgset_ref=True)
imgset_new1.save(filename,'-4-H', imgset_ref=False)



