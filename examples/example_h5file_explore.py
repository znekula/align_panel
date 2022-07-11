# Created by Zdenek Nekula 2022
# Shows how to explore content of an HDF5 file

from align_panel.imgsetlib import H5file

# define a path to a h5 datafile
myfilename = 'test_data/holography.h5'
mydatafile = H5file(myfilename)




# Print all imagesets inside the h5 file
print ("\nthis is the content:")
print ("reference imageset name: " + str(mydatafile.ref_imageset_name))
print ("imagesets names: " + str(mydatafile.imageset_names))
print ("the rest: " + str(mydatafile.rest))

print ("\nHere are the full names of imagesets:")
print ("reference imageset full name: " + str(mydatafile.ref_imageset_fullname))
print ("imagesets full names: " + str(mydatafile.imageset_fullnames) + '\n')





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