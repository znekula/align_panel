"""
Here I load h5 file and do alignment of image sets in respect to the reference one.
"""

import h5py
import numpy as np
#from skimage import transform, io, exposure
from pystackreg import StackReg
#import cv2 as cv
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage import transform as sktransform


#plt.close("all")


class Imgset:
    """loads image set (labeled by order number), and 
    reference image set (labeled defaultly by number 0) from h5  file, and 
    makes automatic alignment by StackReg. 
    Then saves transformation matrix and transformed unwrapped phase 
    to the same h5 file"""
    def __init__(self, filename, order):
        """file name is the name of the h5 file. Order is the number of the image set"""
        """read data"""
        self.filename = filename    
        self.order = order
        f = h5py.File(filename, 'r')

        dset_unwrapped_phase_stat = f['imageset' + str(0) +'/unwrapped_phase']
        self.unwrapped_phase_stat = np.asarray(dset_unwrapped_phase_stat)
        dset_unwrapped_phase = f['imageset' + str(order) +'/unwrapped_phase']
        self.unwrapped_phase = np.asarray(dset_unwrapped_phase)
        
        dset_amplitude_stat = f['imageset' + str(0) +'/amplitude']
        self.amplitude_stat = np.asarray(dset_amplitude_stat)
        dset_amplitude = f['imageset' + str(order) +'/amplitude']
        self.amplitude = np.asarray(dset_amplitude)

        dset_img_stat = f['imageset' + str(0) +'/img']
        self.img_stat = np.asarray(dset_img_stat)
        dset_img = f['imageset' + str(order) +'/img']
        self.img = np.asarray(dset_img)

        dset_ref_stat = f['imageset' + str(0) +'/ref']
        self.ref_stat = np.asarray(dset_ref_stat)
        dset_ref = f['imageset' + str(order) +'/ref']
        self.ref = np.asarray(dset_ref)

        dset_phase_stat = f['imageset' + str(0) +'/phase']
        self.phase_stat = np.asarray(dset_phase_stat)
        dset_phase = f['imageset' + str(order) +'/phase']
        self.phase = np.asarray(dset_phase)

        dset_img_metadata_stat = f['imageset' + str(0) +'/img_metadata']
        self.img_metadata_stat = np.asarray(dset_img_metadata_stat)
        dset_img_metadata = f['imageset' + str(order) +'/img_metadata']
        self.img_metadata = np.asarray(dset_img_metadata)

        dset_ref_metadata_stat = f['imageset' + str(0) +'/ref_metadata']
        self.ref_metadata_stat = np.asarray(dset_ref_metadata_stat)
        dset_ref_metadata = f['imageset' + str(order) +'/ref_metadata']
        self.ref_metadata = np.asarray(dset_ref_metadata)

        group = f['/imageset'+ str(self.order)]
        if "tmat" in group.keys():
            dset_tmat = f['imageset' + str(order) +'/tmat']
            self.tmat = np.asarray(dset_tmat)

        f.close()
        
    """define help functions:"""
    def make_same_size(self,img_refsize, img_changesize):
        refsize0 = img_refsize[0].size; refsize1 = img_refsize[1].size  
        changesize0 = img_changesize[0].size; changesize1 = img_changesize[1].size
        img_changed = img_changesize
        dif0 = changesize0 - refsize0
        if dif0 < 0 :
            img_changed = np.append(img_changed,np.zeros((dif0,changesize1)),axis = 0)
        elif dif0 > 0 :
            img_changed = img_changed[dif0:,:] 
        changesize0 = img_changesize[0].size; changesize1 = img_changesize[1].size   
        dif1 = changesize1 - refsize1    
        if dif1 < 0 :
            img_changed = np.append(img_changed,np.zeros((changesize0,dif1)),axis = 1)
        elif dif1 > 0 :
            img_changed = img_changed[:,dif1:]
        return img_changed
    
    def delete_background(self,img_original, reduction = 500, value_high = 1, value_low = 0 ):
        """img is input image
        reduction means reduction of number of bins in image histogram
        this function automatically find a treshold to distinguis background and the wire
        all pixel-values lower than thrashold will be set to value_low
        all pixel-values higher than thrashold will be set to value_high
        """    
        img = deepcopy(img_original)
        img_1D = img.flatten()
        img_1D_sort = np.sort(img_1D)
        for i in range(50): #delete extremes
            img_1D_sort = np.delete(img_1D_sort, 0)
            img_1D_sort = np.delete(img_1D_sort, -1)
        reduction = 500
        img_1D_sort_reduced = np.zeros(int(img_1D_sort.size/reduction))
        for a in range(int(img_1D_sort.size/reduction)):
            img_1D_sort_reduced[a] = np.average(img_1D_sort[a*reduction:a*reduction+reduction])
        derivative = np.convolve(img_1D_sort_reduced,[1,-1],'same')
        derivative[0]=0; derivative[-1]=0
        trashold = img_1D_sort_reduced[np.argmax(derivative)]
        for i in range(img[:,0].size):
            for j in range(img[0,:].size):
                if img[i,j] <  trashold:
                    img[i,j] = value_low
                else:
                    img[i,j] = value_high            
        return img
    
    def get_transformation(self,tmat):
        """tmat = transformation matrix from stackreg, 
        gives transformation in pixels"""
        shift_x = tmat[0,2]
        shift_y = tmat[1,2]
        rotation = np.arccos(tmat[0,0])
        return [shift_x,shift_y,rotation]
        
    # def inverse_coordinate_tmat(self,tmat):
    #     """takes the opposite rotation angel and opositte shift direstion"""
    #     tmat_inv = deepcopy(tmat)
    #     tmat_inv[0,1] = tmat_inv[0,1]*(-1)
    #     tmat_inv[1,0] = tmat_inv[1,0]*(-1)
    #     tmat_inv[0,2] = tmat_inv[0,2]*(-1)
    #     tmat_inv[1,2] = tmat_inv[1,2]*(-1)
    #     return tmat_inv

    def autoalign(self, rougness=500, del_back=True, keeporiginalsize = False, **images):
        """makes autoalignment of unwrapped phases images, aligning just objest without background
        roughnes = rougness of estimation the border betwen object and background. 
        Low roughness estimation can be disrupted by noise.
        """
        try:
            img_stat = images['img_stat']
            img_move = images['img_move']
            print(">>> using user values")

        except:
            img_stat = self.amplitude_stat
            img_move = self.amplitude
            print(">>> using default values")

        img_move = self.make_same_size(self.amplitude_stat, img_move )
        
        sr = StackReg(StackReg.RIGID_BODY)
        if del_back==True:
            img_stat_noback = self.delete_background(img_stat, rougness)
            self.img_stat_noback=img_stat_noback
            img_move_noback = self.delete_background(img_move, rougness)
            self.img_move_noback=img_move_noback
            reg = sr.register_transform(img_stat_noback, img_move_noback)
        else:            
            reg = sr.register_transform(img_stat, img_move)
        reg = reg.clip(min=0)

        self.tmat = sr.get_matrix()
        dimensions = img_move.shape
        dimensions = (dimensions[1],dimensions[0])#inverse order of coordinates
        #self.tmat_inv = self.inverse_coordinate_tmat(self.tmat)
        self.img_aligned = sktransform.warp(img_move, self.tmat)
        
        
        """Save data to h5 file"""
        # f = h5py.File(self.filename, "a")  
        # datasets = ["tmat", "tmat_invertcoord", "unwrapped_phase_aligned"]
        # group = f['/imageset'+ str(self.order)]
        # for dataset_name in datasets:            
        #     if dataset_name in group.keys():                
        #         del group[dataset_name]
        # f.create_dataset('imageset' + str(self.order)+'/tmat', data = self.tmat)
        # #f.create_dataset('imageset' + str(self.order)+'/tmat_invertcoord', data = self.tmat_inv)
        # #f.create_dataset('imageset' + str(self.order)+'/unwrapped_phase_aligned', data = self.img_aligned)        
        # f.close()

        self.savedata(["tmat"],[self.tmat])
        if keeporiginalsize == False:
            datasets = [self.img, self.ref, self.amplitude, self.phase, self.unwrapped_phase]
            count=0
            for dataset in datasets:
                datasets[count] = self.make_same_size(self.amplitude_stat, dataset)
                count+=1
        self.savedata(['img','ref','amplitude','phase','unwrapped_phase'], datasets)
    
    def savedata(self, datasets_names, datasets_data):
        """Save data to the h5 file. dataset_names = list of datasets, for example ["tmat", "unwrapped_phase_aligned"]
        datasets_data = list of data, for example [self.tmat, "self.unwrapped_phase_aligned"]"""
        f = h5py.File(self.filename, "a")  
        #datasets_names = ["tmat"]
        #datasets_data = [self.tmat]

        group = f['/imageset'+ str(self.order)]
        count=0
        for dataset_name in datasets_names:
            if dataset_name in group.keys():                
                del group[dataset_name]
            f.create_dataset('imageset' + str(self.order)+'/'+dataset_name, data = datasets_data[count])
            count +=1       
        f.close()

    def manual_fine(self, img_stat, img_move, initial_transform_matrix=np.identity(3)):
        from align_panel import fine_adjust
        initial_transform = sktransform.AffineTransform(matrix=initial_transform_matrix) #make enclosed variable containing the transformation matrix
        layout, fine_getter = fine_adjust(img_stat, img_move, initial_transform = initial_transform)
        print(">>> to kill server pres ctrl+c into terminal")
        layout.show(threaded=False)
        tmat_new = fine_getter()['transform'].params
        print(">>> New transformation matrix from manual fine adjustment:"); print(tmat_new)
        userinput = input('>>> Do you want to save a new transformation matrix?  y/n ')
        if userinput == "y":
            self.tmat = tmat_new
            self.savedata(['tmat'],[tmat_new]); print(">>> Transformation matrix saved")
        else:
            print(">>> New transformation matrix rejected")

    def manual_point(self, img_stat, img_move):
        from align_panel import point_registration
        layout, transform_getter = point_registration(img_stat, img_move)
        print(">>> to kill server pres ctrl+c into terminal")
        layout.show()
        tmat_new = transform_getter().get('transform', None).params
        print(">>> New transformation matrix from manual point alignment:"); print(tmat_new)
        userinput = input('>>> Do you want to save a new transformation matrix?  y/n ')
        if userinput == "y":
            self.tmat = tmat_new
            self.savedata(['tmat'],[tmat_new]); print(">>> Transformation matrix saved")
        else:
            print(">>> New transformation matrix rejected")

    def apply_tmat(self, image):
        """image = image which should be aligne by the transformation matrix"""
        if 'tmat' in self.__dict__:
            #self.tmat_inv = self.inverse_coordinate_tmat(self.tmat)
            dimensions = image.shape
            dimensions = (dimensions[1],dimensions[0])#inverse order of coordinates
            self.img_aligned = sktransform.warp(image, self.tmat)
            return (self.img_aligned)
        else:
            print(">>> Error: no transformation matrix")



        



    

# ##############################################################################
# imgset1 = Imgset("mytestfile5.h5",1)
# imgset1.autoalign(1000,del_back=True)
# ##############################################################################

# """try to display some results:"""


# fig, ax = plt.subplots(1, 5, figsize=(20, 9))

# im0 = ax[0].imshow(imgset1.unwrapped_phase_stat, cmap='gist_rainbow')
# ax[0].set_title('unwrapped_phase_stat')
# ax[0].axis('off')
# fig.colorbar(im0, ax=ax[0])

# im1 = ax[1].imshow(imgset1.unwrapped_phase, cmap='gist_rainbow')
# ax[1].set_title('unwrapped_phase')
# ax[1].axis('off')
# fig.colorbar(im1, ax=ax[1])

# im2 = ax[2].imshow(imgset1.img_aligned, cmap='gist_rainbow')
# ax[2].set_title('img_aligned')
# ax[2].axis('off')
# fig.colorbar(im2, ax=ax[2])

# im3 = ax[3].imshow((imgset1.unwrapped_phase_stat + imgset1.img_aligned)/2, cmap='gist_rainbow')
# ax[3].set_title('sum')
# ax[3].axis('off')
# fig.colorbar(im3, ax=ax[3])

# im4 = ax[4].imshow((imgset1.unwrapped_phase_stat - imgset1.img_aligned)*2, cmap='gist_rainbow')
# ax[4].set_title('dif')
# ax[4].axis('off')
# fig.colorbar(im4, ax=ax[4])


# fig, ax = plt.subplots(1, 5, figsize=(20, 9))

# im0 = ax[0].imshow(imgset1.img_stat_noback, cmap='gist_rainbow')
# ax[0].set_title('unwrapped_phase_stat')
# ax[0].axis('off')
# fig.colorbar(im0, ax=ax[0])

# im1 = ax[1].imshow(imgset1.img_move_noback, cmap='gist_rainbow')
# ax[1].set_title('unwrapped_phase')
# ax[1].axis('off')
# fig.colorbar(im1, ax=ax[1])

# im2 = ax[2].imshow(imgset1.img_aligned, cmap='gist_rainbow')
# ax[2].set_title('img_aligned')
# ax[2].axis('off')
# fig.colorbar(im2, ax=ax[2])

# im3 = ax[3].imshow((imgset1.unwrapped_phase_stat + imgset1.img_aligned)/2, cmap='gist_rainbow')
# ax[3].set_title('sum')
# ax[3].axis('off')
# fig.colorbar(im3, ax=ax[3])

# im4 = ax[4].imshow((imgset1.unwrapped_phase_stat - imgset1.img_aligned)*2, cmap='gist_rainbow')
# ax[4].set_title('dif')
# ax[4].axis('off')
# fig.colorbar(im4, ax=ax[4])