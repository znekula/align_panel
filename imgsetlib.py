"""
Here I load h5 file and do alignment of image sets in respect to the reference one.
"""

import h5py
import numpy as np
from pystackreg import StackReg
from copy import deepcopy
from skimage import transform as sktransform    

class H5file:
    def __init__(self,filename):
        f = h5py.File(filename, 'r')


class Imgset:
    """loads image set (labeled by order number), and 
    reference image set (labeled defaultly by number 0) from h5  file, and 
    makes automatic alignment by StackReg. 
    Then saves transformation matrix and transformed unwrapped phase 
    to the same h5 file"""
    def __init__(self, filename, imgset_name):
        """Creates an object of an imageset
        Parameters
        ----------
        filename : str
            name of the h5 file
        imgset_name : str
            name of the imageset
        """
        # read data
        self.filename = filename
        self.imgset_name = imgset_name 
        f = h5py.File(filename, 'r')

        # find refernce imageset
        for groupname in list(f.keys()):
            if groupname[0:13] == 'ref_imageset_':
                imgsetref_fullname = groupname
                imgsetref_name = groupname[13:]
                if imgsetref_name == imgset_name:
                    self.imgset_fullname = imgsetref_fullname
                else:
                    self.imgset_fullname = 'imageset_' + imgset_name
                print(">>> "+ groupname)
                break

            else:
                print(">>> "+ groupname)
                print(groupname[0:13] )
        

        dset_unwrapped_phase_stat = f[imgsetref_fullname +'/unwrapped_phase']
        self.unwrapped_phase_stat = np.asarray(dset_unwrapped_phase_stat)
        dset_unwrapped_phase = f[self.imgset_fullname +'/unwrapped_phase']
        self.unwrapped_phase = np.asarray(dset_unwrapped_phase)
        
        dset_amplitude_stat = f[imgsetref_fullname +'/amplitude']
        self.amplitude_stat = np.asarray(dset_amplitude_stat)
        dset_amplitude = f[self.imgset_fullname +'/amplitude']
        self.amplitude = np.asarray(dset_amplitude)

        dset_img_stat = f[imgsetref_fullname +'/img']
        self.img_stat = np.asarray(dset_img_stat)
        dset_img = f[self.imgset_fullname +'/img']
        self.img = np.asarray(dset_img)

        dset_ref_stat = f[imgsetref_fullname +'/ref']
        self.ref_stat = np.asarray(dset_ref_stat)
        dset_ref = f[self.imgset_fullname +'/ref']
        self.ref = np.asarray(dset_ref)

        dset_phase_stat = f[imgsetref_fullname +'/phase']
        self.phase_stat = np.asarray(dset_phase_stat)
        dset_phase = f[self.imgset_fullname +'/phase']
        self.phase = np.asarray(dset_phase)

        dset_img_metadata_stat = f[imgsetref_fullname +'/img_metadata']
        self.img_metadata_stat = np.asarray(dset_img_metadata_stat)
        dset_img_metadata = f[self.imgset_fullname +'/img_metadata']
        self.img_metadata = np.asarray(dset_img_metadata)

        dset_ref_metadata_stat = f[imgsetref_fullname +'/ref_metadata']
        self.ref_metadata_stat = np.asarray(dset_ref_metadata_stat)
        dset_ref_metadata = f[self.imgset_fullname +'/ref_metadata']
        self.ref_metadata = np.asarray(dset_ref_metadata)

        group = f['/imageset_'+ self.imgset_name]
        if "tmat" in group.keys():
            dset_tmat = f[self.imgset_fullname +'/tmat']
            self.tmat = np.asarray(dset_tmat)

        f.close()
        
    # define help functions:
    def make_same_size(self,img_refsize, img_changesize):
        print(">>> start make same size")
        refsize0 = img_refsize[0].size; 
        refsize1 = img_refsize[:,0].size  
        changesize0 = img_changesize[0].size; 
        changesize1 = img_changesize[:,0].size
        dif0 = changesize0 - refsize0
        print(">>> dif 0 = "+str(dif0))
        dif1 = changesize1 - refsize1
        print(">>> dif 1 = "+str(dif0))
        img_changed = deepcopy(img_changesize)
        if dif0 < 0 :
            print(">>> smaller")
            img_changed = np.append(img_changed,np.zeros((-dif0,changesize1)),axis = 0)
        elif dif0 > 0 :
            print(">>> bigger")
            img_changed = img_changed[:-dif0,:]    
        if dif1 < 0 :
            img_changed = np.append(img_changed,np.zeros((changesize0 - dif0,-dif1)),axis = 1)
        elif dif1 > 0 :
            img_changed = img_changed[:,:-dif1]
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


            # """makes autoalignment of unwrapped phases images, aligning just objest without background
        # roughnes = roughness of estimation the border betwen object and background. 
        # Low roughness estimation can be disrupted by noise.
        # """

    def autoalign(self, roughness=500, del_back=True, keeporiginalsize = False, **images):
        """Makes autoalignment of selected images, aligning just object without background
        roughnes = roughness of estimation the border betwen object and background. 
        Low roughness estimation can be disrupted by noise.

        Parameters
        ----------
        roughness : int
            roughness of estimation the border betwen object and background. 
            Low roughness estimation can be disrupted by noise.
        del_back : Boolean
            True = delete background and proceed autoalignment only with boleen images (blakc and white);
            this is only for autoalignment, images are saved with original walues;
            False = proceed alignment on original images
        keeporiginalsize : Boolean
            True = keeps oringinal size of images;
            False = crop the images that tey will have the same number of pixels as an reference imageset 
            and saves them in this new shape
        **images : ['img_stat', 'img_move', 'transformation']
            img_stat = reference image;
            img_move = moving image;
            transformation = ['RIGID_BODY', 'TRANSLATION', 'SCALED_ROTATION', 'AFFINE', 'BILINEAR']
        """
        try:
            img_stat = images['img_stat']
            img_move = images['img_move']
            print(">>> autoalign: using user kind of images")
        except:
            img_stat = self.amplitude_stat
            img_move = self.amplitude
            print(">>> autoalign: using default kind of images")

        img_move = self.make_same_size(self.amplitude_stat, img_move )
        
        try:
            transformation = images['transformation']
        except:
            transformation = 'RIGID_BODY'

        if transformation == 'TRANSLATION':
            sr = StackReg(StackReg.TRANSLATION)
        elif transformation == 'SCALED_ROTATION':
            sr = StackReg(StackReg.SCALED_ROTATION)
        elif transformation == 'AFFINE':
            sr = StackReg(StackReg.AFFINE)
        elif transformation == 'BILINEAR':
            sr = StackReg(StackReg.BILINEAR)
        else:
            sr = StackReg(StackReg.RIGID_BODY)
        


        if del_back==True:
            img_stat_noback = self.delete_background(img_stat, roughness)
            self.img_stat_noback=img_stat_noback
            img_move_noback = self.delete_background(img_move, roughness)
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
        
        
        # Save data to h5 file:

        self.savedata(["tmat"],[self.tmat])
        print(">>> saving")
        if not keeporiginalsize:
            datasets = [self.amplitude, self.phase, self.unwrapped_phase]
            count=0
            for dataset in datasets:
                datasets[count] = self.make_same_size(self.amplitude_stat, dataset)
                count+=1
            self.savedata(['amplitude','phase','unwrapped_phase'], datasets)
            # rewrite images in memory
            self.amplitude = datasets[0]
            self.phase = datasets[1]
            self.unwrapped_phase = datasets[2]
    
    def savedata(self, datasets_names, datasets_data):
        """Save data to the h5 file. dataset_names = list of datasets, for example ["tmat", "unwrapped_phase_aligned"]
        datasets_data = list of data, for example [self.tmat, "self.unwrapped_phase_aligned"]"""
        f = h5py.File(self.filename, "a")  
        #datasets_names = ["tmat"]
        #datasets_data = [self.tmat]

        group = f['/imageset_'+ self.imgset_name]
        count=0
        for dataset_name in datasets_names:
            if dataset_name in group.keys():                
                del group[dataset_name]
            f.create_dataset('imageset_' + self.imgset_name +'/'+dataset_name, data = datasets_data[count])
            count +=1       
        f.close()

    def manual_fine(self, img_stat, img_move, initial_transform_matrix=np.identity(3)):
        from align_panel import fine_adjust
        #make enclosed variable containing the transformation matrix
        initial_transform = sktransform.AffineTransform(matrix=initial_transform_matrix) 
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