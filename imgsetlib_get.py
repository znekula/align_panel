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
        """This class is used for browsing the content of h5 file and 
        listing imagesets inside

        Parameters
        ----------
        filename : str
            data path to the h5 file
        """
        f = h5py.File(filename, 'r')
        groups = list(f.keys())
        f.close()
        self.ref_imageset_name=[]
        self.ref_imageset_fullname=[]
        self.imageset_names=[]
        self.imageset_fullnames = []
        self.rest=[]
        
        for group in groups:
            parts = group.split('_', 2)
            if 'ref'== parts[0] and 'imageset'==parts[1]:
                self.ref_imageset_name.append(parts[2])
                self.ref_imageset_fullname.append(group)
            elif 'ord'==parts[0] and 'imageset'==parts[1] :
                self.imageset_names.append(parts[2])
                self.imageset_fullnames.append(group)
            else:
                self.rest.append(group)





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
                self.imgsetref_fullname = groupname
                self.imgsetref_name = groupname[13:]
                if self.imgsetref_name == imgset_name:
                    self.imgset_fullname = self.imgsetref_fullname
                else:
                    self.imgset_fullname = 'ord_imageset_' + imgset_name
                break
        f.close()
        


    def get_content(self):
        f=h5py.File(self.filename, 'r')
        group = f[self.imgset_fullname]
        content = list(group.keys())
        f.close()
        return content

    def get_data(self, dataname:str, stat=False):
        """Get data directly from the h5 file of the imageset.

        Parameters
        ----------
        dataname : str
            Name of wanted data. One of theese 
            [img, ref, img_metadata, ref_metadata, amplitude, phase, unwrapped_phase, tmat]
        stat : bool, optional
            True=data from the reference static imageset, 
            False=data from the imageset itself, by default False
        """
        f=h5py.File(self.filename, 'r')
        try:
            # choose group as an imageset or a stat. imageset
            if stat:
                group = f[self.imgsetref_fullname]
            else:
                group = f[self.imgset_fullname]
            # load data
            if dataname in ['img_metadata', 'ref_metadata']:
                dset = group[dataname]
                data = str(np.asarray(dset))[2:-1]
            else:
                dset = group[dataname]
                data = np.asarray(dset)
        finally:
            f.close()
        return data
     
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

    def autoalign(self, roughness=500, del_back=True, keeporiginalsize = False, **kwargs):
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
        **kwargs : ['img_stat', 'img_move', 'transformation']
            img_stat = reference image;
            img_move = moving image;
            transformation = ['RIGID_BODY', 'TRANSLATION', 'SCALED_ROTATION', 'AFFINE', 'BILINEAR']
        """
        try:
            img_stat = kwargs['img_stat']
            img_move = kwargs['img_move']
            print(">>> autoalign: using user kind of images")
        except:
            img_stat = self.amplitude_stat
            img_move = self.amplitude
            print(">>> autoalign: using default kind of images")

        img_move = self.make_same_size(img_stat, img_move )
        
        try:
            transformation = kwargs['transformation']
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
        if not keeporiginalsize and self.imageset_kind == "holography":
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

        group = f['/'+ self.imgset_fullname]
        count=0
        for dataset_name in datasets_names:
            if dataset_name in group.keys():                
                del group[dataset_name]
            f.create_dataset(self.imgset_fullname +'/'+dataset_name, data = datasets_data[count])
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

    def apply_tmat(self, image, tmat=None):
        """image = image which should be aligne by the transformation matrix"""
        if tmat is None:
            tmat = self.tmat

        if isinstance(image, np.ndarray):
            pass
        else:
            # image == (imgset_7", "phase")
            imgset, img = image
            image = self.get_image(imgset, img)

        img_aligned = sktransform.warp(image, tmat)
        return img_aligned