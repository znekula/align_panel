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

    def get_2d_image_keys(self, alignable=True):
        not_alignable = ['ref', 'img']
        keys = []
        f=h5py.File(self.filename, 'r')
        group = f[self.imgset_fullname]
        for key, obj in group.items():
            try:
                if obj.ndim == 2:
                    keys.append(key)
            except AttributeError:
                pass
        f.close()
        if alignable:
            keys = [k for k in keys if k not in not_alignable]
        return keys

    def get_data(self, dataname:str, stat=False, aligned=False):
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
        if aligned:
            assert not stat
            assert 'metadata' not in dataname
            assert 'tmat' not in dataname

        f=h5py.File(self.filename, 'r')
        try:
            # choose group as an imageset or a stat. imageset
            if stat:
                group = f[self.imgsetref_fullname]
            else:
                group = f[self.imgset_fullname]
            # load data (loading metadata is different than loading images)
            if dataname in ['img_metadata', 'ref_metadata','img_metadataoriginal',  'ref_metadataoriginal']:
                dset = group[dataname]
                data = str(np.asarray(dset))[2:-1] #cutout added sybolss
            else:
                dset = group[dataname]
                data = np.asarray(dset)
        finally:
            f.close()
        if aligned:
            return self.apply_tmat(data)
        return data
     
    # define help functions:
    @staticmethod
    def make_same_size(img_refsize, img_changesize):
        refsize0 = img_refsize[0].size; 
        refsize1 = img_refsize[:,0].size  
        changesize0 = img_changesize[0].size; 
        changesize1 = img_changesize[:,0].size
        dif0 = changesize0 - refsize0
        dif1 = changesize1 - refsize1
        img_changed = deepcopy(img_changesize)
        if dif0 < 0 :
            img_changed = np.append(img_changed,np.zeros((-dif0,changesize1)),axis = 0)
        elif dif0 > 0 :
            img_changed = img_changed[:-dif0,:]    
        if dif1 < 0 :
            img_changed = np.append(img_changed,np.zeros((changesize0 - dif0,-dif1)),axis = 1)
        elif dif1 > 0 :
            img_changed = img_changed[:,:-dif1]
        return img_changed
    
    @staticmethod
    def delete_background(img_original, bins:int = 62, value_high = 1, value_low = 0, deadpix = 0.0005 ):
        """ 1) sort all pixels of image in 1D array  
        2) crop the extremes from both sides (hot and dead pixels)  
        3) split the array into bins, in each is an equal number of pixels
        4) sum pixel values in each bin and crate histogram
        5) derivate the histogram
        6) set thrashold where the derivation has extreme
        7) to all pixels which are under the trashold is given value = "value_low"
        8) to all pixels which are in or above the trashold is given value = "value_high"
        9) this boolean image is then returned
        
        Useful for images where object has a strong contrast compare with background.
        
        Parameters
        ==========
        img_original : 2DArray
            image where the background should be deleted
        bins : int
            number of bins in histogram
        value_high : float
            all bright pixels will be set to this value
        value_low : float
            all dark pixels will be set to this value
        deadpix : float
            percentage of dead pixels in the image
        """  
        # flat and sort 2D array into 1D array  
        img = deepcopy(img_original)
        img_1D = img.flatten()
        img_1D_sort = np.sort(img_1D)
        # remove hot and dead pixels ... 2/1000 of the pixels
        pxs_del = int(np.size(img_1D_sort) *deadpix) 
        img_1D_sort = np.delete(img_1D_sort, np.s_[:pxs_del]) #delete dead pixels
        img_1D_sort = np.delete(img_1D_sort, np.s_[-pxs_del:]) #delete hot pixels
        # binpixels is number of pixels in one bin
        binpixels = int(np.size(img_1D_sort) / bins)
        img_1D_sort_reduced = np.zeros(int(img_1D_sort.size/binpixels))
        for a in range(bins):
            img_1D_sort_reduced[a] = np.average(img_1D_sort[a*binpixels:a*binpixels+binpixels])
        # do derivation
        derivative = np.convolve(img_1D_sort_reduced,[1,-1],'same')
        derivative[0]=0; derivative[-1]=0
        # set trashold
        trashold = img_1D_sort_reduced[np.argmax(derivative)]
        for i in range(img[:,0].size):
            for j in range(img[0,:].size):
                if img[i,j] <  trashold:
                    img[i,j] = value_low
                else:
                    img[i,j] = value_high            
        return img # returning the boolean image

    @staticmethod
    def autoalign_methods():
        return ['RIGID_BODY', 'TRANSLATION', 'SCALED_ROTATION', 'AFFINE', 'BILINEAR']

    def autoalign(self, img_stat, img_move, transformation='RIGID_BODY', bins=62, del_back=True):
        """Makes autoalignment of selected images, aligning just object without background
        roughnes = roughness of estimation the border betwen object and background. 
        Low roughness estimation can be disrupted by noise.

        Parameters
        ----------
        img_stat : np2DArray
            reference image
        img_move : np2DArray
            moving image
        bins : int
            number of bins while making image histogram to find border between image and background
            High number of bins can lead to high sensitivity to noise. Recomended number is 62.
        del_back : Boolean
            True = delete background and proceed autoalignment only with boleen images (blakc and white);
            this is only for autoalignment, images are saved with original walues;
            False = proceed alignment on original images
        transformation : str
            transformation in ['RIGID_BODY', 'TRANSLATION', 'SCALED_ROTATION', 'AFFINE', 'BILINEAR']

        Return
        ------
        tmat: 2DArray
        """
        if transformation == 'RIGID_BODY':
            sr = StackReg(StackReg.RIGID_BODY)
        elif transformation == 'TRANSLATION':
            sr = StackReg(StackReg.TRANSLATION)
        elif transformation == 'SCALED_ROTATION':
            sr = StackReg(StackReg.SCALED_ROTATION)
        elif transformation == 'AFFINE':
            sr = StackReg(StackReg.AFFINE)
        elif transformation == 'BILINEAR':
            sr = StackReg(StackReg.BILINEAR)
        
        if del_back is True:
            img_stat_noback = self.delete_background(img_stat, bins)
            img_move_noback = self.delete_background(img_move, bins)
            reg = sr.register_transform(img_stat_noback, img_move_noback)
        else:            
            reg = sr.register_transform(img_stat, img_move)
        reg = reg.clip(min=0)

        tmat = sr.get_matrix()
        return tmat

    def get_tmat(self) -> np.ndarray:
        try:
            return self.get_data('tmat')
        except KeyError:
            return np.eye(3)

    def clear_tmat(self):
        self.save_tmat(np.eye(3))
    
    def savedata(self, datasets_names:list, datasets_data:list):
        """Save data to the h5 file. dataset_names = list of datasets, for example ["tmat", "unwrapped_phase"]
        datasets_data = list of data, for example [tmat, unwrapped_phase]"""
        f = h5py.File(self.filename, "a")  
        #datasets_names = ["tmat"]
        #datasets_data = [tmat]

        try:
            group = f[self.imgset_fullname]
            count=0
            for dataset_name in datasets_names:
                if dataset_name in group.keys():                
                    del group[dataset_name]
                f.create_dataset(self.imgset_fullname +'/'+dataset_name, data = datasets_data[count])
                count +=1     
        finally:  
            f.close()

    def save_tmat(self, tmat):
        self.savedata(['tmat'], [tmat])

    def manual_fine(self, img_stat, img_move, initial_transform_matrix=np.identity(3)):
        """Do manual fine alignment. Runs a server with a GUI for fine alignment. 
        To kill server, pres ctrl+c into terminal. 

        Parameters
        ----------
        img_stat : 2DArray
            static image
        img_move : 2DArray
            moving image
        initial_transform_matrix : 2DArray, optional
            initial transformation, by default np.identity(3)

        Returns
        -------
        tmat_new : 2DArray
            new transformation matrix
        """
        from align_panel.align_panel import fine_adjust
        #make enclosed variable containing the transformation matrix
        initial_transform = sktransform.AffineTransform(matrix=initial_transform_matrix) 
        layout, fine_getter = fine_adjust(img_stat, img_move, initial_transform = initial_transform)
        print(">>> to kill server pres ctrl+c into terminal")
        layout.show(threaded=False)
        tmat_new = fine_getter()['transform'].params
        return tmat_new

    def manual_point(self, img_stat, img_move):
        """Do manual point alignment. Runs a server with a GUI for fine alignment. 
        To kill server, pres ctrl+c into terminal. 

        Parameters
        ----------
        img_stat : 2DArray
            static image
        img_move : 2DArray
            moving image
        initial_transform_matrix : 2DArray, optional
            initial transformation, by default np.identity(3)

        Returns
        -------
        tmat_new : 2DArray
            new transformation matrix
        """
        from align_panel.align_panel import point_registration
        layout, transform_getter = point_registration(img_stat, img_move)
        print(">>> to kill server pres ctrl+c into terminal")
        layout.show()
        tmat_new = transform_getter().get('transform', None).params
        return tmat_new

    def apply_tmat(self, image, tmat=None):
        """ Applies transformation matrix to the image.

        Parameters
        ----------
        image : np.ndarray or str
            input directly any image as np.ndarray, or name of the image which is in the same imageset as str.
        tmat : 2DArray or None
            input any 3x3 transformation matrix. If None, the saved tmat of the imageset from the file will be used.

        Returns
        -------
        img_aligned : 2DArray
            aligned image
        """
        if tmat is None:
            try:
                tmat = self.get_data('tmat')
            except KeyError:
                return image

        if isinstance(image, np.ndarray):
            img = image
        else:
            # image == ("unwrapped_phase")
            img = self.get_data(image)

        img_aligned = sktransform.warp(img, tmat)
        return img_aligned
