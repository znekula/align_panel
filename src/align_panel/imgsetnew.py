# Use hyperspy to holography image processing
# Load raw data (images) and get phase and save all in one h5 file 

import hyperspy.api as hs
import h5py
import numpy as np


class Imgset_new_holography:
    def __init__(self, img_path, ref_path):
        """Creates a new image set consisting of one image of sample and one reference image. 
        Theese two images can be used to calculate amplitude, phase and unwrapped phase images.

        Parameters
        ----------
        img_path : str
            path to the raw data sample image file
        ref_path : str
            path to the raw data reference image file
        """
        self.img_raw = hs.load(img_path, signal_type='hologram')
        self.ref_raw = hs.load(ref_path, signal_type='hologram')
        
        self.img = self.img_raw.data #image of a sample
        self.ref = self.ref_raw.data #reference image
        
        self.img_meta = self.img_raw.metadata #metadata
        self.ref_meta = self.ref_raw.metadata #metadata

        self.img_metaoriginal = self.img_raw.original_metadata #original metadata
        self.ref_metaoriginal = self.ref_raw.original_metadata #original metadata
        
    def phase_reconstruction(self):
        """Makes phase reconstruction of loaded images in the imageset"""
        # Find side band - position and size in FFT of the ref image      
        sb_position = self.ref_raw.estimate_sideband_position(sb='upper')
        sb_size = self.ref_raw.estimate_sideband_size(sb_position)
        
        # Reconstruction"""
        # wave = self.img_raw.reconstruct_phase(self.ref_raw, sb_position=sb_position, sb_size=sb_size,
        #                         output_shape=(int(sb_size.data*2), int(sb_size.data*2)))
        wave = self.img_raw.reconstruct_phase(self.ref_raw, sb_position=sb_position, sb_size=sb_size,
                                output_shape=np.shape(self.img))


        # Reconstructed images"""
        self.real = wave.real.data
        self.imag = wave.imag.data
        self.amplitude = wave.amplitude.data
        self.phase = wave.phase.data
        self.unwrapped_phase = wave.unwrapped_phase().data
    
    # Save data to h5 file:
    def save(self, filename, imgset_name, imgset_ref = False):
        """save data into hdf5 file into supgroup with specified name

        Parameters
        ----------
        filename : str
            path and name of h5 file where you want to save data
        imgset_name : str
            Name of the imageset
        imgset_ref = bool
            True = imageset is supposed to be used as a reference for alignment of the other imagesets;
            False = just an ordinary imageset, supposed to be aligned according to another reference-imageset.
            In an h5 file can be only one reference imgageset.
        """
        # check what is allready inside of the h5 file
        f = h5py.File(filename, 'a')
        try:
            groups = list(f.keys())
            img_shape_identical = True
            img_shape = np.shape(self.img)
            ref_imgsets = 0
            for groupfullname in groups:
                if any(img_shape != f[groupfullname].attrs['img_shape']):
                    img_shape_identical = False
                parts = groupfullname.split('_', 1)
                if parts[0] == 'ref':
                    ref_imgsets +=1
        finally:
            f.close()
        # Check if the data to save are compatible with the file
        if ref_imgsets > 1:
            raise Exception("Error: file allready contains more than 1 reference imagesets. Maximum 1 is allowed. Delete one of them or create a new file.")
        elif ref_imgsets ==1 and imgset_ref is True:
            raise Exception("Error: file allready contains 1 reference imageset. Maximum 1 is allowed. Change your command to save the imageset as an ordinary")
        elif img_shape_identical == False:
            raise Exception("Error: The shape of the image different than images inside of the file. Change the shape of the image.")

        if imgset_ref:
            prefix = 'ref_'
        else:
            prefix = 'ord_'

        f = h5py.File(filename, "a")
        imgset_fullname = prefix + 'imageset_' + imgset_name

        try:
            group = f.create_group(imgset_fullname)
            group.attrs['img_shape']= np.shape(self.img)
            group.create_dataset('img', data = self.img)
            group.create_dataset('ref', data = self.ref)
            group.create_dataset('amplitude', data = self.amplitude)
            group.create_dataset('phase', data = self.phase)
            group.create_dataset('unwrapped_phase', data = self.unwrapped_phase)
            group.create_dataset('img_metadata', data = str(self.img_meta.as_dictionary()))
            group.create_dataset('ref_metadata', data = str(self.ref_meta.as_dictionary()))
            group.create_dataset('img_metadataoriginal', data = str(self.img_metaoriginal.as_dictionary()))
            group.create_dataset('ref_metadataoriginal', data = str(self.ref_metaoriginal.as_dictionary())) 
        finally:
            f.close()

        return imgset_fullname


class Imgset_new_synchrotron:

    def __init__(self,img_path):
        """For images from a synchrotron. 
        Creates a new image set consisting of one image of sample and metadata. 

        Parameters
        ----------
        img_path : str
            path to the raw data sample image file
        """
        self.img_raw =  hs.load(img_path)
        self.img = self.img_raw.data #image
        self.img_meta = self.img_raw.metadata #metadata
        self.img_metaoriginal = self.img_raw.original_metadata #original metadata

    def save(self, filename, imgset_name, imgset_ref = False):
        """Saves data into hdf5 file into supgroup with specified name.

        Parameters
        ----------
        filename : str
            path and name of h5 file where you want to save data
        imgset_name : str
            Name of the imageset
        imgset_ref = bool
            True = imageset is supposed to be used as a reference for alignment of the other imagesets;
            False = just an ordinary imageset, supposed to be aligned according to another reference-imageset.
            In an h5 file can be only one reference imgageset.
        """
        # check what is allready inside of the h5 file
        f = h5py.File(filename, 'a')
        try:
            groups = list(f.keys())
            img_shape_identical = True
            img_shape = np.shape(self.img)
            ref_imgsets = 0
            for groupfullname in groups:
                if any(img_shape != f[groupfullname].attrs['img_shape']):
                    img_shape_identical = False
                parts = groupfullname.split('_', 1)
                if parts[0] == 'ref':
                    ref_imgsets +=1
        finally:
            f.close()
        # Check if the data to save are compatible with the file
        if ref_imgsets > 1:
            raise Exception("Error: file allready contains more than 1 reference imagesets. Maximum 1 is allowed. Delete one of them or create a new file.")
        elif ref_imgsets ==1 and imgset_ref is True:
            raise Exception("Error: file allready contains 1 reference imageset. Maximum 1 is allowed. Change your command to save the imageset as an ordinary")
        elif img_shape_identical == False:
            raise Exception("Error: The shape of the image different than images inside of the file. Change the shape of the image.")

        # choose prefix
        if imgset_ref:
            prefix = 'ref_'
        else:
            prefix = 'ord_'
        # save data
        f = h5py.File(filename, "a")
        try:
            group = f.create_group(prefix + 'imageset_' + imgset_name)
            group.attrs['img_shape']= np.shape(self.img)
            group.create_dataset('img', data = self.img)
            group.create_dataset('img_metadata', data = str(self.img_meta.as_dictionary()))    
            group.create_dataset('img_metadataoriginal', data = str(self.img_metaoriginal.as_dictionary())) 
        finally:
            f.close()
