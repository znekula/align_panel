# Use hyperspy to holography image processing
# Load raw data (images) and get phase and save all in one h5 file 

import hyperspy.api as hs
import h5py


class Imgset_new:
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
        
    def phase_reconstruction(self):
        """Makes phase reconstruction of loaded images in the imageset"""
        # Find side band - position and size in FFT of the ref image      
        self.sb_position = self.ref_raw.estimate_sideband_position(sb='upper')
        self.sb_size = self.ref_raw.estimate_sideband_size(self.sb_position)
        
        # Reconstruction"""
        wave = self.img_raw.reconstruct_phase(self.ref_raw, sb_position=self.sb_position, sb_size=self.sb_size,
                                output_shape=(int(self.sb_size.data*2), int(self.sb_size.data*2)))
        # Reconstructed images"""
        self.real = wave.real.data
        self.imag = wave.imag.data
        self.amplitude = wave.amplitude.data
        self.phase = wave.phase.data
        self.unwrapped_phase = wave.unwrapped_phase().data
    
    # Save data to h5 file:
    def save(self, filename, imgset_name):
        """save data into hdf5 file into supgroup of specified order
        if order = 0 then it is the reference imageset (later for alignment with other imagesets)
        for nonreference imagesets use as order any integer

        Parameters
        ----------
        filename : str
            path and name of h5 file where you want to save data
        imgset_name : str
            Name of the imageset
        """

        f = h5py.File(filename, "a")


        
        f.create_dataset('imageset_' + imgset_name+'/img', data = self.img)
        f.create_dataset('imageset_' + imgset_name+'/ref', data = self.ref)
        f.create_dataset('imageset_' + imgset_name+'/amplitude', data = self.amplitude)
        f.create_dataset('imageset_' + imgset_name+'/phase', data = self.phase)
        f.create_dataset('imageset_' + imgset_name+'/unwrapped_phase', data = self.unwrapped_phase)
        f.create_dataset('imageset_' + imgset_name+'/img_metadata', data = str(self.img_meta))
        f.create_dataset('imageset_' + imgset_name+'/ref_metadata', data = str(self.ref_meta))
        
        f.close()