"""
Use hyperspy to holography image processing
Load raw data (images) and get phase and save all in one h5 file 
"""
import hyperspy.api as hs
import h5py


class Imgset_new:
    """Complete image set consisting of one image of sample and reference image. 
    Those two are used to calculate amplitude, phase and unwrapped phase images"""
    def __init__(self, img_path, ref_path):
        self.img_raw = hs.load(img_path, signal_type='hologram')
        self.ref_raw = hs.load(ref_path, signal_type='hologram')
        
        self.img = self.img_raw.data #image of a sample
        self.ref = self.ref_raw.data #reference image
        
        self.img_meta = self.img_raw.metadata #metadata
        self.ref_meta = self.ref_raw.metadata #metadata
        
    def phase_reconstruction(self):
        """Find side band - position and size in FFT of the ref image"""
        self.sb_position = self.ref_raw.estimate_sideband_position(sb='upper')
        self.sb_size = self.ref_raw.estimate_sideband_size(self.sb_position)
        
        """Reconstruction"""
        wave = self.img_raw.reconstruct_phase(self.ref_raw, sb_position=self.sb_position, sb_size=self.sb_size,
                                output_shape=(int(self.sb_size.data*2), int(self.sb_size.data*2)))
        """Reconstructed images"""
        self.real = wave.real.data
        self.imag = wave.imag.data
        self.amplitude = wave.amplitude.data
        self.phase = wave.phase.data
        self.unwrapped_phase = wave.unwrapped_phase().data
    
    """Save data to h5 file"""
    def save(self, filename, order):
        """save data into hdf5 file into supgroup of specified order
        if order = 0 then it is the reference imageset (later for alignment with other imagesets)
        for nonreference imagesets use as order any integer"""
        if order == 0:
            f = h5py.File(filename, "w")
        else:
            f = h5py.File(filename, "a")


            
        f.create_dataset('imageset' + str(order)+'/img', data = self.img)
        f.create_dataset('imageset' + str(order)+'/ref', data = self.ref)
        f.create_dataset('imageset' + str(order)+'/amplitude', data = self.amplitude)
        f.create_dataset('imageset' + str(order)+'/phase', data = self.phase)
        f.create_dataset('imageset' + str(order)+'/unwrapped_phase', data = self.unwrapped_phase)
        f.create_dataset('imageset' + str(order)+'/img_metadata', data = str(self.img_meta))
        f.create_dataset('imageset' + str(order)+'/ref_metadata', data = str(self.ref_meta))
        
        f.close()
            
        
##############################################################################
# """define data paths"""
# img0_path = '-4-H.dm3'
# ref0_path = '-4-R2.dm3'

# img1_path = '-2-H.dm3'
# ref1_path = '-2-R2.dm3'

# """create new imagesets"""
# imgset_new0 = Imgset_new(img0_path,ref0_path); print("<<< imgset_new0 done")
# imgset_new1 = Imgset_new(img1_path,ref1_path); print("<<< imgset_new1 done")

# """do phase reconstruction """
# imgset_new0.phase_reconstruction()
# imgset_new1.phase_reconstruction()

# """save data"""
# filename = "mytestfile5.h5"
# imgset_new0.save(filename,0)
# imgset_new1.save(filename,1)
##############################################################################
