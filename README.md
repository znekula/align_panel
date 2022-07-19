Align Panel
===========
align_panel is an python library for processing images from TEM-electron-holography and from synchrotron. 

# 1 Installation
Tis manual suppose that you are using the anaconda python distribution. To begin the installation, open the **Anaconda Powershell Prompt** command line.

## 1.1 Create a new environment

This is optional. Instalation of the library into a new environment will prevent any interaction with another libraries.

To crate a new environment by anaconda:
`conda create --name myenv`
(see <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html> for more information).

Then, activate the new environment.

## 1.2 Installation
First, install pystackreg library:
```bash
conda install -c conda-forge pystackreg
```

Second, install the align_panel library:
```bash
pip install https://github.com/znekula/align_panel/releases/download/0.0.1/align_panel-0.0.1-py3-none-any.whl
```

# 2 Run application # under the development

To run the app:
```bash
align_panel
```

within the environment into which the package was installed.

# 3 Examples

Examples are available at:
<https://github.com/znekula/align_panel/tree/master/examples>

Recomended logical sequence of examples depends on your usecase:
## Electron holography
1) example_new_holography  --> load raw images, do phase reconstruction, save them into a new hdf5 file
2) example_samesize  --> check if all images have the same shape (same number of pixels), if not, it will modify them
3) example_alignment_holography  --> align images automatically and manually
4) example_h5file_explore  --> show what is inside of the newly created hdf5 file
5) example_imageset_explore_holography  --> show results and data

## Synchrotron
1) example_new_synchrotron  --> load raw images and save them into a new hdf5 file
2) example_alignment_synchrotron  --> align images automatically and manually
3) example_h5file_explore  --> show what is inside of the newly created hdf5 file
4) example_imageset_explore_synchrotron  --> show results and data


# 4 h5 file
All data are stored in hdf5 file (shortly h5 file). Those files have inner hierarchy. The whole experiment is stored in one h5 file, consisting of several imagesets. The following paragraphs shows what is inside.



## 4.1 Electron holography
In electron holography, one allways makes imageges which are paired with their reference images. Each images and its ref. image are used for creating one imageset. Each h5 file contains as many imagesets as many images one made in the experiment. 


(f) datafile.h5  
|—— (g) ref_imageset_name  
|     |—— (d) img  
|     |—— (d) ref  
|     |—— (d) img_metadata  
|     |—— (d) img_metadataoriginal  
|     |—— (d) ref_metadata  
|     |—— (d) ref_metadataoriginal  
|     |—— (d) amplitude  
|     |—— (d) phase  
|     |—— (d) unwrapped_phase  
|  
|—— (g) ord_imageset_name1  
|     |—— (d) img  
|     |—— (d) ref  
|     |—— (d) img_metadata  
|     |—— (d) img_metadataoriginal  
|     |—— (d) ref_metadata  
|     |—— (d) ref_metadataoriginal  
|     |—— (d) amplitude  
|     |—— (d) phase  
|     |—— (d) unwrapped_phase  
|     |—— (d) tmat  
|  
|—— (g) ord_imageset_name2  
|     |—— (d) img  
|     |—— (d) ref  
|     |—— (d) img_metadata  
|     |—— (d) img_metadataoriginal  
|     |—— (d) ref_metadata  
|     |—— (d) ref_metadataoriginal  
|     |—— (d) amplitude  
|     |—— (d) phase  
|     |—— (d) unwrapped_phase  
|     |—— (d) tmat   
  
## 4.2 Synchrotron
All images from one experiment are stored in one h5 file. Each image is in its own imageset which contain: image, original metadata, metadata and transformation matrix.

 (f) datafile.h5  
 |—— (g) ref_imageset_name  
 |     |—— (d) img  
 |     |—— (d) img_metadata  
 |     |—— (d) img_metadataoriginal  
 |  
 |—— (g) ord_imageset_name1  
 |     |—— (d) img  
 |     |—— (d) img_metadata  
 |     |—— (d) img_metadataoriginal  
 |     |—— (d) tmat  
 |  
 |—— (g) ord_imageset_name2  
 |     |—— (d) img  
 |     |—— (d) img_metadata  
 |     |—— (d) img_metadataoriginal  
 |     |—— (d) tmat    
 

