Align Panel
===========
align_panel is an python library for processing images from TEM-electron-holography and from synchrotron. 

# Installation

## Create a new environment

This is optional. Instalation of the library into a new environment will prevent any interaction with another libraries.

To create a new environmetn by python:
`python -m venv ./path/to/environment` 
(see <https://docs.python.org/3/library/venv.html> for more information).

To crate a new environment by anaconda:
`conda create --name myenv`
(see <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html> for more information).

Then, activate the new environment.

## Installation

To install library, run:

```bash
pip install https://github.com/matbryan52/align_panel/releases/download/0.0.1/align_panel-0.0.1-py3-none-any.whl
```

# Run application # under the development

To run the app:
```bash
align_panel
```

within the environment into which the package was installed.

# Examples

Examples are available at:
<https://github.com/matbryan52/align_panel/tree/master/examples>

Recomended logical sequence of examples depends on your usecase:
## Electron holography
1) example_new_holography --> load raw images, do phase reconstruction, save them into a new hdf5 file
2) example_samesize --> check if all images have the same shape (same number of pixels), if not, it will modify them
3) example_alignment_holography --> align images automatically and manually
4) example_h5file_explore --> show what is inside of the newly created hdf5 file
5) example_imageset_explore_holography --> show results and data

## Synchrotron
1) example_new_synchrotron
2) example_alignment_synchrotron
3) example_h5file_explore
4) example_imageset_explore_synchrotron





