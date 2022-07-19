import argparse
import pathlib
from aperture.cli import launch_external
import logging

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


def main():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='align_panel')
    subparsers = parser.add_subparsers(title='subcommands',
                                       description='valid subcommands',
                                       required=True)
    
    # create the parser for the launch command
    parser_launch = subparsers.add_parser('launch', help='Launch the alignment workflow')
    parser_launch.set_defaults(func=launch_align_workflow)
    parser_launch.add_argument('hdf5_path', type=str, help='The HDF5 path to launch with')
    
    # create the parser for the load command
    parser_load = subparsers.add_parser('load',
                                        help='Load files into an HDF5 container')
    parser_load.set_defaults(func=load_or_create_hdf5)
    
    parser_load.add_argument('hdf5_path',
                             type=str,
                             help='The HDF5 path to load data into')
    parser_load.add_argument('file_path',
                             type=str,
                             help='The data to load into the hdf5')
    parser_load.add_argument('imgset_name',
                             type=str,
                             help='The name to give to this imageset in the HDF5')
    parser_load.add_argument('-r',
                             '--ref',
                             type=str,
                             help='File path to use as vacuum reference')
    parser_load.add_argument('-s',
                             '--static',
                             help='set these data as the static image in the HDF5',
                             action='store_true')

    # create the parser for the inspect command
    parser_inspect = subparsers.add_parser('inspect', help='List the contents of an HDF5 file')
    parser_inspect.set_defaults(func=inspect_hdf5)
    parser_inspect.add_argument('hdf5_path', type=str, help='The HDF5 path to inspect')                             

    args = parser.parse_args()
    args.func(args)


def launch_align_workflow(args):
    from .imgsetlib import H5file
    hdf5_path = as_path(args.hdf5_path, exists=True)
    check_is_hdf5(hdf5_path)
    file_obj = H5file(hdf5_path)
    assert file_obj.ref_imageset_name, 'Must have at least one static imgset in hdf5 file'
    assert file_obj.imageset_names, 'Must have at least one moving imgset in hdf5 file'

    from .align_workflow import build_workflow
    launch_external(build_workflow, build_kwargs=dict(hdf5_path=hdf5_path))


def as_path(str_path: str, exists=False) -> pathlib.Path:
    try:
        path = pathlib.Path(str_path).resolve()
    except TypeError:
        raise TypeError(f'Unable to interpret {str_path} as a file')
    if path.is_dir():
        raise TypeError(f'File {path} is a directory!')
    if exists and not path.is_file():
        raise TypeError(f'File {path} does not exist')
    return path


def check_is_hdf5(path: pathlib.Path):
    hdf5_suffixes = ('.h5', '.hdf5')
    if not path.suffix in hdf5_suffixes:
        raise TypeError(f'Must use suffix in {hdf5_suffixes} for hdf5 files')
    return True


def load_or_create_hdf5(args):
    from .imgsetnew import Imgset_new_holography, Imgset_new_synchrotron

    hdf5_path = as_path(args.hdf5_path)
    check_is_hdf5(hdf5_path)
    if hdf5_path.is_file():
        logger.info(f'Loading data into existing HDF5 file @ {hdf5_path}')
    else:
        logger.info(f'Will create new HDF5 file @ {hdf5_path}')
    file_path = as_path(args.file_path, exists=True)
    ref_path = args.ref
    if ref_path is not None:
        ref_path = as_path(args.ref, exists=True)
    is_static = args.static
    imgset_name = args.imgset_name

    if '/' in imgset_name:
        raise ValueError('Cannot use imgset names with / characters '
                         'as this breaks the HDF5 tree structure')

    if ref_path is not None:
        logger.info(f'Interpreting data as hologram with reference')
        imgset = Imgset_new_holography(file_path, ref_path)
        logger.info(f'Performing phase reconstruction')
        imgset.phase_reconstruction()
    else:
        logger.info(f'Interpreting data as synchrotron format (no reference)')
        imgset = Imgset_new_synchrotron(file_path)

    imgset_fullname = imgset.save(hdf5_path, imgset_name, imgset_ref=is_static)
    logger.info(f'Saving data into {imgset_fullname} in HDF5 file {hdf5_path}')


def inspect_hdf5(args):
    from align_panel.imgsetlib import H5file
    hdf5_path = as_path(args.hdf5_path, exists=True)
    check_is_hdf5(hdf5_path)
    mydatafile = H5file(hdf5_path)

    # Print all imagesets inside the h5 file
    print ("\nthis is the content:")
    print ("reference imageset name: " + str(mydatafile.ref_imageset_name))
    print ("imagesets names: " + str(mydatafile.imageset_names))
    print ("the rest: " + str(mydatafile.rest))

    print ("\nHere are the full names of imagesets:")
    print ("reference imageset full name: " + str(mydatafile.ref_imageset_fullname))
    print ("imagesets full names: " + str(mydatafile.imageset_fullnames) + '\n')
