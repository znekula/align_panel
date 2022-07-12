from setuptools import setup, find_packages

setup(
    name="align_panel",
    version='0.0.1',
    license='MIT',
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.7.5',
    install_requires=[
            "numpy",
            "hyperspy",
            "h5py",
            "scikit-image",
            "panel",
            "pystackreg",
            "aperture @ git+https://github.com/matbryan52/aperture.git",
        ],
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    entry_points={
        'console_scripts': [
            'align_panel=align_panel.cli:main',
        ]},
    description="Package to align images in HDF5 format",
    long_description='''
Package to align images in HDF5 format
''',
    url="https://github.com/matbryan52/align_panel",
    author="Zdeněk Nekula, Matthew Bryan",
    keywords="electron holography, image alignment",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License (MIT)',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Environment :: Web Environment',
        'Environment :: Console',
    ],
)
