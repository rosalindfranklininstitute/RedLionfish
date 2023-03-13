"""
Copyright 2021 Rosalind Franklin Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    version = '0.9',
    name = 'RedLionfish',
    description = 'Fast Richardson-Lucy deconvolution of 3D volume data using GPU or CPU with napari plugin.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = 'https://github.com/rosalindfranklininstitute/RedLionfish',
    author = 'Luis Perdigao',
    author_email='luis.perdigao@rfi.ac.uk',
    packages=['RedLionfishDeconv'],
    classifiers=[
        'Development Status :: 4 - Beta ',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Framework :: napari'
    ],
    license='Apache License, Version 2.0',
    zip_safe=False,
    install_requires=[
        
    ] ,

    #for the napari plugin
    entry_points={'napari.plugin': 'RedLionfish = RedLionfishDeconv.napari_plugin'}
)
