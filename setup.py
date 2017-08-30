
import os
from os.path import join as pjoin
import glob
from distutils.core import setup, Extension
import numpy.distutils.misc_util
import subprocess
from subprocess import Popen

prefix = '/home/pafluxa/anaconda2/envs/atsec'

extensions = []
# Correlation extension
# *****************************************************************************
# Compile cuda library
corr_source_files = glob.glob( './extensions/correlation/src/*.c' )

corr_lib_dirs = []
corr_lib_dirs.append( os.path.join( prefix, 'lib' ) )
corr_lib_dirs.append( os.path.join( prefix, 'lib64' ) )
corr_lib_dirs.append( '/usr/local/cuda/lib64' )

corr_inc_dirs = []
corr_inc_dirs.append( os.path.join( prefix, 'include' ) )
corr_inc_dirs.append( numpy.distutils.misc_util.get_numpy_include_dirs()[0] )
corr_inc_dirs.append( './extensions/correlation/include' )

_correlation =  Extension( 
        "atsec.correlation._correlation",
        corr_source_files ,
        language = "c",
        libraries=['gomp', 'cudacorr', 'cuda', 'cufft' ],
        library_dirs = corr_lib_dirs ,
        include_dirs = corr_inc_dirs , 
        extra_compile_args=["-fopenmp", '-std=c99', '-lm', '-fPIC', '-Wall', '-O3'] , )

# *****************************************************************************
extensions.append( _correlation )

# Disutils setup specs
# *****************************************************************************
setup(
    name = 'atsec',
    package_dir = \
        { 'atsec'             : 'pythonsrc'   , },
    packages = ['atsec',
                # Core modules
                #=============================================
                'atsec.correlation',
               ],
    author  = 'Pedro A. Fluxa Rojas.',
    author_email = 'pafluxa@astro.puc.cl',
    version = '0.1',
    ext_modules = extensions,   
    )
# *****************************************************************************
