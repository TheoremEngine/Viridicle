# Copyright 2021, Theorem Engine
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
from pkg_resources import resource_filename
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys


class DeferredBuildExt(build_ext):
    '''
    This is based on the class with the same name in statsmodel's setup.py.
    Briefly: viridicle._C needs to link against numpy libaries. To find those
    libraries and add them to extension.include_dirs and library_dirs, we need
    numpy to have already been installed. If numpy has not been installed, we
    are going to install it as a prerequisite - but in the default build_ext,
    we need to have set those variables BEFORE the prerequisites are installed.
    This class updates those variables AFTER the prerequisites have been
    installed but BEFORE the actual compilation has occurred.
    '''
    def build_extensions(self):
        self._update_extensions()
        build_ext.build_extensions(self)

    def _update_extensions(self):
        from numpy.distutils.misc_util import get_info

        np_paths = get_info('npymath')
        random_lib_path = resource_filename('numpy', 'random/lib')

        for extension in self.extensions:
            extension.include_dirs += np_paths['include_dirs']
            extension.library_dirs.append(random_lib_path)
            extension.library_dirs += np_paths['library_dirs']
            extension.libraries.append('npyrandom')
            extension.libraries += np_paths['libraries']


if __name__ == '__main__':
    # Get root path
    root_path = os.path.dirname(os.path.abspath(__file__))

    # Get version
    init_path = os.path.join(root_path, 'viridicle/__init__.py')
    with open(init_path, 'r') as init_file:
        for line in init_file:
            if line.startswith('__version__'):
                _, version = line.split('=')
                version = version.strip(" '\n")
                break
        else:
            raise RuntimeError('Could not find version')

    cmdclass = {'build_ext': DeferredBuildExt}

    viridicle_ext = Extension(
        'viridicle._C',
        sources=[
            'viridicle/data_prep.c',
            'viridicle/graph_ops.c',
            'viridicle/viridicle.c',
        ],
        include_dirs=[
            'viridicle/',
        ],
        extra_compile_args=['-O3'],
    )

    if '--warn' in sys.argv:
        sys.argv.remove('--warn')
        viridicle_ext.extra_compile_args.append('-Wall')

    if '--profile' in sys.argv:
        sys.argv.remove('--profile')
        viridicle_ext.extra_compile_args.extend(['-g', '-DWITHVALGRIND'])

    setup(
        name='viridicle',
        version=version,
        packages=['viridicle'],
        ext_modules=[viridicle_ext],
        description='Fast stochastic ecological simulations on graphs in '
                    'Python and numpy',
        setup_requires=['numpy'],
        install_requires=['networkx', 'numpy'],
        cmdclass=cmdclass,
    )
