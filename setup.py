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
        import numpy as np

        include_dirs = np.get_include()
        library_dirs = resource_filename('numpy', 'random/lib')

        for extension in self.extensions:
            extension.include_dirs.append(include_dirs)
            extension.library_dirs.append(library_dirs)
            extension.libraries.append('npyrandom')


if __name__ == '__main__':
    # Read in requirements.txt
    with open('requirements.txt', 'r') as reqs_file:
        requirements = list(reqs_file.readlines())

    cmdclass = {'build_ext': DeferredBuildExt}

    viridicle_ext = Extension(
        'viridicle._C',
        sources=[
            'viridicle/graph_ops.c',
            'viridicle/data_prep.c',
            'viridicle/viridicle.c'
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
        version='0.0',
        packages=['viridicle'],
        ext_modules=[viridicle_ext],
        description='Fast stochastic ecological simulations on graphs in '
                    'Python and numpy',
        setup_requires=requirements,
        install_requires=requirements,
        cmdclass=cmdclass,
    )
