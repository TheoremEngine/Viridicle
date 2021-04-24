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

'''
Contains custom Geography subclasses.
'''
from typing import Dict, Sequence, Union

import numpy as np

import viridicle

__all__ = ['KleinGeography', 'SphericalGeography']


class KleinGeography(viridicle.LatticeGeography):
    '''
    Generates a lattice graph on a Klein bottle. This is the same as a periodic
    lattice, except that the left and right edges are twisted when they wrap
    around.
    '''
    is_directed: bool = False

    def _coerce_to_sites(self, sites: Union[Sequence[int], np.ndarray]) \
            -> np.ndarray:
        '''
        Coerces sites to a :class:`numpy.ndarray`, randomly generating one if
        necessary.

        :param sites: Object to be coerced.
        :type sites: :class:`numpy.ndarray` or int

        :return: Sites coerced to a :class:`numpy.ndarray`.
        :rtype: :class:`numpy.ndarray`
        '''
        rtn = super()._coerce_to_sites(sites)
        if (rtn.ndim != 2):
            raise ValueError('Sites must be 2-dimensional.')
        return rtn

    def encode(self) -> Dict:
        '''
        Encodes the graph to a dictionary of parameters to be passed to the C
        layer. The dictionary for :class:`LatticeGeography` s will include the
        keys 'graph_type', 'sites', and 'neighborhood'.

        :return: Parameter dictionary for C layer.
        :rtype: dict
        '''
        # Generate the edges.
        h, w = self.shape
        # x, y: Coordinates of vertices, in shape HW.
        x, y = np.meshgrid(np.arange(h), np.arange(w))
        # n_x, n_y: These will be arrays of shape HW4, where n_x[x, y, :],
        # n_y[x, y, :] contain the x, y coordinates of the neighbors of the
        # vertex (x, y). We implement the vertical wrapping here, and implement
        # the horizontal wrapping next.
        n_x = np.stack([x + 1, x - 1, x, x], axis=2)
        n_y = np.stack([y, y, y + 1, y - 1], axis=2) % h
        # Implement the horizontal wrapping.
        n_y[(n_x == w) | (n_x == -1)] = (h - 1) - n_y[(n_x == w) | (n_x == -1)]
        n_x %= w
        # Convert the neighbors from (x, y) coordinates to flat indices.
        n_i = (n_x + (n_y * w))
        # Generate edge_idxs: each vertex will have exactly 4 neighbors.
        edge_idxs = np.arange(0, 4 * (h * w + 1), 4)
        edges = n_i.flatten()

        return {
            'graph_type': 2,
            'sites': self.sites,
            'edge_idxs': edge_idxs,
            'edges': edges
        }


class SphericalGeography(viridicle.LatticeGeography):
    '''
    Generates a spherical graph on the 2-dimensional lattice. This is not a
    true sphere, but is topologically a sphere:

          /-----------\
          |   /---\   |
        - O - O - O - O -
          |   |   |   |
        - O - O - O - O -
          |   |   |   |
        - O - O - O - O -
          |   |   |   |
        - O - O - O - O -
          |   \---/   |
          \-----------/

    The horizontal sides wrap around as in a 2-dimensional periodic lattice.
    '''
    is_directed: bool = False

    def _coerce_to_sites(self, sites: Union[Sequence[int], np.ndarray]) \
            -> np.ndarray:
        '''
        Coerces sites to a :class:`numpy.ndarray`, randomly generating one if
        necessary.

        :param sites: Object to be coerced.
        :type sites: :class:`numpy.ndarray` or int

        :return: Sites coerced to a :class:`numpy.ndarray`.
        :rtype: :class:`numpy.ndarray`
        '''
        rtn = super()._coerce_to_sites(sites)
        if (rtn.ndim != 2):
            raise ValueError('Sites must be 2-dimensional.')
        if (rtn.shape[1] % 2 != 0):
            raise ValueError('Width of sites must be even.')
        return rtn

    def encode(self):
        '''
        Encodes the graph to a dictionary of parameters to be passed to the C
        layer. The dictionary for :class:`LatticeGeography` s will include the
        keys 'graph_type', 'sites', and 'neighborhood'.

        :return: Parameter dictionary for C layer.
        :rtype: dict
        '''
        # Generate the edges.
        h, w = self.shape
        # x, y: Coordinates of vertices, in shape HW.
        x, y = np.meshgrid(np.arange(h), np.arange(w))
        # n_x, n_y: These will be arrays of shape HW4, where n_x[x, y, :],
        # n_y[x, y, :] contain the x, y coordinates of the neighbors of the
        # vertex (x, y). Additionally, implement the wrap around the horizontal
        # edges.
        n_x = np.stack([x + 1, x - 1, x, x], axis=2) % w
        n_y = np.stack([y, y, y + 1, y - 1], axis=2)
        # Implement the polar wrap.
        is_up = (n_y < 0)
        is_down = (n_y >= h)
        n_y[is_down], n_x[is_down] = n_y[is_down] - 1, (w - 1) - n_x[is_down]
        n_y[is_up], n_x[is_up] = n_y[is_up] + 1, (w - 1) - n_x[is_up]
        # Convert the neighbors from (x, y) coordinates to flat indices.
        n_i = (n_x + (n_y * w))
        # Generate edge_idxs: each vertex will have exactly 4 neighbors.
        edge_idxs = np.arange(0, 4 * (h * w + 1), 4)
        edges = n_i.flatten()

        return {
            'graph_type': 2,
            'sites': self.sites,
            'edge_idxs': edge_idxs,
            'edges': edges
        }
