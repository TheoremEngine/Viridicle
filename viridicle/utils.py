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

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

from . import _C  # type: ignore
from .viridicle import Geography

__all__ = ['cluster_geography', 'grow_clusters', 'merge_small_clusters']


def cluster_geography(geo: Geography, out: Optional[np.ndarray] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''
    We define a cluster as a maximal connected subgraph, all of whose vertices
    have the same state. This function calculates clusters of the geography,
    and returns a numpy.ndarray of the same size and shape as the sites array
    whose entries are the cluster ids for the corresponding sites.

    Args:
        geo (:class:`Geography`): The model to be processed.
        out (optional, :class:`numpy.ndarray`): The :class:`numpy.ndarray` to
          hold the cluster ids. If this is not provided, a new one is generated
          from scratch, but the user may specify this themselves if desired to
          stabilize memory use.

    Returns:
        A pair of :class:`numpy.ndarray` s. The first array is the same shape
        as the sites array, and each entry gives the cluster id of the
        corresponding entry in the sites array. The second array maps from the
        cluster id to the original state.
    '''
    clusters, cluster_map = _C.cluster_geo(**geo.encode(), out=out)

    return geo.decode(clusters), cluster_map


def grow_clusters(geo: Geography, num_steps: int, empty_state: int = 0) \
        -> Geography:
    '''
    Grows the clusters in a graph. Briefly, at every step, we look at every
    empty vertex that is adjacent to one or more non-empty vertices. If the
    adjacent non-empty vertices have only one state, then we set that vertex to
    that state. We then repeat for the desired number of steps. For example, on
    a 2-dimensional periodic lattice, which we run for only one step, we would
    see:

    .. code-block::

        0000
        0110
        0002
        0000

    Become:

    .. code-block::

        0110
        1110
        2102
        0002

    Args:
        geo (:class:`Geography`): The model to be processed.
        num_steps (int): Number of steps to grow clusters.
        empty_state (int): State to be used as the empty value. Default 0.

    Returns:
        A new :class:`Geography` of the same state with the underlying sites
        changed.
    '''

    # Avoid making this an in-place operation.
    geo = deepcopy(geo)

    _C.grow_clusters(
        num_steps=num_steps,
        empty_state=empty_state,
        **geo.encode()
    )

    return geo


def merge_small_clusters(geo: Geography, min_size: int,
                         merge_size: Optional[int] = None,
                         empty_state: int = 0) -> Geography:
    '''
    We define a cluster as a maximal connected subgraph, all of whose vertices
    have the same state. This function eliminates clusters that lie below a
    specified minimum number of vertices, in order to clean the graph for
    further processing. The cleaning process works in two stages:

        1. First, we identify all clusters that fall below the minimum size,
           and set them to an empty_state value.
        2. Second, we take this new graph and identify all clusters of the
           empty state which fall below the minimum size, and check if they are
           adjacent to more than one neighboring cluster. If they are adjacent
           to only one other cluster, we set them to that cluster's value.

    Args:
        geo (:class:`Geography`): The model to be processed.
        min_size (int): Minimum cluster size.
        merge_size (int): Empty areas below this size that are fully surrounded
          by another cluster will be set to that cluster's state. If not set,
          defaults to min_size.
        empty_state (int): State to be used as the empty value. Default 0.

    Returns:
        A new :class:`Geography` of the same state with the underlying sites
        changed.
    '''
    if merge_size is None:
        merge_size = min_size

    # Avoid making this an in-place operation.
    geo = deepcopy(geo)

    _C.merge_small(
        min_size=min_size,
        merge_size=merge_size,
        empty_state=empty_state,
        **geo.encode()
    )

    return geo
