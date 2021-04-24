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

from typing import Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np

from . import _C  # type: ignore


__all__ = [
    'ArbitraryGeography', 'FullyConnectedGeography', 'Geography',
    'LatticeGeography', 'may_leonard_rules'
]


# Define type for what can be coerced into a generator
CoercableToGenerator = Union[int, np.random.Generator]

# Type for rules that have been decoded from strings; the tuple is:
# (cur_state_1, cur_state_2, new_state_1, new_state_2, rate)
Rule = Tuple[int, int, int, int, float]


class Geography:
    '''
    This is a base class for all viridicle simulations. It must be used in a
    subclass.

    In all :class:`Geography` s, the local behavior of the system is specified
    using one property, the 4-dimensional :class:`numpy.ndarray` beta. beta
    specifies the rate at which transitions occur: in every time step, we
    randomly select a pair of neighboring sites. If (cur_state_1, cur_state_2)
    is the current states of those sites, then the probability that they
    transition in that time step to (new_state_1, new_state_2) is given by
    beta[cur_state_1, cur_state_2, new_state_1, new_state_2].

    Besides beta, the current state of the system is specified by a property
    sites. In most :class:`Geography` s, sites will be a
    :class:`numpy.ndarray`, but it may have other types.

    All subclasses of :class:`Geography` must provide the following methods:

    ``_coerce_to_sites``: Should coerce the input provided as sites into a
    sites object of the appropriate type.
    ``encode``: This function should return a dictionary of graph-specific
    parameters that must be passed to _C.run_system: the sites
    :class:`numpy.ndarray`, the graph_type integer, and optionally a
    neighborhoods, edges, and edge_idxs :class:`numpy.ndarrays`. Further
    requirements are discussed in the docstring for the ``encode`` method.
    ``decode``: This function should accept a dictionary in the same form
    returned by ``encode`` and convert it back to the sites property, of
    whatever class is used by this :class:`Geography`. Further requirements are
    discussed in the docstring for the ``decode`` method.

    Subclasses must also provide the following properties:

    ``num_directed_edges``: Number of directed edges in the graph. This is used
        in calculating rate scaling.

    Subclasses should provide the following properties:

    ``num_sites``: Number of vertices in the graph.

    Args:
        sites: The current state of the system.
        rules (Listr[str] or :class:`numpy.ndarray`): Rules for the system
          transition. This can be provided either as a 4-dimensional
          :class:`numpy.ndarray` where entry rules[i, j, k, l] gives the
          probability of the transition (i,j)->(k,l) in a single time step, or
          it can be provided as a list of strings given transition rules. See
          the add_rule method for how these strings are interpreted and
          converted into an array.
        generator (optional, int or :class:`numpy.random.Generator`): Random
          number generator. If an integer is provided, a new generator is
          constructed using that number as the seed for a default numpy random
          number generator.
        num_states (optional, int): Number of possible states in the system.
          This is only needed if you are specifying your rules as a list of
          strings.
    '''
    def __init__(self, sites, rules: Union[np.ndarray, Sequence[str]],
                 generator: Optional[CoercableToGenerator] = None,
                 num_states: Optional[int] = None):
        # Coerce generator into an np.random.Generator
        self.generator: np.random.Generator = \
            self._coerce_to_generator(generator)

        # Set the beta array
        if isinstance(rules, np.ndarray):
            self.beta = rules
            if (num_states is not None) and (num_states != self.num_states):
                raise ValueError(
                    'num_states was explicitly specified but does not match '
                    f'the shape of the rules array: {num_states} vs. '
                    f'{self.beta.shape}.'
                )
        else:
            if num_states is None:
                raise ValueError(
                    'num_states must be provided if rules are provided as a '
                    'list of strings.'
                )
            self.beta = np.zeros(
                (num_states, num_states, num_states, num_states)
            )
            for rule in rules:
                self.add_rule(rule)

        # Check beta array. This is checked again in the C layer, but better to
        # fail earlier if possible.
        if not all((s == self.num_states for s in self.beta.shape)):
            raise ValueError(
                f'beta must have equal dimensions; received {self.beta.shape}.'
            )
        if self.beta.ndim != 4:
            raise ValueError(
                f'beta must be 4-dimensional; received array of shape '
                f'{self.beta.shape}.'
            )

        # Note that the order here is important, since _coerce_to_sites may use
        # generator and beta for its work.
        self.sites = self._coerce_to_sites(sites)

    @staticmethod
    def _coerce_to_generator(g: Optional[CoercableToGenerator] = None) -> \
            np.random.Generator:
        '''
        Coerces an object into a :class:`numpy.random.Generator`.

        Args:
            g (optional, int or :class:`numpy.random.Generator`): Object to be
            coerced.

        Returns:
            A :class:`numpy.random.Generator`.
        '''
        # Coerce generator into existence.
        if g is None:
            g = 0
        if isinstance(g, int):
            g = np.random.default_rng(g)
        if not isinstance(g, np.random.Generator):
            raise TypeError(
                'If provided, generator must be either an np.random.Generator'
                f' or an integer random seed, not a {type(g)}.'
            )
        return g

    def _coerce_to_sites(self, sites):
        '''
        This must take an arbitrary input provided for sites and coerce it into
        the sites property.

        Args:
            sites: Object to be coerced.
        '''
        raise NotImplementedError()

    def add_rule(self, rule: str):
        '''
        Adds a transition rule. Rules are expected to be strings of the form:

        .. code-block::

            "i,j->k,l@r"

        Where i, j, k, and l are integers giving a state, and r is a float
        giving the transition probability in a time step. A wildcard "*" is
        also allowed, but only one wildcard is allowed on either side of the
        ->, and is assumed to represent the same value on both sides - so, for
        example, if our system has two states, and we are provided the rule:

        .. code-block::

            "*1->*0@0.1"

        This is interpreted as the pair of rules:

        .. code-block::

            "01->00@0.1",
            "11->10@0.1"

        Args:
            rule (str): Rule to be added.
        '''
        rules = _encode_rule(rule, num_states=self.num_states)
        for i, j, k, l, r in rules:
            self.beta[i, j, k, l] += r

    def decode(self, sites: np.ndarray):
        '''
        Converts a :class:`numpy.ndarray` of site data to the form to be
        stored as the sites property. For most :class:`Geography` s, this is a
        no-op, but for :class:`Geography` s that store sites as a different
        class (e.g. :class:`ArbitraryGeography`, where it is a
        :class:`networkx.Graph`), this may require some additional work.

        In addition, decode must be able to accept an array that has an
        additional dimension at the end relative to the internal sites array,
        for use in decoding a returned sites record. For example, suppose we
        have an :class:`ArbitraryGeography` which we run with
        return_sites=True. sites is a :class:`networkx.Graph`, which the encode
        method converts to a 1-dimensional :class:`numpy.ndarray`, of shape
        (N,). The sites record is then returned as a 2-dimensional array of
        shape (N, T). We want to then decode the array into a new
        :class:`networkx.Graph` where each vertex has associated to it a 1-
        dimensional array of shape (T,).

        Args:
            sites (:class:`numpy.ndarray`): Array to be decoded.

        Returns:
            Sites decoded to the proper format for this graph type.
        '''
        raise NotImplementedError()

    def encode(self) -> Dict:
        '''
        Returns a dictionary of parameters to be passed to the C layer.

        The dictionary must include the following keys:

         *  'graph_type' (:class:`int`): This must be a numerical value
            instructing the C layer in how to interpret the contents of the
            parameters. The options are:

                0: The graph is fully-connected.
                1: The graph is a lattice. An additional key, 'neighbors', must
                also be provided, containing a :class:`numpy.ndarray`; see the
                :class:`LatticeGeography` for details.
                2: The graph has arbitrary structure. Two additional keys,
                'edges' and 'edge_idxs', must also be provided, containing
                :class:`numpy.ndarray`s; see the :class:`ArbitraryGeography`
                for details.

         * 'sites' (:class:`numpy.ndarray`): This must encode the states of the
           different vertices. Note that this must be a :class:`numpy.ndarray`,
           even though the C layer can technically accept other classes that
           will be coerced, because the `_C.run_system` function will modify it
           in place, and this is how the new sites array is returned to update
           the sites parameter.

        Returns:
            Dictionary of parameters.
        '''
        raise NotImplementedError()

    @property
    def is_directed(self) -> bool:
        '''
        Whether the graph is directed.
        '''
        raise NotImplementedError()

    @property
    def num_directed_edges(self) -> int:
        '''
        Number of directed edges in the system.
        '''
        raise NotImplementedError()

    @property
    def num_states(self) -> int:
        '''
        Number of possible system states.
        '''
        return self.beta.shape[0]

    def run(self, elapsed_time: Optional[float] = None,
            num_steps: Optional[int] = None,
            report_every: Optional[float] = None,
            return_counts: bool = True, return_sites: bool = False,
            allow_diffusion_trick: bool = True):
        '''
        Runs the system for a specified amount of time using the Gillespie
        algorithm. Depending on the choice of parameters, may return a
        :class:`numpy.ndarray` containing the counts of each site state at
        every reporting interval, and/or a :class:`numpy.ndarray` containing
        the entire system at each reporting interval.

        The user may specify either the amount of time to run the system, or
        the number of steps, but may not specify both.

        Args:
            elapsed_time (optional, float): Amount of time to run the system.
            num_steps (optional, int): Number of steps to run the system. May
              be specified in place of elapsed_time.
            report_every (optional, float): How often to record telemetry. If
              not set, return only the initial and final states. Is interpreted
              as either time or number of steps, depending on which of
              elapsed_time or num_steps is set.
            return_counts (bool): Whether to return population counts at every
              reporting interval. Default True.
            return_sites (bool): Whether to return the full sites array at
              every reporting interval. Default False.
            allow_diffusion_trick (bool): Whether to allow the use of the
              diffusion trick. This reduces runtime, but can cause very slight
              inaccuracy in the model if the diffusion rate is extremely high.

        Returns:
            Depending on parameters set, can return a record of the population
            counts over the run, a record of the system state over the run, or
            both.
        '''
        if (elapsed_time and num_steps) or not (elapsed_time or num_steps):
            raise ValueError(
                'Must specify elapsed_time or num_steps, but not both.'
            )

        report_every = report_every or (elapsed_time or num_steps)

        # Get beta. If the system is undirected, we can improve performance
        # by splitting transitions (a, b)->(c, d) into pairs of transitions
        # (a, b)->(c, d) and (b, a)->(d, c), each of half rate.
        if self.is_directed:
            # Make sure this is a copy since we're going to modify it in-
            # place below.
            beta = self.beta.copy()
        else:
            beta = (self.beta + np.transpose(self.beta, (1, 0, 3, 2))) / 2

        # We can also remove null transitions: transitions (a, b)->(a, b).
        i, j = np.meshgrid(range(self.num_states), range(self.num_states))
        i, j = i.flatten(), j.flatten()
        beta[i, j, i, j] = 0.0

        if allow_diffusion_trick:
            # Calculate diffusion rate.
            i_0, i_1 = np.meshgrid(*((range(self.num_states),) * 2))
            i_0, i_1 = i_0.flatten(), i_1.flatten()
            i_0, i_1 = i_0[i_0 != i_1], i_1[i_0 != i_1]
            diffusion_rate = beta[i_0, i_1, i_1, i_0].min()

            # Calculate the maximum non-diffusion rate and, from that, the
            # diffusion probability.
            beta[i_0, i_1, i_1, i_0] -= diffusion_rate
            max_reaction_rate = beta.sum((2, 3)).max()
            max_total_rate = max_reaction_rate + diffusion_rate
            diffusion_probability = diffusion_rate / max_total_rate

            # If diffusion is the only event possible, then the default
            # diffusion implementation is not going to work. It's not really
            # worth implementing a separate diffusion-only C layer pathway, so
            # instead we'll treat diffusion as an ordinary reaction, and not
            # use the diffusion trick. It's also not worth using the trick if
            # the diffusion probability is too low.
            if (max_reaction_rate == 0) or (diffusion_probability < 0.5):
                # Undo the change to beta
                beta[i_0, i_1, i_1, i_0] += diffusion_rate
                diffusion_probability = 0.0
                max_reaction_rate = max_total_rate

        else:
            diffusion_probability = 0.0
            max_total_rate = max_reaction_rate = beta.sum((2, 3)).max()

        # Convert beta from rates to probabilities. We include a small epsilon
        # in the max_total_rate to prevent floating point errors causing the
        # max total rate to slightly exceed 1.0, which raises a ValueError in
        # the C layer.
        beta /= (1 + 1e-9) * max_reaction_rate

        # Convert elapsed_time to number of steps. Note that we expect rates to
        # be specified PER EDGE, not PER VERTEX, hence the use of
        # num_directed_edges in the time_scale calculation, instead of
        # num_sites.
        if elapsed_time:
            time_scale = self.num_directed_edges * max_total_rate
            num_steps = int(elapsed_time * time_scale)
            report_every = int(report_every * time_scale)
        else:
            num_steps, report_every = int(num_steps), int(report_every)

        # Pass control to C layer.
        params = self.encode()
        rtn = _C.run_system(
            bitgen=self.generator.bit_generator.capsule,
            beta=beta,
            num_steps=num_steps,
            report_every=report_every,
            diffusion_probability=diffusion_probability,
            return_counts=return_counts,
            return_sites=return_sites,
            **params
        )
        self.sites = self.decode(params['sites'])
        return rtn


class ArbitraryGeography(Geography):
    '''
    :class:`ArbitraryGeography` works with arbitrary underlying site graphs,
    specified as an :class:`nx.Graph`. The state of each site in the graph must
    be given as a property. By default this property is simply 'state', but it
    can be specified by the parameter 'state_key'.

    Args:
        sites (:class:`networkx.Graph`): The current state of the system. This
          must be a graph encoding the system structure.
        rules (list or :class:`numpy.ndarray`): Rules for the system
          transition. This can be provided either as a 4-dimensional
          :class:`numpy.ndarray` where entry rules[i, j, k, l] gives the
          probability of the transition (i,j)->(k,l) in a single time step, or
          it can be provided as a list of strings giving transition rules. See
          the add_rule method for how these strings are interpreted and
          converted into an array.
        generator (optional, int or :class:`numpy.random.Generator`): Random
          number generator. If an integer is provided, a new generator is
          created using that number as the seed for a default numpy random
          number generator.
        num_states (optional, int): Number of possible states in the system.
          This is only needed if you are specifying your rules as a list of
          strings.
        state_key (optional, str): The property of the nodes that specifies the
          starting state of the node. If this is None, then initialize the
          states randomly, and use 'state' as the state_key.
    '''
    def __init__(self, sites: nx.Graph,
                 rules: Union[np.ndarray, Sequence[str]],
                 generator: Optional[CoercableToGenerator] = None,
                 state_key: Optional[str] = None,
                 num_states: Optional[int] = None):

        self.state_key = state_key
        super().__init__(sites, rules, generator, num_states=num_states)

    def _coerce_to_sites(self, sites: nx.Graph) -> nx.Graph:
        '''
        Coerces the sites input to an nx.Graph where each node has a
        state_key entry.

        Args:
            sites (:class:`networkx.Graph`): Object to be coerced.

        Returns:
            The sites graph.
        '''
        if not isinstance(sites, nx.Graph):
            raise TypeError(
                f'Sites must be a networkx.Graph, not a {type(sites)}.'
            )

        if self.state_key is None:
            self.state_key = 'state'

        # Make sure that every vertex has a state, and if they do not, generate
        # them randomly.
        if not all(self.state_key in sites.nodes[n] for n in sites.nodes):
            for node in sites.nodes:
                sites.nodes[node][self.state_key] = self.generator.integers(
                    self.num_states, dtype=np.uint8
                )

        return sites

    def decode(self, sites: np.ndarray) -> nx.Graph:
        '''
        Converts the sites :class:`numpy.ndarray` used to pass the graph state
        to the C layer back to a :class:`networkx.Graph`.

        Args:
            sites (:class:`numpy.ndarray`): Array to be decoded.

        Returns:
            The decoded :class:`networkx.Graph`.
        '''
        graph = self.sites.copy()
        # Go through and update the node statuses
        for idx, node in enumerate(graph.nodes):
            graph.nodes[node][self.state_key] = sites[idx]
        return graph

    def encode(self) -> Dict:
        '''
        Encodes the graph to a dictionary of parameters to be passed to the C
        layer. The dictionary for :class:`ArbitraryGeography` s will include
        the keys 'graph_type', 'sites', 'edges', and 'edge_idxs'.

        Returns:
            The parameter dictionary.
        '''
        # Construct a numerical index for the nodes
        node_to_idx = {n: idx for idx, n in enumerate(self.nodes)}

        # Encode the node state into an array
        sites = np.array(
            [self.nodes[n][self.state_key] for n in self.nodes], dtype=np.uint8
        )

        # Construct the arrays encoding the graph structure. edges contains the
        # actual edges, while edge_idxs indexs within edges where to find the
        # neighbors of a particular node. Briefly, if we want to find the
        # neighbors of site k, then we should do:
        #
        # neighbors = edges[edge_idxs[k]:edge_idxs[k + 1]]

        edge_idxs = np.cumsum([0] + [self.sites.degree[n] for n in self.nodes])

        edges = np.empty((self.num_directed_edges,), dtype=np.intp)
        for idx, node in enumerate(self.nodes):
            edges[edge_idxs[idx]:edge_idxs[idx + 1]] = \
                [node_to_idx[_n] for _n in self.sites.adj[node]]

        return {
            'graph_type': 2,
            'sites': sites,
            'edge_idxs': edge_idxs,
            'edges': edges
        }

    @property
    def is_directed(self) -> bool:
        '''
        Whether the graph is directed.
        '''
        return nx.is_directed(self.sites)

    @property
    def nodes(self) -> nx.classes.reportviews.NodeView:
        '''
        The nodes of the graph.
        '''
        return self.sites.nodes

    @property
    def num_directed_edges(self) -> int:
        '''
        The number of directed edges in the system.
        '''
        # if self.sites is directed, then number_of_edges counts directed
        # edges, otherwise it counts undirected edges.
        if self.is_directed:
            return self.sites.number_of_edges()
        else:
            return 2 * self.sites.number_of_edges()

    @property
    def num_sites(self) -> int:
        '''
        The number of sites in the system.
        '''
        return len(self.sites.nodes)


class FullyConnectedGeography(Geography):
    '''
    This is a geography for a fully-connected graph. All vertices are connected
    to every other vertex in the graph.

    Args:
        sites (int or :class:`numpy.ndarray`): The current state of the system.
          If an integer, a new graph of that size is randomly created.
        rules (list or :class:`numpy.ndarray`): Rules for the system
          transition. This can be provided either as a 4-dimensional
          :class:`numpy.ndarray` where entry rules[i, j, k, l] gives the
          probability of the transition (i,j)->(k,l) in a single time step, or
          it can be provided as a list of strings giving transition rules. See
          the add_rule method for how these strings are interpreted and
          converted into an array.
        generator (optional, int or :class:`numpy.random.Generator`): Random
          number generator. If an integer is provided, a new generator is
          created using that number as the seed for a default numpy random
          number generator.
        num_states (optional, int): Number of possible states in the system.
          This is only needed if you are specifying your rules as a list of
          strings.
    '''
    is_directed: bool = False

    def _coerce_to_sites(self, sites: Union[np.ndarray, int]) -> np.ndarray:
        '''
        Coerces sites to a :class:`numpy.ndarray`, randomly generating one if
        necessary.

        Args:
            sites (int or :class:`numpy.ndarray`): Object to be coerced.

        Returns:
            The sites :class:`numpy.ndarray`.
        '''
        if isinstance(sites, int):
            return self.generator.integers(
                0, self.num_states, sites, dtype=np.uint8
            )
        elif not isinstance(sites, np.ndarray):
            return np.array(sites, dtype=np.uint8)

    def decode(self, sites: np.ndarray) -> np.ndarray:
        '''
        Decodes an array returned from the C layer back to the system state.

        Args:
            sites (:class:`numpy.ndarray`): The array to be decoded.

        Returns:
            The :class:`numpy.ndarray`.
        '''
        return sites

    def encode(self) -> Dict:
        '''
        Encodes the graph to a dictionary of parameters to be passed to the C
        layer. The dictionary for :class:`FullyConnectedGeography` s will
        include only the keys 'graph_type' and 'sites'.

        Returns:
            The parameter dictionary.
        '''
        return {
            'graph_type': 0,
            'sites': self.sites,
        }

    @property
    def num_directed_edges(self) -> int:
        '''
        The number of directed edges in the system.
        '''
        n = self.sites.shape[0]
        return n * (n - 1)

    @property
    def num_sites(self) -> int:
        '''
        The number of sites in the system.
        '''
        return self.sites.size


class LatticeGeography(Geography):
    '''
    This is a geography for a periodic, arbitrarily-dimensioned lattice. This
    consists of a collection of evenly spaced vertices, usually but not
    necessarily 2-dimensional. The edges are specified by a set of offsets;
    the default offsets are [[0, 1], [1, 0], [0, -1], [-1, 0]], meaning the
    horizontal and vertical neighbors, for example:

    .. code-block::

         | | | | | |
        -N-N-N-N-N-N-
         | | | | | |
        -N-N-N-N-N-N-
         | | | | | |
        -N-N-N-N-N-N-
         | | | | | |
        -N-N-N-N-N-N-
         | | | | | |
        -N-N-N-N-N-N-
         | | | | | |
        -N-N-N-N-N-N-
         | | | | | |

    However, the user may specify a different neighborhood.

    Args:
        sites (sequence of int or :class:`numpy.ndarray`): The current state of
          the system. If a sequence of integers is provided, a new random sites
          array of that shape will be created.
        rules (list or :class:`numpy.ndarray`): Rules for the system
          transition. This can be provided either as a 4-dimensional
          :class:`numpy.ndarray` where entry rules[i, j, k, l] gives the
          probability of the transition (i,j)->(k,l) in a single time step, or
          it can be provided as a list of strings giving transition rules. See
          the add_rule method for how these strings are interpreted and
          converted into an array.
        generator (optional, int or :class:`numpy.random.Generator`): Random
          number generator. If an integer is provided, a new generator is
          created using that number as the seed for a default numpy random
          number generator.
        num_states (optional, int): Number of possible states in the system.
          This is only needed if you are specifying your rules as a list of
          strings.
        neighborhood (optional, :class:`numpy.ndarray`): The neighborhood of
          each vertex. If provided, this should be a 2-dimensional array such
          that each row gives an offset value. If not provided, defaults to the
          horizontally and vertically adjacent sites.
    '''
    is_directed: bool = False

    def __init__(self, sites: Union[Sequence[int], np.ndarray],
                 rules: Union[np.ndarray, Sequence[str]],
                 generator: Optional[CoercableToGenerator] = None,
                 neighborhood: Optional[np.ndarray] = None,
                 num_states: Optional[int] = None):
        super().__init__(sites, rules, generator, num_states=num_states)

        # If neighborhood is not set, use the default.
        if neighborhood is None:
            neighborhood = np.zeros(
                (2 * self.sites.ndim, self.sites.ndim), dtype=np.int64
            )
            for i in range(self.sites.ndim):
                neighborhood[2 * i, i] = 1
                neighborhood[2 * i + 1, i] = -1

        self.neighborhood: np.ndarray = np.array(neighborhood, dtype=np.int64)

    def _coerce_to_sites(self, sites: Union[Sequence[int], np.ndarray]) \
            -> np.ndarray:
        '''
        Coerces sites to a :class:`numpy.ndarray`, randomly generating one if
        necessary.

        Args:
            sites (sequence of ints or :class:`numpy.ndarray`): Object to be
              coerced.

        Returns:
            The sites :class:`numpy.ndarray`.
        '''
        if isinstance(sites, Sequence):
            if all(isinstance(s, int) for s in sites):
                return self.generator.integers(
                    0, self.num_states, sites, dtype=np.uint8
                )
        if not isinstance(sites, np.ndarray):
            sites = np.array(sites, dtype=np.uint8)
        return sites

    def decode(self, sites: np.ndarray) -> np.ndarray:
        '''
        Decodes an array returned from the C layer back to the system state.

        Args:
            sites (:class:`numpy.ndarray`): The array to be decoded.

        Returns:
            The :class:`numpy.ndarray`.
        '''
        return sites

    def encode(self) -> Dict:
        '''
        Encodes the graph to a dictionary of parameters to be passed to the C
        layer. The dictionary for :class:`LatticeGeography` s will include the
        keys 'graph_type', 'sites', and 'neighborhood'.

        Returns:
            The parameter dictionary.
        '''
        return {
            'graph_type': 1,
            'sites': self.sites,
            'neighborhood': self.neighborhood,
        }

    @property
    def ndim(self) -> int:
        '''
        The dimension of the lattice.
        '''
        return self.sites.ndim

    @property
    def num_directed_edges(self) -> int:
        '''
        The number of directed edges in the system.
        '''
        return self.neighborhood.shape[0] * self.num_sites

    @property
    def num_sites(self) -> int:
        '''
        The number of sites in the system.
        '''
        return self.sites.size

    @property
    def shape(self) -> Sequence[int]:
        '''
        The shape of the lattice.
        '''
        return self.sites.shape


def _encode_rule(rule: str, num_states: Optional[int] = None) -> List[Rule]:
    '''
    Converts a rule string into a list of 5-tuples encoding it as entries in a
    beta array. Rules are expected to be strings of the form:

    .. code-block::

        "i,j->k,l@r"

    Where i, j, k, and l are integers giving a state, and r is a float
    giving the transition probability in a time step. This is then encoded into
    the 5-tuple:

    .. code-block::

        (i, j, k, l, r)

    We also allow the use of wildcards '*'. Only one wildcard is allowed on
    either side of the '->', and it is assumed to represent the same value on
    both sides - so, for example, if our system has two states, and we are
    provided the rule:

    .. code-block::

        "*,1->*,0@0.1"

    This is interpreted as the pair of rules:

    .. code-block::

        "0,1->0,0@0.1",
        "1,1->1,0@0.1"

    Note that, to use wildcards, num_states MUST be explicitly provided; if it
    is not, the function will raise a ValueError on encountering a wildcard.

    Args:
        num_states (optional, int): Total number of states in the system. Only
          required if wildcard values are allowed.

    Returns:
        A list of 5-tuples encoding the rules.
    '''
    # Break rule along '@', '->'
    try:
        pairs, rate = rule.split('@')
        side_1, side_2 = pairs.split('->')
    except ValueError:
        raise ValueError(f'Could not parse rule {rule}.')

    # Break rule along ','
    try:
        (a, b), (c, d) = side_1.split(','), side_2.split(',')
    except ValueError:
        raise ValueError(f'Could not parse rule {rule}.')

    # Check for too many wildcards.
    if (a == b == '*') or (c == d == '*'):
        raise ValueError('rule contains too many wildcards.')

    rules: List[Rule]

    # If wildcards are present, unpack them.
    if '*' in {a, b, c, d}:
        if num_states is None:
            raise ValueError(
                'Wildcards require that num_states be specified.'
            )

        def unpack(s: str, i: int) -> int:
            return i if (s == '*') else int(s)

        rules = [
            (*(unpack(s, i) for s in (a, b, c, d)),  # type: ignore
             float(rate))
            for i in range(num_states)
        ]

    else:
        rules = [(int(a), int(b), int(c), int(d), float(rate))]

    return rules


def may_leonard_rules(n: int, r: int, diffusion_rate: float,
                      empty_zero: bool = True) -> np.ndarray:
    '''
    This constructs the rules tensor for the generalized (n, r) May-Leonard
    system, as defined in:

    Ahmed, Debanjan, and Pleimling. "Interplay Between Partnership Formation
    and Competition in Generalized May-Leonard Games." Physical Review E Vol.
    87 No. 3. https://arxiv.org/abs/1303.3139

    Args:
        n (int): Number of species.
        r (int): Number of species that each species will prey on. Species k
          will predate on species k + 1, k + 2, ..., k + r, calculated modulo.
        diffusion_rate (float): Rate of diffusion in the system.
        empty_zero (bool): If set, treat zero as empty, rather than as a
          distinct species. Default True.

    Returns:
        A rules :class:`numpy.ndarray` suitable for use in a
        :class:`Geography`.
    '''
    # Check inputs
    if n <= 0:
        raise ValueError(f'n must be positive; got {n}')
    if r < 0:
        raise ValueError(f'r must be non-negative; got {r}')
    if diffusion_rate < 0:
        raise ValueError(
            f'diffusion_rate must be non-negative; got {diffusion_rate}'
        )

    if empty_zero:
        beta = np.zeros((n + 1, n + 1, n + 1, n + 1))
        # Add reproduction term
        for s in range(1, n + 1):
            beta[0, s, s, s] = beta[s, 0, s, s] = 0.2
        # Add diffusion term
        for s_1 in range(n + 1):
            for s_2 in range(n + 1):
                if s_1 != s_2:
                    beta[s_1, s_2, s_2, s_1] = diffusion_rate
        # Add predation term
        for s_1 in range(1, n + 1):
            for shift in range(1, r + 1):
                s_2 = s_1 + shift
                s_2 = s_2 if (s_2 <= n) else (s_2 - n)
                beta[s_1, s_2, s_1, 0] = 0.2
                beta[s_2, s_1, 0, s_1] = 0.2

    else:
        beta = np.zeros((n, n, n, n))
        # Add diffusion term
        for s_1 in range(n):
            for s_2 in range(n):
                beta[s_1, s_2, s_2, s_1] = diffusion_rate
        # Add predation term
        for s_1 in range(n):
            for shift in range(1, r + 1):
                s_2 = (s_1 + shift) % n
                beta[s_1, s_2, s_1, s_1] = 0.2
                beta[s_2, s_1, s_1, s_1] = 0.2

    return beta
