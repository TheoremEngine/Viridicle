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

from contextlib import suppress
import sys
import unittest
import weakref

import networkx as nx
import numpy as np

from viridicle._C import cluster_geo, merge_small, run_system, grow_clusters
from viridicle.viridicle import _encode_rule
import viridicle


class ReferenceLeakTest(unittest.TestCase):
    '''
    This tests for reference leaks in the C layer.
    '''
    def test_run_system(self):
        # First, check for reference leaks in error condition. We use weakrefs
        # to preserve the objects, and check if they're garbage collected. We
        # can't create a weakref to a PyCapsule, so we count the references
        # instead, which is a less reliable method of checking for errors.

        sites = np.array([0, 1, 0, 1], dtype=np.uint8)
        beta = np.zeros((3, 3, 3, 3))
        rng = np.random.default_rng().bit_generator.capsule

        sites_ref = weakref.ref(sites)
        beta_ref = weakref.ref(beta)
        n_rng_ref = sys.getrefcount(rng)

        with suppress(ValueError):
            run_system(-1, sites, rng, beta, 1, 1)

        del sites, beta

        self.assertTrue(sys.getrefcount(rng) == n_rng_ref)
        self.assertTrue(sites_ref() is None)
        self.assertTrue(beta_ref() is None)

        # Check for reference leaks if process works.

        for return_counts in [True, False]:
            for return_sites in [True, False]:
                sites = np.array([0, 1, 0, 1], dtype=np.uint8)
                beta = np.zeros((3, 3, 3, 3))
                rng = np.random.default_rng().bit_generator.capsule

                sites_ref = weakref.ref(sites)
                beta_ref = weakref.ref(beta)

                data = run_system(
                    0,
                    sites,
                    rng,
                    beta,
                    1,
                    1,
                    return_counts=return_counts,
                    return_sites=return_sites,
                )

                if return_counts and return_sites:
                    count_records, site_records = data
                elif return_counts:
                    count_records = data
                elif return_sites:
                    site_records = data
                if return_counts:
                    counts_ref = weakref.ref(count_records)
                    del count_records
                if return_sites:
                    record_ref = weakref.ref(site_records)
                    del site_records

                del sites, beta, data

                self.assertTrue(sites_ref() is None)
                self.assertTrue(beta_ref() is None)
                self.assertTrue(sys.getrefcount(rng) == n_rng_ref)
                if return_counts:
                    self.assertTrue(counts_ref() is None)
                if return_sites:
                    self.assertTrue(record_ref() is None)

    def test_merge_small(self):
        nhd = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        sites = np.array([
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 2, 0, 1],
            [0, 0, 2, 1]
        ], dtype=np.uint8)

        nhd_ref = weakref.ref(nhd)
        sites_ref = weakref.ref(sites)

        merge_small(1, sites, 0, 0, 255, neighborhood=nhd)

        del sites, nhd

        self.assertTrue(sites_ref() is None)
        self.assertTrue(nhd_ref() is None)

    def test_cluster(self):
        # Base test
        nhd = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        sites = np.array([
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ], dtype=np.uint8)

        nhd_ref = weakref.ref(nhd)
        sites_ref = weakref.ref(sites)

        clusters, cluster_map = cluster_geo(
            1, sites, neighborhood=nhd, out=None
        )

        clusters_ref = weakref.ref(clusters)
        cluster_map_ref = weakref.ref(cluster_map)

        del sites, nhd, clusters, cluster_map

        self.assertTrue(sites_ref() is None)
        self.assertTrue(nhd_ref() is None)
        self.assertTrue(clusters_ref() is None)
        self.assertTrue(cluster_map_ref() is None)

        # Now, what if we provide out as an argument?
        nhd = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        sites = np.array([
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ], dtype=np.uint8)
        out = np.zeros_like(sites, dtype=np.uint64)

        nhd_ref = weakref.ref(nhd)
        sites_ref = weakref.ref(sites)
        out_ref = weakref.ref(out)

        clusters, cluster_map = cluster_geo(
            1, sites, neighborhood=nhd, out=out
        )

        # clusters and out should be not just equal, but actually the same
        # object.
        self.assertTrue(clusters is out)

        cluster_map_ref = weakref.ref(cluster_map)

        del sites, nhd, clusters, cluster_map, out

        self.assertTrue(sites_ref() is None)
        self.assertTrue(nhd_ref() is None)
        self.assertTrue(out_ref() is None)
        self.assertTrue(cluster_map_ref() is None)

    def test_grow(self):
        nhd = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        sites = np.array([
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 2, 0, 1],
            [0, 0, 2, 1]
        ], dtype=np.uint8)

        nhd_ref = weakref.ref(nhd)
        sites_ref = weakref.ref(sites)

        grow_clusters(1, sites, num_steps=1, empty_state=255, neighborhood=nhd)

        del sites, nhd

        self.assertTrue(sites_ref() is None)
        self.assertTrue(nhd_ref() is None)


class ParameterChecks(unittest.TestCase):
    '''
    This unit test tests the system for parsing parameters at the C layer,
    making sure that appropriate errors and no seg faults are raised.
    '''
    def test_run_system(self):
        sites = [0, 1, 0, 1]
        beta = np.zeros((3, 3, 3, 3))
        beta[0, 1, 1, 0] = 1.0
        rng = np.random.default_rng().bit_generator.capsule

        # Check that appropriate errors are raised if graph_type is not an
        # available option
        with self.assertRaises(ValueError):
            run_system(-1, sites, rng, beta, 1, 1)
        with self.assertRaises(ValueError):
            run_system(3, sites, rng, beta, 1, 1)

        for graph_type in [0, 1, 2]:
            params = {
                0: {},
                1: {'neighborhood': np.array([[1, 0], [0, 1]])},
                2: {'edge_idxs': np.array([0, 2, 4, 6], dtype=np.intp),
                    'edges': np.array([1, 2, 0, 2, 0, 1], dtype=np.uint8)}
            }[graph_type]
            sites = {
                0: [1, 0, 2],
                1: [[1, 0], [0, 2]],
                2: [1, 0, 2]
            }[graph_type]
            beta = np.zeros((3, 3, 3, 3))
            beta[0, 1, 1, 0] = 1.0

            # Check that appropriate errors are raised if num_steps or
            # report_every are non-positive.
            with self.assertRaises(ValueError):
                run_system(graph_type, sites, rng, beta, 0, 1, **params)
            with self.assertRaises(ValueError):
                run_system(graph_type, sites, rng, beta, 1, 0, **params)
            # Check that appropriate error is raised if an argument is missing.
            with self.assertRaises(TypeError):
                run_system(graph_type, sites, rng, beta, 1, **params)
            if params:
                with self.assertRaises(TypeError):
                    run_system(graph_type, sites, rng, beta, 1, 1)

            # Check that appropriate error is raised if sites array contains
            # entry outside of range [0, num_species].
            beta = np.zeros((2, 2, 2, 2))
            beta[0, 1, 1, 0] = 1.0
            with self.assertRaises(ValueError):
                run_system(
                    graph_type,
                    sites,
                    rng,
                    beta=beta,
                    num_steps=1,
                    report_every=1,
                    **params,
                )
            # Check that appropriate error is raised if dimensions of beta is
            # inconsistent: beta must be 4-d and all dimensions must be equal.
            beta = np.zeros((3, 3, 3))
            beta[0, 1, 1] = 1.0
            with self.assertRaises(ValueError):
                run_system(
                    graph_type,
                    sites,
                    rng,
                    beta=beta,
                    num_steps=1,
                    report_every=1,
                    **params,
                )
            beta = np.zeros((3, 2, 3, 3))
            beta[0, 1, 1, 0] = 1.0
            with self.assertRaises(ValueError):
                run_system(
                    graph_type,
                    sites,
                    rng,
                    beta=beta,
                    num_steps=1,
                    report_every=1,
                    **params,
                )


class ReturnTest(unittest.TestCase):
    '''
    Tests return data parsing.
    '''
    def test_lattice(self):
        beta = np.zeros((2, 2, 2, 2))
        beta[1, 0, 0, 0] = beta[0, 1, 0, 0] = 0.5
        beta[1, 1, 0, 1] = beta[1, 1, 1, 0] = 0.5
        for return_counts in [True, False]:
            for return_sites in [True, False]:
                rng = np.random.default_rng(0)
                sites = np.array([[0, 0], [1, 1]], dtype=np.uint8)
                neighborhood = np.array(
                    [[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.intp
                )
                rtn = run_system(
                    1,
                    sites,
                    rng.bit_generator.capsule,
                    beta=beta,
                    num_steps=1,
                    report_every=1,
                    neighborhood=neighborhood,
                    return_counts=return_counts,
                    return_sites=return_sites,
                )
                if return_counts and return_sites:
                    counts, sites = rtn
                elif return_counts:
                    counts = rtn
                elif return_sites:
                    sites = rtn
                else:
                    self.assertTrue(rtn is None)

                if return_counts:
                    self.assertTrue(counts.shape == (2, 2))
                    self.assertTrue(
                        (counts == np.array([[2, 2], [3, 1]])).all()
                    )
                if return_sites:
                    self.assertTrue(sites.shape == (2, 2, 2))
                    self.assertTrue(
                        (sites == np.array([[[0, 0], [1, 1]],
                                            [[0, 0], [1, 0]]])).all()
                    )


class FullyConnectedGraphTest(unittest.TestCase):
    '''
    Tests fully-connected graphs by running them through a single iteration
    with an RNG that will produce known values. With the default RNG
    initialized with seed 0 and num_sites=4, the number of edges will be 12.
    Then, we should see the following during a single step:

    site_idx_1:         3
    site_idx_2:         1
    roll:               0.04

    We can use this to test the run system.
    '''
    def test_init(self):
        # Tests that the geography will end up with a generator if we supply it
        # with a valid but non-generator input, and that a random sites array
        # will be generated if we do not supply one.
        beta = np.zeros((2, 2, 2, 2))
        beta[0, 0, 1, 0] = beta[1, 0, 1, 0] = 1.0
        geo = viridicle.FullyConnectedGeography(4, beta)
        self.assertTrue(isinstance(geo.generator, np.random.Generator))
        self.assertTrue(isinstance(geo.sites, np.ndarray))
        self.assertTrue(geo.sites.shape == (4,))
        self.assertTrue(geo.beta is beta)

        geo = viridicle.FullyConnectedGeography(4, beta, 0)
        self.assertTrue(isinstance(geo.generator, np.random.Generator))
        self.assertTrue(isinstance(geo.sites, np.ndarray))
        self.assertTrue(geo.sites.shape == (4,))
        self.assertTrue(geo.beta is beta)

    def test_python(self):
        # RNGs return different values on Windows, so tests will not be valid.
        if sys.platform != 'linux':
            return
        # Test transition 01->11.
        beta = np.zeros((2, 2, 2, 2))
        beta[0, 1, 1, 1] = beta[1, 0, 1, 1] = 0.05
        # Ensure maximum total transition is 1.0 to prevent rescaling.
        beta[1, 1, 0, 0] = 1.0
        geo = viridicle.FullyConnectedGeography(
            [0, 0, 0, 1], beta, generator=np.random.default_rng(0)
        )
        geo.run(1 / 12.0)
        self.assertTrue((geo.sites == np.array([0, 1, 0, 1])).all(), geo.sites)

        # Test that this transition will not occur if we reduce the
        # probability.
        beta[0, 1, 1, 1] = beta[1, 0, 1, 1] = 0.03
        geo = viridicle.FullyConnectedGeography(
            [0, 0, 0, 1], beta, generator=np.random.default_rng(0)
        )
        geo.run(1 / 12.0)
        self.assertTrue((geo.sites == np.array([0, 0, 0, 1])).all())

    def test_c_layer(self):
        # RNGs return different values on Windows, so tests will not be valid.
        if sys.platform != 'linux':
            return
        # Test transition 01->11.
        beta = np.zeros((2, 2, 2, 2))
        beta[0, 1, 1, 1] = beta[1, 0, 1, 1] = 0.05
        rng = np.random.default_rng(0)
        sites = np.array([0, 0, 0, 1], dtype=np.uint8)
        run_system(
            0,
            sites,
            rng.bit_generator.capsule,
            beta=beta,
            num_steps=1,
            report_every=1
        )
        self.assertTrue((sites == np.array([0, 1, 0, 1])).all())

        # Now test reducing probability prevents the transition.
        beta[0, 1, 1, 1] = beta[1, 0, 1, 1] = 0.03
        rng = np.random.default_rng(0)
        sites = np.array([0, 0, 0, 1], dtype=np.uint8)
        run_system(
            0,
            sites,
            rng.bit_generator.capsule,
            beta=beta,
            num_steps=1,
            report_every=1
        )
        self.assertTrue((sites == np.array([0, 0, 0, 1])).all())


class LatticeGraphTest(unittest.TestCase):
    '''
    Tests lattice graphs by running them through a single iteration with an RNG
    that will produce known values. With the default RNG initialized with seed
    0 and num_sites=4, we should see the following during a single step:

    edge_idx:           15
    site_idx_1:         3 -> (0, 3)
    site_idx_2:         2 -> (0, 2)
    roll:               0.95

    We can use this to test the run system.
    '''
    def test_init(self):
        # Tests that the geography will end up with a generator if we supply it
        # with a valid but non-generator input, and that a random sites array
        # will be generated if we do not supply one.
        beta = np.zeros((2, 2, 2, 2))
        beta[0, 1, 1, 1] = beta[1, 0, 1, 1] = 0.2
        geo = viridicle.LatticeGeography((2, 2), beta)
        self.assertTrue(isinstance(geo.generator, np.random.Generator))
        self.assertTrue(isinstance(geo.sites, np.ndarray))
        self.assertTrue(geo.sites.shape == (2, 2))
        self.assertTrue(geo.beta is beta)

        geo = viridicle.LatticeGeography((2, 2), beta, 0)
        self.assertTrue(isinstance(geo.generator, np.random.Generator))
        self.assertTrue(isinstance(geo.sites, np.ndarray))
        self.assertTrue(geo.sites.shape == (2, 2))
        self.assertTrue(geo.beta is beta)

    def test_python(self):
        # RNGs return different values on Windows, so tests will not be valid.
        if sys.platform != 'linux':
            return
        # Test transition 01->11.
        beta = np.zeros((4, 4, 4, 4))
        beta[0, 1, 1, 1] = beta[1, 0, 1, 1] = 0.96
        # Ensure maximum total transition is 1.0 to prevent rescaling.
        beta[1, 1, 0, 0] = 1.0
        geo = viridicle.LatticeGeography(
            [[0, 0], [0, 1]], beta, generator=np.random.default_rng(1)
        )
        geo.run(1 / 16.0)
        self.assertTrue((geo.sites == np.array([[0, 0], [1, 1]])).all())

        # Test that this transition will not occur if we reduce the
        # probability.
        beta[0, 1, 1, 1] = beta[1, 0, 1, 1] = 0.90
        # Ensure maximum total transition is 1.0 to prevent rescaling.
        beta[1, 1, 0, 0] = 1.0
        geo = viridicle.LatticeGeography(
            [[0, 0], [0, 1]], beta, generator=np.random.default_rng(1)
        )
        geo.run(1 / 16.0)
        self.assertTrue((geo.sites == np.array([[0, 0], [0, 1]])).all())

    def test_c_layer(self):
        # RNGs return different values on Windows, so tests will not be valid.
        if sys.platform != 'linux':
            return
        # Test transition 01->11.
        beta = np.zeros((4, 4, 4, 4))
        beta[0, 1, 1, 1] = beta[1, 0, 1, 1] = 0.96
        # Ensure maximum total transition is 1.0 to prevent rescaling.
        beta[1, 1, 0, 0] = 1.0
        neighborhood = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
        ], dtype=np.int64)
        # First, test 1->0 using no interaction terms
        rng = np.random.default_rng(1)
        sites = np.array([[0, 0], [0, 1]], dtype=np.uint8)
        run_system(
            1,
            sites,
            rng.bit_generator.capsule,
            beta=beta,
            num_steps=1,
            report_every=1,
            neighborhood=neighborhood
        )
        self.assertTrue((sites == np.array([[0, 0], [1, 1]])).all())

        # Test that this transition will not occur if we reduce the
        # probability.
        beta[0, 1, 1, 1] = beta[1, 0, 1, 1] = 0.90
        rng = np.random.default_rng(1)
        sites = np.array([[0, 0], [0, 1]], dtype=np.uint8)
        run_system(
            1,
            sites,
            rng.bit_generator.capsule,
            beta=beta,
            num_steps=1,
            report_every=1,
            neighborhood=neighborhood
        )
        self.assertTrue((sites == np.array([[0, 0], [0, 1]])).all())

    def test_diffusion(self):
        # RNGs return different values on Windows, so tests will not be valid.
        if sys.platform != 'linux':
            return

        # Set rules so that diffusion probability is 2/3, and the non-diffusion
        # reaction is abiogenesis.
        rules = np.zeros((2, 2, 2, 2))
        rules[0, 1, 1, 0] = rules[1, 0, 0, 1] = 2
        rules[0, 0, 0, 1] = rules[0, 0, 1, 0] = 0.5

        # Test diffusion actually occurs. The first roll will be 0.636, so with
        # diffusion probability of 2/3, there will be two diffusion events
        # before any other event. The two diffusion events will be:
        #
        # (1, 0) <-> (2, 0)
        # (0, 2) <-> (3, 2)
        sites = np.zeros((4, 4), dtype=np.uint8)
        sites[1, 0] = sites[0, 2] = 1

        rng = np.random.default_rng(0)
        geo = viridicle.LatticeGeography(sites, rules, rng)

        geo.run(num_steps=2)

        should_be = np.zeros((4, 4), dtype=np.uint8)
        should_be[2, 0] = should_be[3, 2] = 1
        self.assertTrue((geo.sites == should_be).all(), geo.sites)

        # Now test the case that neither site in the diffusion event has any
        # contents.
        sites = np.zeros((4, 4), dtype=np.uint8)
        rng = np.random.default_rng(0)
        geo = viridicle.LatticeGeography(sites, rules, rng)
        geo.run(num_steps=2)

        self.assertTrue((geo.sites == 0).all(), geo.sites)

        # And test that non-diffusion events can occur. The first edge that
        # will be selected after the two diffusion events is, again:
        #
        # (0, 2) <-> (3, 2)
        #
        # The roll is very small.
        sites = np.zeros((4, 4), dtype=np.uint8)
        rng = np.random.default_rng(0)
        geo = viridicle.LatticeGeography(sites, rules, rng)
        geo.run(num_steps=3)

        should_be = np.zeros((4, 4), dtype=np.uint8)
        should_be[0, 2] = 1

        self.assertTrue((geo.sites == should_be).all(), geo.sites)


class ArbitraryGraphTest(unittest.TestCase):
    '''
    Tests arbitrary graphs by running them through a single iteration with an
    RNG that will produce known values.
    '''
    def test_init(self):
        # Tests that the geography will end up with a generator if we supply it
        # with a valid but non-generator input
        sites = nx.Graph()
        sites.add_nodes_from([0, 1, 2, 3])
        sites.add_edges_from([(0, 1), (1, 2), (2, 3)])
        for idx, node in enumerate(sites.nodes):
            sites.nodes[node]['state'] = [0, 0, 1, 1][idx]

        beta = np.zeros((2, 2, 2, 2))
        beta[1, 1, 0, 0] = 1.0

        geo = viridicle.ArbitraryGeography(sites, beta, state_key='state')
        self.assertTrue(isinstance(geo.generator, np.random.Generator))

        geo = viridicle.ArbitraryGeography(sites, beta, 0, state_key='state')
        self.assertTrue(isinstance(geo.generator, np.random.Generator))

    def test_encoding(self):
        geo = _get_arbitrary_graph()

        params = geo.encode()
        self.assertTrue((params['sites'] == np.array([0, 1, 0, 1])).all())
        edge_idxs, edges = params['edge_idxs'], params['edges']
        self.assertTrue((edge_idxs == np.array([0, 1, 3, 5, 6])).all())
        self.assertTrue((edges == np.array([1, 0, 2, 1, 3, 2])).all())

        params['sites'][1:3] = [1, 0]
        sites = geo.decode(params['sites'])
        self.assertTrue(sites.nodes[0]['state'] == 0)
        self.assertTrue(sites.nodes[1]['state'] == 1)
        self.assertTrue(sites.nodes[2]['state'] == 0)
        self.assertTrue(sites.nodes[3]['state'] == 1)

    def test_python_constant_degree(self):
        '''
        With the graph structure returned by _get_constant_degree_graph, the
        expected rolls are:

        edge_idx:           7
        site_idx_1:         3
        site_idx_2:         0
        roll:               0.95

        The graph has initial node states:

            [0, 1, 0, 1]
        '''
        # RNGs return different values on Windows, so tests will not be valid.
        if sys.platform != 'linux':
            return
        # Test transition will occur if probability is high enough
        beta = np.zeros((2, 2, 2, 2))
        beta[0, 1, 1, 1] = beta[1, 0, 1, 1] = 0.96
        # Ensure maximum total transition is 1.0 to prevent rescaling.
        beta[1, 1, 0, 0] = 1.0
        geo = _get_constant_degree_graph(beta)
        geo.run(1 / 8.0)

        states = tuple((geo.sites.nodes[n]['state'] for n in range(4)))
        self.assertTrue(states == (1, 1, 0, 1), states)

        # Test transition will not occur if probability is not high enough
        beta = np.zeros((2, 2, 2, 2))
        beta[0, 1, 1, 1] = beta[1, 0, 1, 1] = 0.95
        # Ensure maximum total transition is 1.0 to prevent rescaling.
        beta[1, 1, 0, 0] = 1.0
        geo = _get_constant_degree_graph(beta)
        geo.run(1 / 8.0)

        states = tuple((geo.sites.nodes[n]['state'] for n in range(4)))
        self.assertTrue(states == (0, 1, 0, 1), states)


class RuleEncodingTest(unittest.TestCase):
    def test_encoding(self):
        # Basic case
        rules = _encode_rule('1,2->2,1@0.1')
        self.assertTrue((1, 2, 2, 1, 0.1) in rules)

        # Wildcard unpacking
        rules = _encode_rule('*,1->0,1@0.1', num_states=3)
        self.assertTrue(len(rules) == 3)
        self.assertTrue((0, 1, 0, 1, 0.1) in rules)
        self.assertTrue((1, 1, 0, 1, 0.1) in rules)
        self.assertTrue((2, 1, 0, 1, 0.1) in rules)

        # Error handling
        with self.assertRaises(ValueError):
            _encode_rule('*,1->0,1@0.1')


class MergingTest(unittest.TestCase):
    def test_c_layer(self):
        nhd = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        sites = np.array([
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 2, 0, 1],
            [0, 0, 2, 1]
        ], dtype=np.uint8)
        correct_sites = sites.copy()

        merge_small(1, sites, 0, 0, 255, neighborhood=nhd)
        self.assertTrue(_array_equals(sites, correct_sites), sites)

        correct_sites = np.array([
            [0, 0, 0, 1],
            [0, 255, 0, 1],
            [0, 255, 0, 1],
            [0, 0, 255, 1]
        ], dtype=np.uint8)
        orig_sites = sites.copy()
        merge_small(1, sites, 1, 1, 255, neighborhood=nhd)
        self.assertTrue(_array_equals(sites, correct_sites), sites)

        correct_sites = np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 255, 1]
        ], dtype=np.uint8)
        sites = orig_sites.copy()
        merge_small(1, sites, 2, 2, 255, neighborhood=nhd)
        self.assertTrue(_array_equals(sites, correct_sites), sites)

    def test_python_layer(self):
        rules = ['0,1->1,2@0.1', '1,1->2,2@0.2']
        orig_sites = np.array([
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 2, 0, 1],
            [0, 0, 2, 1]
        ], dtype=np.uint8)

        correct_sites = orig_sites.copy()
        geo = viridicle.LatticeGeography(
            orig_sites.copy(), rules, num_states=3
        )
        new_geo = viridicle.merge_small_clusters(geo, 0, empty_state=255)
        self.assertTrue(
            _array_equals(new_geo.sites, correct_sites), new_geo.sites
        )
        self.assertTrue(_array_equals(new_geo.neighborhood, geo.neighborhood))
        self.assertTrue(new_geo.sites is not geo.sites)
        self.assertTrue(_array_equals(new_geo.beta, geo.beta))
        self.assertTrue(new_geo.beta is not geo.beta)

        correct_sites = np.array([
            [0, 0, 0, 1],
            [0, 255, 0, 1],
            [0, 255, 0, 1],
            [0, 0, 255, 1]
        ], dtype=np.uint8)
        geo = viridicle.LatticeGeography(
            orig_sites.copy(), rules, num_states=3
        )
        new_geo = viridicle.merge_small_clusters(geo, 1, empty_state=255)
        self.assertTrue(
            not _array_equals(geo.sites, new_geo.sites), new_geo.sites
        )
        self.assertTrue(
            _array_equals(new_geo.sites, correct_sites), new_geo.sites
        )

        correct_sites = np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 255, 1]
        ], dtype=np.uint8)
        geo = viridicle.LatticeGeography(
            orig_sites.copy(), rules, num_states=3
        )
        new_geo = viridicle.merge_small_clusters(geo, 2, empty_state=255)
        self.assertTrue(
            _array_equals(new_geo.sites, correct_sites), new_geo.sites
        )


class ClusterTest(unittest.TestCase):
    def test_c_layer(self):
        nhd = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        sites = np.array([
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ], dtype=np.uint8)
        correct_clusters = np.array([
            [0, 0, 0, 1],
            [0, 2, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ], dtype=np.uint64)
        correct_map = np.array([0, 1, 1], dtype=np.uint8)

        clusters, cluster_map = cluster_geo(1, sites, neighborhood=nhd)

        self.assertTrue(_array_equals(clusters, correct_clusters), clusters)
        self.assertTrue(_array_equals(cluster_map, correct_map), cluster_map)

    def test_python_layer(self):
        sites = np.array([
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ], dtype=np.uint8)
        rules = ['0,1->1,2@0.1', '1,1->2,2@0.2']
        orig_sites = sites.copy()

        geo = viridicle.LatticeGeography(sites, rules, num_states=3)
        correct_clusters = np.array([
            [0, 0, 0, 1],
            [0, 2, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ], dtype=np.uint64)
        correct_map = np.array([0, 1, 1], dtype=np.uint8)
        clusters, cluster_map = viridicle.cluster_geography(geo)
        self.assertTrue(_array_equals(geo.sites, orig_sites))
        self.assertTrue(_array_equals(clusters, correct_clusters), clusters)
        self.assertTrue(_array_equals(cluster_map, correct_map), cluster_map)


class ClusterGrowthTest(unittest.TestCase):
    def test_grow_clusters(self):
        sites = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 2],
            [0, 0, 0, 0]
        ], dtype=np.uint8)
        rules = ['0,1->1,2@0.1', '1,1->2,2@0.2']
        geo = viridicle.LatticeGeography(sites, rules, num_states=3)
        orig_sites = sites.copy()

        new_geo = viridicle.grow_clusters(geo, 1)
        new_sites = np.array([
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [2, 1, 0, 2],
            [0, 0, 0, 2]
        ], dtype=np.uint8)
        self.assertTrue(_array_equals(geo.sites, orig_sites), geo.sites)
        self.assertTrue(_array_equals(new_geo.sites, new_sites), new_geo.sites)

        new_geo = viridicle.grow_clusters(geo, 2)
        new_sites = np.array([
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [2, 1, 0, 2],
            [2, 1, 0, 2]
        ], dtype=np.uint8)
        self.assertTrue(_array_equals(new_geo.sites, new_sites), new_geo.sites)

        new_geo = viridicle.grow_clusters(geo, 3)
        self.assertTrue(_array_equals(new_geo.sites, new_sites), new_geo.sites)


def _array_equals(a_1, a_2):
    if a_1.shape != a_2.shape:
        return False
    if a_1.dtype != a_2.dtype:
        return False
    return (a_1 == a_2).all()


def _get_arbitrary_graph(beta=None):
    '''
    Generates an arbitrary graph with structure:

        0 - 1 - 2 - 3
    '''
    if beta is None:
        beta = np.zeros((2, 2, 2, 2))
        beta[1, 1, 0, 0] = 1.0
    sites = nx.Graph()
    sites.add_nodes_from([0, 1, 2, 3])
    sites.add_edges_from([(0, 1), (1, 2), (2, 3)])
    for idx, node in enumerate(sites.nodes):
        sites.nodes[node]['state'] = idx % 2

    return viridicle.ArbitraryGeography(sites, beta, 1, state_key='state')


def _get_constant_degree_graph(beta=None):
    '''
    This generates a circular graph:

        0 - 1
        |   |
        3 - 2
    '''
    if beta is None:
        beta = np.ones((2, 2, 2, 2))
        beta[1, 1, 0, 0] = 1.0
    sites = nx.Graph()
    sites.add_nodes_from([0, 1, 2, 3])
    sites.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    for idx, node in enumerate(sites.nodes):
        sites.nodes[node]['state'] = idx % 2

    return viridicle.ArbitraryGeography(sites, beta, 1, state_key='state')


if __name__ == '__main__':
    if sys.platform != 'linux':
        print(
            'Warning: tests using RNGs are only run on Linux, as the RNGs '
            'will generate different, unexpected values. Those tests will be '
            'skipped.'
        )
    unittest.main()
