#!/usr/bin/python3

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
This provides a simple Python implementation of the May-Leonard system for
benchmarking purposes. Benchmark times are from running on an AWS EC2
c6g.4xlarge.
'''
# name=Viridicle-benchmark
# image=viridicle
# size=c6g.4xlarge
# maximum_runtime=96:00:00

from itertools import repeat
import multiprocessing as mp
import time

import numpy as np
from tabulate import tabulate

# Total time for test: 14411 sec at 100 time

ELAPSED_TIME = 1000
TEST_CASES = [
    {
        'width': 256,
        'diffusion-rate': 1.0,
        # 1820.9 sec
    }, {
        'width': 128,
        'diffusion-rate': 10.0,
        # 3897.7 sec
    }, {
        'width': 128,
        'diffusion-rate': 1.0,
        # 460.1 sec
    }
]

# Number of experiments to run per combination of parameters
NUM_EXPERIMENTS = 256
# Number of workers in multi-threaded experiments
NUM_WORKERS = 16


# Offsets for von Neumann neighborhood
NHD = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
# If we randomly sample a directed edge (v_0, v_1) when we have rolled a
# reaction, SHIFT[s_1, s_2, i] gives the state that vertex v_i will shift to.
SHIFT = np.empty((4, 4, 2))
# Both vertices are the same state and no reaction occurs
SHIFT[0, 0, :], SHIFT[1, 1, :], SHIFT[2, 2, :], SHIFT[3, 3, :] = 0, 1, 2, 3
# Reproduction reactions
SHIFT[0, 1, :], SHIFT[0, 2, :], SHIFT[0, 3, :] = 1, 2, 3
SHIFT[1, 0, :], SHIFT[2, 0, :], SHIFT[3, 0, :] = 1, 2, 3
# Predation reactions
SHIFT[3, 1, :], SHIFT[1, 2, :], SHIFT[2, 3, :] = (3, 0), (1, 0), (2, 0)
SHIFT[1, 3, :], SHIFT[2, 1, :], SHIFT[3, 2, :] = (0, 3), (0, 1), (0, 2)


def run(width: int, diffusion_rate: float, elapsed_time: float):
    '''
    A simple implementation of the (3, 1) May-Leonard system in pure Python on
    a periodic lattice. Since this is used only for benchmarking, the function
    returns nothing.

    Args:
        width (int): Width of the lattice.
        diffusion_rate (float): Diffusion rate (not mobility).
        elapsed_time (float): Runtime of system.
    '''
    rng = np.random.default_rng(0)

    sites = rng.integers(0, 4, (width, width))
    n_steps = int(elapsed_time * sites.size * (diffusion_rate + 0.2))

    # We use the diffusion trick from:
    #
    # Reichenbach, Mobilia, and Frey. "Self-Organization of Mobile Populations
    # in Cyclic Competition." Journal of Theoretical Biology, Vol. 254 No. 2,
    # pp. 368-383.
    #
    # To improve runtime. They make the observation that, in a system with high
    # diffusion, most reactions will be diffusion reactions. If we calculate
    # the probability that the next reaction will be a diffusion reaction, then
    # the number of diffusion reactions before the next non-diffusion reaction
    # will be a geometric random variable. So generate that number, perform
    # that many reactions, then perform a single non-diffusion reaction, and
    # regenerate the number of diffusions. In our case, this is especially
    # efficient because, for every pair of possible states, we have only 0 or 1
    # possible non-diffusion reactions, so there's no need to sample another
    # random variable to determine what happens.

    # The maximum total rate for any pair of states is diffusion_rate + 0.2.
    non_diff_prob = 0.2 / (diffusion_rate + 0.2)
    n_diffs = rng.geometric(non_diff_prob)

    for _ in range(n_steps):
        # Select the directed edge
        x_1, y_1 = rng.integers(0, width, (2,))
        d_x, d_y = NHD[rng.integers(0, 4)]
        x_2, y_2 = (x_1 + d_x) % width, (y_1 + d_y) % width

        # If we're performing a diffusion operation, then swap the sites'
        # contents.
        if n_diffs:
            sites[x_1, y_1], sites[x_2, y_2] = sites[x_1, y_1], sites[x_2, y_2]
        # Otherwise, a) regenerate the number of diffusions and b) perform the
        else:
            n_diffs = rng.geometric(non_diff_prob)
            sites[x_1, y_1], sites[x_2, y_2] = \
                SHIFT[sites[x_1, y_1], sites[x_2, y_2]]


if __name__ == '__main__':
    origin_time = time.perf_counter()
    exps_per_worker = NUM_EXPERIMENTS / NUM_WORKERS

    # Multi-process runs
    for test_case in TEST_CASES:
        args = (test_case['width'], test_case['diffusion-rate'], ELAPSED_TIME)
        args = repeat(args, NUM_EXPERIMENTS)
        start_time = time.perf_counter()
        with mp.Pool(NUM_WORKERS) as pool:
            pool.starmap(run, args)
        mean_runtime = (time.perf_counter() - start_time) / exps_per_worker
        test_case['Runtime'] = f'{mean_runtime} sec'

    total_time = time.perf_counter() - origin_time

    print(tabulate(TEST_CASES, headers='keys'))

    print('Total Elapsed Time:', total_time, 'sec')
