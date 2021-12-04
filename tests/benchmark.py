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
Runs simple benchmarks to test the effect of changes to the code on runtime
performance. Benchmark times are from running on an AWS EC2 c6g.4xlarge.
'''
# name=Viridicle-benchmark
# image=viridicle
# size=c6g.4xlarge
# maximum_runtime=24:00:00

from copy import deepcopy
import multiprocessing as mp
import time
from typing import Dict

from tabulate import tabulate
import viridicle

TEST_CASES = [
    {
        'width': 128,
        'diffusion-rate': 1.0,
        # 1.8 sec
    }, {
        'width': 128,
        'diffusion-rate': 10.0,
        # 9.7 sec
    }, {
        'width': 256,
        'diffusion-rate': 1.0,
        # 16.9 sec
    }, {
        'width': 256,
        'diffusion-rate': 10.0,
        # 134.5 sec
    }, {
        'width': 512,
        'diffusion-rate': 1.0,
        # 95.2 sec
    }, {
        'width': 512,
        'diffusion-rate': 10.0,
        # 739.8 sec
    }
]
# Units of time for each experiment
ELAPSED_TIME = 1000.0
# Number of experiments to run per combination of parameters
NUM_EXPERIMENTS = 256
# Number of workers in multi-threaded experiments
NUM_WORKERS = 16


def worker(params: Dict):
    shape = (params['width'], params['width'])
    beta = viridicle.may_leonard_rules(3, 1, params['diffusion-rate'])
    geo = viridicle.LatticeGeography(shape, beta, params['seed'])
    geo.run(elapsed_time=ELAPSED_TIME, return_counts=False)


def add_seed(params: Dict, seed: int) -> Dict:
    params = deepcopy(params)
    params['seed'] = seed
    return params


if __name__ == '__main__':
    origin_time = time.perf_counter()
    exps_per_worker = NUM_EXPERIMENTS / NUM_WORKERS

    for test_case in TEST_CASES:
        start_time = time.perf_counter()

        if NUM_WORKERS > 1:
            exps = [add_seed(test_case, i) for i in range(NUM_EXPERIMENTS)]
            with mp.Pool(NUM_WORKERS) as pool:
                pool.map(worker, exps)
        else:
            for i in range(NUM_EXPERIMENTS):
                worker(test_case, i)

        runtime = (time.perf_counter() - start_time) / exps_per_worker
        test_case['Runtime'] = f'{runtime} sec'

    total_time = time.perf_counter() - origin_time

    print(tabulate(TEST_CASES, headers='keys'))

    print('Total Elapsed Time:', total_time, 'sec')
