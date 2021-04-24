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

from itertools import chain
import os
import subprocess
import time

from tabulate import tabulate

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
# Common argments used in all test cases.
COMMON_ARGUMENTS = {
    'initialization': 'random',
    'elapsed-time': 1000,
}
# Number of experiments to run per combination of parameters
NUM_EXPERIMENTS = 256
# Number of workers in multi-threaded experiments
NUM_WORKERS = 16
# Install location of Viridicle on AWS image.
OPT_PATH = '/opt/viridicle'


if __name__ == '__main__':
    origin_time = time.perf_counter()

    # Locate run_may_and_leonard script.
    if os.path.exists(OPT_PATH):
        script_root = os.path.join(OPT_PATH, 'examples')
    else:
        script_root = os.path.join(os.path.dirname(__file__), '../examples')
    script_path = os.path.join(script_root, 'run_may_and_leonard.py')

    # Construct command
    command = [
        'python3', script_path, '--benchmark',
        '--num-workers', str(NUM_WORKERS),
        '--num-experiments', str(NUM_EXPERIMENTS)
    ]
    command += list(chain(*([f'--{key}', str(value)] for key, value in
                            COMMON_ARGUMENTS.items())))

    exps_per_worker = NUM_EXPERIMENTS / NUM_WORKERS

    for test_case in TEST_CASES:
        start_time = time.perf_counter()
        subprocess.run(
            command + list(chain(*([f'--{key}', str(value)]
                                   for key, value in test_case.items()
                                   if ' ' not in key))),
            check=True,
            cwd=script_root,
            stdout=subprocess.DEVNULL
        )
        runtime = (time.perf_counter() - start_time) / exps_per_worker
        test_case['Runtime'] = f'{runtime} sec'

    total_time = time.perf_counter() - origin_time

    print(tabulate(TEST_CASES, headers='keys'))

    print('Total Elapsed Time:', total_time, 'sec')
