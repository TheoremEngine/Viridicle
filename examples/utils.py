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
Contains generic tooling for running multiple viridicle experiments in
parallel.
'''
import argparse
import os
import time
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm
import multiprocessing as mp
import numpy as np
from PIL import Image
import viridicle


CoercableToGenerator = Union[int, np.random.Generator]


def devnull(*args, **kwargs):
    '''
    Dummy function, which replaces make_gif when example scripts are run in
    benchmarking mode.
    '''
    return


def get_block_initialization(num_states: int, shape: Sequence[int],
                             rng: Optional[CoercableToGenerator] = None) \
        -> np.ndarray:
    '''
    Constructs an initialization state for a 2-dimensional lattice using blocks
    instead of random initialization. The blocks are then placed randomly, and
    will not overlap. This typically leads to greater likelihood of species
    coexistence than a random initialization.

    Args:
        num_states (int): Number of system states. We assume state 0 is empty
        and the number of species equals (num_states - 1).
        shape (sequence of ints): Shape of the lattice.
        rng (optional, int or :class:`numpy.random.Generator`): The random
        number generator to use. Integers will be used as a seed for the
        default RNG.

    Returns:
        The :class:`numpy.ndarray` containing the initial lattice state.
    '''
    if len(shape) != 2:
        raise ValueError('Block init only supports 2 dimensions.')
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    height, width = shape
    sites = np.zeros((height, width), dtype=np.uint8)
    b_h = height // (4 * (num_states - 1))
    b_w = width // (4 * (num_states - 1))
    ul_x, ul_y = rng.integers(0, (width - b_w, height - b_h))

    for sp in range(1, num_states):
        while sites[ul_y:ul_y + b_h, ul_x:ul_x + b_w].any():
            ul_x, ul_y = rng.integers(0, (width - b_w, height - b_h))
        sites[ul_y:ul_y + b_h, ul_x:ul_x + b_w] = sp

    return sites


def get_colormap(num_states: int = 3, include_empty: bool = True,
                 zero_is_clear: bool = False) -> matplotlib.colors.Colormap:
    '''
    Generates viridis color map for making pretty images.

    Args:
        num_states (int): Number of system states.
        include_empty (bool): Treat the zero state as an "empty" state whose
        color is white or transperant.
        zero_is_clear (bool): Render white as transperant.

    Returns:
        A :class:`matplotlib.colors.Colormap`.
    '''
    num_states -= include_empty
    cmap = matplotlib.cm.get_cmap('viridis', num_states)
    colors = cmap(np.linspace(0, 1, num_states))
    if include_empty:
        a = 0 if zero_is_clear else 1
        colors = np.concatenate((np.array([[1, 1, 1, a]]), colors))
    return matplotlib.colors.ListedColormap(colors)


def get_default_arg_parser() -> argparse.ArgumentParser:
    '''
    Constructs an argparse.ArgumentParser with arguments used by all of the
    various example scripts.

    Returns:
        The :class:`argparse.ArgumentParser` with common arguments added.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-experiments', type=int, default=1,
                        help='Number of experiments to run')
    parser.add_argument('--width', type=int, default=256,
                        help='Width of lattice')
    parser.add_argument('--elapsed-time', type=float, default=1000,
                        help='Runtime for system')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of worker processes')
    parser.add_argument('--output-path', type=str, default='.',
                        help='Path to write output to')
    parser.add_argument('--random-seeds', type=int, nargs='+', default=None,
                        help='Random seeds to use - defaults to 0, 1, ...')
    parser.add_argument('--report-every', type=float, default=1.0,
                        help='How often to take a snapshot')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run for runtime benchmarking - do not generate '
                             'output file(s)')
    return parser


def get_neighborhood(radius: float) -> np.ndarray:
    '''
    Builds a neighborhood array including all entries of within a specified
    radius, excluding the site itself.

    Args:
        radius (float): Radius of neighborhood in Euclidean norm.

    Returns:
        A :class:`numpy.ndarray` giving the offsets for a neighborhood of the
        specified radius.
    '''
    neighborhood = np.stack(np.meshgrid(
        *([np.arange(-int(radius), int(radius) + 1)] * 2)
    ), axis=2).reshape(-1, 2)
    neighborhood = neighborhood[(neighborhood**2).sum(1) > 0, :]
    neighborhood = neighborhood[(neighborhood**2).sum(1) <= radius**2, :]

    return neighborhood


def make_gif(geo: viridicle.Geography, out: np.ndarray, path: str,
             zero_is_clear: bool = True):
    '''
    Generates a gif from the output of a Viridicle run. The signature is
    required by the worker function.

    Args:
        geo (:class:`Geography`): The system used in the run.
        out (:class:`numpy.ndarray` or pair of :class:`numpy.ndarray`): The
        values returned by the run method.
        naming the gif; if not provided, a UUID will be used.
        path (str): The prefix for the file; the extension '.gif' will be
        appended.
        zero_is_clear (bool): Whether to treat state 0 as empty.
    '''
    # We only want the sites record, not the counts.
    if isinstance(out, tuple):
        out, _ = out

    path = path + '.gif'
    if os.path.exists(path):
        raise FileExistsError(path)
    # Ensure output_path exists to receive output file.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    colors = get_colormap(geo.num_states, zero_is_clear=zero_is_clear)

    # Construct the gif. The dimensions of gif will be THWC, where T is the
    # time, H and W are space, and C is the color.
    t, h, w = out.shape
    # Fill in the gif
    gif = colors(out)
    gif = (256 * gif).clip(min=0.0, max=255.0).astype(np.uint8)

    # Construct the gif as a file. This is based on:
    #
    # https://note.nkmk.me/en/python-pillow-gif
    images = [Image.fromarray(_gif) for _gif in gif]
    images[0].save(path, save_all=True, append_images=images[1:])


def run(graph_class: type, init_params: Dict, run_params: Dict,
        output_func: Callable, output_path: str, num_workers: int,
        num_experiments: int, random_seeds: Optional[Sequence[int]] = None):
    '''
    Organizes and performs a set of experimental runs.

    Args:
        graph_class (type): Specifies class of graph to use.
        init_params (dict): Parameters to pass to the :class:`Geography` during
        initialization. Note that generator, if present, will be overwritten.
        run_params (dict): Parameters to pass to geography when calling the run
        method.
        output_func (callable): Function to pass outputs to. This will receive
        as input a tuple (geo, out, seed, path), where geo out is the
        :class:`Geography` after the run, rtn is the values returned by the
        run, seed is the integer random number seed, and path is a path to
        write output files to.
        output_path (str): Root directory to write results to.
        num_workers (int): Number of worker processes to use.
        num_experiments (int): Number of experiments to run.
        random_seeds (optional, sequence of ints): Random seeds to use.
    '''
    start_time = time.perf_counter()

    # Set up default random seeds if not set
    if random_seeds is None:
        random_seeds = list(range(num_experiments))
    elif len(random_seeds) != num_experiments:
        raise ValueError(
            'Number of random seeds does not match number of experiments: '
            f'{len(random_seeds)} vs. {num_experiments}.'
        )

    def make_args(rank):
        '''
        Constructs the arguments to be passed to the worker function.
        '''
        _init = init_params.copy()
        _init['generator'] = random_seeds[rank]
        path = os.path.join(output_path, str(random_seeds[rank]))
        return (graph_class, _init, run_params, output_func, path)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            pool.map(worker, [make_args(i) for i in range(num_experiments)])
    else:
        for i in range(num_experiments):
            worker(make_args(i))

    elapsed_time = time.perf_counter() - start_time

    print(f'Finished - Elapsed Time: {elapsed_time} sec')


def worker(args: Tuple[type, Dict, Dict, Callable, str]):
    '''
    This function handles actually running the experiment and then collecting
    and saving the outputs. Due to the mp.Pool.map API, it must take a single
    argument, args. We pass it a tuple of arguments, which we then unpack:

        graph_class (type): Specifies the type of graph to use.
        init_params (dict): Parameters to pass to Geography during
        initialization.
        run_params (dict): Parameters to pass to :class:`Geography` during run.
        output_func (callable): Function to pass outputs to. This will receive
        as input a tuple (geo, out, seed, path), where geo out is the
        :class:`Geography` after the run, rtn is the values returned by the
        run, seed is the integer random number seed, and path is a path to
        write output files to.
        output_path (str): Path to write results to, if any.
    '''
    graph_class, init_params, run_params, output_func, output_path = args

    try:
        n = init_params['num_states']
    except KeyError:
        n = init_params['rules'].shape[0]

    # Initialize the lattice.
    init = init_params.pop('initialization', 'random')
    if init == 'block':
        init_params['sites'] = get_block_initialization(
            n, init_params['sites'], init_params.get('generator', None)
        )
    elif init == 'random':
        pass
    else:
        raise ValueError(f'Did not recognize initialization {init}.')

    geo = graph_class(**init_params)
    rtn = geo.run(**run_params)
    output_func(geo, rtn, output_path)
