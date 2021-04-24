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
Performs experiments using the Lotka-Volterra competition model from:

    Avelino, Bazeia, Menezes, and de Oliveira. "String Networks in Z_N Lotka-
    Volterra Competition Models." Physics Letters A Vol. 378 No. 4, pp.
    393-397. https://arxiv.org/abs/1305.4253v2

This is similar to the May-Leonard model, but with symmetric competition. The
model allows empty sites and up to N species, which interact with the following
reactions:

    1. Birth: Species may give birth into empty sites, i0->ii, at rate 0.1.
    2. Mobility: Two sites may swap contents, ij->ji, at rate 0.15.
    3. Predation: Each species i preys on species (i + 2), (i + 2), ...,
       (i + N - 2), where addition is modulo N - so in a 5-species model,
       species 3 preys on species 1 and 5. The predation reaction has form
       ij->i0, at rate 0.75.

In addition, this model is examined in both 2- and 3-dimensional lattices.
'''
from itertools import product

import viridicle

import utils


if __name__ == '__main__':
    parser = utils.get_default_arg_parser()
    parser.add_argument('--num-species', type=int, default=4,
                        help='Number of species in the system')
    parser.add_argument('--dimension', type=int, default=2,
                        help='Dimension of the lattice')
    args = parser.parse_args()

    n = args.num_species

    # Predation
    rules = [
        f'{i},{((i + j - 1) % n) + 1}->{i},0@0.75'
        for i in range(1, n + 1) for j in range(2, n - 1)
    ]
    # Birth rate
    rules += [f'{i},0->{i},{i}@0.1' for i in range(1, n + 1)]
    # Mobility
    rules += [
        f'{i},{j}->{j},{i}@0.15'
        for i, j in product(*([range(n + 1)] * 2))
        if (i != j)
    ]

    init_params = {
        'sites': (args.width,) * args.dimension,
        'rules': rules,
        'num_states': args.num_species + 1,
    }
    run_params = {
        'elapsed_time': args.elapsed_time,
        'report_every': args.report_every,
        'return_sites': True,
        'return_counts': False,
    }
    utils.run(
        graph_class=viridicle.LatticeGeography,
        init_params=init_params,
        run_params=run_params,
        output_func=(utils.devnull if args.benchmark else utils.make_gif),
        output_path=args.output_path,
        num_workers=args.num_workers,
        num_experiments=args.num_experiments,
        random_seeds=args.random_seeds,
    )
