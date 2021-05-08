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
Performs experiments using the spatial May & Leonard model for non-transitive
competition. The original non-spatial version of this model is from:

    May and Leonard. "Nonlinear Aspects of Competition Between Three Species."
    SIAM Journal on Applied Mathematics Vol. 29 No. 2, pp. 243-253.

This script implements the spatial generalization of the above model found in:

    Ahmed, Debanjan, and Pleimling. "Interplay Between Partnership Formation
    and Competition in Generalized May-Leonard Games." Physical Review E Vol.
    87 No. 3. https://arxiv.org/abs/1303.3139

An (N, r) May-Leonard model has (N + 1) possible states: state 0 denotes an
empty site while 1, 2, ..., N denote different species. We have the following
possible transitions:

    1. Birth: Species may give birth into empty sites, i0->ii, at rate 0.2.
    2. Mobility: Two sites may swap contents, ij->ji, at rate mu, where mu is a
       hyperparameter.
    3. Predation: Each species i preys on species (i + 1), (i + 2), ...,
       (i + r), where addition is modulo N - so in a (3, 1) model, species 3
       preys on species 1. The predation reaction has form ij->i0, at rate 0.2.
       Note that it is possible for two species to both prey on each other.

In the full generalization the rates for each transition are independently
varied, but we restrict to a symmetric system to keep the number of
hyperparameters tractable. The original paper uses distinct diffusion rates for
interacting and non-interacting species; we use a constant rate but treat this
as a hyperparameter. In this form, the Reichnebach, Mobilia, and Frey model is
the (3, 1) May-Leonard model. There is also an alternative model that does not
allow empty sites; in this case there is no birth reaction, and the predation
reaction replaces the prey with an instance of the predator, ij->ii.

This script runs multiple iterations of these experiments in parallel, to
generate large amounts of results quickly, and save the output in a binary npy
format. It also generates png files containing the final system state and
graphs of system state over time.
'''

import viridicle

import utils


if __name__ == '__main__':
    parser = utils.get_default_arg_parser()
    parser.add_argument('--type', type=int, nargs=2, default=(3, 1),
                        help='Type of May-Leonard system')
    parser.add_argument('--diffusion-rate', type=float, default=0.2,
                        help='Rate of diffusion in the lattice')
    parser.add_argument('--initialization', type=str, default='random',
                        choices=['block', 'random', '1line', '2line'],
                        help='Choice of system initialization method')
    args = parser.parse_args()

    n, r = args.type
    beta = viridicle.may_leonard_rules(n, r, args.diffusion_rate)

    init_params = {
        'sites': (args.width, args.width),
        'rules': beta,
        'initialization': args.initialization,
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
