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
Performs experiments using the generalized May-Leonard system on a lattice that
is a discretization of a topological 2-sphere.
'''

import viridicle

import graphs
import utils


if __name__ == '__main__':
    parser = utils.get_default_arg_parser()
    parser.add_argument('--type', type=int, nargs=2, default=(3, 1),
                        help='Type of May-Leonard system')
    parser.add_argument('--diffusion-rate', type=float, default=0.2,
                        help='Rate of diffusion in the lattice')
    parser.add_argument('--initialization', type=str, default='random',
                        choices=['block', 'random'],
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
        graph_class=graphs.SphericalGeography,
        init_params=init_params,
        run_params=run_params,
        output_func=(utils.devnull if args.benchmark else utils.make_gif),
        output_path=args.output_path,
        num_workers=args.num_workers,
        num_experiments=args.num_experiments,
        random_seeds=args.random_seeds,
    )
