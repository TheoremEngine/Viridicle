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
Performs experiments using the Durret & Levin model for E. Coli bacteria, as
found in:

    Durrett and Levin. "Allelopathy in Spatially Distributed Populations."
    Journal of Theoretical Biology Vol. 185, pp. 165-171.

In this model, we have four states. The 0 state represents an empty point,
while the remaining states are different strains for E. Coli bacteri:

    1: A strain producing the antibiotic colicin. Often called P, for Producer.
    2: A strain that produces a small amount of colicin; it reproduces faster
       since it is not investing as much energy in production. Often called R,
       for Resistant.
    3: A strain that is vulnerable to colicin, but which reproduces much faster
       than the other two strains. Often called S, for susceptible.

These form a rock-paper-scissors model: strain R outcompetes strain P because
it is resistant to the colicin and reproduces faster; strain S outcompetes
strain R because strain R does not produce enough colicin to stop strain S,
which reproduces faster than strain R; and strain P outcompetes strain S
because it produces enough colicin to kill it.

This script runs multiple iterations of these experiments in parallel, to
generate large amounts of results quickly, and save the output in a binary npy
format.
'''
import viridicle

import utils

# In this system, state 1 is a colicin-producing strain of E. Coli; state 2 is
# a colicin-resistant strain of E. Coli; and state 3 is a colicin-vulnerable
# strain of E. Coli. State 1 outcompetes state 3 because the colicin kills the
# vulnerable bacteria; state 2 outcompetes state 1 because it is not effected
# by the antibiotic but is not investing resources in its production; and state
# 3 outcompetes state 2 because it is not investing in colicin resistance.
# State 2 also produces a small amount of colicin, so it also has a small
# effect on the death rate of state 3, but much smaller.

# Birth rates 0->i are beta_if_i, where:
#
# beta_1 = 3
# beta_2 = 3.2
# beta_3 = 4
#
# These can be turned into reactions i0->ii with rate equal to beta_i.
rules = ['1,0->1,1@3.0', '2,0->2,2@3.2', '3,0->3,3@4.0']

# Death rates i->0 are delta_i for i = 1, 2. Death rate 3->0 is:
#
#    delta_3 + gamma_1f_1 + gamma_2f_2
#
#
# delta_1 = delta_2 = delta_3 = 1.0
# gamma_1 = 3.0
# gamma_2 = 0.5
#
# We encode the death rates for 1, 2 as reactions i,*->0,*. We encode the
# death rate for 3 as both a reaction 3,*->0,* and a separate pair of
# reactions 31->01, 32->02.
rules += [
    '1,*->0,*@1.0', '2,*->0,*@1.0', '3,*->0,*@1.0',
    '3,1->0,1@3.0', '3,2->0,2@0.5'
]

if __name__ == '__main__':
    parser = utils.get_default_arg_parser()
    parser.add_argument('--radius', type=float, default=1.0,
                        help='Radius of interaction neighborhood')
    args = parser.parse_args()

    neighborhood = utils.get_neighborhood(args.radius)

    init_params = {
        'sites': (args.width, args.width),
        'rules': rules,
        'neighborhood': neighborhood,
        'num_states': 4,
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
