// Copyright 2021, Theorem Engine
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef VIRIDICLE_GRAPH_OPS_HEADER
#define VIRIDICLE_GRAPH_OPS_HEADER

// For performance reasons, we do not store our probabilities as floating
// point numbers. Instead, when we "roll the dice", we generate a random
// integer, in the range [0, MAX_INT_VALUE), and our "probabilities" are
// integers < MAX_INT_VALUE.

typedef npy_uint64 prob_type;
#define PROB_TYPE_NUM_BITS 64
#define PROB_TYPE_MAX NPY_MAX_UINT64

typedef npy_uint8 state_type;
#define MAX_NUM_STATES 256
#define STATE_NPY_DTYPE NPY_UINT8

// To prevent memory overflow errors, we set a maximum number of possible
// diffusion events between reactions. TODO: Revisit this number.
#define MAX_DIFFUSIONS 1024

// An Edge encodes, well, an edge. It contains only pointers to the two ends of
// the edge. This allows for more efficient memory layout: under the hood, the
// edges array in the Model struct below is essentially an array of 2 *
// num_edges (state_type *) pointers to entries in the sites array. To look up
// the ith edge, one would then go to Model.edges[2 * i], containing the FROM
// entry, and Model.edges[2 * i + 1], containing the TO entry. Recasting this
// as an array of structs instead slightly improves performance and makes
// bookkeeping easier.
typedef struct Edge { state_type * from; state_type * to; } Edge;

// The Model struct is used as a convenience package for passing the system
// around. It does not persist between calls of run_system, and it is not
// accessible from the Python API. You should be sure to use the
// create_model function to create these structs, and free_model to destroy
// them. Be aware that some entries in the struct will be initialized to NULL
// if they are not in use.
//
// sites: The array containing the status of each site. THIS IS NOT FREED BY
//     FREE_MODEL; we expect the user wishes this array to persist and be
//     returned to the Python layer.
// num_sites: The total number of sites, i.e., the length of sites.
// num_edges: The total number of edges, i.e., the length of edges.
// edge_idxs: An array of pointers to pointers, used to look up the neighbors
//     of a site. Each pointer in edge_idxs points to somewhere in the edges
//     array. To look up the neighbors of a site indexed by site_idx,
//     *edge_idxs[site_idx] will point to the place in the edges array where
//     the edges of that site begin, and *edge_idxs[site_idx + 1] will point to
//     one after the place where they stop.
// edges: An array of pointers encoding the edges in the graph. Contains Edges,
//     which each encapsulate a pair of pointers to entries in sites.
//
// To be a little more concrete, suppose you want to iterate over the neighbors
// of a site indexed by site_idx. To do this, use the for loop:
//
// for(Edge *ptr = edge_idxs[site_idx];
//     ptr < edge_idxs[site_idx + 1];
//     ptr++)
// {
//     do_stuff(*ptr);
// }
//
// num_states: The number of possible states in the system.
// n_trans: n_trans[pair_idx] is the number of possible transitions for the
//     pair of states indexed by pair_idx, where:
//
//      pair_idx = (state_0 * NUM_STATES) + state_1
//
//     We index each of the transitions for a state pair by n for consistency.
// trans_states: trans_states[pair_idx] + (2 * n) is a pointer to the
//     destination state of element 0 of the pair (state_0, state_i) in
//     transition n. If you increment the pointer by one, it will point to the
//     destination state of state 1.
// trans_thresh: transition_thresh[pair_idx][n] is the threshold of transition
//     n. We generate a random number in the range [0, MAX_INTEGER_VALUE), then
//     check transitions until we find one where the threshold exceeds the
//     random number and execute that transition. The difference
//     (trans_thresh[i] - trans_thresh[i - 1]) is equal to the transition
//     probability multiplied by MAX_INTEGER_VALUE.
// diffusion_thresh: This is used in the diffusion trick for efficiently
//     generating a geometric variable. A uniform random value over all
//     possible integers is generated, and then compared to each entry in the
//     array to determine the roll.
// run_system: The function used to run the model.

struct Model {
    state_type *sites;
    npy_uint64 num_sites;
    npy_uint64 num_edges;
    Edge **edge_idxs;
    Edge *edges;
    state_type num_states;
    state_type n_trans[MAX_NUM_STATES * MAX_NUM_STATES];
    state_type *trans_state[MAX_NUM_STATES * MAX_NUM_STATES];
    prob_type *trans_thresh[MAX_NUM_STATES * MAX_NUM_STATES];
    prob_type *diffusion_thresh;
    int (*run_system)(struct Model *geo, bitgen_t *rng, long int num_reports,
                      long int report_every, npy_uint64 *count_records,
                      state_type *site_records);
};

// The ActiveList struct keeps track of which edges have the potential to see a
// state change in the next update, so that we can avoid wasting cycles on
// inactive edges. This is used in systems where homogenous edges - edges where
// each site have the same state - have no potential reactions. The struct is
// pre-allocated a large enough block of memory that it can hold all potential
// edges, and uses a swap-and-pop mechanism to keep it updated in O(1) time.
//
// active_edges: Pointer to the array of pointers to the edges that are active.
// num_active: Number of active edges.

struct ActiveList {
    Edge **active_edges;
    int num_active;
}

int run_system_c(struct Model *geo, bitgen_t *rng, long int num_reports,
                 long int report_every, npy_uint64 *count_records,
                 state_type *site_records);
int run_system_fully_connected(struct Model *geo, bitgen_t *rng,
                               long int num_reports, long int report_every,
                               npy_uint64 *count_records,
                               state_type *site_records);
int run_system_with_diffusion(struct Model *geo, bitgen_t *rng,
                              long int num_reports, long int report_every,
                              npy_uint64 *count_records,
                              state_type *site_records);
int run_system_with_active_list(struct Model *geo, bitgen_t *rng,
                                long int num_reports, long int report_every,
                                npy_uint64 *count_records,
                                state_type *site_records);

int check_validity(struct Model *geo, bool is_fully_connected);

void count_population(struct Model *geo, npy_uint64 *counts);

int merge_small_c(struct Model *geo, int min_size, int merge_size,
                  state_type empty_state);
state_type * cluster_graph_c(struct Model *geo, npy_uint64 *clusters,
                             npy_intp *num_clusters);
int grow_clusters_c(struct Model *geo, npy_uint64 num_steps,
                    state_type empty_state);

// __builtin_expected is not defined in MSVC.
#ifndef __GNUC__
#define __builtin_expect(e, c) (e)
#endif

#endif
