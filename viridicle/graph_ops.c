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

//          ********************
//          * Graph Operations *
//          ********************
//
// This file contains the code for actually executing the mathematical
// operations. Ideally, this would be completely divorced from the Python API,
// to the point it could, in principle, be used without it. Note the absence of
// any PyObject or PyArrayObject structs. However, we have some dependencies
// we're currently stuck with:
//
// 1) The numpy random number generator. We want the user to be able to supply
//    their own RNG, and this is the easiest way.
// 2) Setting Python layer errors.
// 3) numpy typedefs. These could easily be eliminated if desired, using
//    inttypes.h, but there's no point as long as we're stuck with issues 1 and
//    2.
//
// An important note: for performance reasons, our probabilities are given as
// uint64s instead of as floats. A probability P as an integer means, as a
// floating point, (P / MAX_INTEGER_VALUE).

#include <Python.h>
#include <structmember.h>
// We need to include the numpy random libraries, but not the numpy array
// libraries. Therefore, we do not need the symbol definition business.
#include "numpy/random/bitgen.h"
#include "numpy/random/distributions.h"

#include "graph_ops.h"

//          ***********************
//          * Run System Function *
//          ***********************

int run_system_c(struct Model *geo, bitgen_t *rng, long int num_reports,
                 long int report_every, npy_uint64 *count_records,
                 state_type *site_records)
{
    // Runs a simulation on an arbitrary graph defined by a lookup table.
    // Returns 0 if successful, and sets an error condition and returns -1 if
    // an error occurs.
    //
    // geo: Model struct encoding the graph structure, transition rules, and
    //     current state.
    // rng: Random number generator.
    // num_reports: Number of reports to be generated.
    // report_every: Number of steps between reports.
    // count_records: Pointer to the array that will hold the count records.
    //     NULL if not returning counts.
    // site_records: Pointer to the array that will hold the site records. NULL
    //     if not returning sites.

    // An Edge is a pair of pointers, edge.from and edge.to, both of type
    // (state_type *), pointing to entries in the sites array. It captures an
    // edge, as the name implies.
    Edge edge;
    // roll: Stores the latest roll to decide on the state transition in the
    // Gillespie algorithm.
    prob_type roll;
    // Indices for for loops
    long int i;
    int n;
    // Index for the state pair
    int pair_idx;
    // Pointer to the transition state
    state_type *trans;
    // Buffer for the number of possible transitions of the current state pair.
    int n_trans;
    // Buffer for a pointer to the transition threshold.
    prob_type *trans_thresh;
    // Convenience variable.
    npy_uint64 num_edges_sub_one;

    if (check_validity(geo, false) != 0)
        return -1;
    
    num_edges_sub_one = geo->num_edges - 1;

    // Run the actual simulation. The outer loop loops through every reporting
    // interval, while the inner loop runs through the steps between each
    // report. Note that i_rep starts at 1 since the first entry of the records
    // is the initial state, before anything has happened.
    for(long int i_rep = 1; i_rep <= num_reports; i_rep++) {
        for(i = 0; i < report_every; i++) {
            // At each time step, we randomly select an edge. random_interval
            // selects an integer in the range [0, MAX] INCLUSIVE, so we need
            // to decrement num_sites by one.
            edge = geo->edges[random_interval(rng, num_edges_sub_one)];

            // Compute the index of the state pair, for looking up values in
            // the Rules struct.
            pair_idx = *edge.from * MAX_NUM_STATES + *edge.to;
            // Look up the number of possible transitions for the pair.
            n_trans = geo->n_trans[pair_idx];

            // n_trans > 0 is likely to be a relatively uncommon ocurrence in
            // many systems of interest, which tend to form single-state
            // domains with no interactions possible within the domains.
            if (n_trans > 0) {
                // Roll the dice. This will fill roll with a randomly-chosen
                // uint64 between 0 and NPY_MAX_UINT64 - 1. As we go through
                // the list of possible events, we decrement roll by the
                // probability of that event. This is equivalent to
                // partitioning the interval [0, 1] into regions for each of
                // the possible events, with width equal to that event's
                // probability, then selecting whichever event roll falls into.
                // However, this approach avoids needing to deal with
                // additional variables.
                roll = random_uint(rng);
                trans_thresh = geo->trans_thresh[pair_idx];

                // Iterate through possible transitions
                for(n = 0; n < n_trans; n++)
                {
                    if (*trans_thresh++ > roll) {
                        trans = geo->trans_state[pair_idx] + 2 * n;
                        *edge.from = *trans++;
                        *edge.to = *trans;
                        break;
                    }
                }
            }
        }

        // At the end of every reporting interval, copy the current counts of
        // states into the numpy.ndarray tracking the counts.
        if (count_records != NULL) {
            // We COULD keep a running count and update it at every time step.
            // But as long as the reporting interval is at least equal to the
            // number of sites in the system - which it seems likely it usually
            // will be - recounting from scratch at every reporting interval is
            // more cost-effective.
            count_population(geo, count_records);
            count_records += geo->num_states;
        }
        // And/or copy the current sites into the telemetry.
        if (site_records != NULL) {
            memcpy(
                site_records,
                geo->sites,
                sizeof(state_type) * geo->num_sites
            );
            site_records += geo->num_sites;
        }
    }

    return 0;
}


int run_system_fully_connected(struct Model *geo, bitgen_t *rng,
                               long int num_reports, long int report_every,
                               npy_uint64 *count_records,
                               state_type *site_records)
{
    // Runs a simulation on a fully-connected graph. Returns 0 if successful,
    // and sets an error condition and returns -1 if an error occurs. This is
    // broken out as a separate function from run_system_c to avoid needing a
    // lookup table for a fully-connected graph, as the memory requirements
    // quickly grow excessive.
    //
    // geo: Model struct encoding the graph structure, transition rules, and
    //     current state.
    // rng: Random number generator.
    // num_reports: Number of reports to be generated.
    // report_every: Number of steps between reports.
    // count_records: Pointer to the array that will hold the count records.
    //     NULL if not returning counts.
    // site_records: Pointer to the array that will hold the site records. NULL
    //     if not returning sites.

    // An Edge is a pair of pointers, edge.sites[0] and edge.sites[1], both of
    // type (state_type *), pointing to entries in the sites array. It captures
    // an edge, as the name implies.
    Edge edge;
    // roll: Stores the latest roll to decide on the state transition in the
    // Gillespie algorithm.
    prob_type roll;
    // Indices for for loops
    long int i;
    int n;
    // Index for the state pair
    int pair_idx;
    // Pointer to the transition state
    state_type *trans;
    // Buffer for the number of possible transitions of the current state pair.
    int n_trans;
    // Buffer for a pointer to the transition threshold.
    prob_type *trans_thresh;
    // Convenience variable.
    npy_uint64 num_sites_sub_one;

    if (check_validity(geo, true) != 0)
        return -1;
    
    num_sites_sub_one = geo->num_sites - 1;

    // Run the actual simulation. The outer loop loops through every reporting
    // interval, while the inner loop runs through the steps between each
    // report. Note that i_rep starts at 1 since the first entry of the records
    // is the initial state, before anything has happened.
    for(long int i_rep = 1; i_rep <= num_reports; i_rep++) {
        for(i = 0; i < report_every; i++) {
            // At each time step, we randomly select an edge. random_interval
            // selects an integer in the range [0, MAX] INCLUSIVE, so we need
            // to decrement num_sites by one. This is less efficient than a
            // lookup table, since it requires two RNG calls, but the memory
            // requirements for a lookup table for a decently-sized fully-
            // connected graph are impractical.
            edge.from = random_interval(rng, num_sites_sub_one) + geo->sites;
            edge.to = random_interval(rng, num_sites_sub_one) + geo->sites;
            while (edge.to == edge.from)
                edge.to = random_interval(rng, num_sites_sub_one) + geo->sites;

            // Compute the index of the state pair, for looking up values in
            // the Rules struct.
            pair_idx = *edge.from * MAX_NUM_STATES + *edge.to;
            // Look up the number of possible transitions for the pair.
            n_trans = geo->n_trans[pair_idx];

            // n_trans > 0 is likely to be a relatively uncommon ocurrence in
            // many systems of interest, which tend to form single-state
            // domains with no interactions possible within the domains.
            if (n_trans > 0) {
                // Roll the dice. This will fill roll with a randomly-chosen
                // uint64 between 0 and NPY_MAX_UINT64 - 1. As we go through
                // the list of possible events, we decrement roll by the
                // probability of that event. This is equivalent to
                // partitioning the interval [0, 1] into regions for each of
                // the possible events, with width equal to that event's
                // probability, then selecting whichever event roll falls into.
                // However, this approach avoids needing to deal with
                // additional variables.
                roll = random_uint(rng);
                trans_thresh = geo->trans_thresh[pair_idx];

                // Iterate through possible transitions
                for(n = 0; n < n_trans; n++)
                {
                    if (*trans_thresh++ > roll) {
                        trans = geo->trans_state[pair_idx] + 2 * n;
                        *edge.from = *trans++;
                        *edge.to = *trans;
                        break;
                    }
                }
            }
        }

        // At the end of every reporting interval, copy the current counts of
        // states into the numpy.ndarray tracking the counts.
        if (count_records != NULL) {
            // We COULD keep a running count and update it at every time step.
            // But as long as the reporting interval is at least equal to the
            // number of sites in the system - which it seems likely it usually
            // will be - recounting from scratch at every reporting interval is
            // more cost-effective.
            count_population(geo, count_records);
            count_records += geo->num_states;
        }
        // And/or copy the current sites into the telemetry.
        if (site_records != NULL) {
            memcpy(
                site_records,
                geo->sites,
                sizeof(state_type) * geo->num_sites
            );
            site_records += geo->num_sites;
        }
    }

    return 0;
}


int run_system_with_diffusion(struct Model *geo, bitgen_t *rng,
                              long int num_reports, long int report_every,
                              npy_uint64 *count_records,
                              state_type *site_records)
{
    // Runs a simulation on an arbitrary graph defined by a lookup table.
    // Returns 0 if successful, and sets an error condition and returns -1 if
    // an error occurs. This function adds a special trick for handling
    // diffusion from:
    //
    // Reichenbach, Mobilia, and Frey. "Self-Organization of Mobile Populations
    // in Cyclic Competition." Journal of Theoretical Biology, Vol. 254 No. 2,
    // pp. 368-383.
    //
    // Specifically, they observe that the number of diffusion events between
    // non-diffusion events is a geometric variable. Since there will typically
    // be many diffusion events for every diffusion event, generating and
    // handling that separately allows significant runtime savings. In the May-
    // Leonard model, in fact, we observed literally an order of magnitude
    // speedup when the diffusion rate was high. This is caused by the fact
    // that the May-Leonard model coalesces quickly into domains, so the
    // majority of interactions will be between members of the same species,
    // and in the high-mobility regime the majority of the interactions will be
    // diffusion events, so if we can handle those diffusions more efficiently
    // - e.g., by not needing to look up values in the transition table - then
    // we can radically reduce runtime costs.
    //
    // An important caveat here is that, for performance reasons, we only check
    // if the counter indicates it's time to make a report after every non-
    // diffusion event. This means our reports will not be precisely evenly
    // spaced. However, even in the high-mobility regime, the raggedness should
    // be minimal: in the May-Leonard model, with a diffusion rate of 60.0, we
    // have a mean of 59 diffusion events between non-diffusion events, and
    // with report_every=1.0 and a 256x256 lattice we have 65,536 steps between
    // reports.
    //
    // geo: Model struct encoding the graph structure, transition rules, and
    //     current state.
    // rng: Random number generator.
    // num_reports: Number of reports to be generated.
    // report_every: Number of steps between reports.
    // count_records: Pointer to the array that will hold the count records.
    //     NULL if not returning counts.
    // site_records: Pointer to the array that will hold the site records. NULL
    //     if not returning sites.

    // An Edge is a pair of pointers, edge.from and edge.to, both of type
    // (state_type *), pointing to entries in the sites array. It captures an
    // edge, as the name implies.
    Edge edge;
    // roll: Stores the latest roll to decide on the state transition in the
    // Gillespie algorithm.
    prob_type roll;
    // Indices for for loops
    long int i = 0;
    int n;
    // Index for the state pair
    int pair_idx;
    // Pointer to the transition state
    state_type *trans;
    // Buffer for the number of possible transitions of the current state pair.
    int n_trans;
    // Buffer for a pointer to the transition threshold.
    prob_type *trans_thresh;
    // Convenience variable.
    npy_uint64 num_edges_sub_one;
    // Buffer for swapping states in diffusion.
    state_type buffer;

    if (check_validity(geo, false) != 0)
        return -1;
    
    num_edges_sub_one = geo->num_edges - 1;
    // Pre-roll for number of diffusion events. The random_geometric method
    // from numpy is inefficient due to its use of floats, so instead we
    // generate a random integer, and we have pre-calculated thresholds for the
    // number of diffusion events. trans_thresh will point to the front of the
    // array of thresholds.
    roll = random_uint(rng);
    trans_thresh = geo->diffusion_thresh;

    // Run the actual simulation. The outer loop loops through every reporting
    // interval, while the inner loop runs through the steps between each
    // report. Note that i_rep starts at 1 since the first entry of the records
    // is the initial state, before anything has happened.
    for(long int i_rep = 1; i_rep <= num_reports; i_rep++) {
        for(i = 0; i < report_every; i++) {
            // Diffusion. We walk through thresholds until we run out of
            // diffusion events.
            if (__builtin_expect(*trans_thresh++ < roll, 1)) {
                edge = geo->edges[random_interval(rng, num_edges_sub_one)];
                if (*edge.to != *edge.from) {
                    buffer = *edge.to;
                    *edge.to = *edge.from;
                    *edge.from = buffer;
                }
            // Now we have a non-diffusion event.
            } else {
                // Select an edge for the reaction step.
                edge = geo->edges[random_interval(rng, num_edges_sub_one)];
                // Compute the index of the state pair, for looking up values
                // in the Rules struct.
                pair_idx = *edge.from * MAX_NUM_STATES + *edge.to;
                // Look up the number of possible transitions for the pair.
                n_trans = geo->n_trans[pair_idx];

                // n_trans > 0 is likely to be a relatively uncommon ocurrence
                // in many systems of interest, which tend to form single-state
                // domains with no interactions possible within the domains.
                if (n_trans > 0) {
                    // Roll the dice. This will fill roll with a randomly-
                    // chosen uint64 between 0 and NPY_MAX_UINT64 - 1. As we go
                    // through the list of possible events, we decrement roll
                    // by the probability of that event. This is equivalent to
                    // partitioning the interval [0, 1] into regions for each
                    // of the possible events, with width equal to that event's
                    // probability, then selecting whichever event roll falls
                    // into. However, this approach avoids needing to deal with
                    // additional variables.
                    roll = random_uint(rng);
                    trans_thresh = geo->trans_thresh[pair_idx];

                    // Iterate through possible transitions
                    for(n = 0; n < n_trans; n++)
                    {
                        if (*trans_thresh++ > roll) {
                            trans = geo->trans_state[pair_idx] + 2 * n;
                            *edge.from = *trans++;
                            *edge.to = *trans;
                            break;
                        }
                    }
                }

                // Roll for next number of diffusion events, and reset
                // trans_thresh to front of the array of diffusion thresholds.
                roll = random_uint(rng);
                trans_thresh = geo->diffusion_thresh;
            }
        }

        // At the end of every reporting interval, copy the current counts of
        // states into the numpy.ndarray tracking the counts.
        if (count_records != NULL) {
            // We COULD keep a running count and update it at every time step.
            // But as long as the reporting interval is at least equal to the
            // number of sites in the system - which it seems likely it usually
            // will be - recounting from scratch at every reporting interval is
            // more cost-effective.
            count_population(geo, count_records);
            count_records += geo->num_states;
        }
        // And/or copy the current sites into the telemetry.
        if (site_records != NULL) {
            memcpy(
                site_records,
                geo->sites,
                sizeof(state_type) * geo->num_sites
            );
            site_records += geo->num_sites;
        }
    }

    return 0;
}


//          *****************
//          * Data Checking *
//          *****************

int check_validity(struct Model *geo, bool is_fully_connected)
{
    // Data checking on the model. This function will set an error condition
    // and return -1 if a problem occurs, and will return 0 if everything is
    // fine. This function only checks conditions that might result in a
    // segfault if the Model struct was passed to run_system_c; it does NOT
    // check for smaller problems. For example, it does not check:
    //
    // 1. Each edge_idx points to an even entry in edges.
    // 2. Edges are organized according to the origin of the edge.
    //
    // This function assumes that the graph structure HAS been defined but the
    // rules may not have been.

    Edge *edge_start = geo->edges;
    Edge *edge_end = geo->edges + geo->num_edges;
    state_type *sites_start = geo->sites;
    state_type *sites_end = geo->sites + geo->num_sites;
    // Buffer for checking maximum state
    state_type max_state = 0;

    // Check that the graph structure has been fully initialized.
    if (
        (geo == NULL) ||
        (geo->sites == NULL) ||
        (geo->num_sites <= 0) ||
        (geo->num_edges <= 0)
     ) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "check_validity received uninitialized Model struct."
        );
        return -1;
    }

    // Check if the maximum entry in sites exceeds num_states - 1.
    if (geo->num_states != 0) {
        for(npy_uint64 i = 0; i < geo->num_sites; i++)
            if (geo->sites[i] > max_state)
                max_state = geo->sites[i];

        if (max_state > geo->num_states - 1) {
            PyErr_SetString(
                PyExc_ValueError,
                "sites contains out-of-bounds value."
            );
            return -1;
        }
    }

    // If not fully-connected, check edges.
    if (!is_fully_connected) {
        if ((geo->edges == NULL) || (geo->edge_idxs == NULL)) {
            PyErr_SetString(
                PyExc_RuntimeError,
                "check_validity received uninitialized Model struct."
            );
            return -1;
        }

        // Check that every entry in edge_idxs gives a valid index in edges.
        // The entry must be between 0 and num_edges - 1, and each entry must
        // be greater than or equal to the previous entry.
        for(npy_uint64 i = 0; i <= geo->num_sites; i++)
            if (
                (geo->edge_idxs[i] < edge_start) ||
                (geo->edge_idxs[i] > edge_end)
            ) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "Invalid entry in edge_idxs: out of range."
                );
                return -1;
            }
        for(npy_uint64 i = 0; i < geo->num_sites; i++)
            if (geo->edge_idxs[i + 1] < geo->edge_idxs[i]) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "Invalid entry in edge_idxs: non-monotonic."
                );
                return -1;
            }

        // Check that every entry in edges points to a valid site. The entry
        // must be between 0 and num_sites - 1.
        for(Edge * edge = edge_start; edge < edge_end; edge++)
            if (((*edge).from < sites_start) || ((*edge).from > sites_end) ||
                ((*edge).to < sites_start) || ((*edge).to > sites_end)
            ) {
                PyErr_SetString(PyExc_ValueError, "Invalid entry in edges.");
                return -1;
            }
    }

    return 0;
}


//          **********************
//          * Analysis Functions *
//          **********************


void count_population(struct Model *geo, npy_uint64 *counts) {
    // Counts the population of different states in the Model.
    //
    // geo: Model struct.
    //
    // counts: Array to be filled with the populations.
    //
    // num_states: Number of states.

    for(int i = 0; i < geo->num_states; i++)
        counts[i] = 0;
    for(npy_uint64 i = 0; i < geo->num_sites; i++)
        counts[geo->sites[i]]++;
}



int merge_small_c(struct Model *geo, int min_size, int merge_size,
                  state_type empty_state) {
    // Runs a small cluster merging operation. We define a cluster as a
    // maximal connected subgraph where every vertex has the same state. We
    // want to eliminate any clusters below a certain size. If they are
    // completely surrounded by another cluster, we want to merge them into
    // that cluster. If not, we want to set them to an empty_state value. This
    // function modifies geo->sites in place. It returns 0 if successful, and
    // sets an error and returns -1 if not.
    //
    // The merging operation proceeds in two stages: first, we identify any
    // clusters below the minimum size and set their state to empty. Then, we
    // identify any clusters of the empty state that are below the minimum size
    // and completely surrounded by another cluster, and set their state to
    // that cluster's state.
    //
    // geo: The Model to be modified.
    // min_size: Clusters below this size will be removed.
    // merge_size: Empty clusters below this size surrounded by another cluster
    //     will be merged into that cluster.
    // empty_state: State to be treated as empty - usually 0.

    // TODO: This is much less efficient than it could be.

    // checked: Workspace holding whether the current site has been checked.
    bool *checked = NULL;
    // sites_to_check: List of potential elements in the cluster, as indices.
    npy_uint64 *sites_to_check = NULL;
    // last_site_to_check: Pointer to last entry in sites_to_check that
    //     has been filled.
    npy_uint64 *last_site_to_check;
    // cluster_size: Size of the current cluster.
    npy_uint64 cluster_size;
    // site_idx: Index of current site.
    npy_uint64 idx;
    // cluster_state: State of the cluster.
    state_type cluster_state;
    // cluster_start: Pointer to array of sites in cluster, as indices.
    npy_uint64 *cluster = NULL;
    // multiple_neighbors: Buffer used in second clustering round, to recall if
    //     we've found multiple neighbors of different states yet.
    bool multiple_neighbors;

    if (check_validity(geo, false) != 0)
        goto error;

    checked = calloc(geo->num_sites, sizeof(bool));
    sites_to_check = malloc(geo->num_edges * sizeof(npy_uint64));
    cluster = malloc(geo->num_sites * sizeof(npy_uint64));
    if ((checked == NULL) ||
        (sites_to_check == NULL) ||
        (cluster == NULL))
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Memory allocation error in merge_small_c."
        );
        goto error;
    }

    // First clustering round: go through the system, find clusters that are
    // below the min_size, and set their contents to empty_state.
    
    for(npy_uint64 i = 0; i < geo->num_sites; i++)
        // Look for a site that has not yet been checked, and which is non-
        // empty.
        if ((!checked[i]) && (geo->sites[i] != empty_state)) {
            // Initialize new cluster.
            cluster_size = 0;
            sites_to_check[0] = i;
            last_site_to_check = sites_to_check;
            cluster_state = geo->sites[i];

            // For every site we need to check, see if it's been already
            // checked and whether it is the right state.
            for(npy_uint64 *site = sites_to_check;
                site <= last_site_to_check;
                site++)
            {
                idx = *site;
                // If it is, mark it checked, add it to the cluster, and add
                // its neighbors to the list of sites to be checked.
                if ((!checked[idx]) && (geo->sites[idx] == cluster_state)) {
                    checked[idx] = true;
                    cluster[cluster_size] = idx;
                    cluster_size++;

                    for(Edge *edge = geo->edge_idxs[idx];
                        edge < geo->edge_idxs[idx + 1];
                        *(++last_site_to_check) = (edge++)->to - geo->sites);
                }
            }

            // We've now got the full cluster. If its size is below min size,
            // then set its state to empty_state.
            if (cluster_size <= min_size)
                for(npy_uint64 j = 0; j < cluster_size; j++)
                    geo->sites[cluster[j]] = empty_state;
        }
    
    // Reset workspace
    for(npy_uint64 i = 0; i < geo->num_sites; i++)
        checked[i] = false;

    // Second clustering round: go through the system, find clusters of
    // empty_state that are below the min_size, check if they have neighbors of
    // multiple states, and if they do not, set them to equal the state of
    // their neighbors.

    for(npy_uint64 i = 0; i < geo->num_sites; i++)
        // Look for a site that has not yet been checked, and which is non-
        // empty.
        if ((!checked[i]) && (geo->sites[i] == empty_state)) {
            // Initialize new cluster.
            cluster_size = 0;
            sites_to_check[0] = i;
            last_site_to_check = sites_to_check;
            // cluster_state now tracks what state we will be changing the
            // cluster to. We initialize it to empty_state. As soon as we find
            // a neighbor, we'll set it to that neighbor's state.
            cluster_state = empty_state;
            // multiple_neighbors tracks whether we've found multiple
            // neighboring clusters or not.
            multiple_neighbors = false;

            // For every site we need to check, see if it's been already
            // checked.
            for(npy_uint64 *site = sites_to_check;
                site <= last_site_to_check;
                site++)
            {
                idx = *site;
                // If it is checked and of empty_state, mark it checked, add it
                // to the cluster, and add its neighbors to the list of sites
                // to be checked.
                if ((!checked[idx]) && (geo->sites[idx] == empty_state)) {
                    checked[idx] = true;
                    cluster[cluster_size] = idx;
                    cluster_size++;

                    for(Edge *edge = geo->edge_idxs[idx];
                        edge < geo->edge_idxs[idx + 1];
                        *(++last_site_to_check) = (edge++)->to - geo->sites);
                }
                // If it is not, update cluster_state and multiple_neighbors.
                // Note that we can't just break out of the loop because we
                // need to make sure we mark the whole cluster as checked.
                else if (geo->sites[idx] != empty_state) {
                    if (cluster_state == empty_state)
                        cluster_state = geo->sites[idx];
                    else if (cluster_state != geo->sites[idx])
                        multiple_neighbors = true;
                }
            }

            // We've now got the full cluster. If its size is below min size,
            // and it does not have multiple neighbors, set its state to the
            // cluster_state.
            if ((cluster_size <= merge_size) && (!multiple_neighbors))
                for(npy_uint64 j = 0; j < cluster_size; j++)
                    geo->sites[cluster[j]] = cluster_state;
        }

    // Free memory.

    free(checked);
    free(sites_to_check);
    free(cluster);

    return 0;

    error:

    if (checked != NULL)
        free(checked);
    if (sites_to_check != NULL)
        free(sites_to_check);
    if (cluster != NULL)
        free(cluster);

    return -1;
}


int grow_clusters_c(struct Model *geo, npy_uint64 num_steps,
                    state_type empty_state) {
    // Grows clusters by a specified number of steps. For example, suppose that
    // the empty state is zero, num_steps, and the graph is the periodic
    // lattice:
    //
    // 000000
    // 011100
    // 000020
    // 000000
    //
    // In the first step, we take every empty vertex that is adjacent to only
    // one state of non-empty vertex, and fill it in with that vertex's state:
    //
    // 011100
    // 111100
    // 011022
    // 000020
    //
    // Note how we don't do this to the two vertices that are next to both 1
    // and 2. Then, in the second step, we do it again:
    //
    // 111100
    // 111102
    // 011022
    // 011022
    //
    // This function modifies the sites array in-place. It returns 0 if
    // successful, and sets an error condition and returns -1 if not.
    
    // distance: This is the distance that we had to travel to fill in each
    // vertex. We use this because we're going to modify sites in place, and
    // during a particular step we need to be able to check if this vertex was
    // filled in in the current step (and so doesn't count when deciding to
    // fill in another vertex) or a previous step (and so does).
    npy_uint64 *distance = NULL;
    // *_neighbors: This is the array of neighbors to check. We need three: the
    // current_neighbors to be checked, the next_neighbors that will be checked
    // in the next round, and the swap_neighbors used to move points between
    // the two. These are stored as indices, not pointers, since we'll need to
    // look them up in geo->edge_idxs.
    npy_intp *cur_neighbors = NULL, *next_neighbors = NULL, *swap_neighbors;
    // Pointers to interior of *_neighbors.
    npy_intp *cur_last_ptr, *next_last_ptr;
    // Buffers holding the state of neighbors, used to check for whether there
    // are neighbors of multiple states.
    state_type neighbor_state, set_state;
    // Buffer holding index.
    npy_intp idx;

    // Allocate memory.
    distance = calloc(geo->num_sites, sizeof(npy_uint64));
    cur_neighbors = malloc(geo->num_edges * sizeof(npy_intp));
    next_neighbors = malloc(geo->num_edges * sizeof(npy_intp));
    if ((distance == NULL) ||
        (cur_neighbors == NULL) ||
        (next_neighbors == NULL))
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Memory allocation error in grow_clusters."
        );
        goto error;
    }

    cur_last_ptr = cur_neighbors;
    next_last_ptr = next_neighbors;

    // Initial iteration to set cur_neighbors for the first time. This will
    // contain every empty site adjacent to a non-empty site. Sites may be
    // repeated. cur_last_ptr will point to one after the last entry.
    for(npy_intp idx = 0; idx < geo->num_sites; idx++)
        if (geo->sites[idx] != empty_state)
            for(Edge *edge = geo->edge_idxs[idx];
                edge < geo->edge_idxs[idx + 1];
                edge++)
                if (*((*edge).to) == empty_state)
                    *cur_last_ptr++ = (*edge).to - geo->sites;

    // Now we start iterating.
    for(npy_uint64 step = 1; step <= num_steps; step++) {
        // Go through each candidate site. Check if it's empty, and if, among
        // the neighboring non-empty sites, there are multiple states. If not,
        // set it to the state of the neighbors, and add its neighbors to the
        // list of neighbors to check in the next step.
        for(npy_intp *cur_ptr = cur_neighbors;
            cur_ptr < cur_last_ptr;
            cur_ptr++)
        {
            // One site may appear in the loop multiple times. If it has, then
            // distance will be set.
            if (distance[*cur_ptr] != 0)
                continue;

            // Initialize set_state to empty.
            set_state = empty_state;
            // Iterate through neighbors.
            for(Edge *edge = geo->edge_idxs[*cur_ptr];
                edge < geo->edge_idxs[*cur_ptr + 1];
                edge++) {
                neighbor_state = *((*edge).to);
                // Check if the neighbor is non-empty, and was not set in this
                // round of checking.
                if ((neighbor_state != empty_state) && 
                    (distance[(*edge).to - geo->sites] != step))
                {
                    // Check if set_state is unset. If it is unset, set it to
                    // the state of the neighbor. If it is set, check if the
                    // neighbor matches that state. If it does, nothing to be
                    // done. If it does not, then we have a conflict, we will
                    // do nothing to this vertex. Reset set_state to empty and
                    // break out of the loop.
                    if (set_state == empty_state)
                        set_state = neighbor_state;
                    else if (set_state != neighbor_state) {
                        set_state = empty_state;
                        break;
                    }
                }
            }

            // Update distance of this vertex and set its state.
            geo->sites[*cur_ptr] = set_state;
            distance[*cur_ptr] = step;

            // If we set this vertex, then we add the neighbors to the roster
            // of vertices to check in the next iteration.
            for(Edge *edge = geo->edge_idxs[*cur_ptr];
                edge < geo->edge_idxs[*cur_ptr + 1];
                edge++) {
                neighbor_state = *((*edge).to);
                idx = (*edge).to - geo->sites;
                if ((neighbor_state == empty_state) && (distance[idx] == 0))
                    *next_last_ptr++ = (*edge).to - geo->sites;
            }
        }

        // Swap cur_neighbors and next_neighbors pointers.
        swap_neighbors = next_neighbors;
        next_neighbors = cur_neighbors;
        cur_neighbors = swap_neighbors;
        cur_last_ptr = next_last_ptr;
        next_last_ptr = next_neighbors;
    }

    free(distance);
    free(cur_neighbors);
    free(next_neighbors);

    return 0;

    error:

    if (distance != NULL)
        free(distance);
    if (cur_neighbors != NULL)
        free(cur_neighbors);
    if (next_neighbors != NULL)
        free(next_neighbors);

    return -1;
}


state_type * cluster_graph_c(struct Model *geo, npy_uint64 *clusters,
                             npy_intp *num_clusters) {
    // Runs a clustering algorithm. We define clusters to be maximal connected
    // subgraphs where every vertex has the same state. clusters must be an
    // array of the same length as geo->sites, and each entry will be filled
    // with the unique id of the cluster containing the corresponding site.
    // There are no guarantees about how cluster ids will be selected except
    // that they will be taken from the set [0, 1, ..., num_clusters - 1]. On
    // success, the function returns a pointer to an array mapping the cluster
    // id to its state; on failure, it sets an error condition and returns
    // NULL.
    //
    // geo: The Model to be modified.
    // clusters: Array to be filled with cluster ids, with the same length as
    //     geo->sites.
    // num_clusters: Pointer to a variable to be filled with the number of
    //     clusters.

    // checked: Workspace holding whether the current site has been checked.
    bool *checked = NULL;
    // sites_to_check: List of potential elements in the cluster, as indices.
    npy_uint64 *sites_to_check = NULL;
    // last_site_to_check: Pointer to last entry in sites_to_check that
    //     has been filled.
    npy_uint64 *last_site_to_check;
    // site_idx: Index of current site.
    npy_uint64 idx;
    // cluster_state: State of the cluster.
    state_type cluster_state;
    // working_cluster_id_to_state: Working array that will hold a map from the
    //     cluster ids to the state. This will be over-sized, and we will
    //     create a new, smaller array to actually pass back to the calling
    //     function.
    state_type *working_cluster_id_to_state = NULL;
    // cluster_id_to_state: This is the actual array that will be returned.
    state_type *cluster_id_to_state = NULL;

    // Input checking.
    if ((clusters == NULL) || (num_clusters == NULL)) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "cluster_graph_c received NULL pointer."
        );
        goto error;
    }
    if (check_validity(geo, false) != 0)
        goto error;

    // Allocate memory.
    *num_clusters = 0;
    checked = calloc(geo->num_sites, sizeof(bool));
    sites_to_check = malloc(geo->num_edges * sizeof(npy_uint64));
    working_cluster_id_to_state = malloc(geo->num_sites * sizeof(state_type));
    if ((checked == NULL) ||
        (sites_to_check == NULL) ||
        (working_cluster_id_to_state == NULL))
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Memory allocation error in cluster_graph."
        );
        goto error;
    }

    // Go through sites array running clustering algorithm.
    for(npy_uint64 i = 0; i < geo->num_sites; i++)
        // Look for a site that has not yet been checked, and which is non-
        // empty.
        if (!checked[i]) {
            // Initialize new cluster.
            sites_to_check[0] = i;
            last_site_to_check = sites_to_check;
            cluster_state = geo->sites[i];
            working_cluster_id_to_state[*num_clusters] = cluster_state;

            // For every site we need to check, see if it's been already
            // checked and whether it is the right state.
            for(npy_uint64 *site = sites_to_check;
                site <= last_site_to_check;
                site++)
            {
                idx = *site;
                // If it is, mark it checked, add it to the cluster, and add
                // its neighbors to the list of sites to be checked.
                if ((!checked[idx]) && (geo->sites[idx] == cluster_state)) {
                    checked[idx] = true;
                    clusters[idx] = *num_clusters;
                    for(Edge *edge = geo->edge_idxs[idx];
                        edge < geo->edge_idxs[idx + 1];
                        *(++last_site_to_check) = (edge++)->to - geo->sites);
                }
            }

            (*num_clusters)++;
        }

    // Construct array to be returned.
    cluster_id_to_state = malloc(*num_clusters * sizeof(state_type));
    if (cluster_id_to_state == NULL) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Memory allocation failure in cluster_graph_c."
        );
        goto error;
    }
    for(npy_uint64 i = 0; i < *num_clusters; i++)
        cluster_id_to_state[i] = working_cluster_id_to_state[i];

    // Free memory.

    free(checked);
    free(sites_to_check);
    free(working_cluster_id_to_state);

    return cluster_id_to_state;

    error:

    if (checked != NULL)
        free(checked);
    if (sites_to_check != NULL)
        free(sites_to_check);
    if (working_cluster_id_to_state != NULL)
        free(working_cluster_id_to_state);
    if (cluster_id_to_state != NULL)
        free(cluster_id_to_state);

    return NULL;
}
