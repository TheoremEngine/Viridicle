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

//          **************
//          * Python API *
//          **************
//
// This file contains the Python-level interface for the library. The comments
// provide, in addition to the usual purpose, documentation on how the Python C
// API works, in hopes that it will help some future person to understand it. A
// brief summary: the main contents of this file is the run_system function,
// which is the entry from the Python layer into the C layer. The run_system
// function calls into functions in data_prep.c to convert the Python objects
// provided by the user into the Model struct, a C layer object that defines
// the simulation behavior and the graph structure respectively. This is then
// passed to the appropriate run_system_* function in graph_ops.c, which does
// the actual mathematics.

#include <Python.h>
#include <structmember.h>
// This suppresses a deprecation warning
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// This define is necessary since we call into the numpy c layer in multiple
// files, not all of which have the import_array line. It must take place
// BEFORE the #include of "numpy/arrayobject.h". It is documented at:
//
// https://numpy.org/devdocs/reference/c-api/array.html
#define PY_ARRAY_UNIQUE_SYMBOL viridicle_ARRAY_API
#include "numpy/arrayobject.h"
#include "numpy/random/bitgen.h"
#include "numpy/random/distributions.h"

#include "graph_ops.h"
#include "data_prep.h"

// To define a function in C that will be accessible from Python, we first
// define the actual function. If the function accepts both positional and
// keyword arguments, then it must have signature:
//
// static PyObject * function(PyObject *self, PyObject *args, PyObject *kwds);
//
// Since this is a function, not a method, self will be the module it lives in;
// ignore this. If an error condition occurs, the function should call the
// function:
//
// PyErr_SetString(ErrType, "Error message");
//
// And then return NULL. Otherwise, it should return a pointer to a PyObject,
// even if that PyObject is None. After defining the function, we define the
// method and module table, and the module initialization function; these are
// documented below the function.


static PyObject *run_system(PyObject *self, PyObject *args, PyObject *kwds)
{
    // This function is the entry from the Python layer into the C layer. It
    // has the Python signature:
    //
    // run_system(graph_type: int, sites: numpy.ndarray, bitgen: PyCapsule,
    //            beta: numpy.ndarray, num_steps: int, report_every: int,
    //            diffusion_probability: float = 0.0,
    //            return_counts: Optional[bool] = True,
    //            return_sites: Optional[bool] = False,
    //            neighborhood: Optional[np.ndarray] = None,
    //            edge_idxs: Optional[np.ndarray] = None,
    //            edges: Optional[np.ndarray] = None) -> np.ndarray
    //
    // graph_type is an integer specifying the graph type:
    //
    // 0: Fully-Connected
    // 1: Lattice
    // 2: Arbitrary
    //
    // sites is the numpy.ndarray containing the current state of the graph.
    // This must be coercable to a numpy.ndarray of dtype np.uint8, and will
    // have additional criteria to satisfy depending on the graph type.
    //
    // bitgen must be a PyCapsule. A PyCapsule is a Python object that
    // encapsulates a pointer to a C struct. In this case it must encapsulate a
    // pointer to a bitgen_t, the C struct underlying a numpy random number
    // generator.
    //
    // beta is a numpy.ndarray specifying the transition probabilities. It must
    // be a 4-dimensional array of floats of shape (num_states, num_states,
    // num_states, num_states), where the entry beta[i, j, k, l] equals the
    // probability that a pair of states (i, j) transition to state (k, l) if
    // they are selected in the Gillespie algorithm. Note that we perform
    // basic validity checks in the C layer to verify that all entries in beta
    // are non-negative and that no state pair has probabilities that sum to
    // more than one, but we expect other checks to be handled in the Python
    // layer - we are mostly concerned with warding off potential segfaults.
    //
    // num_steps is how many time steps to run the system for. report_every is
    // how often to report telemetry, in steps.
    //
    // return_counts and return_sites specify what type of telemetry to return.
    //
    // diffusion_probability is the probability of a given step being a
    // diffusion step. If the diffusion rate is very high, treating diffusion
    // separately can yield runtime improvements. This is ignored if the graph
    // is fully-connected.
    //
    // neighborhood must be specified if and only if the graph is a lattice.
    // This is a numpy.ndarray of dtype np.intp of shape (num_neighbors, 2),
    // which specifies the offsets in the lattice that are a neighbor of a
    // point. For example, the default neighborhood is [[1, 0], [0, 1],
    // [-1, 0], [0, -1]], which means that the neighbors of a point are the
    // points horizontally and vertically adjacent to it.
    //
    // edges and edge_idxs must be specified if and only if the graph is
    // arbitrary. These encode the graph structure, and must be 1-dimensional
    // np.ndarrays of dtype np.intp. Specifically, to look up the neighbors of
    // site k, in Python syntax we would do:
    //
    // idx_start = edge_idxs[k]
    // idx_end = edge_idxs[k + 1]
    // neighbors = edges[idx_start:idx_end]

    // Set up variables. kwlist is the list of keywords accepted by the
    // function, in order.
    static char *kwlist[] = {
        "graph_type", "sites", "bitgen", "beta", "num_steps", "report_every",
        "diffusion_probability", "return_counts", "return_sites",
        "neighborhood", "edge_idxs", "edges", NULL
    };
    // np_*: Temporary variables that will hold PyArrayObject pointers until
    //     they can be added to the Model struct.
    PyArrayObject *np_sites = NULL, *np_beta = NULL;
    PyArrayObject *np_nhd = NULL, *np_edge_idxs = NULL, *np_edges = NULL;
    // rng: Random number generator.
    bitgen_t *rng = NULL;
    // diffusion_probability: Probability that a step is a diffusion step.
    float diffusion_probability = 0.0;
    // num_steps: Number of steps to run.
    // report_every: How often to record the telemetry, in time steps.
    long int num_steps = -1, report_every = -1;
    // graph_type: Integer specifying the type of graph (0: fully-connected,
    //     1: lattice, 2: arbitrary).
    // return_counts: Integer specifying whether to return counts as part of
    //     telemetry (1) or not (0).
    // return_sites: Integer specifying whether to return sites as part of
    //     telemetry (1) or not (0).
    int graph_type = -1, return_counts = 1, return_sites = 0;
    // Pointers to objects storing telemetry as PyArrayObjects.
    PyArrayObject *count_records = NULL, *site_records = NULL;
    // *_ptr: Pointers to objects storing telemetry as C arrays.
    state_type *site_records_ptr = NULL;
    npy_uint64 *count_records_ptr = NULL;
    // Pointer to Model struct. This is used to encapsulate the graph
    // structure, transition rules, and current state, and is documented in
    // graph_ops.h.
    struct Model *geo = NULL;
    // Total number of times telemetry will be recorded.
    npy_intp num_reports;
    // Used for specifying the dimension of arrays to be created.
    npy_intp dims[NPY_MAXDIMS];
    // Buffer for returning value if we're returning a tuple.
    PyObject *rtn = NULL;

    // Parse arguments. PyArg_ParseTupleAndKeywords parses the inputs args and
    // kwds. As used here, it has API:
    //
    // int PyArg_ParseTupleAndKeywords(
    //     PyObject *args, PyObject *kwd, char[] format_string,
    //     char[][] kwlist, ...pointers...
    // );
    // 
    // args and kwd are the inputs to the function. The format string and
    // kwlist define the expected inputs and their types. Each character in the
    // format string specifies the expected type of the corresponding keyword
    // in kwlist; there's a long list of options but we actually make use of:
    //
    //   'f' : float
    //   'i' : int
    //   'l' : long int
    //   'O' : PyObject *
    //   'O&': PyObject with a converter. Unlike the other arguments, TWO
    //         arguments must be supplied: the first an arbitrary pointer and
    //         the second a function with signature:
    //          
    //             int converter(PyObject *, void *)
    //
    //         Where the first argument is the PyObject being parsed, and the
    //         second argument is the pointer that will be filled. The function
    //         should return 1 on success and raise an exception and return 0
    //         on failure.
    //   'p' : Coerces the Python object to a boolean, then returns 1 as a C
    //         int if True, 0 if False.
    //   '|' : Special indicator that all entries after this are optional
    //         keyword arguments
    //
    // All entries prior to '|' are mandatory and an exception will be raised
    // by PyArg_ParseTupleAndKeywords if they are not present. All entries
    // after are optional and the pointers will be left unchanged if they are
    // not present - hence the setting of optional keyword arguments to NULL
    // when they were initialized so we can check if the user provided them.
    //
    // Note that the reference counts for PyObjects parsed in this way are NOT
    // increased, so there is no need to worry about garbage collection for
    // any of the objects it returns - yet.
    //
    // PyArg_ParseTupleAndKeywords will return -1 if an error occurs during
    // parsing. In this case we abort and return NULL; we do not need to raise
    // an exception state ourselves as it has already been done for us by
    // PyArg_ParseTupleAndKeywords.
    //
    // PyArg_ParseTupleAndKeywords is documented at:
    //
    // https://docs.python.org/3.6/c-api/arg.html

    if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "iO&O&O&ll|fppO&O&O&", kwlist,
        &graph_type,
        &coerce_sites, (void *)(&np_sites),
        &coerce_rng, (void *)(&rng),
        &coerce_beta, (void *)(&np_beta),
        &num_steps,
        &report_every,
        &diffusion_probability,
        &return_counts,
        &return_sites,
        &coerce_nhd, (void *)(&np_nhd),
        &coerce_edge_idxs, (void *)(&np_edge_idxs),
        &coerce_edges, (void *)(&np_edges)
    ))
        goto error;

    // Check that num_steps and report_every are positive.
    if (num_steps <= 0) {
        PyErr_SetString(PyExc_ValueError, "Elapsed time must be positive.");
        goto error;
    }
    if (report_every <= 0) {
        PyErr_SetString(PyExc_ValueError, "Report interval must be positive.");
        goto error;
    }

    // Create Model struct. This is used to bundle up the graph's structure and
    // contents. The struct is documented in graph_ops.h and create_model in
    // data_prep.c. create_model will return NULL if an error occurs.
    geo = create_model();
    if (geo == NULL)
        goto error;

    // Add transition rules to the Model struct. This function returns 0 if
    // successful, and sets an error condition and returns -1 if not.
    if (initialize_rules(geo, np_beta, diffusion_probability, graph_type) != 0)
        goto error;

    // Add the graph structure to the Model struct. This function returns 0 if
    // successful, and sets an error condition and returns -1 if not.
    if (
        initialize_graph(geo, graph_type, np_sites, np_nhd, np_edge_idxs,
                         np_edges) != 0
    )
        goto error;

    // Calculate the number of reports to make.
    num_reports = (npy_intp)(num_steps / report_every);
    if (num_reports < 1)
        num_reports = 1;

    // Create arrays for holding records. We'll use dims to hold the dimension
    // of the arrays we're going to build. In both cases, it starts with the
    // number of reports, which we increment by one since we also want to
    // capture the initial state of the system. Note that detailed notes on the
    // numpy C array are found in data_prep.c.
    dims[0] = num_reports + 1;
    if (return_counts == 1) {
        // count_records will have dimension (num_reports, num_states);
        // counts[i, j] will be the population of state j at reporting interval
        // i.
        dims[1] = (npy_intp)geo->num_states;
        // PyArray_EMPTY is a macro that creates an empty numpy array.
        count_records = (PyArrayObject *)PyArray_EMPTY(2, dims, NPY_UINT64, 0);
        if (count_records == NULL)
            goto error;
        // Fill in first entry
        count_records_ptr = (npy_uint64 *)PyArray_GETPTR2(count_records, 0, 0);
        count_population(geo, count_records_ptr);
        // Get the pointer we'll pass to run_system
        count_records_ptr += geo->num_states;
    }
    if (return_sites == 1) {
        // site_records will have dimension (num_reports, *sites.shape). Let's
        // put that into dims:
        for(int i = 0; i < PyArray_NDIM((PyArrayObject *)np_sites); i++)
            dims[i + 1] = PyArray_DIM((PyArrayObject *)np_sites, i);
        // Build the array:
        site_records = (PyArrayObject *)PyArray_EMPTY(
            PyArray_NDIM((PyArrayObject *)np_sites) + 1,
            dims,
            STATE_NPY_DTYPE,
            0
        );
        if (site_records == NULL)
            goto error;
        // In what follows, we will reuse dims to index the record arrays when we
        // fill them with telemetry. We want the index to point to the entry
        // record[0, 0, 0, ..., 0].
        for(int i = 0; i < NPY_MAXDIMS; i++)
            dims[i] = 0;
        // Fill in first entry
        memcpy(
            PyArray_GetPtr(site_records, dims),
            geo->sites,
            sizeof(state_type) * geo->num_sites
        );
        // Get the pointer we'll pass to run_system
        dims[0] = 1;
        site_records_ptr = (state_type *)PyArray_GetPtr(site_records, dims);
    }

    // Actually run the system. These functions are in graph_ops.c. It returns
    // 0 if successful, and returns -1 and sets an error condition if not.
    if (geo->run_system(
        geo, rng, num_reports, report_every, count_records_ptr,
        site_records_ptr
    ) != 0)
        goto error;

    // Free our dynamically allocated variables. This function is in
    // data_prep.c.
    free_model(geo);

    // Python keeps track of the number of references to various objects, which
    // is how it decides when to garbage collect them. The only objects created
    // in this function are the telemetry arrays. If an error occurs, it is
    // possible that error will be caught outside the function, so we still
    // want their references to be counted appropriately. Py_XDECREF will
    // decrement the number of references to the PyObject if it is passed a
    // non-NULL value, and will do nothing if it is passed NULL.
    Py_XDECREF((PyObject *)np_beta);
    Py_XDECREF((PyObject *)np_nhd);
    Py_XDECREF((PyObject *)np_edge_idxs);
    Py_XDECREF((PyObject *)np_edges);
    Py_XDECREF((PyObject *)np_sites);

    // Construct the return value and return it.
    if ((return_counts == 0) && (return_sites == 0)) {
        Py_RETURN_NONE;
    } else if ((return_counts == 1) && (return_sites == 0))
        return (PyObject *)count_records;
    else if ((return_counts == 0) && (return_sites == 1))
        return (PyObject *)site_records;
    else {
        // Py_BuildValue is a sort of inverse of PyArg_ParseTupleAndKeywords.
        // It has signature:
        //
        // PyObject * Py_BuildValue(char[] format_string, ...objects...)
        //
        // It builds a Python tuple out of objects. The format string is very
        // similar to PyArg_ParseTupleAndKeyword's - might even be exactly the
        // same, I haven't checked - main thing is that "O" means that the
        // object we're passing it is a PyObject pointer. So this builds a pair
        // (count_records, site_records). It increments the references to each
        // input by one, which we need to undo.
        rtn = Py_BuildValue(
            "OO",
            (PyObject *)count_records,
            (PyObject *)site_records
        );
        Py_DECREF(count_records);
        Py_DECREF(site_records);

        return rtn;
    }

    error:

    // Clean up in the event of an error condition.
    free_model(geo);

    Py_XDECREF((PyObject *)count_records);
    Py_XDECREF((PyObject *)site_records);
    Py_XDECREF((PyObject *)np_beta);
    Py_XDECREF((PyObject *)np_nhd);
    Py_XDECREF((PyObject *)np_edge_idxs);
    Py_XDECREF((PyObject *)np_edges);
    Py_XDECREF((PyObject *)np_sites);

    return NULL;
}


static PyObject *merge_small(PyObject *self, PyObject *args, PyObject *kwds)
{
    // Python interface to the merge_small_c function. This runs a small
    // cluster merging operation. We define a cluster as a maximal connected
    // subgraph where every vertex has the same state. We want to eliminate any
    // clusters below a certain size. If they are completely surrounded by
    // another cluster, we want to merge them into that cluster. If not, we
    // want to set them to an empty_state value. This function modifies the
    // sites array in place.
    //
    // The merging operation proceeds in two stages: first, we identify any
    // clusters below the minimum size and set their state to empty. Then, we
    // identify any clusters of the empty state that are below the merge_size
    // and completely surrounded by another cluster, and set their state to
    // that cluster's state.
    //
    // This has Python signature:
    //
    // merge_small(graph_type: int, sites: np.ndarray, min_size: int,
    //             merge_size: int, empty_state: int = 0,
    //             neighborhood: Optional[np.ndarray] = None,
    //             edge_idxs: Optional[np.ndarray] = None,
    //             edges: Optional[np.ndarray] = None) -> None

    // Set up variables. kwlist is the list of keywords accepted by the
    // function, in order.
    static char *kwlist[] = {
        "graph_type", "sites", "min_size", "merge_size", "empty_state",
        "neighborhood", "edge_idxs", "edges", NULL
    };
    // np_*: Temporary variables that will hold PyArrayObject pointers until
    //     they can be added to the Model struct.
    PyArrayObject *np_sites = NULL;
    PyArrayObject *np_nhd = NULL, *np_edge_idxs = NULL, *np_edges = NULL;
    // graph_type: Integer specifying the type of graph (0: fully-connected,
    //     1: lattice, 2: arbitrary).
    // min_size: Clusters below this size will be merged into other clusters.
    // merge_size: Empty clusters below this size that are fully surrounded by
    //     another cluster will be merged into that cluster.
    // empty_state: State to be used as empty_state.
    int graph_type = -1, min_size, merge_size, empty_state = 0;
    // Pointer to Model struct. This is used to encapsulate the graph
    // structure, transition rules, and current state, and is documented in
    // graph_ops.h.
    struct Model *geo = NULL;

    if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "iO&ii|iO&O&O&", kwlist,
        &graph_type,
        &coerce_sites, (void *)(&np_sites),
        &min_size,
        &merge_size,
        &empty_state,
        &coerce_nhd, (void *)(&np_nhd),
        &coerce_edge_idxs, (void *)(&np_edge_idxs),
        &coerce_edges, (void *)(&np_edges)
    ))
        goto error;

    // Check inputs
    if (min_size < 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "Minimum size must be non-negative."
        );
        goto error;
    }
    if (empty_state != (int)(state_type)empty_state) {
        PyErr_SetString(
            PyExc_ValueError,
            "empty_state out of valid range."
        );
        goto error;
    }

    // Create Model struct. This is used to bundle up the graph's structure and
    // contents. The struct is documented in graph_ops.h and create_model in
    // data_prep.c. create_model will return NULL if an error occurs.
    geo = create_model();
    if (geo == NULL)
        goto error;

    // Add the graph structure to the Model struct. This function returns 0 if
    // successful, and sets an error condition and returns -1 if not.
    if (
        initialize_graph(geo, graph_type, np_sites, np_nhd, np_edge_idxs,
                         np_edges) != 0
    )
        goto error;

    // Run the merging.
    if (merge_small_c(geo, min_size, merge_size, (state_type)empty_state) != 0)
        goto error;

    // Free memory.

    free_model(geo);

    Py_XDECREF((PyObject *)np_nhd);
    Py_XDECREF((PyObject *)np_edge_idxs);
    Py_XDECREF((PyObject *)np_edges);
    Py_XDECREF((PyObject *)np_sites);

    Py_RETURN_NONE;

    error:

    free_model(geo);

    Py_XDECREF((PyObject *)np_nhd);
    Py_XDECREF((PyObject *)np_edge_idxs);
    Py_XDECREF((PyObject *)np_edges);
    Py_XDECREF((PyObject *)np_sites);

    return NULL;
}


static PyObject *grow_clusters(PyObject *self, PyObject *args, PyObject *kwds)
{
    // Python interface to the grow_clusters_c function.
    //
    // This has Python signature:
    //
    // grow_clusters(graph_type: int, sites: np.ndarray, num_steps: int,
    //               empty_state: int = 0,
    //               neighborhood: Optional[np.ndarray] = None,
    //               edge_idxs: Optional[np.ndarray] = None,
    //               edges: Optional[np.ndarray] = None) -> None

    // Set up variables. kwlist is the list of keywords accepted by the
    // function, in order.
    static char *kwlist[] = {
        "graph_type", "sites", "num_steps", "empty_state", "neighborhood",
        "edge_idxs", "edges", NULL
    };
    // np_*: Temporary variables that will hold PyArrayObject pointers until
    //     they can be added to the Model struct.
    PyArrayObject *np_sites = NULL;
    PyArrayObject *np_nhd = NULL, *np_edge_idxs = NULL, *np_edges = NULL;
    // graph_type: Integer specifying the type of graph (0: fully-connected,
    //     1: lattice, 2: arbitrary).
    // num_steps: Number of steps to grow the clusters.
    // empty_state: State to be used as empty_state.
    int graph_type = -1, num_steps, empty_state = 0;
    // Pointer to Model struct. This is used to encapsulate the graph
    // structure, transition rules, and current state, and is documented in
    // graph_ops.h.
    struct Model *geo = NULL;

    if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "iO&i|iO&O&O&", kwlist,
        &graph_type,
        &coerce_sites, (void *)(&np_sites),
        &num_steps,
        &empty_state,
        &coerce_nhd, (void *)(&np_nhd),
        &coerce_edge_idxs, (void *)(&np_edge_idxs),
        &coerce_edges, (void *)(&np_edges)
    ))
        goto error;

    // Check inputs
    if (num_steps <= 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "Number of steps must be positive."
        );
        goto error;
    }
    if (empty_state != (int)(state_type)empty_state) {
        PyErr_SetString(
            PyExc_ValueError,
            "empty_state out of valid range."
        );
        goto error;
    }

    // Create Model struct. This is used to bundle up the graph's structure and
    // contents. The struct is documented in graph_ops.h and create_model in
    // data_prep.c. create_model will return NULL if an error occurs.
    geo = create_model();
    if (geo == NULL)
        goto error;

    // Add the graph structure to the Model struct. This function returns 0 if
    // successful, and sets an error condition and returns -1 if not.
    if (
        initialize_graph(geo, graph_type, np_sites, np_nhd, np_edge_idxs,
                         np_edges) != 0
    )
        goto error;

    // Run the merging.
    if (grow_clusters_c(geo, num_steps, (state_type)empty_state) != 0)
        goto error;

    // Free memory.

    free_model(geo);

    Py_XDECREF((PyObject *)np_nhd);
    Py_XDECREF((PyObject *)np_edge_idxs);
    Py_XDECREF((PyObject *)np_edges);
    Py_XDECREF((PyObject *)np_sites);

    Py_RETURN_NONE;

    error:

    free_model(geo);

    Py_XDECREF((PyObject *)np_nhd);
    Py_XDECREF((PyObject *)np_edge_idxs);
    Py_XDECREF((PyObject *)np_edges);
    Py_XDECREF((PyObject *)np_sites);

    return NULL;
}


static PyObject *cluster_geo(PyObject *self, PyObject *args, PyObject *kwds)
{
    // Python interface to the cluster_geo function.
    //
    // This has Python signature:
    //
    // cluster_geo(graph_type: int, sites: np.ndarray,
    //             neighborhood: Optional[np.ndarray] = None,
    //             edge_idxs: Optional[np.ndarray] = None,
    //             edges: Optional[np.ndarray] = None,
    //             out: Optional[np.ndarray] = None)
    //      -> Tuple[np.ndarray, np.ndarray]
    // 
    // We define a cluster to be a maximal connected subgraph where all
    // vertices have the same state. cluster_geo calculates the clusters for
    // the graph, and returns a pair consisting of an array of the same shape
    // as sites with the cluster id of each vertex, and a 1-dimensional array
    // of arbitrary length that maps the cluster id back to the original state.
    //
    // graph_type is an integer specifying the graph type:
    //
    // 0: Fully-Connected
    // 1: Lattice
    // 2: Arbitrary
    //
    // sites is the numpy.ndarray containing the current state of the graph.
    // This must be coercable to a numpy.ndarray of dtype np.uint8, and will
    // have additional criteria to satisfy depending on the graph type.
    //
    // neighborhood must be specified if and only if the graph is a lattice.
    // This is a numpy.ndarray of dtype np.intp of shape (num_neighbors, 2),
    // which specifies the offsets in the lattice that are a neighbor of a
    // point. For example, the default neighborhood is [[1, 0], [0, 1],
    // [-1, 0], [0, -1]], which means that the neighbors of a point are the
    // points horizontally and vertically adjacent to it.
    //
    // edges and edge_idxs must be specified if and only if the graph is
    // arbitrary. These encode the graph structure, and must be 1-dimensional
    // np.ndarrays of dtype np.intp. Specifically, to look up the neighbors of
    // site k, in Python syntax we would do:
    //
    // idx_start = edge_idxs[k]
    // idx_end = edge_idxs[k + 1]
    // neighbors = edges[idx_start:idx_end]
    //
    // If specified, out should be a numpy.ndarray of the same shape as sites,
    // but of dtype np.uint64. It will be used to write the cluster ids for
    // each site. A fresh array will be created if out is not specified. This
    // is used to help stabilize memory useage.

    // Set up variables. kwlist is the list of keywords accepted by the
    // function, in order.
    static char *kwlist[] = {
        "graph_type", "sites", "neighborhood", "edge_idxs", "edges", "out",
        NULL
    };
    // np_*: Temporary variables that will hold PyArrayObject pointers until
    //     they can be added to the Model struct.
    PyArrayObject *np_sites = NULL;
    PyArrayObject *np_nhd = NULL, *np_edge_idxs = NULL, *np_edges = NULL;
    // graph_type: Integer specifying the type of graph (0: fully-connected,
    //     1: lattice, 2: arbitrary).
    int graph_type = -1;
    // Pointer to Model struct. This is used to encapsulate the graph
    // structure, transition rules, and current state, and is documented in
    // graph_ops.h.
    struct Model *geo = NULL;
    // cluster_id_to_state: Array mapping cluster ids to original states.
    state_type *cluster_id_to_state = NULL;
    // num_clusters: Buffer for number of clusters found.
    npy_intp num_clusters;
    // np_clusters: Numpy array that will wrap clusters.
    // np_cluster_id_to_state: Numpy array that will wrap cluster_id_to_state.
    PyArrayObject *np_clusters = NULL, *np_cluster_id_to_state = NULL;
    // rtn: Tuple that will be returned.
    PyObject *rtn = NULL;

    if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "iO&|O&O&O&O&", kwlist,
        &graph_type,
        &coerce_sites, (void *)(&np_sites),
        &coerce_nhd, (void *)(&np_nhd),
        &coerce_edge_idxs, (void *)(&np_edge_idxs),
        &coerce_edges, (void *)(&np_edges),
        &coerce_clusters_out, (void*)(&np_clusters)
    ))
        goto error;

    // Create Model struct. This is used to bundle up the graph's structure and
    // contents. The struct is documented in graph_ops.h and create_model in
    // data_prep.c. create_model will return NULL if an error occurs.
    geo = create_model();
    if (geo == NULL)
        goto error;

    // Add the graph structure to the Model struct. This function returns 0 if
    // successful, and sets an error condition and returns -1 if not.
    if (
        initialize_graph(geo, graph_type, np_sites, np_nhd, np_edge_idxs,
                         np_edges) != 0
    )
        goto error;

    // Check if np_clusters was provided. If not, create it. If it was, check
    // that it has the same shape as np_sites.
    if (np_clusters == NULL)
        np_clusters = (PyArrayObject *)PyArray_ZEROS(
            PyArray_NDIM(np_sites),
            PyArray_DIMS(np_sites),
            NPY_UINT64,
            0
        );
    else if (!is_same_shape(np_sites, np_clusters)) {
        PyErr_SetString(
            PyExc_ValueError,
            "Sites and out array must be the same shape."
        );
        goto error;
    }
    // At this point, np_clusters can only be NULL if PyArray_ZEROS returned
    // NULL. PyArray_ZEROS will set an error condition itself if this happens.
    if (np_clusters == NULL)
        goto error;

    // Run the merging.
    cluster_id_to_state = cluster_graph_c(
        geo,
        PyArray_GETPTR1(np_clusters, 0),
        &num_clusters
    );
    if (cluster_id_to_state == NULL)
        goto error;

    // Construct return objects. Note that, by default, if we wrap a numpy
    // array around an already-existing C array, the underlying memory will NOT
    // be freed when the numpy array is garbage-collected after return; the
    // OWNDATA flag must be set manually for that to happen. See:
    //
    // https://stackoverflow.com/questions/8708758/can-i-force-a-numpy-
    //     ndarray-to-take-ownership-of-its-memory
    //
    // Note that we must avoid any goto error statement between a SUCCESSFUL
    // creation of np_cluster_id_to_state and the setting of the OWNDATA flag,
    // or else the cluster_id_to_state array will not be freed. Fortunately
    // this is usually a small array, but memory leaks are still memory leaks.
    np_cluster_id_to_state = (PyArrayObject *)PyArray_SimpleNewFromData(
        1,
        &num_clusters,
        STATE_NPY_DTYPE,
        (void *)cluster_id_to_state
    );
    if (np_cluster_id_to_state == NULL)
        goto error;
    PyArray_ENABLEFLAGS(np_cluster_id_to_state, NPY_ARRAY_OWNDATA);

    rtn = Py_BuildValue(
        "OO",
        (PyObject *)np_clusters,
        (PyObject *)np_cluster_id_to_state
    );

    // Py_BuildValue INCREFs what we pass it, so now there are TWO references
    // to each of the returned arrays. We therefore need to DECREF them, or
    // else they won't be properly garbage-collected since there's no way to
    // get rid of that free-floating reference in the Python layer.

    Py_DECREF((PyObject *)np_clusters);
    Py_DECREF((PyObject *)np_cluster_id_to_state);

    // Free memory.

    free_model(geo);

    Py_XDECREF((PyObject *)np_nhd);
    Py_XDECREF((PyObject *)np_edge_idxs);
    Py_XDECREF((PyObject *)np_edges);
    Py_XDECREF((PyObject *)np_sites);

    return rtn;

    error:

    free_model(geo);

    // This slightly odd syntax is necessary since cluster_id_to_state may
    // exist as an array for some time before it is wrapped as a numpy array.
    if (np_cluster_id_to_state != NULL)
        Py_DECREF((PyObject *)np_cluster_id_to_state);
    else if (cluster_id_to_state != NULL)
        free(cluster_id_to_state);

    Py_XDECREF((PyObject *)np_nhd);
    Py_XDECREF((PyObject *)np_edge_idxs);
    Py_XDECREF((PyObject *)np_edges);
    Py_XDECREF((PyObject *)np_sites);
    Py_XDECREF((PyObject *)np_clusters);

    return NULL;
}


// After defining the functions, we define a method table containing every
// function that will be accessible from the Python layer:
// 
// static PyMethodDef method_table[] = {
//     {
//         "function",
//         (PyCFunction)run_system,
//         METH_VARARGS | METH_KEYWORDS,
//         "docstring"
//     },
//     {NULL, NULL, 0, NULL}
// };
// 
// Each entry in the table has signature:
// 
//     {name, (PyCFunction)function, flags, docstring}
// 
// The flags specify that the function accepts both positional and keyword
// arguments. The {NULL, NULL, 0, NULL} entry is a termination symbol denoting
// the end of the table.


static PyMethodDef viridicle_methods[] = {
    {
        "run_system",
        (PyCFunction)run_system,
        METH_VARARGS | METH_KEYWORDS,
        "Run the system"
    },
    {
        "merge_small",
        (PyCFunction)merge_small,
        METH_VARARGS | METH_KEYWORDS,
        "Merge small clusters"
    },
    {
        "grow_clusters",
        (PyCFunction)grow_clusters,
        METH_VARARGS | METH_KEYWORDS,
        "Grow clusters"
    },
    {
        "cluster_geo",
        (PyCFunction)cluster_geo,
        METH_VARARGS | METH_KEYWORDS,
        "Cluster geography"
    },
    {NULL, NULL, 0, NULL}
};


// Next, we define the module table:
// 
// static struct PyModuleDef module_def = {
//        PyModuleDef_HEAD_INIT,
//        "module_name",
//        "module docstring",
//        -1,
//        method_table
// };
// 
// PyModuleDef_HEAD_INIT is a macro.


static struct PyModuleDef viridicle_module = {
    PyModuleDef_HEAD_INIT,
    "_C",
    "C API for viridicle",
    -1,
    viridicle_methods
};


// Finally, we define the module initialization function:
// 
// PyMODINIT_FUNC
// PyInit_module(void) {
//        PyObject *module = NULL;
// 
//        module = PyModule_Create(&module_def);
//        if (module == NULL)
//        	return NULL;
// 
//        return module;
// }
// 
// PyMODINIT_FUNC is a macro. The module initialization function uses the
// PyModule_Create function to construct the module as a PyObject from the
// module table. We check if an error occurred during module creation, and if
// not, return it.
// 
// When working with numpy, we have the additional requirement that we need to
// include the command:
// 
//        import_array();
// 
// In the module initialization function. This initializes numpy so that our
// other functions can use it. The method table, module table, and module
// initialization function are all at the bottom of this file.


PyMODINIT_FUNC
PyInit__C(void) {
    // Initialize numpy C API.
    import_array();

    // _C_module will be the actual module object when we're done with it.
    PyObject *_C_module = NULL;

    // Create the module using PyModule_Create called on the module table. If
    // an error occurs, then an exception condition will be set inside
    // PyModule_Create - so we don't need to - and _C_module will still be NULL
    // after the call.
    _C_module = PyModule_Create(&viridicle_module);
    if (_C_module == NULL)
        return NULL;

    return _C_module;
}
