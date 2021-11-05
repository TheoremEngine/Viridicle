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
//          * Data Preparation *
//          ********************
//
// This file contains functions for converting PyObjects provided to the Python
// -level interface into the structs that will be passed to the functions in
// graph_ops.c that actually do the math. In addition, it contains copious
// commentary on the numpy C API.
//
// An important note: we leave most data checks - such as scaling the maximum
// total transition rate - to the Python layer, where things are easier. In the
// C layer, we will only perform checks that are necessary to prevent truly
// pathological behavior (like segfaults), not behavior that only leads to
// inefficiencies.

#include <Python.h>
#include <structmember.h>
// This suppresses a deprecation warning
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// These defines are necessary since we call into the numpy c layer in multiple
// files, not all of which have the import_array line. It is documented in:
//
// https://numpy.org/devdocs/reference/c-api/array.html
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL viridicle_ARRAY_API
#include "numpy/arrayobject.h"
#include "numpy/random/bitgen.h"
#include "numpy/random/distributions.h"

#include "data_prep.h"

// numpy arrays at the C level are pointers to structs called PyArrayObjects.
// These structs wrap a one-dimensional array of varying type, and contain
// various metadata such as the dimensions of the array, stride, etc. In
// general, direct access to the underlying array is not recommended; instead,
// we are advised to use various macros and functions provided for that
// purpose. The most important such function is PyArray_GetPtr:
//
// void * PyArray_GetPtr(PyArrayObject *array, npy_intp *site_idx);
//
// Here, site_idx is a pointer to an array of length equal to the dimension of
// array, containing the indices of the entry we wish to retrieve.
// PyArray_GetPtr converts that array of integers into a single integer index
// of the specific entry in the C array underlying the numpy.ndarray, and
// returns a pointer to that entry. We refer to site_idx as the "array index",
// and to the single integer index used by PyArray_GetPtr as the "flat index".
// For our operations, converting from the array index to the flat index can
// actually be somewhat expensive, and in our mathematics we will work with
// flat indices exclusively. However, we need to be able to work with array
// indices during the data prep stage so that we can get everything ready.
//
// In addition to the PyArray_GetPtr function, we are provided three
// convenience macros, PyArray_GETPTR1, PyArray_GETPTR2, and PyArray_GETPTR3,
// which allow access to 1-, 2-, or 3-dimensional arrays using ints without the
// need to construct an array to hold the index, e.g.:
//
// void * PyArray_GETPTR2(PyArrayObject *array, npy_intp i_0, npy_intp i_1);
//
// (Note that these macros do not check the shape of the array, so we can,
// e.g., use PyArray_GETPTR1(array, 0) to get the start of the array regardless
// of its shape.)
//
// PyArray_GetPtr is located in:
//
// https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/multiarraymodule.c
//
// And documented at:
//
// https://numpy.org/devdocs/reference/c-api/array.html
//
// We are also provided with macros for creating arrays. The most important is
// PyArray_FROMANY, which wraps the function PyArray_FromAny. Both attempt to
// coerce an arbitrary Python object to a numpy.ndarray and return a pointer to
// it. Note that it returns it as a PyObject pointer, NOT a PyArrayObject
// pointer, but we can immediately cast it to one. The macro has signature:
//
// PyObject * PyArray_FROMANY(
//     PyObject * object, int dtype, int min_dim, int max_dim, int flags
// );
//
// object is the object to be coerced.
//
// dtype is an integer indicating the expected dtype of the numpy.ndarray.
// A collection of constants are provided. This is the key difference
// between PyArray_FROMANY, which uses these integer dtypes, and the
// function, which expects to be passed an actual dtype PyObject instead.
//
// min_dim and max_dim are self-explanatory; these can be set to 0 if no
// minimum or maximum is needed.
//
// The flags specify various conditions about data layout etc.;
// NPY_ARRAY_CARRY is a constant that sets the flags indicating the data
// should be laid out like a C array. We generally wish to enforce this, as we
// will assume it is true when accessing the C arrays underlying these numpy
// arrays in graph_ops.c.
//
// If object is already a numpy.ndarray that conforms to the desired
// properties, then the object itself is returned, not a copy. If it is
// not, then the function attempts to coerce it to one. In each case, the
// reference count for the object is incremented, so we need to remember to
// DECREF it after we are done with it if it is not being returned. If an error
// occurs during coercion, the function sets an exception - so we do not need
// to - and returns NULL.
//
// PyArray_FROMANY is documented at:
//
// https://numpy.org/doc/stable/reference/c-api/array.html
//
// We also have various functions and macros for constructing numpy arrays from
// scratch. The simplest is PyArray_EMPTY, which unpacks to work with the
// function PyArray_Empty. The macro has signature:
//
// PyObject * PyArray_EMPTY(
//     int num_dims, npy_intp *dims, int dtype, int is_fortran
// );
//
// num_dims is the number of dimensions in the shape.
//
// dims is an array holding the actual desired shape.
//
// dtype is an integer specifying the dtype. This is the distinction
// between the macro PyArray_EMPTY, which expects this typenum integer, and
// the function PyArray_Empty, which expects a dtype struct.
//
// is_fortran specifies whether to create the array in Fortran array layout
// or C array layout.
//
// Again, the primary distinction between the macro and the function is that
// the function expects to be passed a dtype PyObject, while the macro works
// with an integer constant.


//          ******************************
//          * Model Creation and Freeing *
//          ******************************
//
// These functions handle creation and destruction of Model structs. They do
// not handle filling the properties of those structs. See graph_ops.h for
// details of the struct.

struct Model * create_model() {
    // Creates an empty model struct. Used to ensure that all of the properties
    // are initialized to NULL or 0 before we begin to fill them in. Returns
    // NULL if an error occurs.

    struct Model *geo = NULL;

    geo = (struct Model *)malloc(sizeof(struct Model));
    // Check for malloc failure
    if (geo == NULL)
        goto error;

    geo->sites = NULL;
    geo->num_sites = 0;
    geo->num_edges = 0;
    geo->edge_idxs = NULL;
    geo->edges = NULL;
    geo->num_states = 0;
    geo->diffusion_thresh = NULL;

    for(int i = 0; i < MAX_NUM_STATES * MAX_NUM_STATES; i++) {
        geo->n_trans[i] = 0;
        geo->trans_state[i] = NULL;
        geo->trans_thresh[i] = NULL;
    }

    return geo;

    error:
    
    free_model(geo);

    return NULL;
}


void free_model(struct Model *geo) {
    // Frees the model struct, if it exists. This should be able to accept the
    // model struct at any stage in its construction.

    if (geo != NULL) {
        if (geo->edges != NULL)
            free(geo->edges);
        if (geo->edge_idxs != NULL)
            free(geo->edge_idxs);
        if (geo->diffusion_thresh != NULL)
            free(geo->diffusion_thresh);

        // trans_state, trans_thresh are pointers to the interior of single
        // blocks that were allocated all at once.
        if (geo->trans_state[0] != NULL)
            free(geo->trans_state[0]);
        if (geo->trans_thresh[0] != NULL)
            free(geo->trans_thresh[0]);

        free(geo);
    }
}


//          *********************
//          * Argument Coercion *
//          *********************
//
// These functions are used in PyArg_ParseTuple and related functions to coerce
// arguments. These functions are used with 'O&' in format strings, and must
// have the signature:
//
//     int function(PyObject *obj, void* ptr)
//
// obj is the PyObject in the argument. ptr is the pointer that must be filled
// by the function. The function must return 1 if successful, and must set an
// error and return 0 if it fails.


int coerce_sites(PyObject *obj_sites, PyArrayObject **site_ptr) {
    // Coerces the sites object provided in the argument tuple into a
    // PyArrayObject.

    *site_ptr = (PyArrayObject *)PyArray_FROMANY(
        obj_sites, STATE_NPY_DTYPE, 1, NPY_MAXDIMS - 1, NPY_ARRAY_CARRAY
    );

    if (*site_ptr == NULL)
        return 0;
    return 1;
}


int coerce_rng(PyObject *capsule, bitgen_t **rng_ptr) {
    // Get the random number generator out of the capsule. PyCapsule_IsValid
    // checks if capsule is a valid PyCapsule rather than some other PyObject.
    // "BitGenerator" is the name of the capsule we expect to find, and makes
    // sure we weren't passed some other kind of PyCapsule. These bit
    // generators are documented at:
    //
    // https://numpy.org/devdocs/reference/random/c-api.html
    //
    // PyCapsules are documented at:
    //
    // https://docs.python.org/3/c-api/capsule.html

    if (!PyCapsule_IsValid(capsule, "BitGenerator")) {
        PyErr_SetString(
            PyExc_TypeError,
            "Run function was not passed a BitGenerator capsule."
        );
        return 0;
    }

    // We can now unpack the PyCapsule to get the underlying pointer, then cast
    // it to bitgen_t.
    *rng_ptr = (bitgen_t *)PyCapsule_GetPointer(capsule, "BitGenerator");

    if (*rng_ptr == NULL)
        return 0;
    return 1;
}


int coerce_beta(PyObject *obj_beta, PyArrayObject **beta_ptr) {
    // Coerces the beta object provided in the argument tuple into a
    // PyArrayObject.

    *beta_ptr = (PyArrayObject *)PyArray_FROMANY(
        obj_beta, NPY_FLOAT64, 4, 4, NPY_ARRAY_CARRAY
    );

    if (*beta_ptr == NULL)
        return 0;
    return 1;
}


int coerce_nhd(PyObject *obj_nhd, PyArrayObject **nhd_ptr) {
    // Coerces the neighborhood object provided in the argument tuple into a
    // PyArrayObject.

    if (obj_nhd == NULL)
        return 0;

    *nhd_ptr = (PyArrayObject *)PyArray_FROMANY(
        obj_nhd, NPY_INTP, 2, 2, NPY_ARRAY_CARRAY
    );

    if (*nhd_ptr == NULL)
        return 0;
    return 1;
}


int coerce_edge_idxs(PyObject *obj_edge_idxs, PyArrayObject **edge_idxs_ptr) {
    // Coerces the edge_idxs object provided in the argument tuple into a
    // PyArrayObject.

    if (obj_edge_idxs == NULL)
        return 0;

    *edge_idxs_ptr = (PyArrayObject *)PyArray_FROMANY(
        obj_edge_idxs, NPY_INTP, 1, 1, NPY_ARRAY_CARRAY
    );

    if (*edge_idxs_ptr == NULL)
        return 0;
    return 1;
}


int coerce_edges(PyObject *obj_edges, PyArrayObject **edges_ptr) {
    // Coerces the edges object provided in the argument tuple into a
    // PyArrayObject.

    if (obj_edges == NULL)
        return 0;

    *edges_ptr = (PyArrayObject *)PyArray_FROMANY(
        obj_edges, NPY_INTP, 1, 1, NPY_ARRAY_CARRAY
    );

    if (*edges_ptr == NULL)
        return 0;
    return 1;
}


int coerce_clusters_out(PyObject *obj_out, PyArrayObject **out_ptr) {
    // Coerces the out argument in the cluster_geo function.

    // The object may be None, in which case do nothing.
    if (obj_out == Py_None)
        return 1;

    if (obj_out == NULL)
        return 0;

    *out_ptr = (PyArrayObject *)PyArray_FROMANY(
        obj_out, NPY_UINT64, 1, NPY_MAXDIMS - 1, NPY_ARRAY_CARRAY
    );

    if (*out_ptr == NULL)
        return 0;
    return 1;
}


//          ****************************
//          * Initialization Functions *
//          ****************************
//
// Functions for initializing properties of Model structs.


int initialize_graph(struct Model *geo, int graph_type,
                     PyArrayObject *np_sites, PyArrayObject *np_neighborhood,
                     PyArrayObject *np_edge_idxs, PyArrayObject *np_edges)
{
    // Initializes the components of a Model struct that encode the graph
    // structure. Returns 0 if successful, -1 if not. We assume that we are NOT
    // in an error state when we enter the function. Further, since the geo
    // Model struct is provided by an external source, that source is
    // responsible for calling free_model on the struct in the event of an
    // error.
    //
    // graph_type is an integer specifying the type of graph (0: fully-
    // connected, 1: lattice, 2: arbitrary).
    //
    // np_sites is a pointer to the sites array.
    //
    // np_neighborhood is a pointer to the array for the neighborhood for the
    // lattice. This should be NULL for non-lattice graphs.
    //
    // np_edge_idxs is a pointer to the array containing the edge_idxs for an
    // arbitrary graph. This should be NULL for non-arbitrary graphs.
    //
    // np_edges is a pointer to the array containing the edges for an arbitrary
    // graph. This should be NULL for non-arbitrary graphs.
    //
    // This function is responsible for filling the following properties of the
    // Model struct:
    //
    // sites: The array containing the status of each site.
    // np_sites: A pointer to the numpy.ndarray containing the actual array
    //     defining the status of each site. Used for garbage collection.
    // num_sites: The total number of sites, i.e., the length of sites.
    // num_edges: The total number of edges, i.e., the length of edges.
    // edge_idxs: An array of pointers to pointers, used to look up the
    //     neighbors of a site in a non-fully-connected graph. Each pointer in
    //     edge_idxs points to somewhere in the edges array. To look up the
    //     neighbors of a site indexed by site_idx, *edge_idxs[site_idx] will
    //     point to the place in the edges array where they begin, and
    //     *edge_idxs[site_idx + 1] will point to one after the place where
    //     they stop.
    // edges: An array of pointers encoding the edges in the graph. Contains
    //     pointers to sites in the sites array.

    // Check Model and np_sites exist. All other pointers are technically
    // optional.
    if (geo == NULL) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "initialize_graph received NULL Model pointer."
        );
        goto error;
    }
    if (np_sites == NULL) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "initialize_graph received NULL np_sites pointer."
        );
        goto error;
    }

    // Assign sites object to Model struct.
    geo->sites = (state_type *)PyArray_GETPTR1(np_sites, 0);
    geo->num_sites = (npy_uint64)PyArray_SIZE(np_sites);

    // Check optional parameters are present only if this graph type wants
    // them.

    // neighborhood parameter
    if ((np_neighborhood != NULL) && (graph_type != 1)) {
        PyErr_SetString(
            PyExc_TypeError,
            "neighborhood parameter is only accepted by lattice graphs."
        );
        goto error;
    }

    // edges and edge_idxs parameters
    if (
        (graph_type != 2) && ((np_edge_idxs != NULL) || (np_edges != NULL))
    ) {
        PyErr_SetString(
            PyExc_TypeError,
            "edges, edge_idxs parameters are only taken by arbitrary graphs."
        );
        goto error;
    }

    // Encode the graph structure
    if (graph_type == 0) {
        if (encode_fully_connected(geo) != 0)
            goto error;
    } else if (graph_type == 1) {
        if (encode_lattice(geo, np_sites, np_neighborhood) != 0)
            goto error;
    } else if (graph_type == 2) {
        if (encode_arbitrary(geo, np_edge_idxs, np_edges) != 0)
            goto error;
    } else {
        PyErr_SetString(PyExc_ValueError, "graph_type is not recognized.");
        goto error;
    }

    return 0;

    error:

    return -1;
}


int encode_lattice(struct Model *geo, PyArrayObject *np_sites,
                   PyArrayObject *np_neighborhood)
{
    // This encodes a lattice structure into a lookup table of edges and
    // edge_idxs. Those arrays are documented in graph_ops.h. We assume that
    // neighborhood is a 2-dimensional numpy.ndarray, where each row is an
    // offset specifying which sites are neighbors of the current site. For
    // example, the 2-dimensional square lattice has neighborhood array:
    //
    // [[-1,  0],
    //  [ 1,  0],
    //  [ 0, -1],
    //  [ 0,  1]]
    //
    // This implies that, given a site with index (i, j) in the 2-d array, the
    // neighbors of that site are (i - 1, j), (i + 1, j), (i, j - 1), and
    // (i, j + 1).

    // We begin by noting that we have two types of indices here: flat indices,
    // which are used in the Model struct's internals, and array indices, which
    // are used by numpy.ndarray. In this function, as we iterate through the
    // sites, we will start with a flat index. We convert this to an array
    // index, which we use with np_neighborhood to find the array index of the
    // neighboring sites, and then convert those array indices back to flat
    // indices to set the edges, edge_idxs arrays of the Model struct.

    // site_idx: The current location in the lattice we are encoding.
    // nhd_idx: The location of its neighbor.
    npy_intp site_idx[NPY_MAXDIMS], nhd_idx[NPY_MAXDIMS];
    // num_nhd: Number of neighboring sites.
    npy_intp num_nhd;
    // sites_ndim: Dimension of the lattice.
    npy_intp sites_ndim;
    // sites_dim: Dimensions of the lattice.
    npy_intp *sites_dim;
    // Index into the geo->edges array.
    npy_intp i_edge = 0;
    // strides: Convenience variable pointing to the strides of np_sites.
    npy_intp *strides;

    // Input checking.
    if (np_neighborhood == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "Lattice graph requires the neighborhood parameter."
        );
        return -1;
    }
    if ((geo == NULL) || (np_sites == NULL)) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "encode_lattice received null pointer."
        );
        return -1;
    }

    // Initialize variables.
    num_nhd = PyArray_DIM(np_neighborhood, 0);
    sites_ndim = PyArray_NDIM(np_sites);
    sites_dim = PyArray_DIMS(np_sites);
    strides = PyArray_STRIDES(np_sites);

    // We're going to check if the sites array is in C layout. It SHOULD be,
    // since our coercion function above mandates that, but better safe than
    // sorry.
    for(int i = 1; i < sites_ndim; i++)
        if (strides[i] >= strides[i - 1]) {
            PyErr_SetString(
                PyExc_RuntimeError,
                "encode_lattice received sites array in non-C layout."
            );
            return -1;
        }

    // Neighborhood arrays must be of dimension (num_neighbors,
    // sites.ndim).
    if (PyArray_DIM(np_neighborhood, 1) != sites_ndim) {
        PyErr_SetString(
            PyExc_ValueError,
            "Neighborhood and sites dimensions do not match."
        );
        return -1;
    }

    // Allocate memory for arrays
    geo->num_edges = geo->num_sites * num_nhd;
    geo->edges = malloc(sizeof(Edge) * geo->num_edges);
    geo->edge_idxs = malloc(sizeof(Edge *) * (geo->num_sites + 1));
    // edge_idxs must begin pointing to the beginning of edges
    geo->edge_idxs[0] = geo->edges;

    // Zero out the site_idx to the starting position.
    for(int i = 0; i < sites_ndim; i++)
        site_idx[i] = 0;
    // Make sure that, when we increment the first time, all will be zeros.
    site_idx[sites_ndim - 1] = -1;

    // Go through each site, updating edges and edge_idxs. site will be the
    // flat index of the current site.
    for(npy_intp site = 0; site < geo->num_sites; site++) {
        // Calculate the array index of the current site. This makes the
        // assumption the array is in C layout, which we double-checked above.
        for(npy_intp n = sites_ndim - 1; n >= 0; n--) {
            site_idx[n]++;
            if (site_idx[n] < sites_dim[n])
                break;
            else
                site_idx[n] = 0;
        }

        for(npy_intp i_nhd = 0; i_nhd < num_nhd; i_nhd++) {
            // Use the array index of the current site to find the array index
            // of the neighboring site.
            for(npy_intp d = 0; d < sites_ndim; d++) {
                nhd_idx[d] = site_idx[d] + *(npy_intp *)PyArray_GETPTR2(
                    np_neighborhood, i_nhd, d
                );
                nhd_idx[d] %= sites_dim[d];
                // Annoyingly, % in C is NOT a modulo operation, it's a
                // REMAINDER operation. So -3 % 2 = -1, not 1.
                if (nhd_idx[d] < 0)
                    nhd_idx[d] += sites_dim[d];
            }
            
            // Set the entry in edges.
            geo->edges[i_edge].from = geo->sites + site;
            // The PyArray_GetPtr implicitly converts the array index nhd_idx
            // to the flat index.
            geo->edges[i_edge].to = (state_type *)PyArray_GetPtr(
                np_sites, nhd_idx
            );
            i_edge++;
        }

        // Update edge_idxs
        geo->edge_idxs[site + 1] = geo->edges + i_edge;
    }

    return 0;
}


int encode_arbitrary(struct Model *geo, PyArrayObject *np_edge_idxs,
                     PyArrayObject *np_edges)
{
    // This encodes an arbitrary graph into a lookup table of edges and
    // edge_idxs. Those arrays are documented in graph_ops.h. The numpy arrays
    // np_edge_idxs, np_edges are already almost what we need, but we need to
    // change them from indices into pointers. In addition, we need to check if
    // the degree of each vertex is constant. If it is, then we can use some
    // additional optimizations. The function pointers in geo are set to assume
    // that it is constant by default. If it is not, we need to repoint them.

    // Buffers used in edge encoding. edge_idx_1 will be the index of the entry
    // in geo->edges that begins the edges for the current site, while
    // edge_idx_2 will be the index of the entry beginning the edges for the
    // next site.
    npy_intp edge_idx_1, edge_idx_2;

    // Input checking.
    if ((np_edge_idxs == NULL) || (np_edges == NULL)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Arbitrary graph requires the edges and edge_idxs parameters."
        );
        return -1;
    }
    if (geo == NULL) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "encode_arbitrary received null pointer."
        );
        return -1;
    }

    // Check size of np_edge_idxs matches number of sites
    if ((npy_uint64)PyArray_SIZE(np_edge_idxs) != geo->num_sites + 1) {
        PyErr_SetString(PyExc_ValueError, "edge_idxs has the wrong shape.");
        return -1;
    }

    // Allocate memory to edge_idxs, edges.
    geo->num_edges = (npy_uint64)PyArray_SIZE(np_edges);
    geo->edge_idxs = malloc(sizeof(Edge *) * (geo->num_sites + 1));
    geo->edges = malloc(sizeof(Edge) * geo->num_edges);
    geo->edge_idxs[0] = geo->edges;
    // Initialize edge_idx_2 so that, when we use edge_idx_2 to set edge_idx_1
    // in the first iteration of the loop, it will be set correctly.
    edge_idx_2 = *(npy_intp *)PyArray_GETPTR1(np_edge_idxs, 0);

    // Encode edge_idxs and edges. i will track the FROM side of the edge.
    for(npy_intp i = 0; i < geo->num_sites; i++) {
        // edge_idx_1 will be the index where we start with the edges from site
        // i in np_edges.
        edge_idx_1 = edge_idx_2;
        // edge_idx_2 will be the end.
        edge_idx_2 = *(npy_intp *)PyArray_GETPTR1(np_edge_idxs, i + 1);
        // Set edge_idxs for next site.
        geo->edge_idxs[i + 1] = geo->edges + edge_idx_2;

        // Encode edges.
        for(npy_intp edge_idx = edge_idx_1; edge_idx < edge_idx_2; edge_idx++) {
            geo->edges[edge_idx].from = geo->sites + i;
            geo->edges[edge_idx].to = geo->sites + 
                *(npy_intp *)PyArray_GETPTR1(np_edges, edge_idx);
        }
    }

    return 0;
}


int encode_fully_connected(struct Model *geo)
{
    // This encodes a fully-connected graph.

    // Input checking
    if (geo == NULL) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "encode_fully_connected received null pointer."
        );
        return -1;
    }

    geo->num_edges = (geo->num_sites - 1) * geo->num_sites;
    geo->edge_idxs = NULL;
    geo->edges = NULL;

    return 0;
}


int initialize_rules(struct Model *geo, PyArrayObject *np_beta,
                     float diffusion_probability, int graph_type)
{
    // Initializes the transition rules properties of the Model struct. Returns
    // 0 if successful and sets an error condition and returns -1 if not.
    //
    // It's important to note that this function DOES NO OPTIMIZATIONS. There
    // are various tweaks to your choice of rates that can yield significant
    // runtime optimizations. We expect these to be handled at the Python
    // layer; this function will simply accept what it is given. For example,
    // it will NOT check if there are rules for diffusion in np_beta and then
    // convert them to a diffusion rate itself. It also expects that np_beta
    // has already been normalized so that the maximum total rate is 1.0. It
    // checks to make sure it does not exceed 1.0 - since that can cause an
    // error condition - but it does not check that it is not less than 1.0 -
    // which will waste compute.
    //
    // obj_beta is the pointer to the data underlying the beta array, giving
    // the influence of adjacent states on state transitions.
    //
    // This function is responsible for filling the following properties of the
    // struct:
    //
    // num_states: The number of possible states in the system.
    // n_trans: n_trans[state_0][state_1] is the number of possible transitions
    //     (state_0, state_1) -> ?. We index each of these transitions by n for
    //     consistency.
    // trans_states: trans_states[state_0][state_1][n][j] is the destination
    //     state of element j of the pair (state_0, state_1) in transition n.
    // trans_thresh: transition_thresh[state_0][state_1][n] is the threshold
    //     of transition n. We generate a random number in the range [0,
    //     MAX_INTEGER_VALUE), then check transitions until we find one where
    //     the threshold exceeds the random number and execute that transition.
    //     The difference (trans_thresh[i] - trans_thresh[i - 1]) is equal to
    //     the transition probability multiplied by MAX_INTEGER_VALUE.

    // Convenience variable in initializing diffusion threshold.
    float diff_thresh;

    if (geo == NULL) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "initialize_rules received NULL Model pointer."
        );
        goto error;
    }
    if (np_beta == NULL) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "initialize_rules received NULL np_beta pointer."
        );
        goto error;
    }

    // Set diffusion thresholds.
    if ((diffusion_probability < 0) || (diffusion_probability > 1)) {
        PyErr_SetString(
            PyExc_ValueError,
            "diffusion_probability must be between 0.0 and 1.0."
        );
        goto error;
    }
    if (diffusion_probability > 0) {
        geo->diffusion_thresh = malloc(sizeof(prob_type) * (MAX_DIFFUSIONS + 1));
        if (geo->diffusion_thresh == NULL) {
            PyErr_SetString(
                PyExc_RuntimeError,
                "diffusion_thresh allocation error: please report this error as a git issue."
            );
            goto error;
        }
        diff_thresh = (float)PROB_TYPE_MAX * (1 - diffusion_probability);
        geo->diffusion_thresh[0] = (prob_type)diff_thresh;
        for(int i = 1; i < MAX_DIFFUSIONS - 1; i++) {
            diff_thresh *= diffusion_probability;
            geo->diffusion_thresh[i] = (
                (prob_type)diff_thresh + geo->diffusion_thresh[i - 1]
            );
        }
        geo->diffusion_thresh[MAX_DIFFUSIONS] = PROB_TYPE_MAX;
    }

    // Set run_system.
    if (graph_type == 0)
        geo->run_system = run_system_fully_connected;
    else if (diffusion_probability > 0)
        geo->run_system = run_system_with_diffusion;
    else
        geo->run_system = run_system_c;

    // Something important to discuss: for performance reasons, we keep our
    // probabilities not as floats, but as uint64s, where the probability is
    // equal to ("prob" / NPY_MAX_UINT64). The advantages of using integer over
    // floating point arithmetic are relatively mild, but it allows us to
    // eliminate some type casts.

    // Used to track the total transition probability to ensure it is <= 1.0.
    npy_float64 total_prob = 0;
    // Buffer used to hold a transition probability.
    npy_float64 prob;
    // Index used to index beta tensor.
    npy_intp i[4];
    // Buffer used to index transition. Honestly unnecessary, but it makes the
    // code cleaner.
    npy_intp n;
    // Pointers to blocks - we allocate stuff as a single block for greater
    // efficiency.
    state_type *trans_state_block;
    prob_type *trans_thresh_block;

    // Set num_states, then perform checks on shape and dtype of the beta
    // tensor.
    geo->num_states = (state_type)PyArray_DIM(np_beta, 0);

    // Check that we don't have > MAX_NUM_STATES or <= 0 states.
    if ((npy_intp)geo->num_states != PyArray_DIM(np_beta, 0)) {
        PyErr_SetString(
            PyExc_ValueError,
            "Number of states must be less than the maximum number."
        );
        goto error;
    }
    if (geo->num_states == 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "Number of states must be positive."
        );
        goto error;
    }
    for(int d = 1; d < 4; d++)
        if (PyArray_DIM(np_beta, d) != (npy_intp)geo->num_states) {
            PyErr_SetString(
                PyExc_ValueError,
                "beta must have size (num_states, num_states, num_states, num_states)."
            );
            goto error;
        }


    // We now begin assembling our transition rules. We iterate over every pair
    // of possible states (i[0], i[1]) and calculate the total number of
    // possible transitions.
    n = 0;
    for(i[0] = 0; i[0] < geo->num_states; i[0]++)
        for(i[1] = 0; i[1] < geo->num_states; i[1]++)
            for(i[2] = 0; i[2] < geo->num_states; i[2]++)
                for(i[3] = 0; i[3] < geo->num_states; i[3]++)
                    if (*(npy_float64 *)PyArray_GetPtr(np_beta, i) > 0) {
                        geo->n_trans[MAX_NUM_STATES * i[0] + i[1]]++;
                        n++;
                    }

    // With that done, we allocate the main memory blocks that we will
    // distribute among the subarrays. We use this strategy - of allocating a
    // single large block and then distributing it - because we find it
    // substantially improves runtime, likely due to better caching.
    trans_state_block = (state_type *)malloc(sizeof(state_type) * n * 2);
    if (trans_state_block == NULL) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "trans_state allocation error: please report this error as a git issue."
        );
        goto error;
    }
    // Stash this pointer so we know where to find it to free the block if
    // something goes wrong.
    geo->trans_state[0] = trans_state_block;

    trans_thresh_block = (prob_type *)malloc(sizeof(prob_type) * n);
    if (trans_thresh_block == NULL) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "trans_thresh allocation error: please report this error as a git issue."
        );
        goto error;
    }
    // Stash this pointer so we know where to find it to free the block if
    // something goes wrong.
    geo->trans_thresh[0] = trans_thresh_block;

    for(i[0] = 0; i[0] < geo->num_states; i[0]++)
        for(i[1] = 0; i[1] < geo->num_states; i[1]++) {
            total_prob = 0;
            // If any transitions are possible, then we need to record them.
            if (geo->n_trans[MAX_NUM_STATES * i[0] + i[1]] > 0) {
                // Allocate memory
                geo->trans_state[MAX_NUM_STATES * i[0] + i[1]] =
                    trans_state_block;
                geo->trans_thresh[MAX_NUM_STATES * i[0] + i[1]] =
                    trans_thresh_block;
                for(i[2] = 0; i[2] < geo->num_states; i[2]++)
                    for(i[3] = 0; i[3] < geo->num_states; i[3]++) {
                        prob = *(npy_float64 *)PyArray_GetPtr(np_beta, i);
                        if (prob > 0) {
                            *trans_state_block++ = (state_type)i[2];
                            *trans_state_block++ = (state_type)i[3];
                            total_prob += prob;
                            // Convert the probability from [0, 1] range to
                            // [0, PROB_TYPE_MAX]. This originally used ufromfp
                            // but that's not available in some non-glibc
                            // settings we want to support.
                            if (total_prob >= 1.0)
                                *trans_thresh_block++ = PROB_TYPE_MAX;
                            else
                                *trans_thresh_block++ = (prob_type)(
                                    (npy_float64)PROB_TYPE_MAX * total_prob
                                );
                        }
                    }
            }
            // Check if the total transition probability exceeds 1.0. If this
            // occurs, raise an error and abort.
            if (total_prob > 1.0) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "Maximum total transition rate exceeds 1.0."
                );
                goto error;
            }
        }

    return 0;

    error:
    
    return -1;
}


bool is_same_shape(PyArrayObject *array_1, PyArrayObject *array_2) {
    // Convenience function to check if two PyArrayObjects are the same shape.
    if (PyArray_NDIM(array_1) != PyArray_NDIM(array_2))
        return false;
    for(int d = 0; d < PyArray_NDIM(array_1); d++)
        if (PyArray_DIM(array_1, d) != PyArray_DIM(array_2, d))
            return false;
    return true;
}
