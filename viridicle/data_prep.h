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

#ifndef VIRIDICLE_DATA_PREP_HEADER
#define VIRIDICLE_DATA_PREP_HEADER

#include "graph_ops.h"

struct Model * create_model();
void free_model(struct Model *geo);

int coerce_sites(PyObject *obj_sites, PyArrayObject **site_ptr);
int coerce_rng(PyObject *capsule, bitgen_t **rng_ptr);
int coerce_beta(PyObject *obj_beta, PyArrayObject **beta_ptr);
int coerce_nhd(PyObject *obj_nhd, PyArrayObject **nhd_ptr);
int coerce_edge_idxs(PyObject *obj_edge_idxs, PyArrayObject **edge_idxs_ptr);
int coerce_edges(PyObject *obj_edges, PyArrayObject **edges_ptr);
int coerce_clusters_out(PyObject *obj_out, PyArrayObject **out_ptr);

int initialize_graph(struct Model *model, int graph_type,
                     PyArrayObject *np_sites, PyArrayObject *np_neighborhood,
                     PyArrayObject *np_edge_idxs, PyArrayObject *np_edges);
int encode_fully_connected(struct Model *geo);
int encode_lattice(struct Model *geo, PyArrayObject *np_sites,
                   PyArrayObject *np_neighborhood);
int encode_arbitrary(struct Model *geo, PyArrayObject *np_edge_idxs,
                     PyArrayObject *np_edges);

int initialize_rules(struct Model *geo, PyArrayObject *np_beta,
                     float diffusion_probability, int graph_type);

bool is_same_shape(PyArrayObject *array_1, PyArrayObject *array_2);

#endif
