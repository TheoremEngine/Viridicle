viridicle: A Python Library for Stochastic Ecological Models on Graphs
======================================================================

viridicle is a simple module for running stochastic ecological models on graphs, with a Python API but exploiting C for speed. It can be downloaded from:

`https://github.com/TheoremEngine/viridicle <https://github.com/TheoremEngine/viridicle>`_

Or installed by:

.. code-block:: bash

    pip install viridicle

The basic idea of these models is that you have a graph - usually, but not necessarily, a 2-dimensional lattice - with a state associated to each vertex in the graph. Every pair of adjacent vertices evolves stochastically: for every pair of possible states, there exist a list of possible transitions, and the time until that transition occurs is an exponentially distributed random variable with a specified rate. For example, if 0 denotes an empty site and 1 denotes a site occupied by species 1, then species 1 might reproduce with the transition :math:`(1, 0)\to(1, 1)` at rate 0.2. Then, for any pair of adjacent vertices in state 1 and 0 respectively, the time until the state of the vertex that is currently 0 changes to state 1 is exponentially distributed with rate 0.2, unless some other transition occurs that changes the states of the vertices first. For further details on the mathematics, consult the links below or our paper.

To construct such a model, you can specify the transition rules as either a list of strings, or a 4-dimensional numpy array. If using the first option, then each string needs to be of the form:

.. code-block:: Python

    rules = ['(i,j)->(k,l)@r']

The above rule specifies that an adjacent pair in states :math:`i,j` evolves to states :math:`k,l` at rate :math:`r`. For example, we could write our example above as:

.. code-block:: Python

    rules = ['(1,0)->(1,1)@0.2']

Suppose we also wanted to add diffusion, where individuals of species 1 move around. We could do that with:

.. code-block:: Python

    rules += ['(1,0)->(0,1)@0.3']

And so on. Alternatively, we can specify the rules as a 4-dimensional floating point numpy array ``rules``, each of whose sides has length equal to the number of possible states. Then, ``rules[i, j, k, l]`` gives the rate at which the transition :math:`(i,j)\to(k, l)` occurs. For example, we could encode the above as:

.. code-block:: Python

    import numpy as np
    rules = np.zeros((2, 2, 2, 2))
    rules[0, 1, 1, 1] = 0.2
    rules[0, 1, 1, 0] = 0.3

Once we have designed our rules, we construct a :class:`Geography` to hold them. viridicle offers three :class:`Geography` classes, each corresponding to different types of graphs: :class:`FullyConnectedGeography`, :class:`LatticeGeography`, and :class:`ArbitraryGeography`. Let's suppose we want to run our simulation on a two-dimensional periodic :math:`500\times500` lattice. Then:

.. code-block:: Python

    import viridicle
    geo = viridicle.LatticeGeography((500, 500), rules, num_states=2)

We can then run the system by calling the run method:

.. code-block:: Python

    counts = geo.run(1000, report_every=10)

This will run the simulation for 1000 time, and will return a numpy array containing the counts of the different states over time, with the population captured every 10 time units.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   gillespie
   may_leonard
   api
   utils
   bibliography
