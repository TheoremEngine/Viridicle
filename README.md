## What is Viridicle? <img align="right" src="example.gif">

Viridicle is a library for simulating stochastic graphical ecological models. It implements the continuous time models described in Section 6 of [Durrett and Levin 1994](https://www.researchgate.net/publication/230693004_Stochastic_Spatial_Models_A_User's_Guide_to_Ecological_Applications). The library is written in C for speed, but has a Python API for ease of use. It supports well-mixed systems, multi-dimensional periodic lattices, and arbitrary graphs, with up to 256 species, and is substantially faster than comparable software written in Python alone.

Viridicle models an ecological system as a graph, where each vertex denotes a discrete site - an island, an oasis, a point on a Petri dish, etc. Each site is small enough that we can model its state by assigning it a single number specifying which species is occupying it. Traditionally, 0 denotes an unoccupied site, but neither the mathematics nor the code require this.

The system changes over time as a [continuous-time Markov chain](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain), where each directed edge in the system is a [Poisson point process](https://en.wikipedia.org/wiki/Poisson_point_process). That is, at any given moment, for every edge in the system, the time until the edge's vertices change to state $(s_1,s_2)$ is exponentially distributed with some rate that depends on the state of the system. The model is implemented using the [Gillespie algorithm](https://en.wikipedia.org/wiki/Gillespie_algorithm). For further details, consult the [documentation](https://www.theorem-engine.org/viridicle/index.html).

## Installation

To install Viridicle from the PyPi repo, run:

```
pip install viridicle
```

If you want to compile from source yourself, first make sure that you have gcc installed, along with Python, including its development headers. Then, run:

```
git clone https://github.com/TheoremEngine/viridicle.git
cd viridicle
python3 setup.py install
```
