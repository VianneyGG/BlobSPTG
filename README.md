# Steiner Tree  Sovler by Blob simulation

## Steiner Tree Problem in Graphs (STPG)
The Steiner tree problem in Graph : Given a non-directed connected graph with non negative weighted edges composed of G vertices, and a subset T (terminals) of these vertices, find a tree of minimum weight that contains all terminal vertices but may include additional vertices.

### Example :
![Steiner Tree Example](/media/Graph_Example.PNG)

This graph shows an optimal solution for the Steiner tree problem, with v1, v2, v3 and v5 as terminal nodes

## The Blob (Physarium Polycephalum)
![US roads by a blob](/media/Blob_map.jpg)
The Blob is a single-celled organism that can solve complex problems such as finding the shortest path in a maze or optimizing networks. It has been used as a model for studying decision-making processes and network optimization.

## Blob Simulation for Steiner Tree Problem
in this project, we simulate the Blob's behavior to solve the Steiner tree problem in graphs. The Blob's growth and movement are modeled to find an optimal Steiner tree by simulating its exploration of the graph.

---

## Physarum Solver: Principle and Mathematical Model

In the Physarum Solver, the graph is modeled as a network where each edge carries a protoplasmic flux. For the shortest path problem, two terminals represent nutrient sources for *Physarum polycephalum*: one as the source node and the other as the sink node. Protoplasmic flux enters the graph at the source and exits at the sink. Each vertex has an associated pressure, and the flux $ Q_{ij} $ through edge $(i, j)$ is proportional to the pressure difference between its endpoints, following the Hagen-Poiseuille equation:

$ Q_{ij} = D_{ij} / c_{ij} \cdot (p_i - p_j) $
$ D_{ij} = \frac{\pi r_{ij}^4}{8\xi} $ (1)

where:
- $ D_{ij} $ is the edge conductivity,
- $ c_{ij} $ is the edge length,
- $ p_i, p_j $ are the pressures at vertices $ i $ and $ j $,
- $ r_{ij} $ is the edge radius,
- $ \xi $ is the viscosity coefficient.

The conductivity update equation describes how the tube thickness (and thus conductivity) evolves:

$ \frac{d}{dt} D_{ij} = \alpha |Q_{ij}| - \mu D_{ij} $  (8)

where $ \alpha $ and $ \mu $ are positive constants. This means conductivities increase on edges with higher flux, reflecting the biological mechanism where Physarum's tubes thicken with increased flow.

To update conductivities, pressures are first computed by enforcing conservation of flux at each vertex, leading to the network Poisson equation:

$ \sum_{i \in V(j)} D_{ij} / c_{ij} \cdot (p_i - p_j) =
\begin{cases}
-I_0, & j = \text{source} \\
+I_0, & j = \text{sink} \\
0, & \text{otherwise}
\end{cases}
$  (6)

where $ V(j) $ is the set of neighbors of $ j $, and $ I_0 $ is the total flux.
To ch


After calculating pressures (setting the sink node pressure to zero), fluxes and conductivities are updated iteratively. Edges with conductivity below a threshold $ \varepsilon $ are pruned. For the shortest path problem, this process converges to the unique shortest path. However, the standard Physarum Solver only handles two terminals; solving the Steiner Tree Problem (STPG) with multiple terminals requires new algorithmic adaptations.

## The algorithm

![Physarum Solver Algorithm](/media/algo.png)

## Installation
Run with python3 the script `gui.py` to launch the GUI.

### Dependencies
This project requires the following Python packages:
*   NumPy (for numerical operations)
*   SciPy (for the `scipy.sparse` module)
*   NetworkX (optional, for graph manipulation)
*   Matplotlib (optional, for visualization)

### Code Structure

The code is organized into several modules in the `Fonctions` folder:

- `Initialisation.py`: Contains functions for initializing the graph and parameters.
- `Sink.py`: Contains functions for selecting the sink node.
- `Pression.py`: Contains functions for calculating new pressures based on the current state of the graph.
- `Update.py`: Contains functions for updating the graph based on the current pressures and fluxes.
- `Tools.py`: Contains utility functions, including the conversion from NetworkX graphs to NumPy arrays.
- `Outils.py`: Contains utility functions, including the conversion from NetworkX graphs to NumPy arrays (deprecated, replaced by `Tools.py`).


The files `MS3_PO.py` and `MS3_PO_MT.py` implement the main algorithms for solving the Steiner tree problem using the Physarum Solver approach. The `gui.py` file provides a graphical user interface for interacting with the solver.