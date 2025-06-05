import time
import heapq
from math import *
import time as t
import numpy as np
import networkx as nx

from Fonctions.Initialisation import *
from Fonctions.Sink import selectionPuit
from Fonctions.Pression import calculNouvellesPressions
from Fonctions.Update import miseAJour
from Fonctions.Tools import poids # Ensure poids is imported


def MSTbyPrim(graph_matrix: np.array, start_node_hint: int = 0) -> np.array:
    """
    Computes the Minimum Spanning Tree (MST) of a graph represented by an adjacency matrix
    using Prim's algorithm with a min-heap. Assumes weights represent costs (lower is better).

    Args:
        graph_matrix (np.array): A square NumPy array representing the weighted adjacency matrix.
                                 Use np.inf for non-existent edges. Assumes non-negative weights (costs).
        start_node_hint (int): Preferred node to start the MST search from. Defaults to 0.

    Returns:
        np.array: The adjacency matrix of the MST. Non-edges are represented by np.inf.
                  Returns an empty graph if the input is empty.
                  If the graph is disconnected, returns the MST of the connected component
                  containing the starting node.
    """
    num_nodes = graph_matrix.shape[0]
    if num_nodes == 0:
        return np.full((0, 0), np.inf)

    # Validate start_node_hint or default to 0
    if not (0 <= start_node_hint < num_nodes):
        print(f"Warning: Invalid start_node_hint ({start_node_hint}). Defaulting to node 0.")
        start_node = 0
    else:
        start_node = start_node_hint

    # Initialize MST adjacency matrix with infinity
    mst_matrix = np.full((num_nodes, num_nodes), np.inf)
    # Set of nodes already included in the MST
    visited_nodes = set()
    # min_weight[i] stores the minimum weight (cost) of an edge connecting node i to the MST
    min_weight = np.full(num_nodes, np.inf)
    # parent_node[i] stores the node in the MST that connects to node i via the min_weight edge
    parent_node = np.full(num_nodes, -1, dtype=int)
    # Priority queue storing tuples: (weight, target_node). Uses min-heap property.
    priority_queue = []

    # Start Prim's algorithm from the chosen start_node
    min_weight[start_node] = 0
    # No parent for the start node, parent_node[start_node] remains -1
    heapq.heappush(priority_queue, (0.0, start_node)) # Use float for weight

    while priority_queue and len(visited_nodes) < num_nodes:
        # Get the node 'current_node' closest (minimum weight edge) to the MST
        # that hasn't been visited yet.
        current_weight, current_node = heapq.heappop(priority_queue)

        # If node already visited or a shorter path (lower weight edge) was found
        # previously (due to heap not supporting decrease-key), skip this entry.
        if current_node in visited_nodes or current_weight > min_weight[current_node]:
            continue

        # Add the current node to the MST
        visited_nodes.add(current_node)

        # Add the edge connecting this node to the MST to the result matrix
        # (if it's not the start node, which has no parent).
        if parent_node[current_node] != -1:
            prev_node = parent_node[current_node]
            # The weight of the edge added is the one that caused current_node to be selected.
            edge_weight_in_mst = min_weight[current_node]
            mst_matrix[current_node, prev_node] = edge_weight_in_mst
            mst_matrix[prev_node, current_node] = edge_weight_in_mst # Undirected graph

        # Explore neighbors of the current node to update potential connections to the MST
        for neighbor_node in range(num_nodes):
            # Get the weight (cost) of the edge from the original graph matrix
            edge_weight = graph_matrix[current_node, neighbor_node]

            # Check if neighbor_node is a valid neighbor (edge exists) and not yet in the MST
            if edge_weight != np.inf and neighbor_node not in visited_nodes:
                # If this edge offers a cheaper connection to neighbor_node than previously found
                if edge_weight < min_weight[neighbor_node]:
                    # Update the minimum weight, parent, and push to the priority queue
                    min_weight[neighbor_node] = edge_weight
                    parent_node[neighbor_node] = current_node
                    heapq.heappush(priority_queue, (edge_weight, neighbor_node))

    # Check if the graph was disconnected (not all nodes visited)
    if len(visited_nodes) < num_nodes:
        # Updated warning message
        print(f"Warning: Input graph appears disconnected. MST found only for the component containing the start node ({start_node}).")

    return mst_matrix

def MS3_PO(Graphe: np.array, Terminaux: set[int], M: int = 1, K: int = 3000, alpha: float = 0.15, mu: float = 1, delta: float = 0.2, epsilon: float = 1e-3, débitEntrant: float = 1, modeProba ='unif',modeRenfo='simple' , display_result: bool = True, step_callback=None) -> np.array:
    """itère M fois l'algorithme du Blob pour le probleme de l'arbre de Steiner
    
    Args:
        Graphe (np.array): Matrice d'adjacence pondérée du graphe initial (coûts).
        Terminaux (set[int]): Ensemble des indices des nœuds terminaux.
        M (int): Nombre d'itérations de l'algorithme du Blob (Note: single-threaded version effectively runs M=1).
        K (int): Nombre d'itérations de l'évolution du Blob.
        alpha (float): Paramètre de la loi de renforcement.
        mu (float): Paramètre de la loi de renforcement.
        delta (float): Paramètre de la loi de renforcement.
        epsilon (float): Conductivité minimale des arêtes considérées.
        ksi (float): Paramètre (unused?).
        débitEntrant (float): Débit entrant pour le calcul des pressions.
        modeProba (str): Mode de sélection du puits ('unif', etc.).
        modeRenfo (str): Mode de renforcement ('simple', 'vieillesse').
        display_result (bool): If True and step_callback is provided, calls step_callback.
        step_callback (callable, optional): Callback function called after each evolution step.
                                             Receives (current_blob_state, step_index). Defaults to None.

    Returns:
        np.array: Matrice d'adjacence de l'arbre de Steiner approximatif (MST du meilleur blob).
                  Retourne un graphe vide (rempli de np.inf) si aucune solution valide n'est trouvée.
    """
    t_start = time.time() # Start timer
    n = np.shape(Graphe)[0]
    Pressions = np.array([])
    meilleur_blob = None # Initialize meilleur_blob to None
    meilleur_poids = np.inf

    # Single-threaded version effectively runs only one simulation (M=1)
    num_simulations = 1
    if M > 1:
        print("Warning: MS3_PO is single-threaded, running only 1 simulation (M=1).")

    for i in range(num_simulations): # Loop only once

        #INITIALISATION ---------------------------------------------------------------
        current_blob = initialisation(Graphe)
        current_step_index = 0
        # Initial callback call (step 0)
        if display_result and step_callback is not None:
            step_callback(current_blob.copy(), current_step_index)

        #ITERATION DE L'EVOLUTION DU BLOB ---------------------------------------------
        # Main evolution loop
        for j in range(K):
            current_step_index = j + 1
            Puit = selectionPuit(Graphe, Terminaux, modeProba)
            Pressions = calculNouvellesPressions(Graphe, current_blob, Terminaux, Puit, débitEntrant, epsilon)
            miseAJour(Graphe, current_blob, j, Pressions, alpha, mu, delta, epsilon, modeRenfo)

            # AFFICHAGE STEP BY STEP (every 10 steps) -----------------------------
            if display_result and step_callback is not None:
                if current_step_index % 10 == 0: # Check if step index is a multiple of 10
                    step_callback(current_blob.copy(), current_step_index)


        # --- Fin de la boucle K ---
        # --- Choose a start node hint ---
        mst_start_node = next(iter(Terminaux)) if Terminaux else 0
        # ---------------------------------------------------------

        # --- Calculer l'arbre couvrant maximal (MaxST) ---
        mst_input_matrix = np.full_like(current_blob, np.inf)
        finite_mask = np.isfinite(current_blob)
        current_step_index = K + 1 # Step index after K iterations

        if np.any(finite_mask):
            max_finite_conductivity = np.max(current_blob[finite_mask]) + 1
            mst_input_matrix[finite_mask] = max_finite_conductivity - current_blob[finite_mask]
            mst_sim_structure = MSTbyPrim(mst_input_matrix, start_node_hint=mst_start_node)

            deleted_edges_mask = np.isinf(mst_sim_structure)
            current_blob[deleted_edges_mask] = np.inf
            tree_edges_mask = np.isfinite(mst_sim_structure)
            if np.any(tree_edges_mask):
                 current_blob[tree_edges_mask] = max_finite_conductivity + 1 - mst_sim_structure[tree_edges_mask]

            # Callback after initial MST calculation (always show this step)
            if display_result and step_callback is not None:
                 step_callback(current_blob.copy(), current_step_index)

            # Refinement loop after MaxST
            for j in range(K // 3):
                current_step_index = K + 1 + j + 1 # Step index during refinement
                Puit = selectionPuit(Graphe, Terminaux, modeProba)
                Pressions = calculNouvellesPressions(Graphe, current_blob, Terminaux, Puit, débitEntrant, epsilon)
                miseAJour(Graphe, current_blob, K + j, Pressions, alpha, mu, delta, epsilon, modeRenfo)

                # AFFICHAGE STEP BY STEP (every 10 steps relative to the start of refinement)
                if display_result and step_callback is not None:
                    # Check if the refinement step number (j+1) is a multiple of 10
                    if (j + 1) % 10 == 0:
                        step_callback(current_blob.copy(), current_step_index)
        else:
            current_blob.fill(np.inf)
            print("Blob became empty after K steps. Skipping refinement.")
            # Callback even if empty (always show this step)
            if display_result and step_callback is not None:
                 step_callback(current_blob.copy(), current_step_index)


        # Update best blob found so far
        current_poids = poids(Graphe, current_blob)
        if np.isfinite(current_poids) and current_poids < meilleur_poids and current_poids > 0:
            meilleur_blob = current_blob.copy()
            meilleur_poids = current_poids
        elif meilleur_blob is None and np.isfinite(current_poids) and current_poids > 0:
             meilleur_blob = current_blob.copy()
             meilleur_poids = current_poids


    # --- Fin de la boucle M (runs once) ---
    print(f"\nTemps total d'exécution : {time.time() - t_start:.2f} secondes")

    # Return the best blob found (which is the result of the single simulation)
    if meilleur_blob is not None and meilleur_poids != np.inf:
        print(f"Poids de l'arbre de Steiner final (MST) : {meilleur_poids}")
        return meilleur_blob
    else:
        print("Aucune solution valide trouvée (poids infini ou 0). Retourne un graphe vide.")
        return np.full_like(Graphe, np.inf)
