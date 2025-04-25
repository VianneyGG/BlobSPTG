
import time
import heapq
from math import *
import time as t
import numpy as np
import networkx as nx

from Fonctions.Initialisation import *
from Fonctions.Puit import selectionPuit
from Fonctions.Pression import calculNouvellesPressions
from Fonctions.MiseAJour import miseAJour
from Fonctions.Outils import *


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

def MS3_PO(Graphe: np.array, Terminaux: set[int], M: int = 1, K: int = 3000, alpha: float = 0.15, mu: float = 1, delta: float = 0.2, epsilon: float = 1e-3, ksi: float = 1,  débitEntrant: float = 1, modeProba ='unif',modeRenfo='simple' , display_result: bool = True, step_callback=None) -> np.array:
    """itère M fois l'algorithme du Blob pour le probleme de l'arbre de Steiner

    Args:
        Graphe (np.array): _description_
        Terminaux (list): _description_
        M (int): nombre d'itération de l'algorithme du Blob
        K (int): nombre d'itération de l'évolution du Blob
        alpha (float): paramètre de la loi de renforcement
        mu (float): paramètre de la loi de renforcement
        delta (float): paramètre de la loi de renforcement
        epsilon (float): taille minimale des aretes
        ksi (float): paramètre (unused?)
        débitEntrant (float): Débit entrant
        modeProba (str): Mode de sélection du puit
        modeRenfo (str): Mode de renforcement
        display_result (bool): Whether to display the result using affichage (console/matplotlib).
        step_callback (callable, optional): Callback function called after each evolution step with the current blob state. Defaults to None.

    Returns:
        np.array: meilleur graphe obtenu après M*K itérations de l'algorithme du Blob
    """
    t_start = time.time() # Start timer
    n = np.shape(Graphe)[0]
    Pressions = np.array([])
    meilleur_blob = Graphe.copy()
    meilleur_poids = np.inf

    for i in range(M):

        #INITIALISATION ---------------------------------------------------------------
        current_blob = initialisation(Graphe)

        #ITERATION DE L'EVOLUTION DU BLOB ---------------------------------------------

        for j in range(K):
            Puit = selectionPuit(Graphe, Terminaux, modeProba)
            Pressions = calculNouvellesPressions(Graphe, current_blob, Terminaux, Puit, débitEntrant, epsilon)
            miseAJour(Graphe, current_blob, j, Pressions, alpha, mu, delta, epsilon, modeRenfo)

            # AFFICHAGE STEP BY STEP (Console/Matplotlib) -----------------------------
            if display_result:
                step_callback(current_blob.copy())
                

        # --- Fin de la boucle K ---
        # --- Choose a start node hint (e.g., the first terminal) ---
        mst_start_node = next(iter(Terminaux)) if Terminaux else 0 # Pick first terminal, or 0 if no terminals
        # ---------------------------------------------------------
        
        # --- Calculer l'arbre couvrant maximal (MaxST) sur le blob actuel ---
        mst_input_matrix = np.full_like(current_blob, np.inf) # Initialize with inf
        finite_mask = np.isfinite(current_blob) # Find finite values
        
        max_finite_conductivity = np.max(current_blob[finite_mask]) +1
        mst_input_matrix[finite_mask] = max_finite_conductivity - current_blob[finite_mask]
        mst_sim_structure = MSTbyPrim(mst_input_matrix, start_node_hint=mst_start_node)

        deleted_edges_mask = np.isinf(mst_sim_structure)
        current_blob[deleted_edges_mask] = np.inf # Set deleted edges to inf
        tree_edges_mask = np.isfinite(mst_sim_structure)
        current_blob[tree_edges_mask] = max_finite_conductivity + 1 - mst_sim_structure[tree_edges_mask] # Set tree edges to their original values
        # --- Fin du calcul MaxST ---
        
        for j in range(K//3):
            Puit = selectionPuit(Graphe, Terminaux, modeProba)
            Pressions = calculNouvellesPressions(Graphe, current_blob, Terminaux, Puit, débitEntrant, epsilon)
            miseAJour(Graphe, current_blob, j, Pressions, alpha, mu, delta, epsilon, modeRenfo)

            # AFFICHAGE STEP BY STEP (Console/Matplotlib) -----------------------------
            if display_result:
                step_callback(current_blob.copy())
        
        
        # Update best blob found so far across all M iterations
        current_poids = poids(Graphe, current_blob)
        if current_poids < meilleur_poids:
            meilleur_blob = current_blob.copy()
            meilleur_poids = current_poids

    # --- Fin de la boucle M ---

    # Calculer l'MST sur le meilleur blob trouvé
    if meilleur_poids != np.inf:
        print("Calcul de l'Arbre Couvrant Minimum (MST) sur le meilleur résultat...")
        return meilleur_blob
    else:
        print("Aucune solution valide trouvée (poids infini). Retourne un graphe vide.")
        return np.full_like(Graphe, np.inf)
