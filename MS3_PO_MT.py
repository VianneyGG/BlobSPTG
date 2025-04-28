import sys
import time
import heapq
from math import *
import time as t
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import networkx as nx

from Fonctions.Initialisation import *
from Fonctions.Puit import selectionPuit
from Fonctions.Pression import calculNouvellesPressions
from Fonctions.MiseAJour import miseAJour
from Fonctions.Outils import poids # Ensure poids is imported if not already
from tqdm.auto import tqdm

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

# Change function signature: remove step_callback, add should_record_steps
def run_single_blob_simulation(Graphe, Terminaux, K, alpha, mu, delta, epsilon, débitEntrant, modeProba, modeRenfo, should_record_steps, m_idx):
    """
    Runs a single simulation instance. Returns intermediate steps if requested.
    Steps are recorded as (step_index, blob_state).
    """
    # Store tuples: (step_index, blob_state)
    intermediate_steps = [] if should_record_steps else None

    # INITIALISATION ---------------------------------------------------------------
    current_blob = initialisation(Graphe)
    current_step_index = 0
    if should_record_steps:
        intermediate_steps.append((current_step_index, current_blob.copy())) # Record initial state (step 0)

    # ITERATION DE L'EVOLUTION DU BLOB ---------------------------------------------
    for j in range(K):
        current_step_index = j + 1
        Puit = selectionPuit(Graphe, Terminaux, modeProba)
        Pressions = calculNouvellesPressions(Graphe, current_blob, Terminaux, Puit, débitEntrant, epsilon)
        miseAJour(Graphe, current_blob, j, Pressions, alpha, mu, delta, epsilon, modeRenfo)

        # Record step if needed (every 10 steps)
        if should_record_steps:
            if current_step_index % 10 == 0: # Check if step index is a multiple of 10
                intermediate_steps.append((current_step_index, current_blob.copy()))

    # --- Choose a start node hint ---
    mst_start_node = next(iter(Terminaux)) if Terminaux else 0
    current_step_index = K + 1 # Step index after K iterations

    # --- Calculer l'arbre couvrant maximal (MaxST) ---
    mst_input_matrix = np.full_like(current_blob, np.inf)
    finite_mask = np.isfinite(current_blob)

    if np.any(finite_mask):
        max_finite_conductivity = np.max(current_blob[finite_mask]) + 1
        mst_input_matrix[finite_mask] = max_finite_conductivity - current_blob[finite_mask]
        mst_sim_structure = MSTbyPrim(mst_input_matrix, start_node_hint=mst_start_node)

        deleted_edges_mask = np.isinf(mst_sim_structure)
        current_blob[deleted_edges_mask] = np.inf
        tree_edges_mask = np.isfinite(mst_sim_structure)
        if np.any(tree_edges_mask):
             current_blob[tree_edges_mask] = max_finite_conductivity + 1 - mst_sim_structure[tree_edges_mask]

        # Record state after initial MST (always record this step)
        if should_record_steps:
            intermediate_steps.append((current_step_index, current_blob.copy()))

        # Refinement loop after MaxST
        for j in range(K // 3):
            current_step_index = K + 1 + j + 1 # Step index during refinement
            Puit = selectionPuit(Graphe, Terminaux, modeProba)
            Pressions = calculNouvellesPressions(Graphe, current_blob, Terminaux, Puit, débitEntrant, epsilon)
            miseAJour(Graphe, current_blob, K + j, Pressions, alpha, mu, delta, epsilon, modeRenfo)

            # Record refinement step if needed (every 10 steps relative to start of refinement)
            if should_record_steps:
                # Check if the refinement step number (j+1) is a multiple of 10
                if (j + 1) % 10 == 0:
                    intermediate_steps.append((current_step_index, current_blob.copy()))
    else:
        current_blob.fill(np.inf)
        print(f"Worker {m_idx}: Blob became empty after K steps. Skipping refinement.")
        # Record empty state if needed (always record this step)
        if should_record_steps:
            intermediate_steps.append((current_step_index, current_blob.copy()))

    # Calculate the weight of the resulting blob/tree using original graph weights
    current_poids = poids(Graphe, current_blob)

    # Return final blob, weight, and the list of steps (or None)
    return current_blob, current_poids, intermediate_steps

def MS3_PO_MT(Graphe: np.array, Terminaux: set[int], M: int = 1, K: int = 3000, alpha: float = 0.15, mu: float = 1, delta: float = 0.2, epsilon: float = 1e-3, ksi: float = 1,  débitEntrant: float = 1, modeProba ='unif',modeRenfo='simple' , display_result: bool = True, step_callback=None) -> np.array:
    """
    Itère M fois l'algorithme du Blob en parallèle pour le probleme de l'arbre de Steiner.

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée du graphe initial (coûts).
        Terminaux (set[int]): Ensemble des indices des nœuds terminaux.
        M (int): Nombre d'itérations parallèles de l'algorithme du Blob.
        K (int): Nombre d'itérations de l'évolution du Blob dans chaque simulation.
        alpha (float): Paramètre de la loi de renforcement.
        mu (float): Paramètre de la loi de renforcement.
        delta (float): Paramètre de la loi de renforcement.
        epsilon (float): Conductivité minimale des arêtes considérées.
        ksi (float): Paramètre (unused?).
        débitEntrant (float): Débit entrant pour le calcul des pressions.
        modeProba (str): Mode de sélection du puits ('unif', etc.).
        modeRenfo (str): Mode de renforcement ('simple', 'vieillesse').
        display_result (bool): Si True, active la progression via step_callback pour le premier worker (m_idx=0).
        step_callback (callable, optional): Callback pour l'affichage pas à pas.
                                             Receives (current_blob_state, step_index). Defaults to None.

    Returns:
        np.array: Matrice d'adjacence de l'arbre de Steiner approximatif (MST du meilleur blob).
                  Retourne un graphe vide (rempli de np.inf) si aucune solution valide n'est trouvée.
    """
    t_start = time.time()
    n = np.shape(Graphe)[0]

    # Determine number of workers
    num_workers = min(M, cpu_count()) # Use at most M workers, limited by CPU cores
    print(f"Lancement de {M} simulations sur {num_workers} coeurs...")

    # Prepare arguments for each parallel task
    tasks_args = []
    for i in range(M):
        should_record = display_result and i == 0 and step_callback is not None # Only record if display and callback exist
        tasks_args.append(
            (Graphe, Terminaux, K, alpha, mu, delta, epsilon, débitEntrant, modeProba, modeRenfo,
             should_record,
             i)
        )

    results = []
    recorded_steps = None # To store steps (tuples) from worker 0

    # Run simulations in parallel
    with Pool(processes=num_workers) as pool:
        # results will be a list of tuples: (blob, poids, steps_list_or_None)
        # steps_list_or_None contains tuples: (step_index, blob_state)
        results = list(tqdm(pool.starmap(run_single_blob_simulation, tasks_args), total=M, desc="Simulations Blob"))

    # Process results: find best blob and extract steps from worker 0
    meilleur_blob = None
    meilleur_poids = np.inf
    all_weights = []

    for i, (blob, poids_blob, steps) in enumerate(results):
        all_weights.append(float(poids_blob) if np.isfinite(poids_blob) else float('inf'))
        if poids_blob < meilleur_poids and poids_blob > 0:
            meilleur_poids = poids_blob
            meilleur_blob = blob
        if i == 0 and steps is not None: # Check if this is worker 0's result and steps were recorded
            recorded_steps = steps

    # If steps were recorded and a callback exists, execute the callback for each recorded step
    if recorded_steps and step_callback:
        print(f"Processing {len(recorded_steps)} recorded steps for display...")
        for original_step_index, step_data in recorded_steps: # Unpack the tuple
            step_callback(step_data, original_step_index) # Pass original index to GUI
        print("Finished processing steps.")


    print(f"\nTemps total d'exécution : {time.time() - t_start:.2f} secondes")
    # print(all_weights)

    # Return the best blob found
    if meilleur_blob is not None and meilleur_poids != np.inf:
        print(f"Poids de l'arbre de Steiner final (MST) : {meilleur_poids}")
        # Return the best blob AND its weight, and the recorded steps (for potential future use, though GUI handles it now)
        return meilleur_blob #, meilleur_poids, recorded_steps # Modified return for consistency? Let's stick to just blob for now as GUI expects.
    else:
        print("Aucune solution valide trouvée après M simulations. Retourne un graphe vide.")
        # Return an empty graph AND infinity weight, and recorded steps
        return np.full_like(Graphe, np.inf) #, np.inf, recorded_steps
