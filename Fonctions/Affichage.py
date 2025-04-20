import networkx as nx
import time
from math import *
import numpy as np
import matplotlib.pyplot as plt

def poids(Graphe:np.array, arbre:np.array)->float:
    """Calcule le poids total d'un sous-graphe/arbre défini par `arbre`
       en utilisant les poids (longueurs) de `Graphe` (vectorized).

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée (longueurs) du graphe original.
        arbre (np.array): Matrice d'adjacence indiquant les arêtes présentes dans le sous-graphe/arbre
                          (utilise np.inf pour indiquer l'absence d'arête).

    Returns:
        float: Poids total du sous-graphe/arbre.
    """
    # Create a mask of edges present in the 'arbre'
    mask_arbre = (arbre != np.inf)
    # Ensure corresponding edges exist in Graphe (avoid adding weights for edges only in arbre)
    mask_graphe = (Graphe != np.inf)
    valid_mask = mask_arbre & mask_graphe

    # Sum weights from Graphe where the edge exists in both and in 'arbre'
    # Divide by 2 because the matrix is symmetric and we sum both (i,j) and (j,i)
    poids_total = np.sum(Graphe[valid_mask]) / 2.0
    return poids_total

def np2nx(Graphe:np.array)->nx.Graph:
    """Convertit un graphe représenté par une matrice d'adjacence numpy (avec np.inf pour non-arête)
       en un objet graphe NetworkX. Les poids des arêtes sont stockés dans l'attribut 'poids'.

    Args:
        Graphe (np.array): Matrice d'adjacence numpy.

    Returns:
        nx.Graph: Graphe NetworkX correspondant.
    """
    # Using networkx function is generally more efficient and robust
    # Replace np.inf with 0 temporarily for from_numpy_array, then remove 0-weight edges if needed
    # Or create graph from edge list where weight is not inf
    G_temp = np.where(Graphe == np.inf, 0, Graphe)
    G_nx = nx.from_numpy_array(G_temp)
    # Relabel edges with 'poids' attribute and remove edges where original was inf
    edges_to_remove = []
    for u, v, d in G_nx.edges(data=True):
        original_weight = Graphe[u, v]
        if original_weight == np.inf:
            edges_to_remove.append((u, v))
        else:
            d['poids'] = original_weight
            # Remove the default 'weight' attribute if not needed
            # del d['weight']
    G_nx.remove_edges_from(edges_to_remove)
    return G_nx

def nx2np(Graphe_nx:nx.Graph, nodelist=None)->np.array:
    """Convertit un graphe NetworkX en une matrice d'adjacence numpy.
       Utilise np.inf pour représenter l'absence d'arête. Assume que les poids sont dans l'attribut 'poids'.

    Args:
        Graphe_nx (nx.Graph): Graphe NetworkX.
        nodelist (list, optional): Order of nodes for the matrix. Defaults to sorted(Graphe_nx.nodes()).

    Returns:
        np.array: Matrice d'adjacence numpy correspondante.
    """
    # Using networkx function is generally more efficient
    if nodelist is None:
        nodelist = sorted(Graphe_nx.nodes())
    # Use 'poids' as the weight attribute, set non-edges to np.inf
    G_np = nx.to_numpy_array(Graphe_nx, nodelist=nodelist, weight='poids', nonedge=np.inf)
    # Ensure diagonal is inf (or 0) based on convention
    np.fill_diagonal(G_np, np.inf) # Or 0
    return G_np

def affichage(Graphe:np.array, Blob:np.array, itération:int, pos:dict, Terminaux:list, temps_debut:float, resultat:bool=False)->None:
    """Affiche l'état actuel du Blob superposé au Graphe de base.

    Args:
        Graphe (np.array): Matrice d'adjacence (longueurs) du graphe original.
        Blob (np.array): Matrice d'adjacence (rayons) du réseau actuel.
        itération (int): Numéro de l'itération actuelle (pour titre et affichage conditionnel).
        pos (dict): Dictionnaire des positions des noeuds pour l'affichage {node_id: (x, y)}.
        Terminaux (list): Liste des indices des noeuds terminaux.
        temps_debut (float): Timestamp du début du calcul (obtenu avec time.time()).
        resultat (bool, optional): Si True, affiche le résultat final avec titre et temps. Défaut à False.
    """
    G_nx = np2nx(Graphe) # Convert base graph to networkx
    n = np.shape(Graphe)[0]

    plt.clf() # Clear previous plot

    # --- Draw Base Graph ---
    nx.draw_networkx_nodes(G_nx, pos, node_color="grey", node_size=300, alpha=0.8)
    nx.draw_networkx_nodes(G_nx, pos, nodelist=Terminaux, node_color="red", node_size=400)
    nx.draw_networkx_edges(G_nx, pos, width=1, edge_color="lightgrey", style='dotted')

    # --- Draw Blob Structure (Optimized edge/width collection) ---
    # Create mask for valid edges in Blob (upper triangle to avoid duplicates)
    i_indices, j_indices = np.triu_indices(n, k=1)
    valid_blob_mask = (Blob[i_indices, j_indices] != np.inf) & \
                      ~np.isnan(Blob[i_indices, j_indices]) & \
                      (Blob[i_indices, j_indices] > 1e-6)

    aretesBlob = list(zip(i_indices[valid_blob_mask], j_indices[valid_blob_mask]))
    radiiBlob = Blob[i_indices[valid_blob_mask], j_indices[valid_blob_mask]]

    # Calculate widths (vectorized)
    # Adjust scaling factor (1e6) and base (log10) as needed
    # Add 1 inside log10 to handle radii near 0 gracefully
    widthsBlob = np.maximum(0.1, np.log10(1 + 1e6 * radiiBlob))

    # Collect nodes involved in the blob structure
    noeudsBlob = set(i_indices[valid_blob_mask]) | set(j_indices[valid_blob_mask])

    # Draw Blob nodes (non-terminals involved in Blob)
    noeudsBlob_non_terminaux = list(noeudsBlob - set(Terminaux))
    nx.draw_networkx_nodes(G_nx, pos, nodelist=noeudsBlob_non_terminaux, node_color="green", node_size=300, alpha=0.7)
    # Draw Blob edges
    if aretesBlob: # Check if there are any edges to draw
        nx.draw_networkx_edges(G_nx, pos, edgelist=aretesBlob, width=widthsBlob, alpha=0.7, edge_color="green")

    # --- Labels ---
    labels = {i: str(i) for i in range(n)} # Node labels
    nx.draw_networkx_labels(G_nx, pos, labels, font_size=8, font_color="black")

    # --- Title and Display ---
    plt.axis('off') # Hide axes

    if resultat:
        temps_ecoule = time.time() - temps_debut
        # Calculate weight using the optimized function
        poids_final = poids(Graphe, Blob)
        titre = f'Résultat final ({temps_ecoule:.2f}s) - Poids: {poids_final:.2f}'
        plt.title(titre)
        plt.show() # Show final result, blocks execution

    else:
        # Conditional display for intermediate steps
        # Plotting frequently can significantly slow down the simulation.
        # Increase the interval or save figures to files instead of displaying interactively.
        plot_interval = 500 # Example: Plot every 500 iterations
        if itération % plot_interval == 0 and itération != 0:
            titre = f'Itération {itération}'
            plt.title(titre)
            plt.show(block=False) # Show non-blocking plot
            plt.pause(0.5) # Pause briefly
            # Consider saving the figure instead:
            # plt.savefig(f"output/iteration_{itération:06d}.png", bbox_inches='tight', dpi=150)
            # plt.close() # Close automatically if saving


