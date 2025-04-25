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


