import numpy as np
import scipy.sparse as spa
from scipy.sparse.linalg import spsolve
from math import *

def voisins(S:int,G:np.array)->list:
    """
    Renvoie la liste des voisins d'un sommet S dans le graphe G représenté par une matrice d'adjacence.
    Suppose que G[i, j] == np.inf signifie qu'il n'y a pas d'arête.
    """
    # Optimized version using np.where for dense matrices
    # This is generally faster than a Python loop for large n.
    # Note: This includes S itself if G[S, S] is not inf. Filter if necessary.
    indices = np.where(G[S, :] != np.inf)[0]
    # Exclude self-loops if G[S,S] could be non-infinite and self-loops are not desired
    # return list(indices[indices != S])
    return list(indices)

def calculNouvellesPressions(Graphe:np.array, Blob:np.array, Terminaux:list, puit:int, débitEntrant:float, epsilon:float)->np.array:
    """Calcule les pressions aux noeuds du réseau en résolvant un système linéaire basé sur la loi de Kirchhoff.

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée (longueurs) du graphe original.
        Blob (np.array): Matrice d'adjacence pondérée (rayons) du réseau actuel.
        Terminaux (list): Liste des indices des noeuds terminaux (sources et puit).
        puit (int): Indice du noeud puit (pression fixée à 0).
        débitEntrant (float): Débit entrant à chaque noeud source (Terminaux sauf puit).

    Returns:
        np.array: Vecteur des pressions calculées à chaque noeud.
    """
    n=np.shape(Graphe)[0]

    A = spa.lil_matrix((n, n), dtype=float)  # Sparse matrix for kirchhoff conservation law relations
    B = np.zeros((n,))  # Right-hand side vector for sources/sinks
    
    # Set source terms
    for k in Terminaux:
        if k != puit:
            B[k] = -débitEntrant  # Sources have negative divergence (flow enters node)
    #B[puit] = (len(Terminaux) - 1) * débitEntrant  # Puit has positive divergence (flow exits node)

    # Build the matrix A based on Kirchhoff's current law (sum of flows = 0)
    # Flow = Conductance * Delta_Pressure / Length
    # Sum(C_ij / L_ij * (P_i - P_j)) = Source_i
    for S in range(n):
        
        # Fix pressure at the sink node = 0
        if S == puit:
            A[S, S] = 1.0
            B[S] = 0.0
            
        else:
            sum_conductance_over_length = 0.0
            for V in voisins(S, Blob):
                if Blob[S, V] != np.inf and Graphe[S, V] != np.inf and Graphe[S, V] > 0: # Check edge exists in both and length > 0
                    # Calculate effective conductance for the edge
                    term = Blob[S, V] / Graphe[S, V]  # Conductance / Length
                    
                    # Off-diagonal term
                    A[S, V] += term # Coefficient for P_V
                    sum_conductance_over_length += term

            # Diagonal term: Sum of conductances connected to node S
            A[S, S] = -sum_conductance_over_length # Coefficient for P_S

            # Handle nodes potentially disconnected from the main network but not the sink
            # The original code had a check for empty rows. A better approach
            # might be to ensure the graph connected component containing terminals is used,
            # or handle disconnected nodes explicitly. If a non-sink node `S` has no
            # connections in `Blob`, A[S,S] would be 0. Setting A[S,S]=1 ensures
            # matrix invertibility and sets P_S=0 if B[S]=0.
            if abs(A[S, S]) < epsilon and S not in Terminaux: # Check if row is effectively zero and not a source/sink
                 A[S, S] = 1.0
                 B[S] = 0.0 # Assign zero pressure to isolated nodes

    # Solve the linear system
    try:
        Pression = spsolve(A.tocsr(), B)
    except Exception as e: # spsolve might raise different errors
        print(f"Erreur lors de la résolution du système linéaire creux: {e}")
        Pression = np.full((n,), np.nan)

    return Pression