from math import *
import numpy as np
from Pression import Conductivity

def miseAJourDébits(Graphe:np.array, Blob:np.array, Pressions: np.array, alpha:float, mu:float, arbreDeSteiner:np.array, delta:float, mode)->None:
    """Met à jour la matrice des débits (flux) entre les noeuds (vectorized).

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée (longueurs).
        Blob (np.array): Matrice d'adjacence pondérée (rayons).
        Pressions (np.array): Vecteur des pressions à chaque noeud.

    Returns:
        np.array: Matrice (anti-symétrique) des débits entre les noeuds. Débit[i, j] > 0 signifie flux de i vers j.
    """
    n = np.shape(Graphe)[0]

    # Create masks for valid edges (exist in both Graphe and Blob, and have positive length)
    valid_edge_mask = (Blob != np.inf)

    # Calculate flows for valid edges (vectorized)
    # Qij = Cij * (Pi - Pj) / Lij
    Q_matrix = np.full_like(Blob, 0.0, dtype=float)
    P_diff = Pressions[:, np.newaxis] - Pressions[np.newaxis, :]
    Q_matrix[valid_edge_mask] = Blob[valid_edge_mask] * P_diff[valid_edge_mask] / Graphe[valid_edge_mask]
    
    # Update conducivity 
    Blob[valid_edge_mask] = relationRenforcement(Blob[valid_edge_mask], Q_matrix[valid_edge_mask], alpha, mu, arbreDeSteiner[valid_edge_mask], delta, mode)

def relationRenforcement(Conductivity:float, Flow:float, alpha:float, mu:float,arbreDeSteiner:np.array, delta:float, mode:str) -> float:
    """Applique la relation de renforcement pour mettre à jour la Conductivity d'une arête.

    Args:
        Rayon (float): Rayon actuel de l'arête.
        Debit (float): Débit absolu à travers l'arête.
        alpha (float): Paramètre de renforcement lié au débit.
        mu (float): Paramètre d'affaiblissement (decay).

    Returns:
        float: Nouvelle valeur de Conductivity après renforcement/affaiblissement.
    """
    if mode == 'simple':
        return np.maximum(0.0, alpha * np.abs(Flow) - (mu-1) * Conductivity)

def miseAjourRayons(Blob:np.array, epsilon:float)->None:
    """Met à jour les rayons du réseau (Blob) en fonction des débits.

    Args:
        Blob (np.array): Matrice d'adjacence pondérée (rayons).
        epsilon (float): Valeur minimale pour les rayons.

    Returns:
        None: Modifie la matrice Blob in-place.
    """
    # Set all values below epsilon to np.inf
    Blob[Blob < epsilon] = np.inf

def miseAJour(Graphe:np.array, Blob:np.array, Pression:np.array, alpha:float, mu:float, arbreDeSteiner:np.array, delta:float,epsilon:float , mode:str='simple')->None:
    """Effectue une étape complète de mise à jour du réseau (Blob).

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée (longueurs).
        Blob (np.array): Matrice d'adjacence pondérée (rayons) - sera modifiée in-place.
        Pression (np.array): Vecteur des pressions aux noeuds.
        alpha (float): Paramètre de renforcement alpha.
        mu (float): Paramètre d'affaiblissement mu.
        arbreDeSteiner (np.array): Matrice indiquant l'appartenance à l'arbre de Steiner.
        delta (float): Facteur multiplicatif différentiel lié à l'arbre de Steiner.
    """
    miseAJourDébits(Graphe, Blob, Pression, alpha, mu, arbreDeSteiner, delta, mode)
    miseAjourRayons(Blob,epsilon)