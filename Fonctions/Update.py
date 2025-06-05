from math import *
import numpy as np

def miseAJourDébits(Graphe:np.array, Blob:np.array, step:int, Pressions: np.array, alpha:float, mu:float, delta:float, mode:str, arbreDeSteiner:np.array)->None:
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
    Blob[valid_edge_mask] = relationRenforcement(Blob[valid_edge_mask], Q_matrix[valid_edge_mask], step, alpha, mu, delta, mode)

def relationRenforcement(Conductivity:float, Flow:float, step:int, alpha:float, mu:float, delta:float, mode:str) -> float:
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
        return np.maximum(0.0, alpha * np.abs(Flow) - (mu - 1) * Conductivity)
    if mode == 'vieillesse':
        # Affaiblissement exponentiel de la conductivité
        loss = 1-1/np.log(np.sqrt(step-49)) if step > 50 else 1.0
        return np.maximum(0.0, alpha * np.abs(Flow) - (loss - 1) * Conductivity)

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

def miseAJour(Graphe:np.array, Blob:np.array, step:int, Pression:np.array, alpha:float, mu:float, delta:float,epsilon:float , mode:str='simple', arbreDeSteiner:np.array=([]))->None:
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
    miseAJourDébits(Graphe, Blob, step, Pression, alpha, mu, delta, mode, arbreDeSteiner)
    miseAjourRayons(Blob,epsilon)