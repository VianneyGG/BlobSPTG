from math import *
import numpy as np

def miseAJourDébits(Graphe:np.array, Best_blob: np.array, Blob:np.array, step:int, Pressions: np.array, alpha:float, mu:float, delta:float, evol: bool, mode:str, arbreDeSteiner:np.array)->None:
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
    
    # Create masks for best blob edges (activate delta) and other edges (don't activate delta)
    best_blob_mask = valid_edge_mask & (Best_blob != np.inf)
    
    # Update conductivity for other edges (without delta activation)
    if np.any(valid_edge_mask):
        Blob[valid_edge_mask] = relationRenforcement(Blob[valid_edge_mask], Q_matrix[valid_edge_mask], step, alpha, mu, delta, mode, False)

    # Update conductivity for best blob edges (with delta activation)
    if evol:
        Blob[best_blob_mask] = relationRenforcement(Blob[best_blob_mask], Q_matrix[best_blob_mask], step, alpha, mu, delta, mode, True)
    
    
def relationRenforcement(Conductivity: np.array, Flow: np.array, step:int, alpha:float, mu:float, delta:float, mode:str, activate_delta:bool) -> np.array:
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
        if activate_delta:
            return np.maximum(0.0, ( alpha * np.abs(Flow) - (mu - 1) * Conductivity) * (1 + delta))
        else:
            return np.maximum(0.0, alpha * np.abs(Flow) - (mu - 1) * Conductivity)
    if mode == 'vieillesse':
        # Affaiblissement exponentiel de la conductivité
        loss = 1 * (1-np.exp(- (step-10)/100)) if step > 10 else 0.5
        if activate_delta:
            return np.maximum(0.0, (alpha * np.abs(Flow) - (loss - 1) * Conductivity) * (1 + delta))
        else:
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

def miseAJour(Graphe:np.array,Best_blob: np.array, Blob:np.array, step:int, Pression:np.array, alpha:float, mu:float, delta:float, epsilon:float, evol:bool, mode:str='simple', arbreDeSteiner:np.array=([]))->None:
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
    miseAJourDébits(Graphe,Best_blob, Blob, step, Pression, alpha, mu, delta, evol, mode, arbreDeSteiner)
    miseAjourRayons(Blob,epsilon)