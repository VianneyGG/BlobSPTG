from math import *
import numpy as np
from Fonctions.Pression import conductance

def rayon(Conductance:float)->float:
    """Calcule le rayon correspondant à une conductance donnée. Inverse de la fonction conductance.

    Args:
        Conductance (float): La conductance hydraulique.

    Returns:
        float: Le rayon correspondant. Renvoie 0 si la conductance est négative.
    """
    if Conductance < 0:
        return 0.0
    return (8 * Conductance / pi)**(1/4)

def miseAJourDébits(Graphe:np.array, Blob:np.array, Pressions: np.array) -> np.array:
    """Met à jour la matrice des débits (flux) entre les noeuds (vectorized).

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée (longueurs).
        Blob (np.array): Matrice d'adjacence pondérée (rayons).
        Pressions (np.array): Vecteur des pressions à chaque noeud.

    Returns:
        np.array: Matrice (anti-symétrique) des débits entre les noeuds. Débit[i, j] > 0 signifie flux de i vers j.
    """
    n = np.shape(Graphe)[0]
    Débits = np.zeros((n,n))

    # Create masks for valid edges (exist in both Graphe and Blob, and have positive length)
    valid_edge_mask = (Blob != np.inf) & (Graphe != np.inf) & (Graphe > 0)

    # Calculate conductances for all possible edges (vectorized)
    C_matrix = np.full_like(Blob, 0.0, dtype=float)
    radii_valid = Blob[valid_edge_mask]
    C_matrix[valid_edge_mask] = conductance(radii_valid)

    # Calculate pressure differences (vectorized)
    P_diff = Pressions[:, np.newaxis] - Pressions[np.newaxis, :]

    # Calculate flows only for valid edges
    Débits[valid_edge_mask] = C_matrix[valid_edge_mask] * P_diff[valid_edge_mask] / Graphe[valid_edge_mask]

    return Débits

def relationRenforcement(Rayon:float, Debit:float, alpha:float, mu:float) -> float:
    """Applique la relation de renforcement pour mettre à jour la conductance d'une arête.

    Args:
        Rayon (float): Rayon actuel de l'arête.
        Debit (float): Débit absolu à travers l'arête.
        alpha (float): Paramètre de renforcement lié au débit.
        mu (float): Paramètre d'affaiblissement (decay).

    Returns:
        float: Nouvelle valeur de conductance après renforcement/affaiblissement.
    """
    C_actuelle = conductance(Rayon)
    nouvelle_conductance = C_actuelle + alpha * abs(Debit) - mu * C_actuelle
    return max(0.0, nouvelle_conductance)

def miseAJourRayons(Blob: np.array, Débits: np.array, alpha:float, mu:float, arbreDeSteiner:np.array, delta:float) -> None:
    """Met à jour la matrice des rayons (Blob) en utilisant la relation de renforcement (vectorized).

    Args:
        Blob (np.array): Matrice des rayons (à mettre à jour in-place).
        Débits (np.array): Matrice des débits calculés.
        alpha (float): Paramètre de renforcement alpha.
        mu (float): Paramètre d'affaiblissement mu.
        arbreDeSteiner (np.array): Matrice indiquant l'appartenance à l'arbre de Steiner (np.inf si non membre).
        delta (float): Facteur multiplicatif appliqué aux arêtes (1+delta pour Steiner, 1-delta sinon).
    """
    n = np.shape(Blob)[0]
    # Mask for existing edges in the current Blob
    existing_edge_mask = (Blob != np.inf)

    # Calculate new conductances only for existing edges
    Nouvelles_Conductances = np.full_like(Blob, np.nan, dtype=float)
    Rayons_existants = Blob[existing_edge_mask]
    Debits_existants = Débits[existing_edge_mask]
    Nouvelles_Conductances[existing_edge_mask] = relationRenforcement(Rayons_existants, Debits_existants, alpha, mu)

    # Convert new conductances back to radii
    Nouveaux_Rayons = np.full_like(Blob, np.inf, dtype=float)
    Conductances_valides_mask = ~np.isnan(Nouvelles_Conductances) & existing_edge_mask
    Nouveaux_Rayons[Conductances_valides_mask] = rayon(Nouvelles_Conductances[Conductances_valides_mask])

    # Apply Steiner tree bias (vectorized)
    steiner_mask = (arbreDeSteiner != np.inf) & Conductances_valides_mask
    non_steiner_mask = (arbreDeSteiner == np.inf) & Conductances_valides_mask

    Nouveaux_Rayons[steiner_mask] *= (1 + delta)
    Nouveaux_Rayons[non_steiner_mask] *= (1 - delta)

    # Update Blob symmetrically
    np.copyto(Blob, Nouveaux_Rayons)

def miseAJour(Graphe:np.array, Blob:np.array, Pression:np.array, alpha:float, mu:float, arbreDeSteiner:np.array, delta:float)->None:
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
    Débits = miseAJourDébits(Graphe, Blob, Pression)
    miseAJourRayons(Blob, Débits, alpha, mu, arbreDeSteiner, delta)
