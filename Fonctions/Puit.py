import numpy as np
import numpy.random as rd
from Fonctions.Pression import voisins

def longueurNoeuds(Graphe:np.array,Noeud:int)->float:
    """Renvoie la somme des longueurs des arêtes adjacentes au noeud spécifié.

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée du graphe (longueurs).
        Noeud (int): Indice du noeud.

    Returns:
        float: Somme des longueurs des arêtes connectées au noeud.
    """
    res = 0
    # Consider using np.sum(Graphe[Noeud, Graphe[Noeud, :] != np.inf]) if Graphe uses np.inf for non-edges
    # Or iterate directly if using adjacency lists or sparse matrices for efficiency.
    for k in voisins(Noeud,Graphe):
        res += Graphe[Noeud,k]
    return res


def proba(Graphe:np.array, Terminaux:list)->list:
    """Calcule une loi de probabilité pour les terminaux basée sur la somme des longueurs des arêtes adjacentes.
       Les noeuds avec une plus grande somme de longueurs adjacentes ont une probabilité plus élevée.

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée du graphe (longueurs).
        Terminaux (list): Liste des indices des noeuds terminaux.

    Returns:
        list: Liste des probabilités correspondant à chaque terminal dans l'ordre d'origine.
    """
    n = len(Terminaux)
    if n == 0:
        return []
    lengths = [longueurNoeuds(Graphe, T) for T in Terminaux]
    total_length = sum(lengths)
    if total_length == 0:
        # Handle case where all terminals might be isolated or have zero-length edges
        return [1.0 / n] * n # Assign equal probability
    # Note: The original code sorted lengths, which seemed unnecessary for rd.choice
    # as it takes probabilities corresponding to the original Terminaux list order.
    # Probabilities are proportional to the lengths.
    probabilities = [l / total_length for l in lengths]
    return probabilities

def selectionPuit(Graphe:np.array, Terminaux:list)->int:
    """Sélectionne aléatoirement un puit parmi les terminaux selon la loi de probabilité définie par `proba`.

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée du graphe (longueurs).
        Terminaux (list): Liste des indices des noeuds terminaux.

    Returns:
        int: Indice du noeud sélectionné comme puit.
    """
    probabilities = proba(Graphe, Terminaux)
    if not Terminaux:
        raise ValueError("La liste des terminaux ne peut pas être vide.")
    # Ensure probabilities sum to 1, handling potential floating point inaccuracies
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return rd.choice(Terminaux, p=probabilities)