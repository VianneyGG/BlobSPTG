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


def proba(Graphe:np.array, Terminaux_list:list)->list: # Renamed arg for clarity
    """Calcule une loi de probabilité pour les terminaux basée sur la somme des longueurs des arêtes adjacentes.
       Les noeuds avec une plus grande somme de longueurs adjacentes ont une probabilité plus élevée.

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée du graphe (longueurs).
        Terminaux_list (list): Liste des indices des noeuds terminaux.

    Returns:
        list: Liste des probabilités correspondant à chaque terminal dans l'ordre d'origine.
    """
    n = len(Terminaux_list)
    if n == 0:
        return []
    lengths = [longueurNoeuds(Graphe, T) for T in Terminaux_list]
    total_length = sum(lengths)
    if total_length == 0:
        # Handle case where all terminals might be isolated or have zero-length edges
        return [1.0 / n] * n # Assign equal probability
    # Note: The original code sorted lengths, which seemed unnecessary for rd.choice
    # as it takes probabilities corresponding to the original Terminaux list order.
    # Probabilities are proportional to the lengths.
    probabilities = [l / total_length for l in lengths]
    return probabilities

def selectionPuit(Graphe:np.array, Terminaux:set, mode:str)->int: # Changed type hint to set
    """Sélectionne aléatoirement un puit parmi les terminaux.

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée du graphe (longueurs).
        Terminaux (set): Ensemble des indices des noeuds terminaux.
        mode (str): Mode de sélection ('unif' ou autre pour pondéré).

    Returns:
        int: Indice du noeud sélectionné comme puit.

    Raises:
        ValueError: Si l'ensemble des terminaux est vide.
    """
    if not Terminaux:
        raise ValueError("L'ensemble des terminaux ne peut pas être vide.")

    # Convert the set to a list for consistent ordering and indexing
    terminaux_list = list(Terminaux)

    if mode == 'unif':
        return rd.choice(terminaux_list)
    
    
    else: # Assumes weighted probability mode
        # Calculate probabilities based on the list order
        probabilities = proba(Graphe, terminaux_list)
        if len(probabilities) != len(terminaux_list):
             raise ValueError("Mismatch between number of terminals and probabilities calculated.")

        # Ensure probabilities sum to 1, handling potential floating point inaccuracies
        probabilities_np = np.array(probabilities)
        prob_sum = probabilities_np.sum()
        if prob_sum <= 0: # Handle cases where all probabilities might be zero
             # Fallback to uniform selection if probabilities are invalid
             print("Warning: Probabilities sum to zero or less. Falling back to uniform selection.")
             return rd.choice(terminaux_list)
        probabilities_np /= prob_sum

        # Use the list with rd.choice and the calculated probabilities
        return rd.choice(terminaux_list, p=probabilities_np)