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


def proba(Graphe:np.array, Terminaux_list:list)->list:
    """Calcule une loi de probabilité pour les terminaux basée sur la somme des longueurs des arêtes adjacentes,
       suivant la méthode de l'article : P(i) = l(|T|-i+1)/sum_j l(j), où les terminaux sont triés par longueur croissante.

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée du graphe (longueurs).
        Terminaux_list (list): Liste des indices des noeuds terminaux.

    Returns:
        list: Liste des probabilités correspondant à chaque terminal dans l'ordre d'origine.
    """
    n = len(Terminaux_list)
    if n == 0:
        return []
    # Calcul des longueurs pour chaque terminal
    lengths = [longueurNoeuds(Graphe, T) for T in Terminaux_list]
    # Tri des terminaux par longueur croissante
    sorted_pairs = sorted(zip(Terminaux_list, lengths), key=lambda x: x[1])
    sorted_lengths = [l for _, l in sorted_pairs]
    # Attribution des probabilités selon l'article
    # Pour chaque terminal dans l'ordre d'origine, trouver son rang dans le tri
    index_in_sorted = {node: i for i, (node, _) in enumerate(sorted_pairs)}
    total_length = sum(sorted_lengths)
    if total_length == 0:
        return [1.0 / n] * n
    # Pour le terminal à la position k dans Terminaux_list, sa proba est l(|T|-k+1)
    # c'est-à-dire sorted_lengths[n - index_in_sorted[node] - 1] / total_length
    probabilities = [
        sorted_lengths[n - index_in_sorted[node] - 1] / total_length
        for node in Terminaux_list
    ]
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
        probabilities = proba(Graphe, terminaux_list)
        if len(probabilities) != len(terminaux_list):
            raise ValueError("Mismatch between number of terminals and probabilities calculated.")
        probabilities_np = np.array(probabilities)
        prob_sum = probabilities_np.sum()
        if prob_sum <= 0:
            print("Warning: Probabilities sum to zero or less. Falling back to uniform selection.")
            return rd.choice(terminaux_list)
        probabilities_np /= prob_sum
        return rd.choice(terminaux_list, p=probabilities_np)