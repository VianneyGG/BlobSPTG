from math import*
import numpy as np
from Fonctions.Pression import voisins # Assuming voisins is correctly defined

def composanteConnexe(Graphe:np.array, NoeudsDepart:list)->list:
    """Trouve la composante connexe contenant les noeuds de départ dans le graphe.
       Utilise un parcours (type BFS ou DFS) pour identifier tous les noeuds atteignables.

    Args:
        Graphe (np.array): Matrice d'adjacence (np.inf pour non-arête).
        NoeudsDepart (list): Liste des noeuds à partir desquels commencer le parcours.

    Returns:
        list: Liste des indices des noeuds appartenant à la composante connexe.
    """
    if not NoeudsDepart:
        return []

    n = np.shape(Graphe)[0]
    Vu = set() # Use set for faster 'in' check
    aVisiter = list(NoeudsDepart) # Start with all initial nodes

    while aVisiter:
        s = aVisiter.pop(0) # Use pop(0) for BFS-like behavior, pop() for DFS-like
        if s not in Vu:
            Vu.add(s)
            # Find neighbors using the provided voisins function or directly
            # Voisins_s = list(np.where(Graphe[s, :] != np.inf)[0]) # Example direct check
            Voisins_s = voisins(s, Graphe) # Using the imported function
            for j in Voisins_s:
                if j not in Vu:
                    aVisiter.append(j)
    return list(Vu)

def evolutionBlob(Blob:np.array, epsilon:float, Terminaux:list)->None:
    """Modifie le Blob in-place (vectorized pruning):
       1. Coupe les arêtes dont le rayon est inférieur à epsilon.
       2. Coupe toutes les arêtes connectées à des noeuds qui ne sont pas dans la
          composante connexe contenant les Terminaux.

    Args:
        Blob (np.array): Matrice des rayons (modifiée in-place).
        epsilon (float): Seuil de rayon en dessous duquel une arête est coupée.
        Terminaux (list): Liste des indices des noeuds terminaux.
    """
    n = np.shape(Blob)[0]
    if not Terminaux:
        print("Warning: Liste des Terminaux vide dans evolutionBlob.")
        # Pruning edges below epsilon might still be desired.
        mask_prune_epsilon = (Blob < epsilon) & (Blob != np.inf)
        Blob[mask_prune_epsilon] = np.inf
        return # No component-based pruning

    # 1. Find the connected component containing all terminals
    CC_set = set(composanteConnexe(Blob, Terminaux))
    if not all(t in CC_set for t in Terminaux):
        CC_list = sorted(list(CC_set)) # For printing
        print(f"Warning: Les terminaux ne sont plus tous connectés. Terminaux: {Terminaux}, Composante Connexe: {CC_list}")
        # raise Exception('Terminaux déconnectés') # Or handle differently

    # 2. Prune edges using boolean masks (vectorized)

    # Mask for edges with radius < epsilon
    mask_prune_epsilon = (Blob < epsilon) & (Blob != np.inf)

    # Mask for edges connected to nodes outside the component CC
    # Create a boolean array indicating which nodes are NOT in CC
    nodes_outside_CC = np.ones(n, dtype=bool)
    if CC_set: # Only if CC is not empty
        cc_indices = list(CC_set)
        nodes_outside_CC[cc_indices] = False
    # An edge (i, j) should be pruned if i is outside OR j is outside
    mask_prune_component = nodes_outside_CC[:, np.newaxis] | nodes_outside_CC[np.newaxis, :]
    # Ensure we don't prune non-existent edges
    mask_prune_component &= (Blob != np.inf)

    # Combine masks: prune if below epsilon OR outside component
    mask_prune_final = mask_prune_epsilon | mask_prune_component

    # Apply pruning
    Blob[mask_prune_final] = np.inf