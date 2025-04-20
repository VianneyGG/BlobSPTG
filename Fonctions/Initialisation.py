import numpy as np
import numpy.random as rd
from Fonctions.MiseAJour import rayon # Assuming rayon is correctly defined here

def initialisation(Graphe:np.array) -> np.array:
    """Initialise la matrice des rayons (Blob) basée sur la topologie du Graphe (vectorized).
       Les arêtes existantes dans Graphe reçoivent un rayon initial correspondant
       à une conductance uniforme dans [0.5, 1]. Les non-arêtes restent à np.inf.

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée (longueurs) du graphe de base.

    Returns:
        np.array: Matrice Blob initialisée avec les rayons.
    """
    Blob = np.full_like(Graphe, np.inf) # Start with all inf
    n = np.shape(Graphe)[0]

    # Create a boolean mask for existing edges in the upper triangle
    mask_exist = (Graphe != np.inf) # Mask for all existing edges

    # Generate random conductances for existing edges
    num_existing_edges = np.sum(mask_exist)
    initial_conductances = rd.uniform(0.5, 1.0, size=num_existing_edges)

    # Calculate initial radii
    initial_radii = rayon(initial_conductances) # Assumes rayon is vectorized or handles arrays

    # Assign radii to the Blob matrix using the mask
    Blob[mask_exist] = initial_radii

    # Ensure symmetry if Graphe was symmetric
    Blob = np.maximum(Blob, Blob.T) # Take the element-wise maximum to fill the lower triangle if needed

    # Ensure diagonal remains inf (or 0 if preferred)
    np.fill_diagonal(Blob, np.inf) # Or 0

    return Blob

def traitementTerminaux(Terminaux:list)->list:
    """S'assure que la liste des Terminaux contient des éléments uniques et triés.

    Args:
        Terminaux (list): Liste potentiellement non triée ou avec doublons des indices des terminaux.

    Returns:
        list: Liste triée et sans doublons des indices des terminaux.
    """
    # More Pythonic way using set for uniqueness and then sorting
    if not Terminaux:
        return []
    return sorted(list(set(Terminaux)))
