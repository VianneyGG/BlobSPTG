import numpy as np
import numpy.random as rd

def initialisation(Graphe:np.array) -> np.array:
    """Initialise la matrice des rayons (Blob) basée sur la topologie du Graphe (vectorized).
       Les arêtes existantes dans Graphe reçoivent un rayon initial correspondant
       à une conductivity uniforme dans [0.5, 1]. Les non-arêtes restent à np.inf.

    Args:
        Graphe (np.array): Matrice d'adjacence pondérée (longueurs) du graphe de base.

    Returns:
        np.array: Matrice Blob initialisée avec les rayons.
    """
    n = np.shape(Graphe)[0]
    Blob = np.inf * np.ones((n, n))

    # Create a boolean mask for existing edges in the upper triangle
    mask_exist = (Graphe != np.inf) # Mask for all existing edges

    # Generate a single random conductivity value (uniform in [0.5, 1])
    initial_conductivity_value = rd.uniform(0.5, 1)

    # Initialize all existing edges with the same conductivity value using the boolean mask
    Blob[mask_exist] = initial_conductivity_value

    return Blob
