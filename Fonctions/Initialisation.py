import numpy as np
import numpy.random as rd
from Fonctions.MiseAJour import rayon

def initialisation(Graphe:np.array) -> np.array:
    """
    revoie la matrice des conductances initiales avec des valeurs uniformes dans [0.5,1]
    """
    Blob = np.copy(Graphe)
    n = np.shape(Blob)[0]
    for i in range(n):
       for j in range(i+1, n):
            if Blob[i, j] != np.inf:# on initialise pour chaque arête une valeur aléatoire de conductance dans [0.5,1]
                Blob[i, j] = rayon(rd.uniform(0.5, 1))
                Blob[j, i] = Blob[i, j]  
            else:
                Blob[i, j] = np.inf
                Blob[j,i] = np.inf
    return Blob

def traitementTerminaux(Terminaux:list)->list:
    """s'assure que les Terminaux sont triés et 2 à 2 distincts

    Args:
        Terminaux (list): _description_
    """
    Terminaux.sort()
    res = [Terminaux[0]]
    n = len(Terminaux)
    k=1
    while k <n:
        if Terminaux[k] == res [-1]:
            k += 1
        else:
            res.append(Terminaux[k])
            k += 1
    return res
