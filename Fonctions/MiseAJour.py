from math import *
import numpy as np
from Fonctions.Pression import conductance

def rayon(Conductance:float)->float:
    """renvoie le rayon correspondant à l'conductance

    Args:
        Conductance (float): [description]

    Returns:
        float: [description]
    """
    return 8*Conductance**(1/4)/pi

def miseAJourDébits(Graphe:np.array, Blob:np.array, Pressions: list) -> np.array:
    """met à jour la matrice des débits grâce aux nouvelles valeurs des pressions de chaque noeud et des conductances de chaque arêtes

    Returns:
        [type]: [description]
    """
    n = np.shape(Graphe)[0]
    Débits = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if Blob[i,j] != np.inf:
                Débits[i,j] = (conductance(Blob[i,j]))*(Pressions[i]-Pressions[j])/Graphe[i,j]
                Débits[j,i] = -Débits[i,j]  # matrice anti-symmétrique
    return Débits

def relationRenforcement(Rayon:float, Debit:float, alpha:float, mu:float) -> float:
    """renvoie la nouvelle valeur de conductance en suivant la relation de renforcement

    Args:
        Rayon (float): [description]
        Debit (float): [description]
        alpha (float): [description]
        mu (float): [description]

    Returns:
        float: [description]
    """
    return conductance(Rayon) + alpha*abs(Debit)-mu*conductance(Rayon)

def miseAJourRayons(Blob: np.array, Débits: np.array, Pressions: list, alpha:float, mu:float, arbreDeSteiner:np.array, delta:float) -> None:
    """met à jour la matrice des rayons grâce à la relation de renforcement

    Args:
        Blob (np.array): _description_
        D (_type_): _description_
        Pressions (list): _description_
        alpha (float): _description_
        mu (float): _description_
        arbreDeSteiner (np.array): _description_
        delta (float): _description_
    """
    n = np.shape(Blob)[0]
    for i in range(n):
        for j in range(i+1, n):
            if Blob[i,j] != np.inf:
                if arbreDeSteiner[i,j] != np.inf:
                    Blob[i, j] = rayon(relationRenforcement(Blob[i, j], Débits[i, j],alpha,mu))*(1+delta)
                else:
                    Blob[i, j] = rayon(relationRenforcement(Blob[i, j], Débits[i, j],alpha,mu))*(1-delta)
            Blob[j, i] = Blob[i, j]

def miseAJour(Graphe:np.array, Blob:np.array, Pression:list, alpha:float, mu:float, arbreDeSteiner:np.array, delta:float)->None:
    """met à jour les Rayons du Blob

    Args:
        Graphe (np.array): _description_
        Blob (np.array): _description_
        Pression (list): _description_
        alpha (float): _description_
        mu (float): _description_
        arbreDeSteiner (np.array): _description_
        delta (float): _description_
    """
    Débits=miseAJourDébits(Graphe, Blob, Pression)
    miseAJourRayons(Blob, Débits, Pression, alpha, mu, arbreDeSteiner, delta)
    