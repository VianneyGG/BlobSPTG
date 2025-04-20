import numpy as np
import numpy.random as rd
from Fonctions.Pression import voisins

def longueurNoeuds(Graphe:np.array,Noeud:int)->float:
    """renvoie la somme des longueurs des arêtes adjacentes au noeuds

    Args:
        Graphe (np.array): _description_
        Noeud (int): _description_

    Returns:
        float: _description_
    """
    res = 0
    for k in voisins(Noeud,Graphe):
        res += Graphe[Noeud,k]
    return res
    
    
def proba(Graphe:np.array, Terminaux:list)->list:
    """renvoie la loi de probabilité associée aux terminaux

    Args:
        Graphe (np.array): _description_
        Terminaux (list): _description_

    Returns:
        list: _description_
    """
    Proba=[]
    n = len(Terminaux)
    long= [longueurNoeuds(Graphe,Terminaux[i]) for i in range(n)]
    long.sort()
    for k in range(n):
        Proba.append(long[n-k-1]/(sum(long)))
    return Proba

def selectionPuit(Graphe:np.array, Terminaux:list,)->int:
    """sélectionne le puit parmi les terminaux

    Args:
        Noeuds (list): _description_
        Terminaux (list): _description_

    Returns:
        int: puit
    """
    return rd.choice(Terminaux,p=proba(Graphe,Terminaux))