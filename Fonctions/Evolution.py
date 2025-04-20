from math import*
import numpy as np
from Fonctions.Pression import voisins

def composanteConnexe(Graphe:np.array,Terminaux:list)->list:
    """récupère la composante connexe d'un noeud du graphe

    Args:
        Graphe (np.array): _description_
        Terminaux (list): _description_

    Returns:
        list: _description_
    """
    n=np.shape(Graphe)[0]
    aVisiter = [Terminaux[0]]
    Vu = []
    while len(aVisiter) !=0:
        s = aVisiter.pop()
        if not s in Vu:
            Vu.append(s)
        for j in voisins(s,Graphe):
            if j not in Vu:
                aVisiter.append(j)
    return Vu

def evolutionBlob(Blob:np.array, epsilon:float, Terminaux:list)->None:
    """coupe les arêtes trop fines et celles déconnectées des terminaux

    Args:
        Blob (np.array): _description_
        epsilon (float): _description_
    """
    n = np.shape(Blob)[0]
    CC=composanteConnexe(Blob, Terminaux)
    for i in range(n):
        if i not in CC:
            if i in Terminaux:
                raise Exception('Terminaux déconnectés')
            for j in range(i+1,n):
                Blob[i,j]=np.inf
                Blob[j,i]=np.inf
        else:
            for j in range(i+1,n):
                if Blob[i,j] < epsilon:
                    Blob[i,j]=np.inf
                    Blob[j,i]=np.inf             