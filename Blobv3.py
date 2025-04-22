import sys
import time
from math import *
import time as t
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import networkx as nx

from Fonctions.Initialisation import *
from Fonctions.Puit import selectionPuit
from Fonctions.Pression import calculNouvellesPressions
from Fonctions.MiseAJour import miseAJour
from Fonctions.Evolution import evolutionBlob
from Fonctions.Affichage import *
from tqdm.auto import tqdm

def arbreCouvrant(Graphe: np.array, Terminaux: list) -> np.array:
    """renvoie l'arbre couvrant du Graphe par l'algorithme de Prim

    Args:
        Graphe (np.array): _description_
        Terminaux (list): _description_

    Returns:
        np.array: arbre couvrant
    """
    
    def aretes(Vu: list, Graphe: np.array) -> list:
        """renvoie la liste des aretes potentielles

        Args:
            Vu (list): _description_
            Graphe (np.array): _description_

        Returns:
            list: _description_
        """
        aretes = []
        n = np.shape(Graphe)[0]
        for k in Vu:
            for i in range(n):
                if Graphe[k, i] != np.inf:
                    if not i in Vu:
                        aretes.append([k, i, Graphe[k, i]])
        return aretes

    def nbPointsGraphe(Graphe: np.array) -> int:
        n = np.shape(Graphe)[0]
        res = n
        for i in range(n):
            ligneVide = True
            j = 0
            while ligneVide and j < n:
                if Graphe[i, j] != np.inf:
                    ligneVide = False
                j += 1
            if ligneVide:
                res -= 1
        return res

    n = np.shape(Graphe)[0]
    Arbre = np.inf * np.ones((n, n))
    Vu = [Terminaux[0]]
    while len(Vu) < nbPointsGraphe(Graphe):
        l = aretes(Vu, Graphe)
        l.sort(key=lambda x: x[2], reverse=True)
        arete = l.pop()
        Arbre[arete[0], arete[1]] = arete[2]
        Arbre[arete[1], arete[0]] = arete[2]
        Vu.append(arete[1])
    return Arbre

def evolutionAux(i: int, B: int, Graphe: np.array, Terminaux: list, Pressions: list, alpha: float, mu: float, delta: float, epsilon: float, débitEntrant: float, positions: dict, arbreDeSteiner: np.array):
    global Blob
    Blob = initialisation(Graphe)
    for j in range(B):
        Puit = selectionPuit(Graphe, Terminaux)
        Pressions = calculNouvellesPressions(Graphe, Blob, Terminaux, Puit, débitEntrant)
        miseAJour(Graphe, Blob, Pressions, alpha, mu, arbreDeSteiner, delta)
        evolutionBlob(Blob, epsilon, Terminaux)
    return Blob

def Blob(Graphe: np.array[int], Terminaux: set[int], M: int = 1, K: int = 3000, alpha: float = 0.15, mu: float = 1, delta: float = 0.2, epsilon: float = 1e-3, ksi: float = 1,  débitEntrant: float = 1, modeProba ='unif',modeRenfo='simple' , display_result: bool = True) -> np.array:
    """itère A fois l'algorithme du Blob pour le probleme de l'arbre de Steiner

    Args:
        Graphe (np.array): _description_
        Terminaux (list): _description_
        M (int): nombre d'itération de l'algorithme du Blob
        K (int): nombre d'itération de l'évolution du Blob
        alpha (float): paramètre de la loi de renforcement
        mu (float): paramètre de la loi de renforcement
        delta (float): paramètre de la loi de renforcement
        epsilon (float): taille minimale des aretes
        positions (list): liste des postion des noeuds du graphe
        display_result (bool): Whether to display the result using affichage.
        step_callback (callable, optional): Callback function called after each evolution step. Defaults to None.

    Returns:
        np.array: meilleur graphe obtenu après A itérations de l'algorithme du Blob
    """
    t_start = time.time() # Start timer
    n = np.shape(Graphe)[0]
    Pressions = np.array([])
    meilleur_blob = Graphe.copy()
    meilleur_poids = np.inf
    
    for i in range(M):

        #INITIALISATION ---------------------------------------------------------------
        current_blob = initialisation(Graphe)
        
        #ITERATION DE L'EVOLUTION DU BLOB ---------------------------------------------
        
        for j in range(K):
            Puit = selectionPuit(Graphe, Terminaux, modeProba)
            Pressions = calculNouvellesPressions(Graphe, current_blob, Terminaux, Puit, débitEntrant, epsilon)
            miseAJour(Graphe, current_blob, Pressions, alpha, mu, delta, epsilon, modeRenfo)

            # AFFICHAGE STEP BY STEP --------------------------------------------------
            if display_result and j%10 ==0:
                affichage(Graphe, current_blob, Terminaux)


        current_poids = poids(current_blob)
        if current_poids < meilleur_poids:
            meilleur_blob = current_blob.copy()
            meilleur_poids = current_poids


    return arbreCouvrant(meilleur_blob)