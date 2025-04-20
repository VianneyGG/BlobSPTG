import sys
import time
from math import *
import time as t
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool,cpu_count
import networkx as nx

from Fonctions.Initialisation import *
from Fonctions.Puit import selectionPuit
from Fonctions.Pression import calculNouvellesPressions
from Fonctions.MiseAJour import miseAJour
from Fonctions.Evolution import evolutionBlob
from Fonctions.Affichage import *
from tqdm.auto import tqdm

def arbreCouvrant(Graphe: np.array,Terminaux:list) -> np.array:
    """renvoie l'arbre couvrnat du Graphe par l'algorithme de Prim

    Args:
        Graphe (np.array): _description_
        Terminaux (list): _description_

    Returns:
        np.array: arbre couvrant
    """
    
    def aretes(Vu:list,Graphe:np.array)->list:
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
                if Graphe[k,i] != np.inf:
                    if not i in Vu:
                        aretes.append([k,i,Graphe[k,i]])
        return aretes

    def nbPointsGraphe(Graphe:np.array)->int:
        n=np.shape(Graphe)[0]
        res = n
        for i in range(n):
            ligneVide = True
            j=0
            while ligneVide and j<n:
                if Graphe[i,j] != np.inf:
                    ligneVide = False
                j += 1
            if ligneVide:
                res -= 1
        return res


    n=np.shape(Graphe)[0]
    Arbre = np.inf*np.ones((n,n))
    Vu=[Terminaux[0]]
    while len(Vu) < nbPointsGraphe(Graphe):
        l =  aretes(Vu,Graphe)
        l.sort(key=lambda x : x[2],reverse= True)
        arete = l.pop()
        Arbre[arete[0],arete[1]]=arete[2]
        Arbre[arete[1],arete[0]]=arete[2]
        Vu.append(arete[1])
    return Arbre


def evolutionAux(i:int,B:int,Graphe:np.array, Terminaux:list, Pressions:list, alpha:float, mu:float, delta:float, epsilon:float, débitEntrant:float, positions:dict,arbreDeSteiner:np.array):
    global Blob
    Blob =initialisation(Graphe)
    for j in range(B):
        print(i+1,'-',j)
        Puit=selectionPuit(Graphe,Terminaux)
        Pressions = calculNouvellesPressions(Graphe,Blob,Terminaux,Puit,débitEntrant)
        miseAJour(Graphe, Blob, Pressions, alpha, mu, arbreDeSteiner,delta)
        evolutionBlob(Blob, epsilon, Terminaux)
        affichage(Graphe,Blob,j,positions,Terminaux,t)      
    return Blob

def Blob(Graphe: np.array, Terminaux: list, A:int=12, B:int=3000, alpha:float=0.15, mu: float=1, delta:float=0.2, epsilon: float=1e-3, débitEntrant: float = 1, positions:dict={}) -> np.array:
    """itère A fois l'algorithme du Blob pour le probleme de l'arbre de Steiner

    Args:
        Graphe (np.array): _description_
        Terminaux (list): _description_
        A (int): nombre d'itération de l'algorithme du Blob
        B (int): nombre d'itération de l'évolution du Blob
        alpha (float): paramètre de la loi de renforcement
        mu (float): paramètre de la loi de renforcement
        delta (float) : paramètre de la loi de renforcement
        epsilon (float): taille minimale des aretes
        positions (list): liste des postion des noeuds du graphe

    Returns:
        np.array : meilleur arbre de Steiner obtenu après A itérations de l'algorithme du Blob
    """
    t=time.time()
    n=np.shape(Graphe)[0]
    Terminaux = traitementTerminaux(Terminaux)
    Pressions = np.array([])
    arbredeSteiner= Graphe
    nbCoeurs=cpu_count()
    k=0
    
    global evolution
    def evolution(i:int)->None:
        """_summary_

        Args:
            i (int): _description_

        Returns:
            _type_: _description_
        """
        global evolutionAux
        return evolutionAux(i,B,Graphe,Terminaux,Pressions,alpha,mu,delta,epsilon,débitEntrant,positions, arbredeSteiner)
    
    
    p= Pool()
    while k<A:
        resultats=p.imap_unordered(evolution,[j for j in range(A)])
        for resultat in resultats:
            if poids(resultat) < poids(arbredeSteiner):
                arbredeSteiner = resultat

    '''
    while k<A:
        for i in range(A//nbCoeurs):
            p= Pool(processes=12)
            resultats=p.imap_unordered(evolution,[j for j in range(i*nbCoeurs,min((i+1)*nbCoeurs,A))])
            for resultat in resultats:
                if poids(resultat) < poids(arbredeSteiner):
                    arbredeSteiner = resultat
    '''

    res = arbreCouvrant(resultats[0],Terminaux)
    affichage(Graphe,res,0,positions,Terminaux,t,True)
    return res



g=nx.Graph()
for k in range(10):
    g.add_node(k)
g.add_edge(0,1,poids=9)
g.add_edge(0,2,poids=8)
g.add_edge(0,3,poids=9)
g.add_edge(0,6,poids=18)

g.add_edge(1,3,poids=3)
g.add_edge(1,5,poids=6)

g.add_edge(2,3,poids=9)
g.add_edge(2,4,poids=8)
g.add_edge(2,6,poids=10)
g.add_edge(2,7,poids=7)
g.add_edge(2,9,poids=9)

g.add_edge(3,4,poids=2)
g.add_edge(3,5,poids=4)

g.add_edge(4,5,poids=2)
g.add_edge(4,7,poids=9)

g.add_edge(5,7,poids=9)

g.add_edge(6,8,poids=4)
g.add_edge(6,9,poids=3)

g.add_edge(7,8,poids=4)
g.add_edge(7,9,poids=5)

g.add_edge(8,9,poids=1)

Terminaux=[0,5,8]
pos={}
pos[0]=np.array([7,0])
pos[1]=np.array([0,5.75])
pos[2]=np.array([11.25,7.75])
pos[3]=np.array([2.25,7.75])
pos[4]=np.array([3.25,9.75])
pos[5]=np.array([2.25,11.75])
pos[6]=np.array([19,13])
pos[7]=np.array([11,13.75])
pos[8]=np.array([15.25,15.5])
pos[9]=np.array([16,13.75])

Blob(nx2np(g),Terminaux,B=100,positions=pos)