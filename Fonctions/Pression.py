from math import *
from scipy import sparse as sp
import numpy as np

def voisins(S:int,G:np.array)->list:
    """
    renvoie la liste des voisins d'un sommet S du graphe G
    """
    n=np.shape(G)[0]
    V=[]
    for k in range(n):
       if G[S,k] !=np.inf :
           V.append(k)
    return V

def conductance(Rayon:float)->float:
    """
    renvoie la conductance correspondant au rayon
    """
    return pi*Rayon**(4)/8

def calculNouvellesPressions(Graphe:np.array, Blob:np.array, Terminaux:list, puit:int, débitEntrant)->tuple:
    """Actualisation des pressions

    Args:
        Graphe (np.array): _description_
        Blob (np.array): _description_
        Terminaux (list): _description_
        Pression (list): _description_
        puit (int):  indice du puit dans la liste des Terminaux

    Returns:
        list : liste des pressions actually
        np.array : Graphe
        
    """
    n=np.shape(Graphe)[0]
    A =np.zeros((n,n)) # matrice des coefficients
    B =np.zeros((n,)) # second membre
    for k in Terminaux:
        B[k] = -débitEntrant  # Tous les autre terminaux sont des sources
    B[puit]= 0
     
    for S in range(n): # Equations du réseau de Poisson
        if S != puit : # la pression au niveau du puit est nulle
            for V in voisins(S,Blob):
                A[S,V] += conductance(Blob[S,V])/Graphe[S,V]
                A[S,S] -= conductance(Blob[S,V])/Graphe[S,V]
    
    for i in range(n):
        ligneVide = True
        j=0
        while j<n and ligneVide:
            if A[i,j] != 0:
                ligneVide = False
            j+=1
        if ligneVide:
            A[i,i]=1                
                    
    for k in range(n): 
        A[k,puit]=0
    A[puit, puit] = 1 
    
    Pression=np.linalg.solve(A,B)
    return Pression