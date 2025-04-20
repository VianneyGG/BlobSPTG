import networkx as nx
import time
from math import *
import numpy as np
import matplotlib.pyplot as plt

def poids(Graphe:np.array,arbre:np.array)->float:
    """renvoie le poids du graphe

    Args:
        Graphe (np.array): _description_
        arbre (np.array): _description_

    Returns:
        float: poids de l'arbre
    """
    poids=0
    n=np.shape(Graphe)[0]
    for i in range(n):
        for j in range(i+1,n):
            if arbre[i,j] != np.inf:
                poids += Graphe[i,j]
    return poids

def np2nx(Graphe:np.array)->nx.Graph:
    """transforme le graphe numpy en un graphe networkx

    Args:
        Graphe (np.array): _description_

    Returns:
        nx.Graph: _description_
    """
    n=np.shape(Graphe)[0]
    G = nx.Graph()
    G.add_nodes_from([i for i in range(n)])
    for i in range(n):
        for j in range(i+1,n):
            if Graphe[i,j] != np.inf:
                G.add_edge(i,j,poids=Graphe[i,j])
    return G

def nx2np(Graphe:nx.Graph)->np.array:
    """transforme le graphe networkx en un graphe numpy

    Args:
        Graphe (nx.Graph): _description_

    Returns:
        np.array: _description_
    """
    n=Graphe.number_of_nodes()
    noeuds=list(Graphe.nodes)
    G= np.inf*np.ones((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if nx.is_path(Graphe,[noeuds[i],noeuds[j]]):
                G[i,j]=Graphe[noeuds[i]][noeuds[j]]['poids']
                G[j,i]=G[i,j]
    return G

def affichage(Graphe:np.array,Blob:np.array,itération:int,pos:list,Terminaux:list,temps:time,resultat:bool=False)->None:
    """affiche le Blob sur son graphe
    Args:
        Graphe (np.array): _description_
        Blob (np.array): _description_
        it (_type_): _description_
        pos (list): _description_
        Terminaux (list): _description_
    """
    G=np2nx(Graphe)
    n = np.shape(Graphe)[0]
    noeudsBlob = []
    aretesBlob = []
    labels={}
    edgesLabels={}
    
    nx.draw_networkx_nodes(G, pos, node_color="black", node_size=500)
    nx.draw_networkx_nodes(G, pos, node_color="red", node_size=500,nodelist=Terminaux)
    nx.draw_networkx_edges(G, pos, width=2, edge_color="grey")
    for i in range(n):
        labels[i]=i
        for j in range(i+1, n):
            if G.get_edge_data(i, j, default=np.inf) != np.inf:
                edgesLabels[(i,j)]=G.get_edge_data(i, j, default=np.inf)['poids']
            if Blob[i,j] != np.inf:
                if Blob[i,j] == np.nan:
                    Blob[i,j] = np.inf
                else:
                    edgesLabels[i,j]=G[i][j]['poids']
                    if i not in noeudsBlob and i not in Terminaux:
                        noeudsBlob.append(i)
                        nx.draw_networkx_nodes(G, pos, nodelist=[i],node_color="green", node_size=500, alpha=0.7)
                    if j not in noeudsBlob and j not in Terminaux:
                        noeudsBlob.append(j)
                        nx.draw_networkx_nodes(G, pos, nodelist=[j],node_color="green", node_size=500, alpha=0.7)
                    nx.draw_networkx_edges(G, pos, edgelist=[(i,j)], width=log10(abs(1e8*Blob[i,j])) ,alpha=0.65,edge_color="green")

    nx.draw_networkx_labels(G, pos, labels, font_size=10,font_color="whitesmoke")
    nx.draw_networkx_edge_labels(G,pos,edgesLabels,font_size=15,font_color='k') 

    if resultat:
        titre = 'résultat en '
        titre += str(time.time()-temps)
        titre += 'secondes, poids de'
        titre += str(poids(Graphe,Blob))
        plt.title(titre)
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()

    else:   
        if itération%10000 == 0 and itération != 0:
            titre =  str(itération)
            titre += '-ème itération'
            plt.title(titre)
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.show(block=False)
            plt.pause(3)
            plt.close()
        plt.clf()

    
    