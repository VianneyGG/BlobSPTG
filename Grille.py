import time
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from Blobv2 import Blob
from Fonctions.Outils import *


Terminaux=[]
for k in range(50):
    Terminaux.append(rd.randint(0,399))

G=np.array

def grapheGrille(n,p):
    g=nx.grid_2d_graph(n,p)
    pos={}
    for i in range(n):
        for j in range(p):
            pos[n*i+j]=np.array([i,j])
    return g, pos


g,pos = grapheGrille(20,20)  
nx.set_edge_attributes(g,1,'poids')

res = Blob(nx2np(g),Terminaux,2,600,0.05,1.0,1e-2,1,pos)
