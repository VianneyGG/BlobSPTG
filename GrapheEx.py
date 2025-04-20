import networkx as nx
import numpy as np
from Blobv2 import Blob
from Fonctions.Affichage import *

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