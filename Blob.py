from math import*
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import networkx as nx

def aretesGraphes(G:np.array,Lignes:list)->list:
    """
    renvoie la liste des arêtes du graphe donné en entrée
    """
    Aretes=[]
    n=np.shape(G)[0]
    for i in range(n):
        for j in range(n):
            if G[i,j] != np.inf:
                Aretes.append((Lignes[i],Lignes[j]))
    return Aretes

    """
    affiche le graphe entré et les rayons séléctionnés
    """
    
def grille(n:int,p:int)->np.array:
    Res=np.inf*np.ones((n*p,n*p))
    for i in range(n-1):
        k=10
        for j in range(p-1):
            Res[p*i+j,p*i+j+1]=k
            Res[p*i+j+1,p*i+j]=k
            Res[p*i+j,p*(i+1)+j]=k
            Res[p*(i+1)+j,p*i+j]=k
            Res[p*(n-1)+j,p*(n-1)+1+j]=k
            Res[p*(n-1)+1+j,p*(n-1)+j]=k
        Res[p*(i+1)-1,p*(i+2)-1]=k
        Res[p*(i+2)-1,p*(i+1)-1]=k
    return Res

def affichegrille(G:np.array,p:int,Blob:np.array,Lignes:list)->None:
    n=int(np.shape(G)[0]//p)
    plt.ion()
    X=[ i%p for i in range(n*p)]
    Y=[ n-(i//p) for i in range(n*p)]
    plt.scatter(X,Y,data=[i for i in range(n)])
    for (a,b) in aretesGraphes(G,[i for i in range(n*p)]):
        P1=[X[a],X[b]]
        P2=[Y[a],Y[b]]
        plt.plot(P1,P2,color="black")
    for (a,b) in aretesGraphes(Blob,Lignes):
        P1=[X[a],X[b]]
        P2=[Y[a],Y[b]]
        plt.plot(P1,P2,color="red", linewidth = 2)
    fig =plt.figure()

def conductance(Rayon:float)->float:
    """
    renvoie la conductance correspondant au rayon
    """
    return pi*Rayon**(4)/8

def rayon(Conductance:float)->float:
    """
    renvoie le rayon correspondant à l'conductance
    """
    return 8*Conductance**(1/4)/pi

def rayonsInitiaux(G:np.array)->np.array:
    """
    revoie la matrice des conductances initiales avec des valeurs uniformes dans [0.5,1]
    """
    n=np.shape(G)[0]
    Rayons=np.zeros((n,n),dtype=np.float32) #Matrice de conductance vide
    for i in range(n):
       for j in range(i+1,n):
            if G[i,j] != np.inf:
                Rayons[i,j] = rd.uniform(0.5,1) #on initialise pour chaque arête une valeur aléatoire de conductance dans [0.5,1]
                Rayons[j,i] = Rayons[i,j] #la matrice est symmétrique
    return Rayons

def composanteConnexe(G:np.array,Terminaux:list)->list:
    n=np.shape(G)[0]
    aVisiter = [Terminaux[0]]
    Vu = []
    while len(aVisiter) !=0:
        s = aVisiter.pop()
        if not s in Vu:
            Vu.append(s)
        for j in range(n):
            if G[s,j] != np.inf:
                if j not in Vu:
                    aVisiter.append(j)
    return Vu

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

def relationRenforcement(Rayon:float,Debit:float)->float:
    """
    renvoie la nouvelle valeur de conductance en suivant la relation de renforcement
    """
    a = -1.196e-22
    b=-0.002332
    c=1.72e-5
    d=3.282e1
    e=-2.338e3
    return abs(Rayon +0.6*Debit-0.5*Rayon)

def miseAJourRayons(G:np.array,Rayons:np.array,Débits:np.array,Pressions:list)->np.array:
    """
    met à jour la matrice des rayons grâce à la relation de renforcement
    """
    RayonMinimal = 0.001
    n=np.shape(G)[0]
    for i in range(n):
        for j in range(i+1,n):
            if Rayons[i,j]>RayonMinimal : 
                Rayons[i,j] = relationRenforcement(Rayons[i,j],abs(Débits[i,j]))
                Rayons[j,i] = Rayons[i,j]
            else :#si la conductance est trop faible on supprime l'arête
                Rayons[i,j]= 0
                Rayons[j,i] = Rayons[i,j]
                G[i,j]=np.inf
                G[j,i]=np.inf
    return Rayons,G

def miseAJourDébits(G:np.array,Rayons:np.array,Débits:np.array,Pressions:list)->np.array:
    """
    met à jour la matrice des débits grâce aux nouvelles valeurs des pressions de chaque noeud et des conductances de chaque arêtes
    """
    n=np.shape(G)[0]
    for i in range(n):
        for j in range(i+1,n):
            if Rayons[i,j] != 0:
                Débits[i,j] = (conductance(Rayons[i][j]))*(Pressions[i]-Pressions[j])/G[i,j]
                Débits[j][i] = -Débits[i][j] #matrice anti-symmétrique
    return Débits

def pointsSupprimés(G:np.array,Terminaux:list)->list:
    """
    revoie la liste des points qui on été supprimés
    """
    res=[]
    n = np.shape(G)[0]
    for i in range(n):
        vide = True
        k=0
        while k < n and vide:
            if G[i,k] != np.inf:
                vide = False
            k += 1
        if vide :
            res.append(i)    
    cc=composanteConnexe(G,Terminaux)
    for k in range(n):
        if not k in cc and not k in res:
            res.append(k)
    return res

def terminauxRéindéxés(G:np.array,Terminaux:list,supr:list)->list:
    res = []
    for t in Terminaux:
        k=0
        for s in range(t):
            if s in supr:
                k +=1
        res.append(t-k)         
    return res

def miseAJourGraphe(G:np.array,Rayons:np.array,Debits:np.array,Lignes:list,Terminaux:list)->np.array:
    """
    réindexe le graphe entré fonction des points supprimés
    """
    supr = pointsSupprimés(G,Terminaux)
    Terminaux=terminauxRéindéxés(G,Terminaux,supr)
    n=len(supr)
    supr.sort
    for s in range(n):
        G= np.delete(G, (supr[n-1-s]), axis=1)
        Rayons= np.delete(Rayons, (supr[n-1-s]), axis=1)
        Debits= np.delete(Debits, (supr[n-1-s]), axis=1)
    for s in range(n):
        G= np.delete(G, (supr[n-1-s]), axis=0)
        Rayons= np.delete(Rayons, (supr[n-1-s]), axis=0)
        Debits= np.delete(Debits, (supr[n-1-s]), axis=0)
        Lignes.pop(supr[n-1-s])
    return G,Rayons,Debits,Lignes,Terminaux

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
    B[Terminaux[puit]]= débitEntrant*(len(Terminaux)-1)
     
    for S in range(n): # Equations du réseau de Poisson
        if S != Terminaux[puit] :
            for V in voisins(S,Blob):
                A[S,V] += conductance(Blob[S,V])/Graphe[S,V]
                A[S,S] -= conductance(Blob[S,V])/Graphe[S,V]
   
    for k in range(n): # la pression au niveau du puit est nulle
        A[k,Terminaux[puit]]=0
    A[Terminaux[puit], Terminaux[puit]] = 1 
    
    Pression=np.linalg.solve(A,B)
    return Pression

def itération(k:int,G:np.array,p:int,Terminaux:list,DébitEntrant:float)->np.array:
    n=np.shape(G)[0]
    Blob=np.copy(G)
    Rayons=rayonsInitiaux(G)
    Débits=np.zeros((n,n))
    Pression=[0 for i in range(n)]
    Lignes=[i for i in range(n)]
    for i in range(k): 
        print(Rayons,Blob)
        Pression,Blob,Terminaux = calculNouvellesPressions(Blob,Rayons,Terminaux,Pression,DébitEntrant)
        Débits = miseAJourDébits(Blob,Rayons,Débits,Pression)
        Rayons,Blob = miseAJourRayons(Blob,Rayons,Débits,Pression)
        Blob,Rayons,Débits,Lignes,Terminaux=miseAJourGraphe(Blob,Rayons,Débits,Lignes,Terminaux)

        if i%5 == 0:
            print(affichegrille(G,p,Blob,Lignes))
    return Blob

#X=np.array([[1,2,3],[4,5,6],[7,8,9]])
#x=np.delete(x,(1),axis=1)
#print(x)

print(affichegrille(grille(7,5),itération(30,grille(7,5),5,[2,30,34],100)))
"""G=nx.grid_graph(dim=(5,5))
pos =nx.planar_layout(G)
aretes=nx.draw_networkx_edge_labels(G,pos)
nx.draw_networkx(G,pos,with_labels=True)
plt.show()
print(G)

G=np.array([[np.inf,1,np.inf,1,np.inf,np.inf,np.inf,np.inf,np.inf],
            [1,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf],
            [np.inf,np.inf,np.inf,np.inf,np.inf,1,np.inf,np.inf,np.inf],
            [1,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf],
            [np.inf,np.inf,np.inf,np.inf,np.inf,1,np.inf,1,np.inf],
            [np.inf,np.inf,1,np.inf,1,np.inf,np.inf,np.inf,1],
            [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1,np.inf],
            [np.inf,np.inf,np.inf,np.inf,1,np.inf,1,np.inf,1],
            [np.inf,np.inf,np.inf,np.inf,np.inf,1,np.inf,1,np.inf] ])
print(affichegrille(G,itération(200,G,3,[2,6],100)))
"""