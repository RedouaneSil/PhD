# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as scp
from multiprocessing import Pool
import multiprocessing
from functools import partial
import time


def lsv_line(x,y,W,t,line,R,A): #x forward, y backward
    size=int(1+x.dimX +((x.dimX+1)*x.dimX /2))
    solution=np.zeros((size,1))
    U=np.zeros(x.Lambda)
     #general case
    """if(y.dimY==1):
        for j in  range(x.Lambda):
            U[j]=(1/y.h)*y.valueY[j,0,t]*(W[t,j,line]-W[t-1,j,line]) 
            #print(y.valueY[j])
    elif(np.shape(W[0])[0]==1):
        for j in  range(x.Lambda):
            U[j]=(1/y.h)*y.valueY[j,line,t]*(W[t,j,0]-W[t-1,j,0])
    else:
        for j in  range(x.Lambda):
            U[j]=(1/y.h)*np.dot(y.valueY[j,:,t],(W[t,j,:]-W[t-1,j,:]))[line]"""
    U = (1 / y.h) * y.valueY[:, line, t] * (W[t, :, 0] - W[t - 1, :, 0])
    solution=np.linalg.lstsq(A, U, rcond=None)[0]
    return solution



def lsu_line(x,y,f,t,line,R,A):
    size=int(1+x.dimX +((x.dimX+1)*x.dimX /2))
    #computation of U
    U=np.zeros(x.Lambda)
    f_vect=f(t-1,x,y)
    U=y.valueY[:,line,t]+(y.h)*f_vect[:,line]
    solution=np.array(scp.lsq_linear(A,U).x)
    return solution



def lsv(x,y,W,t,R,A):
    for i in range(y.dimZ):
        sol=lsv_line(x, y, W , t, i, R,A)
        sol_v=np.dot(A,sol)
        x.v[:,i,t-1]=sol_v
        

def lsu(x,y,f,W,t,R,A):
    for i in range(y.dimY):
        sol=lsu_line(x,y,f,t,i,R,A)
        sol_u=np.dot(A,sol)
        x.u[:,i,t-1]=sol_u
        

        
class forward :
    def __init__(self,n,m,Lambda,dimX,dimY,dimZ,x0,T,D,alpha,beta,gamma):
        self.n=n # time subdivision
        self.m=m # number of iterations
        self.Lambda=Lambda #number of BM
        self.D=D
        self.dimX=dimX #dimension of the forward vector X
        self.dimY=dimY #dimension of the backward vector Y
        self.dimZ=dimZ #dimension of the backward vector Z
        self.valueX=np.zeros((Lambda,dimX+1,n))  #X for all lambda
        self.valueX[:,1,:]=D
        self.u=np.zeros((Lambda,dimY,n)) #decoupling field u(x_{\lambda}) for all lambda
        self.v=np.zeros((Lambda,dimZ,n)) #decoupling field v(x), fixed =0
        self.x0=x0
        #self.x=x
        self.T=T
        self.h=T/(n-1)
        self.dimX=dimX
        self.alpha=alpha
        self.gamma=gamma
        self.beta=beta
        self.u_g=(np.sqrt(alpha*beta)-gamma)/(np.sqrt(alpha*beta)+gamma)
    ###update X    
    def update(self,W,b,sigma):
        t1=time.time()
        self.valueX[:,0,0]=self.x0 #starting point
        #here sigma is constant, hence the next line
        sigm_global=sigma(0,self)
        for i in range(self.n-1):
            self.valueX[:,0,i+1]=self.valueX[:,0,i]+b(i,self)*self.h+np.transpose((W[i+1,:,0]-W[i,:,0]))*sigm_global
        t2=time.time()
        #print(t2-t1)
class backward:
    def __init__(self,n,m,Lambda,dimX,dimY,dimZ,x0,T,alpha,beta,gamma):
        self.n=n # time subdivision
        self.m=m # number of iterations
        self.Lambda=Lambda #number of BM
        self.dimX=dimX #dimension of the forward vector X
        self.dimY=dimY #dimension of the backward vector Y
        self.dimZ=dimZ #dimension of the backward vector Z
        self.valueY=np.zeros((Lambda,dimY,n)) #Y for all lambda
        self.valueZ=np.zeros((Lambda,dimZ,n)) #Z for all lambda
        self.x0=x0
        self.T=T
        self.h=T/(n-1)
        self.alpha=alpha
        self.gamma=gamma
        self.beta=beta
        self.u_g=(np.sqrt(alpha*beta)-gamma)/(np.sqrt(alpha*beta)+gamma)
        
    def update(self,W,f,forward,R):
        t1=time.time()
        ##exemple in the article : Y_T != 0
        g=np.zeros((forward.Lambda,1))
        #terminal condition for Y
        self.valueY[:,:,self.n-1]=g
        #print(self.u)
        #update from n-1 to 1
        size=int(1+forward.dimX +((forward.dimX+1)*forward.dimX /2))
        shape0 = forward.valueX[:,:,0].shape[0]
        shape1 = forward.valueX[:,:,0].shape[1]
        indices = np.triu_indices(shape1)  # Obtient les indices des éléments supérieurs ou égaux à la diagonale
        for i in range(self.n-2,0,-1):
            # Crée une matrice de toutes les multiplications possibles xip * xjp
            products = np.einsum('pi,pj->pij', forward.valueX[:,:,i], forward.valueX[:,:,i])[:, indices[0], indices[1]]
            products = np.maximum(np.minimum(products, R), -R)
            # Crée une matrice de taille (p, n + 1 + n*(n+1)//2)
            A = np.zeros((shape0, shape1 + 1 + shape1 * (shape1 + 1) // 2))
            # Remplit la matrice avec les valeurs appropriées
            A[:, 0] = 1  # Ajoute le 1 en première colonne
            A[:, 1:shape1 + 1] = forward.valueX[:,:,i]  # Ajoute les valeurs xip pour i <= n
            A[:, shape1 + 1:] = products  # Ajoute les produits xip * xjp
            self.valueY[:, :, i + 1] = forward.u[:, :, i + 1]
            lsv(forward,self,W,i+1,R,A)
            #print(self.valueY)
            self.valueZ[:, :, i] = forward.v[:, :, i]
            lsu(forward,self,f,W,i+1,R,A)
        #example
        z=np.zeros(self.dimZ)
        y=np.zeros(self.dimY)
        #update first value of Y and Z at time 0 and 1
        self.valueY[:,:,1]=forward.u[:,:,1]
        Ydot=np.transpose(self.valueY[:,:,1])
        W_10=np.diag(W[1,:,0]-W[0,:,0])
        z=(1/self.Lambda)*(1/self.h)*np.sum(np.dot(Ydot,W_10),axis=1)
        Zlam=np.tile(z, (self.Lambda, 1))
        self.valueZ[:,:,0]=Zlam
        forward.v[:,:,0]=Zlam
        vect_f=f(0,forward,self)
        y=(1/self.Lambda)*np.sum(self.valueY[:,:,1]+vect_f*self.h,axis=0)
        yLam=np.tile(y,(self.Lambda,1))
        self.valueY[:,:,0]=yLam
        forward.u[:,:,0]=yLam
        t2=time.time()
        #print(t2-t1)
        
