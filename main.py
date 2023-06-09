import numpy as np
from Class import*
import matplotlib.pyplot as plt
import scipy.optimize as scp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import random
import multiprocessing
from multiprocessing import Pool,Manager
from functools import partial
import time
import timeit

beta=5
alpha=5
gamma=100000
vector_beta=[1,2,5,10,20]
vector_alpha=[3,5,10,20]
vector_gamma=[0,0.1,500,1000]
Lambda=1000
n=2000














##problem constants for our problem
T=7
theta1=80*np.pi
theta=2
mu=50
rho=0.4
sigma_vol=10
#FBSDE of the form dX=b(t,X_t,Y_t)dt +\sigma(t,X_t,Y_t)dW_t
#conventional producers
F0=30
F=0.5
kappa=1/(F+(1/(2*alpha)))
#initial conditions
Q0=400
D0=300



m=30
dimX=1
dimY=2
dimZ=2
dim_BM=2
R=100 #basis functions
maxP=10000
P0=-100

u=(np.sqrt(alpha*beta)-gamma)/(np.sqrt(alpha*beta)+gamma)

Price=np.zeros((Lambda,n))
"""
####exemple from article:
x_0=np.pi/2
D=10
r=0
T=1
sigma_vol=0.1
dimX=D
dimY=1
dimZ=D
dim_BM=D
Lambda=4000
R=10
n=50
m=16
"""




def conventional(P):
     #return F*(P**2) #quadratic
   return F0+F*P #linear


#Auxiliary functions 

def F1(t,T,alpha,beta,gamma,u):
    num=np.exp(np.sqrt(beta/alpha)*t)*np.sqrt(beta/alpha)
    denom=1+u*np.exp(-2*np.sqrt(beta/alpha)*(T-t))
    return num/denom
    
def F2(t,T,alpha,beta,gamma,u):
    num=u*np.exp(-np.sqrt(beta/alpha)*(2*T-t))*np.sqrt(beta/alpha)
    denom=1+u*np.exp(-2*np.sqrt(beta/alpha)*(T-t))
    return num/denom


def fp(t,T,alpha,beta,gamma,u):
    num=1-u*np.exp(-2*np.sqrt(beta/alpha)*(T-t))
    denom=1+u*np.exp(-2*np.sqrt(beta/alpha)*(T-t))
    return(num/denom)



###function b
def b(i,forward):
    # my problem
    alpha=forward.alpha
    beta=forward.beta
    gamma=forward.gamma
    u=forward.u_g
    kappa=1/(F+(1/(2*alpha)))
    image=np.zeros((Lambda))
    valueX_at_t=forward.valueX[:,:,i]
    t=i*T/(forward.n-1)
    valueX_l_at_t=forward.valueX[:,:,i]
    valueY_l_at_t=forward.u[:,:,i]
    Y1=valueY_l_at_t[:,0]
    Y2=valueY_l_at_t[:,1]
    Qt=valueX_l_at_t[:,0]
    Dt=forward.D[:,i]
    Pt=kappa*(Dt-F0-np.sqrt(beta/alpha)*fp(t,T,alpha,beta,gamma,u)*(Qt-Q0)+F1(t,T,alpha,beta,gamma,u)*Y1-F2(t,T,alpha,beta,gamma,u)*Y2)
    image=-(Dt-conventional(Pt))
    #exemple from articl
    #image=np.zeros((D,1))
    return image


##function sigma
def sigma(i,forward):
    #our fbsde
    #sigm=np.zeros((1,2))
    #example from article
    """
    t=i*T/forward.n
    valueY_l_at_t=forward.u[l,:,i]
    sigm=sigma_vol*valueY_l_at_t[0]"""
    return rho


def f(i,forward,backward):
    alpha=forward.alpha
    beta=forward.beta
    gamma=forward.gamma
    u=forward.u_g
    kappa=1/(F+(1/(2*alpha)))
    valueX_at_t=forward.valueX[:,:,i]
    t=i*T/(forward.n-1)
    valueX_l_at_t=forward.valueX[:,:,i]
    valueY_l_at_t=forward.u[:,:,i+1]
    Y1=valueY_l_at_t[:,0]
    Y2=valueY_l_at_t[:,1]
    Qt=valueX_l_at_t[:,0]
    Dt=forward.D[:,i]
    Pt=kappa*(Dt-F0-np.sqrt(beta/alpha)*fp(t,T,alpha,beta,gamma,u)*(Qt-Q0)+F1(t,T,alpha,beta,gamma,u)*Y1-F2(t,T,alpha,beta,gamma,u)*Y2)
    #test for a non linear price:
    #Pt=pricefunction(Dt-np.sqrt(beta/alpha)*fp(t,T)*(Qt-Q0)+F1(t,T)*Y1-F2(t,T)*Y2)
    image=np.zeros((Lambda,dimY))
    image[:,0]=np.exp(-np.sqrt(beta/alpha)*t)*(Pt/(2*alpha))
    image[:,1]=np.exp(np.sqrt(beta/alpha)*t)*(Pt/(2*alpha))
    #remise à l'échelle
    #image[1]=np.sqrt(beta/alpha)*Y2+(Pt/(2*alpha))
    #image[1]=Y2+(Pt/(2*np.sqrt(alpha*beta)))
    #image[2]=(Pt/(2*alpha))
    #example from article
    """
    valueX_l_at_t=forward.valueX[l,:,i]
    partial_sum=0 
    for d in range(D):
        partial_sum+=np.sin(valueX_l_at_t[d])
    t=i*T/forward.n
    image=-r*backward.valueY[l,:,i+1]+0.5*np.exp(-3*r*(T-t))*(sigma_vol**2)*((partial_sum)**3)
    """
    return image


 
def brownian(n,T,Lambda,dim):
    dt=T/(n-1)
    dB = np.sqrt(dt) * np.random.randn(n, Lambda,dim)  # Incréments browniens
    B = np.cumsum(dB, axis=0)
    return B

def Demand():
    W=brownian(n,T,Lambda,1)
    result=np.zeros((Lambda,n))
    times=np.arange(0,n)*(T/(n-1))
    cos=np.tile((theta1*np.sin(2*np.pi*times)/(2*np.pi))+D0,(Lambda,1))
    return cos + sigma_vol*np.transpose(W[:,:,0])
    
    
def main(alpha,beta,gamma):
    x0=np.array([Q0])
    #x0=np.array([x_0 for i in range(D)])
    #exemple from article
    W=brownian(n,T, Lambda,dim_BM)
    D=Demand()
    backward_variables=backward(n,m,Lambda,dimX,dimY,dimZ,x0,T,alpha,beta,gamma)
    forward_variables=forward(n,m,Lambda,dimX,dimY,dimZ,x0,T,D,alpha,beta,gamma)
    previousY=np.zeros(dimY)
    previousX=np.zeros(dimX)
    previousX[0]=-100
    mmax=0
    valeur=-100
    while max(max([abs(previousY[i]-backward_variables.valueY[0,i,0]) for i in range(dimY)]),max([abs(previousX[i]-forward_variables.valueX[0,i,n-1]) for i in range(dimX)]))>0.01:
        mmax+= 1
        for i in range(dimY):
            previousY[i]=backward_variables.valueY[0,i,0]
        for i in range(dimX):
            previousX[i]=forward_variables.valueX[0,i,n-1]
        forward_variables.update(W,b,sigma)
        backward_variables.update(W,f,forward_variables,R)
        #print("finalX=" +str(forward_variables.valueX[0,i,n-1]))
        #print("initialYdiff= " + str([previousY[i]-backward_variables.valueY[0,i,0] for i in range(dimY)]))
        print("ITERATION : " + str(mmax))
        print("diff max="+str(max(max([abs(previousY[i]-backward_variables.valueY[0,i,0]) for i in range(dimY)]),max([abs(previousX[i]-forward_variables.valueX[0,i,n-1]) for i in range(dimX)]))))
    print("iteration max="+str(mmax))
    #real_value=[np.exp(-r*(T-(i*T/n)))*sum(np.sin(forward_variables.valueX[0,d,i]) for d in range(D)) for i in range(n)]
    #plt.plot(time1,real_value,'r-', label="real")
    #plt.plot(time1,forward_variables.valueX[0,0],'b-', label="simu")   
    return (backward_variables.valueY,forward_variables.valueX,W,backward_variables.valueZ,mmax,D)


time1=np.linspace(0,T,n)
dt=T/(n-1)


#######utilisation des résultats


def computePt(X,Y,D,l,alpha,beta,gamma):
    Pt=np.zeros(n)
    u=(np.sqrt(alpha*beta)-gamma)/(np.sqrt(alpha*beta)+gamma)
    for i in range(n):
        t=i*T/(n-1)
        valueX_l_at_t=X[l,:,i]
        valueY_l_at_t=Y[l,:,i]
        Y1=valueY_l_at_t[0]
        Y2=valueY_l_at_t[1]
        Qt=valueX_l_at_t[0]
        Dt=D[l,i]
        Pt[i]=kappa*(Dt-F0-np.sqrt(beta/alpha)*fp(t,T,alpha,beta,gamma,u)*(Qt-Q0)+F1(t,T,alpha,beta,gamma,u)*Y1-F2(t,T,alpha,beta,gamma,u)*Y2)

        #test for a non linear conventional function:
        #Pt[i]=pricefunction(Dt-np.sqrt(beta/alpha)*fp(t,T)*(Qt-Q0)+F1(t,T)*Y1-F2(t,T)*Y2)
    return(Pt)

 #première exploitation : tracé de chaque variable dans un scénario donné décidé par les paramètres tout en haut

def simulation(alpha=alpha,beta=beta,gamma=gamma,n=n):
    t1=time.time()      
    time1=np.linspace(0,T,n)
    result=main(alpha,beta,gamma)
    Y=result[0]
    X=result[1]
    W=result[2]
    Z=result[3]
    mmax=result[4]
    D=result[5]
    Pt=computePt(X,Y,D,0,alpha,beta,gamma)
    conventional_energy=[conventional(Pt[i]) for i in range(n)]
    valueX_l_at_t=X[0,:,:]
    valueY_l_at_t=Y[0,:,:]
            
    Y1=valueY_l_at_t[0]
            
    Y2=valueY_l_at_t[1]
            
            
    #Dt=valueX_l_at_t[1]
    
    ##tracé
    fig, axs = plt.subplots(1,3)
    fig.suptitle('Plot with alpha='+str(alpha)+', beta='+str(beta)+', F(Pt)='+str(F0)+'+'+str(F)+'*Pt')
    axs[0].plot(time1,X[0,0],'b-', label="alpha=" + str(alpha))
    axs[0].set_title('Storage level')
    axs[1].plot(time1,Pt,'b-', label="alpha=" + str(alpha))
    axs[1].set_title('Price')
    axs[2].plot(time1,D[0],'b-', label="alpha=" + str(alpha))
    axs[2].set_title('Demand')
    # axs[3].plot(time1,Y[0,0],'b-', label="simu")
    # axs[3].set_title('Y1')
    # axs[4].plot(time1,Y[0,1],'b-', label="simu")
    # axs[4].set_title('Y2')
    axs[0].grid()   
    axs[1].grid()
    axs[2].grid()
    # axs[3].grid()   
    # axs[4].grid()  
    t2=time.time()
    print("execution time="+str(t2-t1))



#seconde exploitation de l'algo, tracé de l'évolution du prix selon les valeurs de beta

def simulation_beta(alpha=alpha,vector=vector_beta,gamma=gamma,n=n):
    t1=time.time()
    time1=np.linspace(0,T,n)
    fig, axs = plt.subplots(1,3)
    kappa=1/(F+(1/(2*alpha)))
    fig.suptitle('Plot with parameters beta changing, alpha='+str(alpha)+', F(Pt)='+str(F0)+'+'+str(F)+'*Pt')
    for beta in vector_beta:
        np.random.seed(10)
        u=(np.sqrt(alpha*beta)-gamma)/(np.sqrt(alpha*beta)+gamma)
        result=main(alpha,beta,gamma)
        Y=result[0]
        X=result[1]
        W=result[2]
        Z=result[3]
        D=result[5]
        Pt=computePt(X,Y,D,0,alpha,beta,gamma)
        axs[0].plot(time1,X[0,0],'-', label="beta=" + str(beta))
        axs[0].set_title('Storage level')
        axs[0].legend()
        axs[1].plot(time1,Pt,'-', label="beta=" + str(beta))
        axs[1].set_title('Electricity price')
        axs[1].legend()
        axs[2].plot(time1,D[0],'-', label="beta=" + str(beta))
        axs[2].set_title('Demand')
        axs[2].legend()
    t2=time.time()
    print("execution time="+str(t2-t1))    
    
    
 
    

#tracé de l'évolution selon les valeurs de alpha

def simulation_alpha(vector=vector_alpha,beta=beta,gamma=gamma):
    t1=time.time()
    time1=np.linspace(0,T,n)
    fig, axs = plt.subplots(1,3)
    fig.suptitle('Plot with parameters alpha changing, beta='+str(beta)+', F(Pt)='+str(F0)+'+'+str(F)+'*Pt')
    for alpha in vector:
        u=(np.sqrt(alpha*beta)-gamma)/(np.sqrt(alpha*beta)+gamma)
        np.random.seed(4)
        kappa=1/(F+(1/(2*alpha)))
        result=main(alpha,beta,gamma)
        Y=result[0]
        X=result[1]
        W=result[2]
        Z=result[3]
        D=result[5]
        # qt=np.zeros(n)
        # for j in range(n-1):
        #     qt[j]=(X[0,0,j+1]-X[0,0,j])/(T/(n-1))
        Pt=computePt(X,Y,D,0,alpha,beta,gamma)
        axs[0].plot(time1,X[0,0],'-', label="alpha=" + str(alpha))
        axs[0].set_title('Storage level')
        axs[0].legend()
        axs[1].plot(time1,Pt,'-', label="alpha=" + str(alpha))
        axs[1].set_title('Electricity price')
        axs[1].legend()
        axs[2].plot(time1,D[0],'-', label="alpha=" + str(alpha))
        axs[2].set_title('Demand')
        axs[2].legend() 
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    t2=time.time()
    print("execution time="+str(t2-t1))



 #6ème exploitation, tracé de l'évolution du prix selon les valeurs de gamma


 
  
def simulation_gamma(alpha=alpha,beta=beta,vector=vector_gamma,n=n): 
    t1=time.time()
    time1=np.linspace(0,T,n)
    kappa=1/(F+(1/(2*alpha)))
    fig, axs = plt.subplots(1,3)
    fig.suptitle('Plot with parameters gamma changing, alpha='+str(alpha)+', beta='+str(beta)+', F(Pt)='+str(F0)+'+'+str(F)+'*Pt')
    for gamma in vector:
        np.random.seed(10)
        u=(np.sqrt(alpha*beta)-gamma)/(np.sqrt(alpha*beta)+gamma)
        result=main(alpha,beta,gamma)
        Y=result[0]
        X=result[1]
        W=result[2]
        Z=result[3]
        D=result[5]
        Pt=computePt(X,Y,D,0,alpha,beta,gamma)
        axs[0].plot(time1,X[0,0],'-', label="gamma=" + str(gamma))
        axs[0].set_title('Storage level')
        axs[0].legend()
        axs[1].plot(time1,Pt,'-', label="gamma=" + str(gamma))
        axs[1].set_title('Electricity price')
        axs[1].legend()
        axs[2].plot(time1,D[0],'-', label="gamma=" + str(gamma))
        axs[2].set_title('Demand')
        axs[2].legend()
    t2=time.time()
    print("execution time="+str(t2-t1))
  
def d(t):
    return (theta1*np.sin(2*np.pi*t)/(2*np.pi))+D0
"""
def d(t):
    return (theta1*np.sin(2*np.pi*t)/(2*np.pi))+D0
def mysys(z,t):
    res=[]
    P=min(d(t)*(kappa)-kappa*F0-kappa*np.sqrt(beta/alpha)*fp(t,T)*(z[0]-Q0)+kappa*F1(t,T)*z[1]-kappa*F2(t,T)*z[2]-kappa*F3(t,T)*z[3],maxP)
    res.append(-d(t)+F0+F*P)
    res.append(-np.exp(-np.sqrt(beta/alpha)*t)*(1/(2*alpha))*P)
    res.append(-z[2]-(1/(2*np.sqrt(alpha*beta)))*P)
    res.append(-(1/(2*alpha))*P)
    #print(alpha)
    return res

def solver_forward_backward(xf):
    t=np.linspace(T,0,n)
    yf=[xf,0,0,0]
    sol=odeint(mysys, yf, t)
    x0=sol[n-1,0]
    #print(x0)
    return x0
def systemtooptimize(x):
    return(solver_forward_backward(x)-Q0)

res=scp.root_scalar(systemtooptimize,bracket=[-100000,100000])

time2=np.linspace(T,0,n)
yf=[res.root,0,0,0]
deterministicplot=odeint(mysys, yf, time2)
Qtheo=deterministicplot[:,0]
Dtheo=[d(i) for i in time2]
Y1theo=deterministicplot[:,1]
Y2theo=deterministicplot[:,2]
Y3theo=deterministicplot[:,3]
fig, axs = plt.subplots(2,4)
fig.suptitle('Plot with parameters beta='+str(beta)+',alpha='+str(alpha) + ', Q_0='+str(Q0)+', D0=' + str(D0)+ ',vol_storage=' + str(rho)+',vol_demand=' + str(sigma_vol))
axs[0,0].plot(time2,Qtheo,'-', label="alpha=" + str(alpha))
axs[0,0].set_title('Storage level')
axs[0,0].grid()
axs[0,1].plot(time2,Y1theo,'-', label="alpha=" + str(alpha))
axs[0,1].set_title('Y1theo')
axs[0,1].grid()
axs[0,2].plot(time2,Y2theo,'-', label="alpha=" + str(alpha))
axs[0,2].set_title('Y2theo')
axs[0,2].grid()
axs[0,3].plot(time2,Y3theo,'-', label="alpha=" + str(alpha))
axs[0,3].set_title('Y3theo')
axs[0,3].grid()
axs[1,0].plot(time1,X[0,0],'-', label="alpha=" + str(alpha))
axs[1,0].set_title('STORAGE')
axs[1,0].grid()
axs[1,1].plot(time1,Y[0,0],'-', label="alpha=" + str(alpha))
axs[1,1].set_title('Y1')
axs[1,1].grid()
axs[1,2].plot(time1,Y[0,1],'-', label="alpha=" + str(alpha))
axs[1,2].set_title('Y2')
axs[1,2].grid()
axs[1,3].plot(time1,Y[0,2],'-', label="alpha=" + str(alpha))
axs[1,3].set_title('Y3')
axs[1,3].grid()
"""
"""
beta=0.001
time2=np.linspace(T,0,n)

fig, axs = plt.subplots(1,3)

fig.suptitle('Plot with parameters beta='+str(beta)+ ', Q_0='+str(Q0)+', D0=' + str(D0)+ ',vol_storage=' + str(rho)+',vol_demand=' + str(sigma_vol))
for alpha in [100]:
    u=(np.sqrt(alpha*beta)-gamma)/(np.sqrt(alpha*beta)+gamma)
    np.random.seed(10)
    kappa=1/(F+(1/(2*alpha)))
    res=scp.root_scalar(systemtooptimize,bracket=[-100000,10000])
    yf=[res.root,0,0,0]
    deterministicplot=odeint(mysys, yf, time2)
    Qtheo=deterministicplot[:,0]
    Dtheo=[d(i) for i in time2]
    Y1theo=deterministicplot[:,1]
    Y2theo=deterministicplot[:,2]
    Y3theo=deterministicplot[:,3]
    Pttheo=np.zeros(n)
    for i in range(n):
        Y1=Y1theo[i]
        Y2=Y2theo[i]
        Y3=Y3theo[i]
        Qt=Qtheo[i]
        Dt=Dtheo[i]
        t=time2[i]
        Pttheo[i]=min(kappa*(Dt-F0-np.sqrt(beta/alpha)*fp(t,T)*(Qt-Q0)+F1(t,T)*Y1-F2(t,T)*Y2-F3(t,T)*Y3),maxP)
    axs[0].plot(time2,Qtheo,'-', label="alpha=" + str(alpha))
    axs[0].set_title('Storage level')
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(time2,Pttheo,'-', label="alpha=" + str(alpha))
    axs[1].set_title('Price')
    axs[1].grid()
    axs[1].legend()
    axs[2].plot(time2,Dtheo,'-', label="alpha=" + str(alpha))
    axs[2].set_title('Demand')
    axs[2].legend()
    axs[2].grid()

"""
"""
beta=5
time2=np.linspace(T,0,n)

alpha=1
u=(np.sqrt(alpha*beta)-gamma)/(np.sqrt(alpha*beta)+gamma)
kappa=1/(F+(1/(2*alpha)))
fig, axs = plt.subplots(1,3)
fig.suptitle('Plot with parameters beta='+str(beta)+',alpha='+str(alpha) + ', Q_0='+str(Q0)+', D0=' + str(D0)+ ',vol_storage=' + str(rho)+',vol_demand=' + str(sigma_vol))
for maxP in [700,900,1000,1200,1800,3000,4000]:
    np.random.seed(10)
    res=scp.root_scalar(systemtooptimize,bracket=[-100000,10000])
    yf=[res.root,0,0,0]
    deterministicplot=odeint(mysys, yf, time2)
    Qtheo=deterministicplot[:,0]
    Dtheo=[d(i) for i in time2]
    Y1theo=deterministicplot[:,1]
    Y2theo=deterministicplot[:,2]
    Y3theo=deterministicplot[:,3]
    Pttheo=np.zeros(n)
    for i in range(n):
        Y1=Y1theo[i]
        Y2=Y2theo[i]
        Y3=Y3theo[i]
        Qt=Qtheo[i]
        Dt=Dtheo[i]
        t=time2[i]
        Pttheo[i]=min(kappa*(Dt-F0-np.sqrt(beta/alpha)*fp(t,T)*(Qt-Q0)+F1(t,T)*Y1-F2(t,T)*Y2-F3(t,T)*Y3),maxP)
    axs[0].plot(time2,Qtheo,'-', label="maxP=" + str(maxP))
    axs[0].set_title('Storage level')
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(time2,Pttheo,'-', label="maxP=" + str(maxP))
    axs[1].set_title('Price')
    axs[1].grid()
    axs[1].legend()
    axs[2].plot(time2,Dtheo,'-', label="maxP=" + str(maxP))
    axs[2].set_title('Demand')
    axs[2].legend()
    axs[2].grid()
    """



"""
###test with cubic F
def fconv(P):
    if (P<maxP/10):
        return F0
    elif(P<maxP/5):
        return F*P/2 +F0
    elif(P<maxP/2):
        return F*P+F0
    else:
        return 2*F*max(min(P,maxP),0)+F0
    return F0+F*P #F*(max(min(P,maxP),0)**3)+F0

def inverseP(y):
    def to_optimize(x):
        return (x/(2*alpha))+ fconv(x)-max(min(y,maxP),0)
    return inverse(to_optimize)
def d(t):
    return (theta1*np.sin(2*np.pi*t)/(2*np.pi))+D0
def mysys(z,t):
    res=[]
    P=max(min(inverseP(d(t)-np.sqrt(beta/alpha)*fp(t,T)*(z[0]-Q0)+F1(t,T)*z[1]-F2(t,T)*z[2]-F3(t,T)*z[3]).root,maxP),0)
    res.append(-d(t)+fconv(P))
    res.append(-np.exp(-np.sqrt(beta/alpha)*t)*(1/(2*alpha))*kappa*(min(d(t)-F0-np.sqrt(beta/alpha)*fp(t,T)*(z[0]-Q0)+F1(t,T)*z[1]-F2(t,T)*z[2]-F3(t,T)*z[3],maxP)))
    res.append(-z[2]-(1/(2*np.sqrt(alpha*beta)))*kappa*min(d(t)-F0-np.sqrt(beta/alpha)*fp(t,T)*(z[0]-Q0)+F1(t,T)*z[1]-F2(t,T)*z[2]-F3(t,T)*z[3],maxP))
    res.append(-(1/(2*alpha))*kappa*min(d(t)-F0-np.sqrt(beta/alpha)*fp(t,T)*(z[0]-Q0)+F1(t,T)*z[1]-F2(t,T)*z[2]-F3(t,T)*z[3],maxP))
    return res

def solver_forward_backward(xf):
    t=np.linspace(T,0,n)
    yf=[xf,0,0,0]
    sol=odeint(mysys, yf, t)
    x0=sol[n-1,0]
    return x0
def systemtooptimize(x):
    return(solver_forward_backward(x)-Q0)


time2=np.linspace(T,0,n)


beta=5
time2=np.linspace(T,0,n)

fig, axs = plt.subplots(1,3)

fig.suptitle('Plot with parameters beta='+str(beta)+ ', Q_0='+str(Q0)+', D0=' + str(D0)+ ',vol_storage=' + str(rho)+',vol_demand=' + str(sigma_vol))
for alpha in [0.5,1,3,5,10]:
    u=(np.sqrt(alpha*beta)-gamma)/(np.sqrt(alpha*beta)+gamma)
    np.random.seed(10)
    kappa=1/(F+(1/(2*alpha)))
    res=scp.root_scalar(systemtooptimize,bracket=[-100000,10000])
    yf=[res.root,0,0,0]
    deterministicplot=odeint(mysys, yf, time2)
    Qtheo=deterministicplot[:,0]
    Dtheo=[d(i) for i in time2]
    Y1theo=deterministicplot[:,1]
    Y2theo=deterministicplot[:,2]
    Y3theo=deterministicplot[:,3]
    Pttheo=np.zeros(n)
    for i in range(n):
        Y1=Y1theo[i]
        Y2=Y2theo[i]
        Y3=Y3theo[i]
        Qt=Qtheo[i]
        Dt=Dtheo[i]
        t=time2[i]
        Pttheo[i]=max(min(inverseP(Dt-np.sqrt(beta/alpha)*fp(t,T)*(Qt-Q0)+F1(t,T)*Y1-F2(t,T)*Y2-F3(t,T)*Y3).root,maxP),0)
    axs[0].plot(time2,Qtheo,'-', label="alpha=" + str(alpha))
    axs[0].set_title('Storage level')
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(time2,Pttheo,'-', label="alpha=" + str(alpha))
    axs[1].set_title('Price')
    axs[1].grid()
    axs[1].legend()
    axs[2].plot(time2,Dtheo,'-', label="alpha=" + str(alpha))
    axs[2].set_title('Demand')
    axs[2].legend()
    axs[2].grid()


time2=np.linspace(T,0,n)


t2=time.time()
print("execution time="+str(t2-t1))
"""

##cas deterministe 



def integrsinhDt(t,omega,omegat,alpha,beta):
    res=(1/(1+(4*(np.pi*np.pi)/(omegat*omegat))))*((-theta1/(2*np.pi*omegat))*np.sin(2*np.pi*t)+(theta1/(omegat**2))*np.sinh(omegat*t))+((D0-F0)*(np.cosh(omegat*t)-1)/omegat)
    return res



#     return res

def integrcoshDt(t,omega,omegat,alpha,beta):
    res=(1/(1+(4*(np.pi*np.pi)/(omegat*omegat))))*(theta1/(omegat*omegat))*(-np.cos(2*np.pi*t)+np.cosh(omegat*t))+((D0-F0)*np.sinh(omegat*t)/omegat)
    return res



def integrcoshsinh(t,omega,omegat,alpha,beta):
    res=(1/(2*(omega-omegat)))*(np.cosh(omega*t)-np.cosh(omegat*t)) + (1/(2*(omega+omegat)))*(-np.cosh(omega*t)+np.cosh(omegat*t))
    return res


def integrcoshcosh(t,omega,omegat,alpha,beta):
    res=(1/(2*(omega-omegat)))*(np.sinh(omega*t)-np.sinh(omegat*t)) + (1/(2*(omega+omegat)))*(np.sinh(omega*t)+np.sinh(omegat*t))
    return res
def G(t,omega,omegat,c1,alpha,beta):
    res=(omega/(omegat*(F+(1/alpha))))*integrsinhDt(t,omega,omegat,alpha,beta) + c1*((omega**3)/(omegat*(F+(1/alpha))))*integrcoshsinh(t,omega,omegat,alpha,beta)
    return res


def computedeterPt(t,omega,omegat,c1,alpha,beta):
    return((d(t)-F0-(omega/alpha)*G(t,omega,omegat,c1,alpha,beta)+c1*(omega**2)*np.cosh(omega*t))/(F+(1/alpha)))



##determinist with alpha varying


def deterministic_sim(alpha=alpha,beta=beta,gamma=gamma,n=n):
    alpha1=alpha
    beta1=beta
    beta=beta/2
    alpha=alpha/2
    t1=time.time()
    time1=np.linspace(0,T,n)
    omega=np.sqrt(beta/alpha)
    omegat=omega*np.sqrt(F/(F+(1/alpha)))
    A=(omega/(omegat*(F+(1/alpha))))*integrsinhDt(T,omega,omegat,alpha,beta)
    
    
    
    B=((omega**3)/(omegat*(F+(1/alpha)))) * integrcoshsinh(T,omega,omegat,alpha,beta)
    Aprime=(omega/(F+(1/alpha)))*integrcoshDt(T,omega,omegat,alpha,beta)
    
    Bprime=((omega**3)/(F+(1/alpha)))*integrcoshcosh(T,omega,omegat,alpha,beta)
    
    numc1=gamma*Aprime+beta*A
    denomc1=alpha*omega*(gamma*omega*np.sinh(omega*T)+beta*np.cosh(omega*T)) - gamma*Bprime - beta*B
    c1=numc1/denomc1
    
    valuesG=[G(t,omega,omegat,c1,alpha,beta) for t in time1]
    
    Pricedeter=[computedeterPt(t,omega,omegat,c1,alpha,beta) for t in time1]
    
    
    #Computation of q_t
    # print("B=" + str(B))
    # print("Bprime=" + str(integrcoshcosh(T,omega,omegat)))
    # valueq=[((d(t)-F0)/(1+alpha*F)) + (alpha*F/(1+alpha*F))*((omega/alpha)*G(t,omega,omegat,c1)-c1*omega*omega*np.cosh(omega*t)) for t in time1]
    
    #Computation of Qt :
    valueQ=[Q0-((1/(1+(alpha*F)))*(integrcoshDt(t,omega,omegat,alpha,beta) + c1*omega*omega*integrcoshcosh(t,omega,omegat,alpha,beta)-c1*omega*np.sinh(omega*t)-(c1*beta*F*np.sinh(omega*t)/(omega)))) for t in time1]
    
    fig, axs = plt.subplots(1,3)
    fig.suptitle('Plot with alpha='+str(alpha1)+', beta='+str(beta1)+', F(Pt)='+str(F0)+'+'+str(F)+'*Pt')
    axs[0].plot(time1,Pricedeter,'-', label='alpha=' + str(alpha1))
    axs[0].set_title('Price')
    axs[1].plot(time1,valueQ,'-', label='alpha=' + str(alpha1))
    axs[1].set_title('Q')
    axs[2].plot(time1,d(time1),'-', label='alpha=' + str(alpha1))
    axs[2].set_title('Energy demand')   
    axs[2].grid()
    axs[1].grid()
    axs[0].grid()
    t2=time.time()
    print("execution time="+str(t2-t1))


def deterministic_sim_alpha(vector=vector_alpha,beta_1=beta,gamma=gamma,n=n):
    t1=time.time()
    fig, axs = plt.subplots(1,3)
    time1=np.linspace(0,T,n)
    fig.suptitle('Plot with parameters alpha changing, beta='+str(beta_1)+', F(Pt)='+str(F0)+'+'+str(F)+'*Pt')
    
    time1=np.linspace(0,T,n)
    for alph in vector:
        alpha1=alph
        beta1=beta_1
        beta=beta_1/2
        alpha=alph/2
        omega=np.sqrt(beta/alpha)
        omegat=omega*np.sqrt(F/(F+(1/alpha)))
        A=(omega/(omegat*(F+(1/alpha))))*integrsinhDt(T,omega,omegat,alpha,beta)
        
        
        B=((omega**3)/(omegat*(F+(1/alpha)))) * integrcoshsinh(T,omega,omegat,alpha,beta)
        Aprime=(omega/(F+(1/alpha)))*integrcoshDt(T,omega,omegat,alpha,beta)
        
        Bprime=((omega**3)/(F+(1/alpha)))*integrcoshcosh(T,omega,omegat,alpha,beta)
        
        numc1=gamma*Aprime+beta*A
        denomc1=alpha*omega*(gamma*omega*np.sinh(omega*T)+beta*np.cosh(omega*T)) - gamma*Bprime - beta*B
        c1=numc1/denomc1
        
        valuesG=[G(t,omega,omegat,c1,alpha,beta) for t in time1]
        
        Pricedeter=[computedeterPt(t,omega,omegat,c1,alpha,beta) for t in time1]
        
        
        #Computation of q_t
        # print("B=" + str(B))
        # print("Bprime=" + str(integrcoshcosh(T,omega,omegat)))
        # valueq=[((d(t)-F0)/(1+alpha*F)) + (alpha*F/(1+alpha*F))*((omega/alpha)*G(t,omega,omegat,c1)-c1*omega*omega*np.cosh(omega*t)) for t in time1]
        
        #Computation of Qt :
        valueQ=[Q0-((1/(1+(alpha*F)))*(integrcoshDt(t,omega,omegat,alpha,beta) + c1*omega*omega*integrcoshcosh(t,omega,omegat,alpha,beta)-c1*omega*np.sinh(omega*t)-(c1*beta*F*np.sinh(omega*t)/(omega)))) for t in time1]
        
        axs[0].plot(time1,Pricedeter,'-', label='alpha=' + str(alpha1))
        axs[0].set_title('Price')
        axs[0].legend()
        axs[1].plot(time1,valueQ,'-', label='alpha=' + str(alpha1))
        axs[1].set_title('Q')
        axs[1].legend()
        axs[2].plot(time1,d(time1),'-', label='alpha=' + str(alpha1))
        axs[2].set_title('Energy demand')    
        axs[2].legend()
    axs[2].grid()
    axs[1].grid()
    axs[0].grid()
    t2=time.time()
    print("execution time="+str(t2-t1))




##determinist with beta varying
def deterministic_sim_beta(alpha_1=alpha,vector=vector_beta,gamma=gamma,n=n):
    t1=time.time()
    fig, axs = plt.subplots(1,3)
    fig.suptitle('Plot with parameters beta changing, alpha='+str(alpha_1)+', F(Pt)='+str(F0)+'+'+str(F)+'*Pt')
    time1=np.linspace(0,T,n)
    for bet in vector:
        alpha1=alpha_1
        beta1=bet
        beta=bet/2
        alpha=alpha_1/2
        
        
        
        omega=np.sqrt(beta/alpha)
        omegat=omega*np.sqrt(F/(F+(1/alpha)))
        A=(omega/(omegat*(F+(1/alpha))))*integrsinhDt(T,omega,omegat,alpha,beta)
        
        
        
        B=((omega**3)/(omegat*(F+(1/alpha)))) * integrcoshsinh(T,omega,omegat,alpha,beta)
        Aprime=(omega/(F+(1/alpha)))*integrcoshDt(T,omega,omegat,alpha,beta)
        
        Bprime=((omega**3)/(F+(1/alpha)))*integrcoshcosh(T,omega,omegat,alpha,beta)
        
        numc1=gamma*Aprime+beta*A
        denomc1=alpha*omega*(gamma*omega*np.sinh(omega*T)+beta*np.cosh(omega*T)) - gamma*Bprime - beta*B
        c1=numc1/denomc1
        
        valuesG=[G(t,omega,omegat,c1,alpha,beta) for t in time1]
        
        Pricedeter=[computedeterPt(t,omega,omegat,c1,alpha,beta) for t in time1]
        
        
        #Computation of q_t
        # print("B=" + str(B))
        # print("Bprime=" + str(integrcoshcosh(T,omega,omegat)))
        # valueq=[((d(t)-F0)/(1+alpha*F)) + (alpha*F/(1+alpha*F))*((omega/alpha)*G(t,omega,omegat,c1)-c1*omega*omega*np.cosh(omega*t)) for t in time1]
        
        #Computation of Qt :
        valueQ=[Q0-((1/(1+(alpha*F)))*(integrcoshDt(t,omega,omegat,alpha,beta) + c1*omega*omega*integrcoshcosh(t,omega,omegat,alpha,beta)-c1*omega*np.sinh(omega*t)-(c1*beta*F*np.sinh(omega*t)/(omega)))) for t in time1]
        
    
        axs[0].plot(time1,Pricedeter,'-', label='beta=' + str(beta1))
        axs[0].set_title('Price')
        axs[0].legend()
        axs[1].plot(time1,valueQ,'-', label='beta=' + str(beta1))
        axs[1].set_title('Q')
        axs[1].legend()
        axs[2].plot(time1,d(time1),'-', label='beta=' + str(beta1))
        axs[2].set_title('Energy demand')    
        axs[2].legend()
    axs[2].grid()
    axs[1].grid()
    axs[0].grid()
    t2=time.time()
    print("execution time="+str(t2-t1))











###Utilisation des résultats dans l'exemple pour vérifier l'algo
"""
Xtheo=np.zeros((D,n))
for d in range(D):
    Xtheo[d,0]=np.pi/2
for i in range(1,n):
    t=(i-1)*T/n
    sumX=0
    for d in range(D):
        sumX+=np.sin(Xtheo[d,i-1])
    for d in range(D):
        dx=sigma_vol*np.exp(-r*(T-t))*sumX*(W[0,d,i]-W[0,d,i-1])
        Xtheo[d,i]=Xtheo[d,i-1]+dx
#tracé de Ytheo
Ytheo=np.zeros(n)
for i in range(n):
    t=i*T/n
    sumX=0
    for d in range(D):
        sumX+=np.sin(Xtheo[d,i])
    print(sumX)
    Ytheo[i]=np.exp(-r*(T-t))*sumX
 
    
#quadratic error
diffY=np.zeros(n)
for lam in range(Lambda):
    Xtheolam=np.zeros((D,n))
    for d in range(D):
        Xtheolam[d,0]=np.pi/2
    for i in range(1,n):
        t=(i-1)*T/n
        sumX=0
        for d in range(D):
            sumX+=np.sin(Xtheolam[d,i-1])
        for d in range(D):
            dx=sigma_vol*np.exp(-r*(T-t))*sumX*(W[lam,d,i]-W[lam,d,i-1])
            Xtheolam[d,i]=Xtheolam[d,i-1]+dx
    #tracé de Ytheo
    Ytheolam=np.zeros(n)
    l=0
    for i in range(n):
        t=i*T/n
        sumX=0
        for d in range(D):
            sumX+=np.sin(Xtheolam[d,i])
        Ytheolam[i]=np.exp(-r*(T-t))*sumX
    for i in range(n):
       diffY[i]=diffY[i]+(1/Lambda)*((Ytheolam[i]-Y[lam,0,i])**2)
 
    

fig, axs = plt.subplots(2,2)
fig.suptitle('Plot with parameters D='+str(D)+',sigma='+str(sigma_vol)+',lambda='+str(Lambda) + ',m=' + str(37))
axs[0,0].plot(time1,Y[0,0],'b-', label="simu")
axs[0,0].set_title(' Y via schema in blue and Y via decoupling/Euler in green')
axs[0,0].plot(time1,Ytheo,'-', label="simu")
#axs[0,1].set_title('Y via decoupling')
axs[1,0].plot(time1,X[0,0],'b-', label="simu")
axs[1,0].set_title(' X1[0] schema in blue and via euler in green')
axs[1,0].plot(time1,Xtheo[0],'-', label="simu")
#axs[1,1].set_title(' X1 Euler')
axs[0,1].plot(time1,diffY,'b-', label="simu")
axs[0,1].set_title(' quadratic error Ydecoupling via Euler-Yschema')
axs[1,1].plot(time1,abs(Xtheo[0]-X[0,0]),'b-', label="simu")
axs[1,1].set_title(' X1 Euler[0]-X1schema[0]')

reschema=0
restheo=0
for d in range(D):
    reschema+=np.sin(X[5,d,n-1])
    restheo+=np.sin(Xtheo[d,n-1])
"""
"""
m_max=24
Yfinal=np.zeros(m_max)
absc=[]
for m in range(m_max):
    result=main()
    ###Utilisation des résultats dans l'exemple
    Y=result[0]
    X=result[1]
    W=result[2]
    Z=result[3]
    time1=[i*T/n for i in range(n)]
    Xtheo=np.zeros((D,n))
    for d in range(D):
        Xtheo[d,0]=np.pi/2
    for i in range(1,n):
        t=i*T/n
        sumX=0
        for d in range(D):
            sumX+=np.sin(Xtheo[d,i-1])
        for d in range(D):
            dx=sigma_vol*np.exp(-r*(T-t))*sumX*(W[0,d,i]-W[0,d,i-1])
            Xtheo[d,i]=Xtheo[d,i-1]+dx
    #tracé de Ytheo
    Ytheo=np.zeros(n)
    l=0
    for i in range(n):
        t=i*T/n
        sumX=0
        for d in range(D):
            sumX+=np.sin(Xtheo[d,i-1])
        Ytheo[i]=np.exp(-r*(T-t))*sumX
    Yfinal[m]=abs(Y[0,0,0]-Ytheo[0])
    absc.append(m)
fig, axs = plt.subplots(1,2)
fig.suptitle('evolution of the error between Y0theo and Y0scheme')
axs[0].plot(absc,Yfinal,'b-', label="simu")

"""


