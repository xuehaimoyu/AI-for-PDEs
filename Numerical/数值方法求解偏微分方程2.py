# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 02:26:11 2022
1D vibrating string equation
Runge-Kutta Method  - 4 order 
@author: 17564
"""
from method2 import Normal_method,Runge_Kutta_nonlinearnode_Method,grid_point
import numpy as np
from matplotlib import pyplot as plt

def fun(x):
    return 0

def yfun(x):
    return np.ones(len(x))

def Eqi_BC(t):
    return 0

def standard_func(x,t,l,v0,a):
    sum_=0
    for n in range(10000):
        temp=8*l*v0/np.pi**2/a*1/(2*n+1)**2*np.sin((2*n+1)*np.pi*a*t/2/l)*np.sin((2*n+1)*np.pi*x/2/l)
        sum_+=temp

    return sum_
                                                                                 
                                                                                 
#initial
xmax=1
xmin=0
tmax=1
tmin=0
Nt=100000 #must >= 2
Nx=2**9
dt=(tmax-tmin)/(Nt)
x=grid_point(Nx)
# x=np.linspace(xmin,xmax,Nx+1)
# t=np.linspace(tmin,tmax,Nt)
Nt=100000

a=1
l=1
v0=1

ydata=np.empty((Nx+1,Nt))
data=np.empty((Nx+1,Nt))

ydata[:,0]=yfun(x)
data[:,0]=fun(x)


# data=Normal_method(a,x,xmax,xmin,Nx,Nt,dt,data,ydata,Eqi_BC)
data=Runge_Kutta_nonlinearnode_Method(a,x,xmax,xmin,Nx,Nt,dt,data,ydata,Eqi_BC)
# data=np.load("1D vibrating string equation data.npz")['arr_0']
x=(xmax+xmin)/2+x*(xmax-xmin)/2

num=int(Nt/100)
s=int(Nx/4)
# max_error=[abs(data[s,num*i]-standard_func(x[s],num*i*dt,l,v0,a)) for i in range(100)]
max_error=[np.max(abs(data[:,num*i]-standard_func(x,num*i*dt,l,v0,a))) for i in range(100)]

plt.plot(max_error)
plt.title("error at each time step")
plt.show()

error=np.array([abs(data[:,num*i]-standard_func(x,num*i*dt,l,v0,a)) for i in range(100)])
t=[num*i*dt for i in range(100)]
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111,projection='3d')
x_,t_=np.meshgrid(x,t)
ax.set_xlabel(r"x axis")
ax.set_ylabel(r"t axis")
ax.set_zlabel(r"error axis")
draw2D_err=ax.plot_surface(x_,t_,error,cmap=plt.get_cmap('coolwarm'),vmax=5e-5)
fig.colorbar(draw2D_err,shrink=0.5, aspect=15)
plt.savefig("1Dresult.png",dpi=300,format="png")

np.savez("1D vibrating string equation data.npz",data)






