# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 1:36:19 2022
2D diffusion equation
implicit method with hermite rule- 4 order
@author: 17564
"""
from method2 import ADI_hermite_rule
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
def func(x,y):
    return np.sin(x)*np.sin(y)
def Eqi_BC(t):
    return 0
def standard_func(x,y,t):
    return np.exp(-t)*np.sin(x)*np.sin(y)
def anination_draw(frame_ID,data,draw2D_err):
    draw2D_err[0].remove()
    draw2D_err[0]=ax.plot_surface(X,Y,np.array(abs(data[:,:,frame_ID*100]-standard_func(X,Y,frame_ID*100*dt))),cmap=plt.get_cmap('coolwarm'),vmax=2e-5)

def ini_axis():
    ax.set_xlim(0,np.pi)
    ax.set_ylim(0,np.pi)
    ax.set_zlim(0,2e-5)
    ax.set_xlabel(r"x axis")
    ax.set_ylabel(r"t axis")
    ax.set_zlabel(r"error axis")



#initial 
a=1/2
xmax=np.pi
xmin=0
ymax=np.pi
ymin=0
tmax=2
tmin=0
Nt=40000
Nx=80 #The fraction Nx**2/Nt and Ny**2/Nt should not be too large
Ny=80
dt=(tmax-tmin)/Nt
hx=(xmax-xmin)/Nx
hy=(ymax-ymin)/Ny
x=np.linspace(xmin,xmax,Nx+1)
y=np.linspace(ymin,ymax,Nx+1)

X,Y=np.meshgrid(x,y)
data=np.empty((Ny+1,Nx+1,Nt))
data[:,:,0]=func(X,Y)
# data=ADI_hermite_rule(a,hx,hy,Nx,Ny,Nt,dt,data,Eqi_BC)
data=np.load('2D diffusion equation data.npz')['arr_0']
# sx=int(Nx/4)
# sy=int(Ny/4)
# max_error=[(abs(data[sx,sy,i]-standard_func(x[sx],y[sy],i*dt))) for i in range(Nt)]
max_error=[np.max(abs(data[:,:,i]-standard_func(X,Y,i*dt))) for i in range(Nt)]
plt.plot(max_error)
plt.title("error at each time step")
plt.show()

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111,projection='3d')
ax.set_xlim(0,np.pi)
ax.set_ylim(0,np.pi)
ax.set_zlim(0,2e-5)
ax.set_xlabel(r"x axis")
ax.set_ylabel(r"t axis")
ax.set_zlabel(r"error axis")
draw2D_err=[ax.plot_surface(X,Y,np.array(abs(data[:,:,0]-standard_func(X,Y,0*dt))),cmap=plt.get_cmap('coolwarm'),vmax=2e-5)]
fig.colorbar(draw2D_err[0],shrink=0.5, aspect=15)
# draw2D_err=plt.pcolor(x,y,(abs(data[:,:,0]-standard_func(X,Y,0))),cmap='coolwarm',vmax=1e-5)


ani = animation.FuncAnimation(
    fig=fig,
    func=anination_draw,
    fargs=(data, draw2D_err),
    frames=np.arange(int(Nt/100)),    
    # init_func=ini_axis,
    interval=5,   
    repeat=True,
)
ani.save('2D.gif', dpi=300, writer='Pillow')

np.savez("2D diffusion equation data.npz",data)


