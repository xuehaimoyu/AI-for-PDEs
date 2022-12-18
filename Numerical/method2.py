# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 01:23:53 2022
part of numerical analysis
@author: 17564
"""
import numpy as np
from scipy.sparse.linalg import spsolve
def grid_point(Nx):
    #DifferenceOrder -> Pseudospectral
    
    return np.array([np.cos(i*np.pi/Nx) for i in range(Nx+1)])


def ADI_hermite_rule(a,hx,hy,Nx,Ny,Nt,dt,data,Eqi_BC):
    #only for 2D diffusion equation
    #initial identical matrix
    Ix=np.eye(Nx-1)
    Iy=np.eye(Ny-1)
    
    #initial linear operator
    Lx=(np.eye(Nx-1,k=1)+np.eye(Nx-1,k=-1)-2*np.eye(Nx-1))/hx**2*a
    Ly=(np.eye(Ny-1,k=1)+np.eye(Ny-1,k=-1)-2*np.eye(Ny-1))/hy**2*a
    
    Mx2=1/2*dt*Ly+1/12*dt**2*np.linalg.matrix_power(Lx,2)
    Mx1=Ix-1/2*dt*Lx+1/12*dt**2*np.linalg.matrix_power(Lx,2)
    
    My2=1/2*dt*Ly+1/12*dt**2*np.linalg.matrix_power(Ly,2)
    My1=Iy-1/2*dt*Ly+1/12*dt**2*np.linalg.matrix_power(Ly,2)
    
    
    for i in range(1,Nt):
        U=data[1:-1,1:-1,i-1]
        temp=np.empty((Ny-1,Nx-1))
        result=np.empty((Ny-1,Nx-1))
        
        for j in range(Ny-1):
            b=U[j,:]+np.dot(U,My2[j,:])
            temp[j,:]=spsolve(Mx1,b)
                    
        
        for k in range(Nx-1):
            b=np.transpose(temp[:,k])+np.dot(np.transpose(temp),Mx2[k,:])
            result[:,k]=np.transpose(spsolve(My1,b))
        
        #boundary condition
        data[0,:,i]=Eqi_BC(i)
        data[-1,:,i]=Eqi_BC(i)
        data[:,0,i]=Eqi_BC(i)
        data[:,-1,i]=Eqi_BC(i)
        
        data[1:-1,1:-1,i]=result
    
    return data


def calculate_operator(Nx,x):
    L=np.empty((Nx+1,Nx+1))
    for i in range(Nx+1):
        for j in range(Nx+1):
            if i==0 or i==Nx:
                ci=2
            else :
                ci=1
                
            if j==0 or j==Nx:
                cj=2
            else :
                cj=1

            
            if i != j :
                L[i,j]=(-1)**(i+j)/(x[i]-x[j])*ci/cj
            
            if i==j:
                if  i==0:
                    L[i,j]= (2*(Nx)**2+1)/6
                elif i==Nx:
                    L[i,j]=-(2*(Nx)**2+1)/6
                else:
                    L[i,j]=-x[i]/2/(1-x[i]**2)
    return L
    
def calculate_operator2(Nx,h,a):
    return (np.eye(Nx-1,k=1)+np.eye(Nx-1,k=-1)-2*np.eye(Nx-1))/h**2*a

def Runge_Kutta_Method(a,x,xmax,xmin,Nx,Nt,dt,data,ydata,Eqi_BC):
    #calculate operator
    h=x[0]-x[1]
    L=calculate_operator2(Nx,h,a)
    for i in range(1,Nt):

        # K1=Y
        # L1=a**2*np.dot(L,U)
        
        # K2=Y+1/2*dt*L1
        # L2=a**2*np.dot(L,U+1/2*dt*K1)
        
        # K3=Y+1/2*dt*L2
        # L3=a**2*np.dot(L,U+1/2*dt*K2)
        
        # K4=Y+dt*L3
        # L4=a**2*np.dot(L,U+dt*K3)

        # result=U+1/6*(K1+2*K2+2*K3+K4)*dt
        
        # resulty=Y+1/6*(L1+2*L2+2*L3+L4)*dt
        if i==1:
            for j in range(1,Nx):
                data[j,1]=data[j,0]+1*dt+1/2*(a*dt/h)**2*(data[j+1,0]-2*data[j,0]+data[j-1,0])
        # ydata[1:-1,i]=resulty
            data[0,1]=data[0,0]+1*dt+(a*dt/h)**2*(data[1,0]-data[0,0])
            data[-1,1]=data[-1,0]+1*dt+(a*dt/h)**2*(data[-2,0]-data[-1,0])
        else:
            for j in range(1,Nx):
                data[j,i]=2*data[j,i-1]-data[j,i-2]+(a*dt/h)**2*(data[j+1,i-1]-2*data[j,i-1]+data[j-1,i-1])
            data[0,i]=2*data[0,i-1]-data[0,i-2]+2*(a*dt/h)**2*(data[1,i-1]-data[0,i-1])
            data[-1,i]=2*data[-1,i-1]-data[-1,i-2]+2*(a*dt/h)**2*(data[-2,i-1]-data[-1,i-1])
        
        data[0,:]=Eqi_BC(i)

    
    
    return data,L
    

def Runge_Kutta_nonlinearnode_Method(a,x,xmax,xmin,Nx,Nt,dt,data,ydata,Eqi_BC):
    #calculate operator
    L=np.linalg.matrix_power(calculate_operator(Nx,x),2)
    L=L/((xmax-xmin)**2/4)
    h=x[0]-x[1]
    for i in range(1,Nt):
        
        result=np.empty((Nx-1))
        resulty=np.empty((Nx-1))

        U=data[1:-1,i-1]
        Y=ydata[1:-1,i-1]
        
        
        K1=Y
        L1=a**2*np.dot(L,U)
        
        K2=Y+1/2*dt*L1
        L2=a**2*np.dot(L,U+1/2*dt*K1)
        
        K3=Y+1/2*dt*L2
        L3=a**2*np.dot(L,U+1/2*dt*K2)
        
        K4=Y+dt*L3
        L4=a**2*np.dot(L,U+dt*K3)

        result=U+1/6*(K1+2*K2+2*K3+K4)*dt
        
        resulty=Y+1/6*(L1+2*L2+2*L3+L4)*dt
    
        #boundary condition
        
        
        data[1:-1,i]=result
        ydata[1:-1,i]=resulty
        if i==1:
            data[0,1]=data[0,0]+0+(a*dt/h)**2*(data[1,0]-data[0,0])
            data[-1,1]=data[-1,0]+0+(a*dt/h)**2*(data[-2,0]-data[-1,0])
        else:
            data[0,i]=2*data[0,i-1]-data[0,i-2]+2*(a*dt/h)**2*(data[1,i-1]-data[0,i-1])
            data[-1,i]=2*data[-1,i-1]-data[-1,i-2]+2*(a*dt/h)**2*(data[-2,i-1]-data[-1,i-1])
        
        data[0,:]=Eqi_BC(i)

    
    
    return data,L

















