#!/usr/bin/env python
# coding: utf-8



# Python header (please make sure you are working with Python 3)
import numpy as np # numerical python
from numpy import pi as PI
import pylab as plt # replica of matlab's plotting tools
import scipy
from sys import getsizeof # accesses system tools
import sys
from pdb import set_trace # this stops the code and gives you interactive env.
from scipy.interpolate import pchip_interpolate

# LaTeX setup
from matplotlib import rc as matplotlibrc
from matplotlib import patches
matplotlibrc('text',usetex=False)
matplotlibrc('font', family='serif')

# compressible fluid model
Rgas = 287.0 # J/kg/K
gamma = 1.4
Cv = Rgas/(gamma-1.0) # J/kg/K
Cp = Cv*gamma # assume calorically perfect

P0   = 101000 # Pa, base quiescent air pressure
rho0  = 1.2 # kg/m^3
T0 = P0/rho0/Rgas; # Kelvin
a0 = np.sqrt(gamma*Rgas*T0) # base speed of sound, not the instantaneous/true speed

# Grid generation
Lx  = 4  # characteristic length scale
Ly  = 1

Nxc = 100 # number of cells
Nyc = 40

Nxf = Nxc+1 # number of faces = number of cells plus 1
Nyf = Nyc+1 # number of faces = number of cells plus 1
xf = np.linspace(0,Lx,Nxf) # linearly spaced grid for faces
yf = np.linspace(0,Ly,Nyf) # linearly spaced grid for faces
dx = xf[1]-xf[0]
dy = yf[1]-yf[0]

# cell-centers array in x
xc_int = 0.5*(xf[:-1]+xf[1:]) # internal cell center locations
Ng = 2 # number of ghost cells
if Ng > Nxc:
    sys.exit("Too many ghost cells!")
if Ng == 1:
    sys.exit("Need to run with at least 2 ghost cells!")
xc_ghosts_left = xc_int[:Ng]-Ng*dx
xc_ghosts_right = xc_int[-Ng:]+Ng*dx
xc = np.append(xc_ghosts_left,xc_int)
xc = np.append(xc,xc_ghosts_right) # final xc array

# cell-centers array in y
yc_int = 0.5*(yf[:-1]+yf[1:]) # internal cell center locations
if Ng > Nyc:
    sys.exit("Too many ghost cells!")
yc_ghosts_left = yc_int[:Ng]-Ng*dy
yc_ghosts_right = yc_int[-Ng:]+Ng*dy
yc = np.append(yc_ghosts_left,yc_int)
yc = np.append(yc,yc_ghosts_right) # final yc array

Xc,Yc=np.meshgrid(xc,yc) # this create two, 2D arrays with coordinate values
# these are indexed as: Xc[j,i],Yc[j,i]
# Xf,Yf=np.meshgrid(xf,yf) # this create two, 2D arrays with coordinate values

# from HWK3:
# flux_O1 = 0.0*np.array(xf) # flux array is indexed from 0-->Nxf

Frho_x  = np.zeros([Nyc,Nxf]) # Nxf = Nxc+1
Frhou_x = np.zeros([Nyc,Nxf])
Frhov_x = np.zeros([Nyc,Nxf])
FE_x    = np.zeros([Nyc,Nxf])

Frho_y  = np.zeros([Nyf,Nxc])
Frhou_y = np.zeros([Nyf,Nxc])
Frhov_y = np.zeros([Nyf,Nxc])
FE_y    = np.zeros([Nyf,Nxc])

# define initial conditions
lambda_x,lambda_y = Lx*2.0,Ly*2.0

f_xy = np.cos(2*PI/lambda_x*Xc)*np.cos(2*PI/lambda_y*Yc) # shape of the initial conditions
Amp = 0 #1e-07 # linear acoustic, gentle sound, linear advection
p_fluctuation = Amp*rho0*a0*a0*f_xy # Pa, p'(y,x,t=0)
p_init =  p_fluctuation + P0 # Pa, total initial pressure
rho_init = p_fluctuation/a0/a0 + rho0 # kg/m^3
T_init = p_init/rho_init/Rgas; # applying equation of state
u_init = 0.0*Xc # Amp*a0*f_x # m/s
v_init = 0.0*Yc # Amp*a0*f_x # m/s
sie = Cv*T_init # specific internal energy, J/kg
energy_init = rho_init*sie + 0.5*rho_init*(u_init*u_init+v_init*v_init) # J/m^3

Q_init = np.stack([rho_init,
               rho_init*u_init,
               rho_init*v_init,
               energy_init])

# # dimensions are: Q[var_index, j-y , i-x], and it includes ghost/guard cells

# working variables
rho = np.array(rho_init)
u = np.array(u_init)
v = np.array(v_init)
energy = np.array(energy_init)
Qvec = np.array(Q_init)
rhou = np.array(Qvec[1,:,:])
rhov = np.array(Qvec[2,:,:])
p = np.array(p_init)
t = 0 #seconds

#Initial conditions/parameters for wall
wall_velocity_coeff = 5
wall_position_init = 0
wall_idx_init = Ng
wall_fraction_init = 0

#define inlet BC parameters
hole_position = 0 #int(np.round(3*Nxf/8)) # Location of the opening
hole_width = int(np.round(Nxf/4)) # How many faces to open
hole_velocity = 0 # m/s

def update_ghost_cells(rho,rhou,rhov,p,u,energy,v):
    
    # NaN cell filling -- West Side
    rho[:,:Ng] = np.nan
    rhou[:,:Ng] = np.nan
    rhov[:,:Ng] = np.nan 
    p[:,:Ng] = np.nan
    u[:,:Ng] = np.nan
    energy[:,:Ng] = np.nan
    v[:,:Ng] = np.nan

    # NaN cell filling -- South Side
    rho[:Ng,:] = np.nan
    rhou[:Ng,:] = np.nan
    rhov[:Ng,:] = np.nan
    p[:Ng,:] = np.nan
    u[:Ng,:] = np.nan
    energy[:Ng,:] = np.nan
    v[:Ng,:] = np.nan
  
    # NaN cell filling -- North Side
    rho[-Ng:,:] = np.nan
    rhou[-Ng:,:] = np.nan
    rhov[-Ng:,:] = np.nan
    p[-Ng:,:] = np.nan
    u[-Ng:,:] = np.nan
    energy[-Ng:,:] = np.nan
    v[-Ng:,:] = np.nan
  
    # NaN cell filling -- East Side
    rho[:,-Ng:] = np.nan
    rhou[:,-Ng:] = np.nan
    rhov[:,-Ng:] = np.nan
    p[:,-Ng:] = np.nan
    u[:,-Ng:] = np.nan
    energy[:,-Ng:] = np.nan
    v[:,-Ng:] = np.nan
    
    return rho,rhou,rhov,p,u,energy,v
    

def impose_outlet_BC(Frho_x,Frho_y,Frhou_x,Frhou_y,Frhov_x,Frhov_y,FE_x,FE_y,rho,p,energy):
#BCs from closed box for 3 walls
    Frho_x[:,-1]=0.0 #East wall
    Frho_y[0,:],Frho_y[-1,:]=0.0,0.0 #North and South walls
    Frhou_x[:,-1] = 1.5*p[Ng:-Ng,-Ng-1]-0.5*p[Ng:-Ng,-Ng-2]
    Frhou_y[0,:],Frhou_y[-1,:] = 0.0,0.0 # bug fix!!
    Frhov_x[:,-1] = 0.0 # bug fix!!
    Frhov_y[0,:] = 1.5*p[Ng,Ng:-Ng]-0.5*p[Ng+1,Ng:-Ng] # linear extrapolation to inform momentum flux condition at the wall
    Frhov_y[-1,:] = 1.5*p[-Ng-1,Ng:-Ng]-0.5*p[-Ng-2,Ng:-Ng]
    FE_x[:,-1] = 0.0 #East wall
    FE_y[0,:],FE_y[-1,:], = 0.0,0.0
    
    # BCs from the closed box
    #Frho_x[:,0],Frho_x[:,-1]=0.0,0.0
    #Frho_y[0,:],Frho_y[-1,:]=0.0,0.0
    #Frhou_x[:,0] = 1.5*p[Ng:-Ng,Ng]-0.5*p[Ng:-Ng,Ng+1]
    #Frhou_x[:,-1] = 1.5*p[Ng:-Ng,-Ng-1]-0.5*p[Ng:-Ng,-Ng-2]
    #Frhou_y[0,:],Frhou_y[-1,:] = 0.0,0.0 # bug fix!!
    #Frhov_x[:,0],Frhov_x[:,-1] = 0.0,0.0 # bug fix!!
    #Frhov_y[0,:] = 1.5*p[Ng,Ng:-Ng]-0.5*p[Ng+1,Ng:-Ng] # linear extrapolation to inform momentum flux condition at the wall
    #Frhov_y[-1,:] = 1.5*p[-Ng-1,Ng:-Ng]-0.5*p[-Ng-2,Ng:-Ng]
    #FE_x[:,0],FE_x[:,-1], = 0.0,0.0
    #FE_y[0,:],FE_y[-1,:], = 0.0,0.0
    
    # Actual outlet BC
    
    #THIS IS FOR A VERTICAL HOLE FLUX 
    # Calculate all extrapolated variables at the wall with the outlet
    #energy_extrapolated = 1.5*energy[Ng,Ng:-Ng]-0.5*energy[Ng+1,Ng:-Ng]
    #p_extrapolated = 1.5*p[Ng,Ng:-Ng]-0.5*p[Ng+1,Ng:-Ng]
    #rho_extrapolated = 1.5*rho[Ng,Ng:-Ng]-0.5*rho[Ng+1,Ng:-Ng] 
    #rhou_extrapolated = 1.5*rhou[Ng,Ng:-Ng]-0.5*rhou[Ng+1,Ng:-Ng]
    #tmpy_extrapolated = hole_velocity*(energy_extrapolated+p_extrapolated) #change this so it's hole velocity squared not hole velocity times inside velocity 
    
    # Calculate fluxes at the hole using target velocity and extrapolated variables
    #Frho_y[0,hole_position:hole_position+hole_width] = hole_velocity * rho_extrapolated[hole_position:hole_position+hole_width]
    #Frhou_y[0,hole_position:hole_position+hole_width] = hole_velocity*rhou_extrapolated[hole_position:hole_position+hole_width]
    #Frhov_y[0,hole_position:hole_position+hole_width] = (hole_velocity*hole_velocity* \
    #                rho_extrapolated[hole_position:hole_position+hole_width])\
    #                +p_extrapolated[hole_position:hole_position+hole_width]
    #FE_y[0,hole_position:hole_position+hole_width] = hole_velocity*\
    #        (energy_extrapolated[hole_position:hole_position+hole_width]+\
    #         p_extrapolated[hole_position:hole_position+hole_width])
    
    return(Frho_x,Frho_y,Frhou_x,Frhou_y,Frhov_x,Frhov_y,FE_x,FE_y)

def impose_closed_wall_BC(Frho_x,Frho_y,Frhou_x,Frhou_y,Frhov_x,Frhov_y,FE_x,FE_y,rho,p,energy,wall_idx):
    Frho_x[:,0],Frho_x[:,-1]=0.0,0.0
    Frho_y[0,:],Frho_y[-1,:]=0.0,0.0
    Frhou_x[:,0] = 1.5*p[Ng:-Ng,Ng]-0.5*p[Ng:-Ng,Ng+1]
    Frhou_x[:,-1] = 1.5*p[Ng:-Ng,-Ng-1]-0.5*p[Ng:-Ng,-Ng-2]
    Frhou_y[0,:],Frhou_y[-1,:] = 0.0,0.0 # bug fix!!
    Frhov_x[:,0],Frhov_x[:,-1] = 0.0,0.0 # bug fix!!
    Frhov_y[0,:] = 1.5*p[Ng,Ng:-Ng]-0.5*p[Ng+1,Ng:-Ng] # linear extrapolation to inform momentum flux condition at the wall
    Frhov_y[-1,:] = 1.5*p[-Ng-1,Ng:-Ng]-0.5*p[-Ng-2,Ng:-Ng]
    FE_x[:,0],FE_x[:,-1], = 0.0,0.0
    FE_y[0,:],FE_y[-1,:], = 0.0,0.0
    return(Frho_x,Frho_y,Frhou_x,Frhou_y,Frhov_x,Frhov_y,FE_x,FE_y)

# NEW VERSION OF WALL FUNCTION
# The weighted average is just for velocity
# Pressure is extrapolated 
# Density is based on the volumetric change per time step
def impose_moving_wall_BC(t,rho,rhou,rhov,p,u,energy,v):
    # Find where the wall is
    v_wall = 0 
    u_wall = wall_velocity_coeff # linear velocity profile
    wall_position = wall_velocity_coeff*t # x-coordinate of wall
    wall_idx = np.min(np.where(xc+(dx/2)-wall_position>0)[0]) #the cell the wall is in

    # Find the fraction of the cell taken up by the wall
    wall_fraction = (wall_position - (xc[wall_idx] - (dx/2)))/dx
    
    # Find volumetric change in a time step
    volumetric_change = u_wall * dt / dx 
    
    # Calculate primitives (velocity and rho)
    u[Ng:-Ng,wall_idx] = ((1-wall_fraction)*u[Ng:-Ng,wall_idx]) + (wall_fraction*u_wall)
    v[Ng:-Ng,wall_idx] = ((1-wall_fraction)*v[Ng:-Ng,wall_idx]) + (wall_fraction*v_wall)
    rho[Ng:-Ng,wall_idx] = rho[Ng:-Ng,wall_idx] / (1-(wall_fraction*volumetric_change))
    #find a derivation for this ^^^^ DONE

    #Calculate pressure based on density changes 
    p[Ng:-Ng,wall_idx] = p[Ng:-Ng,wall_idx+1] #simple extrapolation
    
    #momentum 
    rhou[Ng:-Ng,wall_idx] = u[Ng:-Ng,wall_idx] * rho[Ng:-Ng,wall_idx]
    rhov[Ng:-Ng,wall_idx] = v[Ng:-Ng,wall_idx] * rho[Ng:-Ng,wall_idx]
    
    #Energy
    energy[Ng:-Ng,wall_idx] = rho[Ng:-Ng,wall_idx]*sie[Ng:-Ng,wall_idx] + \
            0.5*rho[Ng:-Ng,wall_idx]*(u[Ng:-Ng,wall_idx]*u[Ng:-Ng,wall_idx]+v[Ng:-Ng,wall_idx]*v[Ng:-Ng,wall_idx])
    
    # Mass flux BC to conserve mass
    #mass_flux = rho[Ng:-Ng,wall_idx]*u_wall*Ly
    #volume = Ly*dx
    #rho[Ng:-Ng,wall_idx+1] += mass_flux * dt / volume

    return(rho,rhou,rhov,p,u,energy,v,wall_position, wall_idx, wall_fraction)


#Nt = 10000
fig_skip = 100
CFL= 0.05
dt = CFL*dx/a0
integration_time = 0.2 #seconds
Nt = int(np.round(integration_time / dt))
figure_counter = 0

# Lax-Friedrichs Artificial Viscosity
alpha_art_x = 0.5*(dx**2 / (2*dt)) #Lax-Friedrichs times a scalar 
alpha_art_y = 0 #(dy**2 / (2*dt)) #Lax-Friedrichs

# Variable trackers 
p_tracker = np.zeros([Nt])
t_tracker = np.zeros([Nt])
u_tracker = np.zeros([Nt])
energy_tracker = np.zeros([Nt])
rho_tracker = np.zeros([Nt])
wall_position_array = np.zeros([Nt])
wall_idx_array = np.zeros([Nt])
wall_fraction_array = np.zeros([Nt])
wall_position_array_star = np.zeros([Nt])
wall_idx_array_star = np.zeros([Nt])
wall_fraction_array_star = np.zeros([Nt])

mass_init = np.sum(rho[Ng:-Ng,Ng:-Ng])*dx*dy
total_mass = np.zeros([Nt])
total_mass_new = np.zeros([Nt])

for it in range(0,Nt):

    #Solution 1 for RK2---------------------------------------------------------------------------------------------------
    ## populating ghost cells based on boundary conditions

    rho_star,rhou_star,rhov_star,p_star,u_star,energy_star,v_star = update_ghost_cells(rho,rhou,rhov,p,u,energy,v) #just does the NaNs
    
    rho_star,rhou_star,rhov_star,p_star,u_star,energy_star,v_star,wall_position_array_star[it],wall_idx_array_star[it],wall_fraction_array_star[it] = \
            impose_moving_wall_BC(t,rho_star,rhou_star,rhov_star,p_star,u_star,energy_star,v_star)

    # # continuity equation - rho - mass equation
    Frho_x_star = 0.5*(rhou_star[Ng:-Ng,Ng-1:-Ng]+rhou_star[Ng:-Ng,Ng:-Ng+1]) + \
              -alpha_art_x* (rho_star[Ng:-Ng,Ng:-Ng+1] - rho_star[Ng:-Ng,Ng-1:-Ng])/dx 
    Frho_y_star = 0.5*(rhov_star[Ng-1:-Ng,Ng:-Ng]+rhov_star[Ng:-Ng+1,Ng:-Ng]) 

    # u-momentum
    Frhou_x_star = Frho_x_star*0.5*(u_star[Ng:-Ng,Ng-1:-Ng]+u_star[Ng:-Ng,Ng:-Ng+1]) \
              + 0.5*(p_star[Ng:-Ng,Ng-1:-Ng]+p_star[Ng:-Ng,Ng:-Ng+1]) + \
              - alpha_art_x* (rhou_star[Ng:-Ng,Ng:-Ng+1] - rhou_star[Ng:-Ng,Ng-1:-Ng])/dx  
    Frhou_y_star = Frho_y_star*0.5*(u_star[Ng-1:-Ng,Ng:-Ng]+u_star[Ng:-Ng+1,Ng:-Ng])

    # v-momentum
    Frhov_x_star = Frho_x_star*0.5*(v_star[Ng:-Ng,Ng-1:-Ng]+v_star[Ng:-Ng,Ng:-Ng+1]) + \
              - alpha_art_x* (rhov_star[Ng:-Ng,Ng:-Ng+1] - rhov_star[Ng:-Ng,Ng-1:-Ng])/dx  
    Frhov_y_star = Frho_y_star*0.5*(v_star[Ng-1:-Ng,Ng:-Ng]+v_star[Ng:-Ng+1,Ng:-Ng]) \
              + 0.5*(p_star[Ng-1:-Ng,Ng:-Ng]+p_star[Ng:-Ng+1,Ng:-Ng])
    
    # p[Ng,-Ng-1]
    # energy
    tmpx_star,tmpy_star = u_star*(energy_star+p_star),v_star*(energy_star+p_star)
    FE_x_star = 0.5*(tmpx_star[Ng:-Ng,Ng-1:-Ng]+tmpx_star[Ng:-Ng,Ng:-Ng+1]) + \
              - alpha_art_x* (energy_star[Ng:-Ng,Ng:-Ng+1] - energy_star[Ng:-Ng,Ng-1:-Ng])/dx
    FE_y_star = 0.5*(tmpy_star[Ng-1:-Ng,Ng:-Ng]+tmpy_star[Ng:-Ng+1,Ng:-Ng])

    # All BCs at once: 
    Frho_x_star,Frho_y_star,Frhou_x_star,Frhou_y_star,Frhov_x_star,Frhov_y_star,FE_x_star,FE_y_star = \
                impose_closed_wall_BC(Frho_x_star,Frho_y_star,Frhou_x_star,Frhou_y_star,Frhov_x_star, \
                Frhov_y_star,FE_x_star,FE_y_star,rho_star,p_star,energy_star,wall_idx_array_star[it])
    
    #Cell variables
    #Change indices of where you update fluxes and variables 
    #y indices stay the same, x indices change
    wall_idx = int(wall_idx_array_star[it])
    wall_face = int(wall_idx-Ng)
    rho_star[Ng:-Ng,wall_idx:-Ng] = rho[Ng:-Ng,wall_idx:-Ng] \
                - dt*(Frho_x_star[:,wall_face+1:]-Frho_x_star[:,wall_face:-1])/dx \
                - dt*(Frho_y_star[1:,wall_face:]-Frho_y_star[:-1,wall_face:])/dy

    rhou_star[Ng:-Ng,wall_idx:-Ng] = rhou[Ng:-Ng,wall_idx:-Ng] \
                - dt*(Frhou_x_star[:,wall_face+1:]-Frhou_x_star[:,wall_face:-1])/dx \
                - dt*(Frhou_y_star[1:,wall_face:]-Frhou_y_star[:-1,wall_face:])/dy

    rhov_star[Ng:-Ng,wall_idx:-Ng] = rhov[Ng:-Ng,wall_idx:-Ng] \
                - dt*(Frhov_x_star[:,wall_face+1:]-Frhov_x_star[:,wall_face:-1])/dx \
                - dt*(Frhov_y_star[1:,wall_face:]-Frhov_y_star[:-1,wall_face:])/dy

    energy_star[Ng:-Ng,wall_idx:-Ng] = energy[Ng:-Ng,wall_idx:-Ng] \
                - dt*(FE_x_star[:,wall_face+1:]-FE_x_star[:,wall_face:-1])/dx \
                - dt*(FE_y_star[1:,wall_face:]-FE_y_star[:-1,wall_face:])/dy

    u_star = rhou_star/rho_star
    v_star = rhov_star/rho_star
    kinetic_energy_star = 0.5*(rhou_star*rhou_star+rhov_star*rhov_star)/rho_star
    p_star = (energy_star-kinetic_energy_star)*(gamma-1.0) #bug fix, multiply don't divide 
    
    #Solution 2 for RK2---------------------------------------------------------------------------------------------------
    ## populating ghost cells based on boundary conditions

    rho,rhou,rhov,p,u,energy,v = update_ghost_cells(rho_star,rhou_star,rhov_star,p_star,u_star,energy_star,v_star)
    
    rho,rhou,rhov,p,u,energy,v,wall_position_array[it],wall_idx_array[it],wall_fraction_array[it] = \
            impose_moving_wall_BC(t,rho,rhou,rhov,p,u,energy,v)

    # # continuity equation - rho - mass equation
    Frho_x = 0.5*(rhou[Ng:-Ng,Ng-1:-Ng]+rhou[Ng:-Ng,Ng:-Ng+1]) + \
              -alpha_art_x* (rho[Ng:-Ng,Ng:-Ng+1] - rho[Ng:-Ng,Ng-1:-Ng])/dx 
    Frho_y = 0.5*(rhov[Ng-1:-Ng,Ng:-Ng]+rhov[Ng:-Ng+1,Ng:-Ng])
    
    # u-momentum
    Frhou_x = Frho_x*0.5*(u[Ng:-Ng,Ng-1:-Ng]+u[Ng:-Ng,Ng:-Ng+1]) \
              + 0.5*(p[Ng:-Ng,Ng-1:-Ng]+p[Ng:-Ng,Ng:-Ng+1]) + \
              - alpha_art_x* (rhou[Ng:-Ng,Ng:-Ng+1] - rhou[Ng:-Ng,Ng-1:-Ng])/dx  
    Frhou_y = Frho_y*0.5*(u[Ng-1:-Ng,Ng:-Ng]+u[Ng:-Ng+1,Ng:-Ng])
    
    # v-momentum
    Frhov_x = Frho_x*0.5*(v[Ng:-Ng,Ng-1:-Ng]+v[Ng:-Ng,Ng:-Ng+1]) + \
              - alpha_art_x* (rhov[Ng:-Ng,Ng:-Ng+1] - rhov[Ng:-Ng,Ng-1:-Ng])/dx  
    Frhov_y = Frho_y*0.5*(v[Ng-1:-Ng,Ng:-Ng]+v[Ng:-Ng+1,Ng:-Ng]) \
              + 0.5*(p[Ng-1:-Ng,Ng:-Ng]+p[Ng:-Ng+1,Ng:-Ng])
    
    # p[Ng,-Ng-1]
    # energy
    tmpx,tmpy = u*(energy+p),v*(energy+p)
    FE_x = 0.5*(tmpx[Ng:-Ng,Ng-1:-Ng]+tmpx[Ng:-Ng,Ng:-Ng+1]) + \
              - alpha_art_x* (energy[Ng:-Ng,Ng:-Ng+1] - energy[Ng:-Ng,Ng-1:-Ng])/dx
    FE_y = 0.5*(tmpy[Ng-1:-Ng,Ng:-Ng]+tmpy[Ng:-Ng+1,Ng:-Ng])

    # All BCs at once:
    Frho_x,Frho_y,Frhou_x,Frhou_y,Frhov_x,Frhov_y,FE_x,FE_y = \
                impose_closed_wall_BC(Frho_x,Frho_y,Frhou_x,Frhou_y,Frhov_x,Frhov_y,FE_x,FE_y,rho,p,energy,wall_idx_array[it])
    
    #Cell variables
    #Change indices of where you update fluxes and variables 
    #y indices stay the same, x indices change
    wall_idx = int(wall_idx_array[it])
    wall_face = int(wall_idx-Ng)
    total_mass[it] = np.sum(rho[Ng:-Ng,wall_idx:-Ng])*dx*dy
    rho[Ng:-Ng,wall_idx:-Ng] = rho[Ng:-Ng,wall_idx:-Ng] + 0.5*(\
                - dt*(Frho_x[:,wall_face+1:]-Frho_x[:,wall_face:-1])/dx \
                - dt*(Frho_y[1:,wall_face:]-Frho_y[:-1,wall_face:])/dy
                + dt*(Frho_x_star[:,wall_face+1:]-Frho_x_star[:,wall_face:-1])/dx \
                + dt*(Frho_y_star[1:,wall_face:]-Frho_y_star[:-1,wall_face:])/dy ) 
    
    rhou[Ng:-Ng,wall_idx:-Ng] = rhou[Ng:-Ng,wall_idx:-Ng] + 0.5*( \
                - dt*(Frhou_x[:,wall_face+1:]-Frhou_x[:,wall_face:-1])/dx \
                - dt*(Frhou_y[1:,wall_face:]-Frhou_y[:-1,wall_face:])/dy
                + dt*(Frhou_x_star[:,wall_face+1:]-Frhou_x_star[:,wall_face:-1])/dx \
                + dt*(Frhou_y_star[1:,wall_face:]-Frhou_y_star[:-1,wall_face:])/dy )

    rhov[Ng:-Ng,wall_idx:-Ng] = rhov[Ng:-Ng,wall_idx:-Ng] + 0.5*( \
                - dt*(Frhov_x[:,wall_face+1:]-Frhov_x[:,wall_face:-1])/dx \
                - dt*(Frhov_y[1:,wall_face:]-Frhov_y[:-1,wall_face:])/dy
                + dt*(Frhov_x_star[:,wall_face+1:]-Frhov_x_star[:,wall_face:-1])/dx \
                + dt*(Frhov_y_star[1:,wall_face:]-Frhov_y_star[:-1,wall_face:])/dy )

    energy[Ng:-Ng,wall_idx:-Ng] = energy[Ng:-Ng,wall_idx:-Ng] + 0.5*( \
                - dt*(FE_x[:,wall_face+1:]-FE_x[:,wall_face:-1])/dx \
                - dt*(FE_y[1:,wall_face:]-FE_y[:-1,wall_face:])/dy
                + dt*(FE_x_star[:,wall_face+1:]-FE_x_star[:,wall_face:-1])/dx \
                + dt*(FE_y_star[1:,wall_face:]-FE_y_star[:-1,wall_face:])/dy )
    total_mass_new[it] = np.sum(rho[Ng:-Ng,wall_idx:-Ng])*dx*dy

    u = rhou/rho
    v = rhov/rho
    kinetic_energy = 0.5*(rhou*rhou+rhov*rhov)/rho
    p = (energy-kinetic_energy)*(gamma-1.0) #bug fix, multiply don't divide 
    
    u_tracker[it] = u[wall_idx,wall_idx]
    t_tracker[it] = it*dt
    p_tracker[it] = p[wall_idx,wall_idx]
    rho_tracker[it] = rho[wall_idx,wall_idx]
    energy_tracker[it] = energy[wall_idx,wall_idx]
    
    #TRACK CONSERVATION OF MASS HERE
    #total_mass[it] = np.sum(rho[Ng:-Ng,wall_idx:-Ng])*dx*dy
    
    if not(it % fig_skip):
        
        fig = plt.figure(figsize=(20,20))
        ax  = fig.add_subplot(4,1,1)
        ax.quiver(Xc[Ng:-Ng:2,Ng:-Ng:2],Yc[Ng:-Ng:2,Ng:-Ng:2],u[Ng:-Ng:2,Ng:-Ng:2]/a0,v[Ng:-Ng:2,Ng:-Ng:2]/a0,\
                  scale=np.abs(wall_velocity_coeff/10) ,pivot='tip')
                  #scale=np.abs(hole_velocity/10) ,pivot='tip')
        ax.axvline(x=wall_position_array[it], ymin=0, ymax=1,color='m')
        plt.xlim([0,4])
        plt.ylim([0,1])
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        ax.set_title("Velocity Glyphs")
        ax.set_aspect('equal', 'box')
        #ax.axis('equal')

        ax  = fig.add_subplot(4,1,2)
        plt.axes(ax)
        plt.streamplot(Xc[Ng:-Ng,Ng:-Ng],Yc[Ng:-Ng,Ng:-Ng],u[Ng:-Ng,Ng:-Ng]/a0,v[Ng:-Ng,Ng:-Ng]/a0, density = 0.5,broken_streamlines=False)
        ax.axvline(x=wall_position_array[it], ymin=0, ymax=1,color='m')
        plt.xlim([0,4])
        plt.ylim([0,1])
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        ax.set_title("Velocity Streamlines")
        ax.set_aspect('equal', 'box')
        #ax.axis('equal')

        ax  = fig.add_subplot(4,1,3)
        plt.axes(ax)
        plt.contourf(Xc[Ng:-Ng,Ng:-Ng],Yc[Ng:-Ng,Ng:-Ng],p[Ng:-Ng,Ng:-Ng]) #/rho0/a0/a0)
        ax.axvline(x=wall_position_array[it], ymin=0, ymax=1,color='m')
        ax.set_xlim(0,0.2)
        ax.set_ylim(0,Ly)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        ax.set_title("pressure")
        plt.xlim([0,4])
        plt.ylim([0,1])
        ax.set_aspect('equal', 'box')
        #ax.axis('equal')

        ax  = fig.add_subplot(4,1,4)
        plt.axes(ax)
        plt.contourf(Xc[Ng:-Ng,Ng:-Ng],Yc[Ng:-Ng,Ng:-Ng],u[Ng:-Ng,Ng:-Ng])
        ax.axvline(x=wall_position_array[it], ymin=0, ymax=1,color='m')
        ax.set_xlim(0,Lx)
        ax.set_ylim(0,Ly)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        ax.set_title("U-Velocity")
        plt.xlim([0,4])
        plt.ylim([0,1])
        ax.set_aspect('equal', 'box')
        #ax.axis('equal')
        plt.tight_layout()

        plt.suptitle("t = "+repr(it*dt)+", CFL = "+repr(CFL), fontsize=40.0) #'xx-large')
        
        figure_folder = "./figures/LF2_Wall_Physics/"
        figure_name = "initial_condition_"+repr(it)+".png"
        figure_file_path = figure_folder + figure_name
        print("Saving figure: " + figure_file_path)
        plt.tight_layout()
        #plt.savefig(figure_file_path)
        figure_counter += 1
        plt.close()
        
    t += dt

#Plotting wall position metrics
fig, ax1 = plt.subplots()
plt.suptitle("Wall Index and Fraction") #'xx-large')
color = 'tab:red'
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel("Wall Index", color=color)
ax1.scatter(t_tracker, wall_idx_array, color=color, s=0.2)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Wall Fraction', color=color)  # we already handled the x-label with ax1
ax2.scatter(t_tracker, wall_fraction_array, color=color, s=0.2)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

plt.figure()
plt.plot(t_tracker,rho_tracker)
plt.title("Density Probe at Cell (10,10), CFL = "+str(CFL))
plt.xlabel("Time [sec]")
plt.ylabel("Density [kg/m^3]")
#figure_folder = "./figures/"
#figure_name = "HW4_PT1_RK1_SmallAmp.pdf"
#figure_file_path = figure_folder + figure_name
#print("Saving figure: " + figure_file_path)
plt.tight_layout()
#plt.savefig(figure_file_path)
#plt.close()

