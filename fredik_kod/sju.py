import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
os.system('cls' if os.name == 'nt' else 'clear')

"""Simulation av en planets bana runt solen"""

"""
Utifrån newtons gravitationslag är ṟ'' = -(GM)/r^2 * ȓ (ṟ=avståndsvektorn, r=avståndet, ȓ=enhetsvektorn mot planeten)

Omskrivningen ȓ = ṟ/r låter oss komposantuppdela ṟ = (x,y) till x och y
-(GM)*(x/r^3) och (GM)*(y/r^3) vilket behändigt tillämpas i ett system av D.E:er

:: Solen antas vara stationär
:: Vi arbetar i termer av jordmassor, au och månader

"""

"""INSTÄLLNINGAR"""
startvärden = [1,0,-0.3,0.6] # [x,y,vx,vy]
månader = 200  # hur många månader vi plottar för
""""""


G = 8.22e-7 # gravitationskoinstanten i [AU]^3[jordmassa]^-1[månad]^-2
M = 333000  #solens massa i jordmassor
def gravitation(t,U):

    x, y, v_x, v_y = U # x,y position, v_x,v_y hastighet
    r = np.sqrt(x**2+y**2) # avståndet r beräknas här
    
    ax = -(G*M)*(x/(r**3))  
    ay = -(G*M)*(y/(r**3))

    return np.array([ax,ay])

tmin,tmax = 0,månader
t = np.linspace(0,månader,1000) # spannet blir antal månader vi plottar för
h= månader/(len(t)-1) #steglängd

#setup för semiimplicit euler
lös = np.zeros([2,len(t)])
lös[:,0]=startvärden[:2]
x,y,vx,vy = startvärden
for k in range(1,len(t)): #semiimplicit euler

    ax,ay = gravitation("bajs",[x,y,vx,vy])
    
    vx += h*ax # räknar ut hastigheten före så att man använder värde nr. k+1 för beräkningen
    vy += h*ay # Det gör att energin inte förloras av någon anledning (se rapporten)
    x  += h*vx
    y  += h*vy

    lös[:,k] = [x,y]

plt.plot(lös[0],lös[1],label="semiimplicit euler")
plt.plot(0,0,"o",color="yellow",label="solen")
plt.plot(lös[0][-1],lös[1][-1],"o",color="blue")
plt.grid()
plt.legend()
plt.title(f"Omloppsbana")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
