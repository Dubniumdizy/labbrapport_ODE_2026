import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import os
os.system('cls' if os.name == 'nt' else 'clear')
import warnings
warnings.filterwarnings("ignore")

"""
D.E: y'' + y = 0, y(0)=1, y'(0)=0

Karakäristiska ekvationen ger y = c_1e^(it) + c_2e^(-it). Begynnelsevillkoren ger c_1 = c_2 = 1/2
Detta är identiskt med den komplexa exponential-definitionen av cosinus.

Vi ska försöka beräkna pi genom att hitta begynnelsevärdesproblemets nollställe vid pi/2 och gångra med 2
"""

def diffekv(t,y): # ekvationen som ett linjärt system
    A = np.array([[0,1],[-1,0]]) 
    return A @ y

x0 = 0
y0 = [1,0]
tmin,tmax = 0,2

t=np.linspace(0,7,10000)

"""Lösning med eulers metod (inte särskilt exakt)"""
def systemframåteuler(F,t,y0): # copy-paste från gammal nummelab
    n=len(t)-1
    y=np.zeros((n+1,len(y0)))
    y[0]=y0
    h=np.abs(t[-1]-t[0])/(n)
    for k in np.arange(n):
        y[k+1] = y[k]+h*F(t[k],y[k])
    return y

y1 = systemframåteuler(diffekv,t,y0)[:,0]

for n in range(len(y1)):
    if y1[n]<0: # hittar x-intercept
        y_1,y_2,x_1,x_2 = y1[n-1],y1[n],t[n-1],t[n] #interpolerar en linje genom punkterna närmast nollstället
        liney = lambda t: ((y_2-y_1)/(x_2-x_1))*(t-x_2) + y_2 #riktigt hopkok men vi hoppas inte på nåt mycket
        root = fsolve(liney,1.5)[0]
        print("Euler:",2*root)
        break

"""Lösning med solve_ivp (övervinner inte maskinprecisionen för float64)"""
sol = solve_ivp(diffekv,(tmin,tmax),y0,events=lambda t,y:y[0],method="DOP853",rtol=1e-25,atol=1e-25) 
print("solve_ivp",sol.t_events[0][0]*2)

"""Lösning med mpmath med godtycklig precision"""
mp.mp.dps = 22
funktion = mp.odefun(diffekv,x0,y0) # approximerar problemet med taylor
def cosinus(t):
    return funktion(t)[0] # sinus hade varit funktion(t)[1]
lösning = mp.findroot(cosinus,1.5) # roten hittas av taylorfunktionen
print(f"mpmath: {lösning*2:.20f}")

y=[float(cosinus(x)) for x in t]
plt.plot(t,y,label="cosinus")
plt.grid()
plt.title(f"y\'\'+y = 0, y(0)={y0[0]}, y'(0)={y0[1]}")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.show()

"""Dags att värma upp datorn"""

mp.mp.dps=1e6 # en miljon decimaler gick på nån sekund, 10 miljoner tog några minuter 
lösning = 2*mp.findroot(cosinus,1.5,method="newton")
print(lösning)

# Programmet verkar ha tidskomplexitet O(n^2)