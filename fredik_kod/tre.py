import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import os
os.system('cls' if os.name == 'nt' else 'clear')

"""
Numerisk lösning till BVP:et y' = √|y|, y(0) = 0 och medföljande plot

Funktionen √|y| är kontinuerlig men saknar derivata m.a.p y (inte y Lipschitz) vid y=0
Det ger att lösning existerar men kanske inte är unik

Uppdelning av funktionen pga absolutbeloppet, därefter separering och integrering ger 
y = ±(t/2 + c)^2 med vilkoret y' >= 0 vilket borde framställa en s-formad kurva med 
inflektion i horisontella axeln men om y = 0 är y'= √0 = 0 vilket också menar att 
vi kommer att stanna kvar vid y=0. Tvetydighet!
"""

def func(t,y):
    return np.sqrt(np.abs(y)) 
y0 = [0] # om den numeriska lösningen börjar på 0 stannar den på 0
tmin,tmax=-5,5

sol1 = solve_ivp(func,(0,tmin),y0,t_eval=np.linspace(0,tmin,50))
sol2 = solve_ivp(func,(0,tmax),y0,t_eval=np.linspace(0,tmax,50))

t=np.linspace(tmin,tmax,100)
y = np.append(np.flip(sol1.y[0]),sol2.y[0])

plt.plot(t,y)
plt.grid()
plt.title(f"y\' = √|y|, y(0) = {y0}")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.show()


""" Nu för att se om numeriska lösningen kan fastna kring y=0 med lite olika metoder"""

def fwdeuler(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    h=(t[-1]-t[0])/len(t)
    for n in range(len(t)-1):
        y[n+1] = y[n]+h*f(t[n],y[n])
    return y
def bwdeuler(f,y0,t):
    y=np.zeros_like(t)
    y[0]=y0
    h=(t[-1]-t[0])/len(t)
    for n in range(len(t)-1):
        g = lambda nexty: nexty - y[n] - h*func(t[n+1],nexty)
        litet_tal = 1e-5
        startgissning = y[n]  + litet_tal if np.abs(y[n]) < 1e-8 else y[n] # utan den här raden kan lösningen fastna helt
        y[n+1] = fsolve(g,startgissning)[0] 
        # fsolve behöver en annan startgissning om y är nära 0 eftersom den använder newtons metod
    return y

epsilon=1e-6
y0 = [-1+epsilon]
tmin,tmax=-1,2
num = 10000 

t_eval = np.linspace(tmin,tmax,num)
sol = solve_ivp(func,(tmin,tmax),y0,t_eval=t_eval,method="RK45")
sol2 = solve_ivp(func,(tmin,tmax),y0,t_eval=t_eval,method="DOP853")

eulerfram = fwdeuler(func,y0[0],t_eval)
eulerbak =bwdeuler(func,y0[0],t_eval)

def fastna(arr): # räknar hur många steg kurvan är |y|<1e-6
    count = 0
    for n in arr: 
        if np.abs(n) < 1e-06:
            count += 1
        else:
            if count > 0:
                break
    return count

print("Steg innanför |y|<10^-6")
print("RK45:",fastna(sol.y[0]))
print("DOP853:",fastna(sol2.y[0]))
print("framåteuler:",fastna(eulerfram))
print("bakåteuler:",fastna(eulerbak))

"""Inga inbyggda funktioner fastnade särskilt länge, framåteuler fastnade längre men proportionellt mot
antal steg. Bakåteuler behövde lite intelligens för att komma runt fsolves brister men fastnade därefter lika
länge som framåteuler. """

"""Matematiskt visar det sig att lösningen aggressivt rör sig iväg från 0 om inte ett steg hamnar på exakt 0"""

# plt.plot(t_eval,eulerfram,label="euler")
# plt.plot(t_eval,eulerbak,label="bakåt euler")
# plt.plot(sol.t,sol.y[0],label="RK45")
# plt.legend()
# plt.grid()
# plt.title(f"y\' = √|y|, y(-1) = {y0}")
# plt.xlabel("t")
# plt.ylabel("y(t)")
# plt.show()
