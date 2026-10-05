import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import os
os.system('cls' if os.name == 'nt' else 'clear')

N = 70

"""
D.E: y' = y^2

y^2 är kontinuerlig och lokalt lipschitz överallt => en unik lösning existerar
Separering och integrering ger y = -1/(t+c) med villkoret att y' >= 0,
Kurvan kommer explodera när t går mot -c

Vi har även gränsfallet om y=0 som aldrig rör sig från y=0
"""

def fwdeuler(f,y0,t): # pga singulariteten hade solve_ivp utan ett event krashat, därav min egen euler
    b = np.zeros(len(t))
    b[0] = y0
    h=(t[-1]-t[0])/len(t)
    for n in range(len(t)-1):
        if np.abs(b[n]) > 10: # bryter när y blir stort annars hinner det explodera för mycket för min smak
            b[n:] = None
            break
        b[n+1] = b[n]+h*f(t[n],b[n])
    return b
def func(t,y):
    return y**2
y0 = N/100
tmin,tmax=0,2.5
t=np.linspace(tmin,tmax,1000)

sol = fwdeuler(func,y0,t)
sol_negativ = fwdeuler(func,-6,t)

"""Är begynnelsevillkoret så utformat att vi börjar räkna från ett negativt tal får vi 
undre delen av grafen till y = -1/(x-c)"""

"""om vi har en punkt på kurvan och vill veta var singulariteten är skriver vi om ekvationen
y = -1/(t-c)  <=>  c = t + 1/y   (wlog c = -c eftersom singulariteten är vid t=-c)"""

bruh = sol[~np.isnan(np.copy(sol))] 
for n in range(len(bruh)):
    bruh[n]= t[n]+1/bruh[n]

min = np.min(bruh)
max = np.max(bruh)
avg = np.sum(bruh)/len(bruh)
print(f"Snittgissning {avg:.4f}, min-max: {min:.4f} - {max:.4f}")
print("Analytiskt: 1,4285")

plt.plot(t,sol_negativ,label="negativ")
plt.plot(t,sol,label = "vanlig")
plt.plot(t,np.zeros_like(t),color="black")
plt.grid()
plt.title(f"y\' = y^2, y(0) = {y0}")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.show()

"""Om tidssteget är för stort i förhållande till 1/(C*|epsilon|) så kan negativa startvärden felaktigt
ge upphov till numeriska lösningar som byter tecken och blir positiva, adaptiva metoder är dock för smarta för det"""

t=np.linspace(0,5,8)
C=70
def func2(t,y):
    return C*y**2
epsilon = -0.2
sol_knas = fwdeuler(func2,epsilon,t) # fwdeuler kan byta tecken

plt.plot(t,np.zeros_like(t),color="black")
plt.legend()
plt.plot(t,sol_knas)
plt.title("Felaktigt positiv lösning")
plt.show()