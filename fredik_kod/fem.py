import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
os.system('cls' if os.name == 'nt' else 'clear')

"""
D.E:n y'=y definierar e^x 

Uppgiften är att beräkna N*e^(2^k) där N=70 fast som ett ivp med y(0)=N

Programmet kommer att räkna tills precision med två decimaler uppnås. 
"""

N = 70
def func(t,y):
    return y
y0 = N
k=1
b=4

k=mp.mpf(1)
while True:
    # räknar ut precisionen baserat på att ett tal x har log_{10}(x) antal siffror
    mp.mp.dps = 0.44*(2**k)+4

    funktion = mp.odefun(func,0,y0,verbose=True) # odefun interpolerar en funktion med taylorutveckling
    print(f"k={k:.0f}: {funktion(2**k):.2f}")
    k=k+1

# k = 9 är högsta jag kommer till utan att behöva vänta väldigt länge och det krävde 18 GB ram
#15990895105778296482651260619856823960277671314641286757700395338101
#   17598787054063803479139151507043554189068503024864342352090222771718
#       30233772968609866441517886792258095919607559605363438122530335477546589677404937260569042.30