import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
os.system('cls' if os.name == 'nt' else 'clear')

"""N-body simulation gravitational slingshot"""

"""
Inställningar ligger en bit ner 

Vi ska försöka simulera en gravitationsslunga där ett rymskepp som lyfter från jorden utnyttjar jupiters gravitation för
att öka sin hastighet. Teorin säger att om skeppet närmar sig jupiter vinkelrätt mot sin bana så ska skeppet flyga in bakom
planeten för att öka sin hastighet. 

Jag tog en mer dataspelsaktig approach till problem åtta. Jag tänkte att jag ville efterlika 
steam-spelet universe sandbox som jag haft sen evigheter. Jag försöker att organisera koden likt 
det jag gjorde i sjuan fast med oop. Jag känner också att det blir roligare.
"""

G = 8.22e-7 # gravitationskoinstanten i [AU]^3[jordmassa]^-1[månad]^-2

class Planet:
    def __init__(self,namn,massa,pos,vel,färg=None):
        self.namn = namn
        self.massa=massa
        self.pos=np.array(pos)
        self.vel=np.array(vel)
        self.färg=färg
    def copy(self):
        return Planet(f"{self.namn}",self.massa,self.pos,self.vel,self.färg)
    def __repr__(self):
        return f"{self.namn} at {self.pos}"
    
class Nbodyproblem:
    def __init__(self,månader=12,res=50000):
        self.planeter = []    
        self.res = res
        self.månader = månader
        self.h= månader/(res-1)
        self.initial_state = []

    def addPlanet(self,planet):
        self.planeter.append(planet)
    def setInitialState(self):
        self.initial_state = []
        for planet in self.planeter:
            copy = planet.copy()
            self.initial_state.append(copy)
        # self.initial_state = [k.copy for k in self.planeter]
    def resetState(self):
        for p,pcopy in zip(self.planeter,self.initial_state):
            p.pos = pcopy.pos
            p.vel = pcopy.vel

    def getState(self):
        return np.array([k.pos for k in self.planeter]).reshape(-1,2)
    def getState_vel(self):
        return np.array([k.vel for k in self.planeter]).reshape(-1,2)
        
    def gravitation(self): # summerar gravitationen för varje planet
        N = len(self.planeter)
        g_lista = np.zeros((N,2)) 
 
        for i in range(N): # summan av alla par av planeter i<j
            for j in range(i+1,N):
                p1 = self.planeter[i]
                p2 = self.planeter[j]

                r = p1.pos - p2.pos
                rnorm = np.linalg.norm(r)

                #newtons gravitationsformel
                F = -G*((p1.massa*p2.massa)/rnorm**2)*(r/rnorm)

                # newtons andra lag på de båda planeterna
                g_lista[i] = g_lista[i] + self.h*(F/p1.massa)
                g_lista[j] = g_lista[j] - self.h*(F/p2.massa)
                
        return g_lista

    def semi_euler(self): # räknar ett steg av semiimplicit euler
        acc = self.gravitation()
        for planet,a in zip(self.planeter,acc):
            planet.vel = planet.vel + a # räknar ut nästa hastighet först precis som i sjuan
        for planet in self.planeter:
            planet.pos = planet.pos + self.h*planet.vel

    def trace(self): # ritar kurvorna
        #setup
        plot = np.zeros((len(self.planeter),self.res,2)) # kurvmatrisen
        plot[:,0,:]=self.getState()

        # kör semiimplicita euler
        for k in range(1,self.res): 
            self.semi_euler()
            plot[:,k,:] = self.getState()

        #plotta varje kurva
        for k,planet in enumerate(self.planeter):
            xy=np.transpose(plot[k,:,:])
            
            plt.plot(xy[0],xy[1],color=planet.färg)
            plt.plot(xy[0][-1],xy[1][-1],"o",color=planet.färg,label=planet.namn)
        plt.legend()
        plt.show()
        return self.planeter

    def trace_noplot(self): # ritar kurvorna
        #setup
        plot = np.zeros((len(self.planeter),self.res,2)) # kurvmatrisen
        plot[:,0,:]=self.getState()

        # kör semiimplicita euler
        for k in range(1,self.res): 
            self.semi_euler()
            plot[:,k,:] = self.getState()
        return self.planeter

"""Inställningar""" # kommenterade värden för min gravitationsslunga
theta_jorden = (9/6)*np.pi #9/6
theta_jupiter = (0.835/6)*np.pi #0.835

launch_speed = 0.22 # 0.22 # jordens flykthastighet är typ 0.2 solsystemets är högre
launch_angle = (9.1/6)*np.pi #9.1/6

månader = 30
""""""
def setup_nbp(theta_jorden,theta_jupiter,launch_speed,launch_angle):
    launch_angle += theta_jorden+(1/2)+np.pi # anpassar vinkeln efter jordens position

    jordxy=[np.cos(theta_jorden),np.sin(theta_jorden)]
    jordvel=[-(np.pi/6)*np.sin(theta_jorden),(np.pi/6)*np.cos(theta_jorden)]
    jupiterxy=[5.2*np.cos(theta_jupiter),5.2*np.sin(theta_jupiter)]
    jupitervel=[-(np.pi/13)*np.sin(theta_jupiter),(np.pi/13)*np.cos(theta_jupiter)]

    rocketxy = [jordxy[0]+1e-3*np.cos(launch_angle),jordxy[1]+1e-3*np.sin(launch_angle)] #1e-3
    launch_vel=[jordvel[0]+launch_speed*np.cos(launch_angle),jordvel[1]+launch_speed*np.sin(launch_angle)]
    return jordxy,jordvel,jupiterxy,jupitervel,rocketxy,launch_vel
jordxy,jordvel,jupiterxy,jupitervel,rocketxy,launch_vel = setup_nbp(theta_jorden,theta_jupiter,launch_speed,launch_angle)

jorden=Planet("Jorden",1,jordxy,jordvel,"blue")
solen=Planet("Solen",333000,[0,0],[0,0],"yellow") #solen är en planet
jupiter=Planet("Jupiter",317.8,jupiterxy,jupitervel,"orange")
rymdskepp=Planet("Rymdskepp",1e-11,rocketxy,launch_vel,"black")

nbp = Nbodyproblem(månader=månader,res=1000*månader)
nbp.addPlanet(jorden)
nbp.addPlanet(solen)
nbp.addPlanet(jupiter)
nbp.addPlanet(rymdskepp)

nbp.trace()