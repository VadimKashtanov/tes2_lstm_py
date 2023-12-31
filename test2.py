#!  /usr/bin/python3
#L=5000;f=norme(filtre());plt.plot(norme(ema(prixs[-L:], K=1)),label='prixs K=1');plt.plot(norme(ema(prixs[-L:], K=30)),label='K=?');plt.plot(norme(ema(macds[-L:],K=5)),label='macd');plt.plot(norme(ema(volumes[-L:],K=10)),label='volume');plt.plot([i*(interv:=20) for i in range(8)],[i*abs(min(norme(prixs[-L:])[:interv*8])-max(norme(prixs[-L:])[:interv*8])) for i in f]);plt.legend();plt.show()

from random import random
import struct as st
from math import exp, tanh, atanh, log

logistique = lambda x: 1 / (1+exp(-x))
alogistique = lambda y: -log(1/y -1)

rnd = lambda : 2*random()-1

def lire(f):
    with open(f, 'rb') as co: text = co.read()
    (C,) = st.unpack('I', text[:4])
    p = st.unpack('f'*C, text[4:])
    return p

def ema(arr, K):
    e = [arr[0]]
    for p in arr[1:]:
        e += [e[-1]*(1-1/(1+K)) + p*1/(1+K)]
    return e

class INSTRUCTION:
    POIDS   = None
    VARS    = None
    ENTREES = None
    SORTIES = None
    #
    p       = [None]
    y       = [None]
    dp      = [None]
    dy      = [None]
    #
    entrees = [None]
    sorties = [None]
    dentrees = [None]
    dsorties = [None]
    #
    def impl_correcte(self):
        for i in dir(self):
            exec(f"{}")
        #self en arguments, donc ca marchera que c'est si initialisé
        assert self.POIDS!=None        and self.VARS!=None               and self.SORTIES!=None       and
        assert self.p[0] != None       and self.y[0] != None             and self.sorties[0] != None
        assert len(self.p)==self.POIDS and len(self.y)==self.VARS*self.T and len(self.sorties)==self.SORTIES*self.T

class ENTREE(INSTRUCTION):
    def __init__(self, X, T):
        self.X, self.T = X, T
        self.entree = [0 for _ in range(X*T)]
        self.d_entree = [0 for _ in range(X*T)]

class SORTIE(INSTRUCTION):
    POIDS = 0
    VARS  = 0
    def __init__(self, Y, T):
        self.Y, self.T = Y, T
        self.ENTREE = self.SORTIES = Y
        self.sorties = [0 for _ in range(Y*T)]
        self.d_sorties = [0 for _ in range(Y*T)]

class LSTM(INSTRUCTION):
    def __init__(self, X, Y, T):
        self.X, self.Y, self.T = X, Y, T
        #
        #self.Wf, self.Wi, self.Wo = [np.zeros((X,Y)) for _ in range(3)]
        #self.Uf, self.Ui, self.Uo = [np.zeros((Y,Y)) for _ in range(3)]
        #self.Bf, self.Bi, self.Bo = [np.zeros((Y,)) for _ in range(3)]
        #self.Wc = np.zeros((X,Y))
        #self.Bc = np.zeros((Y,))
        d=0;      self.Wf, self.Wi, self.Wo = d+0*X*Y, d+1*X*Y, d+2*X*Y
        d+=3*X*Y; self.Uf, self.Ui, self.Uo = d+0*Y*Y, d+1*Y*Y, d+2*Y*Y
        d+=3*Y*Y; self.Bf, self.Bi, self.Bo = d+0*Y,   d+1*Y,   d+2*Y
        d+=3*Y  ; self.Wc = d + 0
        d+=1*X*Y; self.Bc = d + 0
        d+=1*Y
        #
        self.POIDS = d  #3*Y*X + 3*Y*Y + 3*Y + X*Y + Y
        #
        #self.Ft = [np.zeros((Y,)) for _ in range(T)]
        #self.It = [np.zeros((Y,)) for _ in range(T)]
        #self.Ot = [np.zeros((Y,)) for _ in range(T)]
        #self.Ct = [np.zeros((Y,)) for _ in range(T)]
        #self.Ht = [np.zeros((Y,)) for _ in range(T)]
        self.Ft = 0*Y
        self.It = 1*Y
        self.Ot = 2*Y
        self.Ct = 3*Y
        self.Ht = 4*Y
        #
        self.SORTIES = Y #les Y dernieres valeurs
        #
        self.VARS = 5*Y
        #
        self.p  = [rnd() for _ in range(self.POIDS   )]
        self.dp = [    0 for _ in range(self.POIDS   )]
        self.y  = [    0 for _ in range(self.VARS * T)]
        self.dy = [    0 for _ in range(self.VARS * T)]
        #
        self.entrees  = [0 for _ in range(self.ENTREES*T)]
        self.sorties  = [0 for _ in range(self.SORTIES*T)]
        self.dsorties = [0 for _ in range(self.SORTIES*T)]
        self.dentrees = [0 for _ in range(self.ENTREES*T)]
        #
    def zero(self):
        for i in range(self.POIDS):    self.p [i] = 0
        for i in range(self.POIDS):    self.dp[i] = 0
        for i in range(self.VARS * T): self.y [i] = 0
        for i in range(self.VARS * T): self.dy[i] = 0
        for i in range(self.SORTIES*T):self.d

    def f(self, x):
        X, Y, T = self.X, self.Y, self.T
        POIDS = self.POIDS
        VARS = self.VARS
        #
        for t in range(1, T):
            depart = self.VARS*t
            #(0) self.Ft[t] = logistic(x@self.Wf + self.Ct[t-1]@self.Uf + self.Bf)
            for i in range(Y):
                #x@self.Wf
                xp = sum(x[X*t + j] * self.p[self.Wf + i*X + j] for j in range(X))
                #self.Ct[t-1]@self.Uf
                cu = sum(self.y[VARS*(t-1) + self.Ct + j] * self.p[self.Uf + i*Y + j] for j in range(Y))
                #self.Bf
                b = self.p[self.Bf + i]
                #
                self.y[depart + self.Ft + i] = logistique(xp + cu + b)
            #(1) self.It[t] = logistic(x@self.Wi + self.Ct[t-1]@self.Ui + self.Bi)
            for i in range(Y):
                #x@self.Wi 
                xp = sum(x[X*t + j] * self.p[self.Wi + i*X + j] for j in range(X))
                #self.Ct[t-1]@self.Ui
                cu = sum(self.y[VARS*(t-1) + self.Ct + j] * self.p[self.Ui + i*Y + j] for j in range(Y))
                #self.Bi
                b = self.p[self.Bi + i]
                #
                self.y[depart + self.It + i] = logistique(xp + cu + b)
            #(2) self.Ot[t] = logistic(x@self.Wo + self.Ct[t-1]@self.Uo + self.Bo)
            for i in range(Y):
                #x@self.Wf
                xp = sum(x[X*t + j] * self.p[self.Wo + i*X + j] for j in range(X))
                #self.Ct[t-1]@self.Uf
                cu = sum(self.y[VARS*(t-1) + self.Ct + j] * self.p[self.Uo + i*Y + j] for j in range(Y))
                #self.Bf
                b = self.p[self.Bo + i]
                #
                self.y[depart + self.Ot + i] = logistique(xp + cu + b)
            #(3) self.Ct[t] = self.Ft[t]*self.Ct[-1] + self.It[t]*tanh(x@self.Wc + self.Bc)
            for i  in range(Y):
                #tanh(x@self.Wc + self.Bc)
                th = tanh(sum(x[X*t + j]*self.p[self.Wc + i*X + j] for j in range(X)) + self.p[self.Bc + i])
                Ft = self.y[depart + self.Ft + i]
                Ct = self.y[VARS*(t-1) + self.Ct + i]
                It = self.y[depart + self.It + i]
                self.y[depart + self.Ct + i] = Ft*Ct + It*th
            #(4) self.Ht[t] = self.Ot[t]*self.Ct[t]
            for i in range(Y):
                self.y[depart + self.Ht + i] = self.y[depart + self.Ot + i] * self.y[depart + self.Ct + i]
        #
        #   return
        #   copie des sorties vers self.sorties (redondant, mais le C/CUDA fixera ca)
        for t in range(T):
            for i in range(self.SORTIES):
                self.sorties[self.SORTIES*t+i] = self.y[self.VARS*t + self.VARS-self.SORTIES+i]
    
    def df(self, dy, dx):
        X, Y, T = self.X, self.Y, self.T
        POIDS = self.POIDS
        VARS = self.VARS
        #
        for t in list(range(1, T))[::-1]:
            depart = self.VARS * t
            
            #(4) self.Ht[t] = self.Ot[t]*self.Ct[t]
            for i in range(Y):
                #self.y[depart + self.Ht + i] = self.y[depart + self.Ot + i] * self.y[depart + self.Ct + i]
                self.y[depart + self.Ot + i] += dy[depart + self.Ht + i]
                self.y[depart + self.Ct + i] += dy[depart + self.Ht + i]
            
            #(3) self.Ct[t] = self.Ft[t]*self.Ct[-1] + self.It[t]*tanh(x@self.Wc + self.Bc)
            for i  in range(Y):
                Ft = self.y[depart + self.Ft + i]
                Ct = self.y[VARS*(t-1) + self.Ct + i]
                It = self.y[depart + self.It + i]
                th = (self.y[depart + self.Ct + i] - Ft*Ct)/It
                #self.y[depart + self.Ct + i] = Ft*Ct + It*th
                self.dy[depart + self.Ft + i] += Ct*self.dy[depart + self.Ct + i]
                self.dy[VARS*(t-1) + self.Ct + i] += Ft*self.dy[depart + self.Ct + i]
                self.y[depart + self.It + i] += th*self.dy[depart + self.Ct + i]
                #
                dth = It*self.dy[depart + self.Ct + i] * ( (1 - th**2) )
                self.dp[self.Bc + i] += dth
                for j in range(X):
                    dx[X*t + j] += dth * self.p[self.Wc + i*X + j]
                    self.dp[self.Bc + i*Y + j] += dth * self.p[self.Wc + i*X + j]
                #th = tanh(sum(x[X*t + j]*self.p[self.Bc + i*X + j] for j in range(X)) + self.p[self.Bc + i])
            
            #(2) self.Ot[t] = logistic(x@self.Wo + self.Ct[t-1]@self.Uo + self.Bo)
            for i in range(Y):
                #x@self.Wf
                #xp = sum(x[X*t + j] * self.p[self.Wo + i*Y + j] for j in range(X))
                #self.Ct[t-1]@self.Uf
                #cu = sum(self.y[VARS*(t-1) + self.Ct + j] * self.p[self.Uo + i*Y + j] for j in range(Y))
                #self.Bf
                #b = self.p[self.Bo + i]
                #
                val = -log(1/self.y[depart + self.Ot + i] - 1)#logistique(xp + cu + b)
                dsomme = self.dy[depart + self.Ot + i] * ((1 - val)*val);
                #
                self.dp[self.Bo + i] += dsomme
                #
                #d(cu)
                for j in range(Y):
                    self.dy[VARS*(t-1) + self.Ct + j] += dsomme * self.p[self.Uo + i*X + j]
                    self.dp[self.Uo + i*X + j] += dsomme * self.y[VARS*(t-1) + self.Ct + j]
                #d(x@self.Wf)
                for j in range(Y):
                    dx[X*t + j] += dsomme * self.p[self.Wo + i*X + j]
                    self.dp[self.Wo + i*X + j] += dsomme * x[X*t + j]
            #(1) self.It[t] = logistic(x@self.Wi + self.Ct[t-1]@self.Ui + self.Bi)
            for i in range(Y):
                #x@self.Wf
                #xp = sum(x[X*t + j] * self.p[self.Wi + i*Y + j] for j in range(X))
                #self.Ct[t-1]@self.Uf
                #cu = sum(self.y[VARS*(t-1) + self.Ct + j] * self.p[self.Ui + i*Y + j] for j in range(Y))
                #self.Bf
                #b = self.p[self.Bi + i]
                #
                val = -log(1/self.y[depart + self.It + i] - 1)#logistique(xp + cu + b)
                dsomme = self.dy[depart + self.It + i] * ((1 - val)*val);
                #
                self.dp[self.Bi + i] += dsomme
                #
                #d(cu)
                for j in range(Y):
                    self.dy[VARS*(t-1) + self.Ct + j] += dsomme * self.p[self.Ui + i*X + j]
                    self.dp[self.Ui + i*X + j] += dsomme * self.y[VARS*(t-1) + self.Ct + j]
                #d(x@self.Wf)
                for j in range(Y):
                    dx[X*t + j] += dsomme * self.p[self.Wi + i*X + j]
                    self.dp[self.Wi + i*X + j] += dsomme * x[X*t + j]
            #(0) self.Ft[t] = logistic(x@self.Wf + self.Ct[t-1]@self.Uf + self.Bf)
            for i in range(Y):
                #x@self.Wf
                #xp = sum(x[X*t + j] * self.p[self.Wf + i*Y + j] for j in range(X))
                #self.Ct[t-1]@self.Uf
                #cu = sum(self.y[VARS*(t-1) + self.Ct + j] * self.p[self.Uf + i*Y + j] for j in range(Y))
                #self.Bf
                #b = self.p[self.Bf + i]
                #
                val = -log(1/self.y[depart + self.Ft + i] - 1)#logistique(xp + cu + b)
                dsomme = self.dy[depart + self.Ft + i] * ((1 - val)*val);
                #
                self.dp[self.Bf + i] += dsomme
                #
                #d(cu)
                for j in range(Y):
                    self.dy[VARS*(t-1) + self.Ct + j] += dsomme * self.p[self.Uf + i*X + j]
                    self.dp[self.Uf + i*X + j] += dsomme * self.y[VARS*(t-1) + self.Ct + j]
                #d(x@self.Wf)
                for j in range(Y):
                    dx[X*t + j] += dsomme * self.p[self.Wf + i*X + j]
                    self.dp[self.Wf + i*X + j] += dsomme * x[X*t + j]
        #
        #   return
        #   copie des sorties vers self.dsorties (redondant, mais le C/CUDA fixera ca)
        for t in range(T):
            for i in range(self.SORTIES):
                self.dsorties[self.SORTIES*t+i] = self.dy[self.VARS*t + self.VARS-self.SORTIES+i]

class MDL:
    def __init__(self, archi, T):
        self.archi = archi
        self.T = T
        self.instructions = [
            LSTM(x, y, T)
            for x,y in zip(archi[:-1], archi[1:])
        ]
        for inst in self.instructions:
            inst.impl_correcte()

    def zero(self):
        for inst in self.instructions:
            inst.zero()
            
    def f(self, x):
        #chaque instruction execute tous les T d'un coup. Car pas besoin de ce qu'il y a apres
        for inst in self.instructions:
            inst.f(x)
            x = inst.sorties

    def df(self, dx, dy):
        for i in range(len(self.instructions)-1, 0-1, -1):
            dx = dy
            self.instructions[i].df(
                self.instructions[i-1], #dy    ##  !!!!!!!!!!!!!!!!!!!!!!
                self.instructions[i-1]  #dx    ##" !!!!!!!!!!!!!!!!!!!
            )
            dy = dx

    def S(self, y, donnees):
        S  = sum((y[i]-donnees[i])**2/2 for i in range(len(y)))/len(y)
        return S

    def dS(self, y, donnees):
        dS = [ (y[i]-donnees[i])/len(y) for i in range(len(y))]
        return dS



if __name__ == "__main__":
    prixs = lire("/home/vadim/Bureau/Filtres-V1.4+ (versions)/2a/prixs/prixs.bin")
    #
    T = 7
    archi = [3,5,5,2]
    X, Y = archi[0], archi[-1]
    mdl = MDL(archi, T)
    x = [rnd() for _ in range(X*T)]
    y = [rnd() for _ in range(Y*T)]
    #
    dx = [0 for _ in range(X*T)]
    #
    mdl.zero()
    mdl.f(x)
    print(mdl.S(mdl.instructions[-1].sorties, y))
    dy = mdl.dS(mdl.instructions[-1].sorties, y)
    mdl.df(dx, dy)
