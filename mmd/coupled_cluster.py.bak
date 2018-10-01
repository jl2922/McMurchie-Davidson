from __future__ import division
from __future__ import print_function
import math
import sys
import itertools
import numpy as np

class CoupledCluster(object):
    """Class for coupled cluster routines"""
    def __init__(self, mol):
        self.mol = mol
        dim = self.mol.nbasis
        self.ts = np.zeros((dim,dim))
        self.td = np.zeros((dim,dim,dim,dim))
        self.spinints = np.zeros((dim,dim,dim,dim))
        occupied = range(self.mol.nocc)
        virtual  = range(self.mol.nocc,self.mol.nbasis)
        basis = range(self.mol.nbasis)
        for p,q,r,s in itertools.product(basis,basis,basis,basis):
            value1 = self.mol.single_bar[p//2,r//2,q//2,s//2] * (p%2 == r%2) * (q%2 == s%2)
            value2 = self.mol.single_bar[p//2,s//2,q//2,r//2] * (p%2 == s%2) * (q%2 == r%2)
            self.spinints[p,q,r,s] = value1 - value2

        fs = np.zeros((dim))
        for i in range(dim):
            fs[i] = self.mol.MO[i//2]
        self.fs = np.diag(fs) # put MO energies in diagonal array
        self.ts = np.zeros((dim,dim))
        self.td = np.zeros((dim,dim,dim,dim))

        for a,b,i,j in itertools.product(virtual,virtual,occupied,occupied):
            self.td[a,b,i,j] += self.spinints[i,j,a,b]/(self.fs[i,i] + self.fs[j,j] - self.fs[a,a] - self.fs[b,b])

        # Make denominator arrays Dai, Dabij
        # Equation (12) of Stanton
        self.Dai = np.zeros((dim,dim))
        for a,i in itertools.product(virtual,occupied):
            self.Dai[a,i] = self.fs[i,i] - self.fs[a,a]

        # Stanton eq (13)
        self.Dabij = np.zeros((dim,dim,dim,dim))
        for a,b,i,j in itertools.product(virtual,virtual,occupied,occupied):
            self.Dabij[a,b,i,j] = self.fs[i,i] + self.fs[j,j] - self.fs[a,a] - self.fs[b,b]

    def MP2(self):
        """Routine to compute MP2 energy from RHF reference"""
        EMP2 = 0.0
        occupied = range(self.mol.nocc)
        virtual  = range(self.mol.nocc,self.mol.nbasis)
        for i,j,a,b in itertools.product(occupied,occupied,virtual,virtual):
            denom = self.mol.MO[i] + self.mol.MO[j] \
                  - self.mol.MO[a] - self.mol.MO[b]
            numer = self.mol.single_bar[i,a,j,b] \
                  * (2.0*self.mol.single_bar[i,a,j,b]
                    - self.mol.single_bar[i,b,j,a])
            EMP2 += numer/denom
        self.mol.emp2 = EMP2 + self.mol.energy
        print('E(MP2) = ', self.mol.emp2.real)

    def CCSD(self):
        ECCSD = 0.0
        ECCSDOld = 1.0
        while abs(ECCSD - ECCSDOld) > 1e-8:
            ECCSDOld = ECCSD
            Fae,Fmi,Fme,Wmnij,Wabef,Wmbej = self.updateintermediates()
            # ts = makeT1(True,ts,td)
            # td = makeT2(True,ts,td)
            # ECCSD = ccsdenergy()
        self.mol.eccsd = ECCSD + self.mol.energy

    # Stanton eq (10)
    def tau(self,a,b,i,j):
      tau = self.td[a,b,i,j] + self.ts[a,i]*self.ts[b,j] - self.ts[b,i]*self.ts[a,j]
      return tau

    # Stanton eq (9)
    def taus(self,a,b,i,j):
      taus = self.td[a,b,i,j] + 0.5*(self.ts[a,i]*self.ts[b,j] - self.ts[b,i]*self.ts[a,j])
      return taus

    def updateintermediates(self):
        occupied = range(self.mol.nocc)
        virtual  = range(self.mol.nocc,self.mol.nbasis)
        dim = self.mol.nbasis
        Nelec = self.mol.nocc

        # Stanton eq (3)
        Fae = np.zeros((dim,dim))
        for a in range(Nelec,dim):
          for e in range(Nelec,dim):
            Fae[a,e] = (1 - (a == e))*self.fs[a,e]
            for m in range(0,Nelec):
              Fae[a,e] += -0.5*self.fs[m,e]*self.ts[a,m]
              for f in range(Nelec,dim):
                Fae[a,e] += self.ts[f,m]*self.spinints[m,a,f,e]
                for n in range(0,Nelec):
                  Fae[a,e] += -0.5*self.taus(a,f,m,n)*self.spinints[m,n,e,f]

        # Stanton eq (4)
        Fmi = np.zeros((dim,dim))
        for m in range(0,Nelec):
          for i in range(0,Nelec):
            Fmi[m,i] = (1 - (m == i))*self.fs[m,i]
            for e in range(Nelec,dim):
              Fmi[m,i] += 0.5*self.ts[e,i]*self.fs[m,e]
              for n in range(0,Nelec):
                Fmi[m,i] += self.ts[e,n]*self.spinints[m,n,i,e]
                for f in range(Nelec,dim):
                  Fmi[m,i] += 0.5*self.taus(e,f,i,n)*self.spinints[m,n,e,f]

        # Stanton eq (5)
        Fme = np.zeros((dim,dim))
        for m in range(0,Nelec):
          for e in range(Nelec,dim):
            Fme[m,e] = self.fs[m,e]
            for n in range(0,Nelec):
              for f in range(Nelec,dim):
                Fme[m,e] += self.ts[f,n]*self.spinints[m,n,e,f]

        # Stanton eq (6)
        Wmnij = np.zeros((dim,dim,dim,dim))
        for m in range(0,Nelec):
          for n in range(0,Nelec):
            for i in range(0,Nelec):
              for j in range(0,Nelec):
                Wmnij[m,n,i,j] = self.spinints[m,n,i,j]
                for e in range(Nelec,dim):
                  Wmnij[m,n,i,j] += self.ts[e,j]*self.spinints[m,n,i,e] - self.ts[e,i]*self.spinints[m,n,j,e]
                  for f in range(Nelec,dim):
                    Wmnij[m,n,i,j] += 0.25*self.tau(e,f,i,j)*self.spinints[m,n,e,f]

        # Stanton eq (7)
        Wabef = np.zeros((dim,dim,dim,dim))
        for a in range(Nelec,dim):
          for b in range(Nelec,dim):
            for e in range(Nelec,dim):
              for f in range(Nelec,dim):
                Wabef[a,b,e,f] = self.spinints[a,b,e,f]
                for m in range(0,Nelec):
                  Wabef[a,b,e,f] += -self.ts[b,m]*self.spinints[a,m,e,f] + self.ts[a,m]*self.spinints[b,m,e,f]
                  for n in range(0,Nelec):
                    Wabef[a,b,e,f] += 0.25*self.tau(a,b,m,n)*self.spinints[m,n,e,f]

        # Stanton eq (8)
        Wmbej = np.zeros((dim,dim,dim,dim))
        for m in range(0,Nelec):
          for b in range(Nelec,dim):
            for e in range(Nelec,dim):
              for j in range(0,Nelec):
                Wmbej[m,b,e,j] = self.spinints[m,b,e,j]
                for f in range(Nelec,dim):
                  Wmbej[m,b,e,j] += self.ts[f,j]*self.spinints[m,b,e,f]
                for n in range(0,Nelec):
                  Wmbej[m,b,e,j] += -self.ts[b,n]*self.spinints[m,n,e,j]
                  for f in range(Nelec,dim):
                    Wmbej[m,b,e,j] += -(0.5*self.td[f,b,j,n] + self.ts[f,j]*self.ts[b,n])*self.spinints[m,n,e,f]

        return Fae, Fmi, Fme, Wmnij, Wabef, Wmbej

    def makeT1(self,ts,td):
        dim = self.mol.nbasis
        Nelec = self.mol.nocc
        tsnew = np.zeros((dim,dim))
        for a in range(Nelec,dim):
          for i in range(0,Nelec):
            tsnew[a,i] = self.fs[i,a]
            for e in range(Nelec,dim):
              tsnew[a,i] += self.ts[e,i]*self.Fae[a,e]
            for m in range(0,Nelec):
              tsnew[a,i] += -ts[a,m]*Fmi[m,i]
              for e in range(Nelec,dim):
                tsnew[a,i] += td[a,e,i,m]*Fme[m,e]
                for f in range(Nelec,dim):
                  tsnew[a,i] += -0.5*td[e,f,i,m]*spinints[m,a,e,f]
                for n in range(0,Nelec):
                  tsnew[a,i] += -0.5*td[a,e,m,n]*spinints[n,m,e,i]
            for n in range(0,Nelec):
              for f in range(Nelec,dim):
                tsnew[a,i] += -ts[f,n]*spinints[n,a,i,f]
            tsnew[a,i] = tsnew[a,i]/Dai[a,i]
      return tsnew

    # Stanton eq (2)
    def makeT2(x,ts,td):
      if x == True:
        tdnew = np.zeros((dim,dim,dim,dim))
        for a in range(Nelec,dim):
          for b in range(Nelec,dim):
            for i in range(0,Nelec):
              for j in range(0,Nelec):
                tdnew[a,b,i,j] += spinints[i,j,a,b]
                for e in range(Nelec,dim):
                  tdnew[a,b,i,j] += td[a,e,i,j]*Fae[b,e] - td[b,e,i,j]*Fae[a,e]
                  for m in range(0,Nelec):
                    tdnew[a,b,i,j] += -0.5*td[a,e,i,j]*ts[b,m]*Fme[m,e] + 0.5*td[a,e,i,j]*ts[a,m]*Fme[m,e]
                    continue
                for m in range(0,Nelec):
                  tdnew[a,b,i,j] += -td[a,b,i,m]*Fmi[m,j] + td[a,b,j,m]*Fmi[m,i]
                  for e in range(Nelec,dim):
                    tdnew[a,b,i,j] += -0.5*td[a,b,i,m]*ts[e,j]*Fme[m,e] + 0.5*td[a,b,i,m]*ts[e,i]*Fme[m,e]
                    continue
                for e in range(Nelec,dim):
                  tdnew[a,b,i,j] += ts[e,i]*spinints[a,b,e,j] - ts[e,j]*spinints[a,b,e,i]
                  for f in range(Nelec,dim):
                    tdnew[a,b,i,j] += 0.5*tau(e,f,i,j)*Wabef[a,b,e,f]
                    continue
                for m in range(0,Nelec):
                  tdnew[a,b,i,j] += -ts[a,m]*spinints[m,b,i,j] + ts[b,m]*spinints[m,a,i,j]
                  for e in range(Nelec,dim):
                    tdnew[a,b,i,j] += td[a,e,i,m]*Wmbej[m,b,e,j] - ts[e,i]*ts[a,m]*spinints[m,b,e,j]
                    tdnew[a,b,i,j] += -td[a,e,j,m]*Wmbej[m,b,e,i] + ts[e,j]*ts[a,m]*spinints[m,b,e,i]
                    tdnew[a,b,i,j] += -td[b,e,i,m]*Wmbej[m,a,e,j] - ts[e,i]*ts[b,m]*spinints[m,a,e,j]
                    tdnew[a,b,i,j] += td[b,e,j,m]*Wmbej[m,a,e,i] - ts[e,j]*ts[b,m]*spinints[m,a,e,i]
                    continue
                  for n in range(0,Nelec):
                    tdnew[a,b,i,j] += 0.5*tau(a,b,m,n)*Wmnij[m,n,i,j]
                    continue
                tdnew[a,b,i,j] = tdnew[a,b,i,j]/Dabij[a,b,i,j]
        return tdnew

    # Expression from Crawford, Schaefer (2000)
    # DOI: 10.1002/9780470125915.ch2
    # Equation (134) and (173)
    # computes CCSD energy given T1 and T2
    def ccsdenergy():
      ECCSD = 0.0
      for i in range(0,Nelec):
        for a in range(Nelec,dim):
          ECCSD += fs[i,a]*ts[a,i]
          for j in range(0,Nelec):
            for b in range(Nelec,dim):
              ECCSD += 0.25*spinints[i,j,a,b]*td[a,b,i,j] + 0.5*spinints[i,j,a,b]*(ts[a,i])*(ts[b,j])
      return ECCSD


