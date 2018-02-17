from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import itertools

class PostSCF(object):
    """Class for post-scf routines"""
    def __init__(self,mol):
        self.mol = mol
        if not self.mol.is_converged:
            sys.exit("SCF not converged, skipping Post-SCF")
        self.ao2mo()

    def ao2mo(self):
        """Routine to convert AO integrals to MO integrals"""
        self.mol.single_bar = np.einsum('mp,mnlz->pnlz',
                                        self.mol.C,self.mol.TwoE)
        temp = np.einsum('nq,pnlz->pqlz',
                         self.mol.C,self.mol.single_bar)
        self.mol.single_bar = np.einsum('lr,pqlz->pqrz',
                                        self.mol.C,temp)
        temp = np.einsum('zs,pqrz->pqrs',
                         self.mol.C,self.mol.single_bar)
        self.mol.single_bar = temp
    
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

    def TDHF(self):
        """Routine to perform TDHF from RHF reference"""

        nocc = self.mol.nocc
        nvir = self.mol.nbasis - self.mol.nocc
        nov  = nocc*nvir

        eo = self.mol.MO[:nocc]
        ev = self.mol.MO[nocc:]

        assert(nocc == len(eo) and nvir == len(ev))

        # (e_a - e_i) * delta_i,j * delta_a,b
        A  = np.einsum('ab,ij->iajb', np.diag(ev), np.diag(np.ones(nocc)))
        A -= np.einsum('ij,ab->iajb', np.diag(eo), np.diag(np.ones(nvir)))

        # Get (ia|jb) and (ij|ab)
        Siajb = np.asarray(self.mol.single_bar[:nocc,nocc:,:nocc,nocc:],dtype='float')
        Sijab = np.asarray(self.mol.single_bar[:nocc,:nocc,nocc:,nocc:],dtype='float')

        # add 2*(ia|jb) - (ij|ab) 
        A += 2*Siajb - Sijab.swapaxes(1, 2)
        A = A.reshape(nov,nov)
       
        # B = -2*(ia|jb) + (ja|ib)
        B  = -2*Siajb
        B += Siajb.swapaxes(0, 2)
        B = B.reshape(nov,nov)

        # Make TDHF matrix
        TDHF = np.vstack((np.hstack((A,B)),np.hstack((-B,-A)))) 

        E,C = np.linalg.eig(TDHF)
        idx = E.argsort() 
        E = E[idx]
        C = C[:,idx]

        print(E[nov:nov+3]*27.2114)


        
        


