#!/usr/bin/env python

import numpy as np
import numpy.linalg as npl
import time
from MMD import *

start_time = time.time()
np.set_printoptions(precision = 5,linewidth = 200)

def orthMat(S):
        s, U = npl.eig(S)
        s_invsqrt = np.diag(s**(-0.5))
        S_invsqrt = np.dot(U, np.dot(s_invsqrt,np.transpose(U)))
        return S_invsqrt

def orthogonalize(M,S_invsqrt):

        M_orth = np.matmul(np.transpose(S_invsqrt), np.matmul(M,S_invsqrt))

        return M_orth

def P(C):
        P_mat = np.zeros(len(C)*len(C), dtype = float)
        P_mat = np.reshape(P_mat, (len(C),len(C)))                      #creates a matrix of the right dimensions filled with zeros

        for mu in range(len(C)):
                for nu in range(len(C)):
                        for m in range(5):                                                              #change to einsum
                                P_mat[mu][nu] += C[mu][m]*C[nu][m]                      #Sums only over the occupied orbitals

        return P_mat    

def energy(H,F,P):

        E_e = np.sum(P * (H + F))

        return E_e

def F_new(H_core,D,rep):

        Fock = H_core + 2*np.einsum('kl,ijkl->ij',D,rep) - np.einsum('kl,ikjl->ij',D,rep)

        return Fock


def enCoef(F, S_invsqrt):

        e_o,c_o = npl.eigh(F)
        C_mo = np.matmul(S_invsqrt,c_o)

        return e_o, C_mo

def MP2(rep, E_mo, C_mo):
        """
        #rep_ao = np.einsum('ip,jq,ijkl,kr,ls->pqrs', C_mo,C_mo,rep,C_mo,C_mo)
        
        #This is equivalent to the noddy algorithm - simple but about 5 times slower than the below algorithm for water with STO-3G case
        """
        rep_ao = np.einsum('ijkl,ls->ijks', rep, C_mo)
        rep_ao = np.einsum('ijks,kr->ijrs', rep_ao, C_mo)
        rep_ao = np.einsum('ijrs,jq->iqrs', rep_ao, C_mo)
        rep_ao = np.einsum('iqrs,ip->pqrs', rep_ao, C_mo)
        

        E_mp2 = 0

        for i in range(5):
                for j in range(5):
                        for a in range(5,len(E_mo)):
                                for b in range(5,len(E_mo)):
                                        E_mp2 += rep_ao[i][a][j][b] * (2*rep_ao[i][a][j][b]-rep_ao[i][b][j][a]) / (E_mo[i] + E_mo[j] - E_mo[a] - E_mo[b])

        #change to einsum
        return E_mp2

def main():
        global Overlap
        molecule = Molecule()
        molecule.read_Molecule('h2o.dat')
        molints = MMDEvaluator(molecule)
        Overlap = molints.S()
        S_invsqrt = orthMat(Overlap)
        H_core = molints.T() + molints.V()
        rep = molints.ERI()
        print(Overlap)
        enuc = 8.002367061810450

        Fock = orthogonalize(H_core,S_invsqrt)
        E_ao, C_mo = enCoef(Fock,S_invsqrt)
        Den = P(C_mo)
        En_e = energy(H_core,H_core,Den)
        E_old = 0.0

        while abs(E_old - En_e) > 0.000000000001:

                E_old = En_e

                Fock = F_new(H_core,Den,rep)
                F_orth = orthogonalize(Fock,S_invsqrt)
                E_ao, C_mo = enCoef(F_orth,S_invsqrt)
                Den = P(C_mo)
                En_e = energy(H_core,Fock,Den)

        E_mp = MP2(rep, E_ao, C_mo)
        print("E(SCF): " + str(En_e))
        print("MP2 Correction: " + str(E_mp))
        print("E(TOT): " + str(En_e + E_mp))
        
        return None

main()
print("--- %s seconds ---" % (time.time() - start_time))
