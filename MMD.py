import numpy as np
from scipy.misc import factorial2 
from scipy.special import hyp1f1
from inteval import *
np.set_printoptions(linewidth=300, suppress=True)

class BasisFunction(object):

    def __init__(self, z, origin=[0,0,0], shell=(0,0,0), exps=[], coefs=[]):
        self.origin = np.asarray(origin)
        self.z = z
        self.shell = shell
        self.exps = exps
        self.coefs = coefs
        self.norm = None
        self.normalise()

    def normalise(self):
        l,m,n = self.shell
        self.norm = np.sqrt(np.power(2,2*(l+m+n)+1.5)*np.power(self.exps,l+m+n+1.5)/factorial2(2*l-1)/factorial2(2*m-1)/factorial2(2*n-1)/np.power(np.pi,1.5))

    def print_BF(self):
        print(self.origin, self.shell, np.asarray(self.exps), np.asarray(self.coefs), self.norm)

class Atom(object):
    '''
    Atom object
    | z - atomic number (int)
    | pos - position vector of atom centre ([float])
    | norb - number of orbitals (int)
    | basisfunctions -  contracted Gaussian-Type Orbitals([BasisFunction])
    '''
    def __init__(self, z = 0, pos=[0.0,0.0,0.0]):
        self.z = z
        self.norb = 0
        self.pos = np.array(pos)
        self.basisfunctions = []

    def __readbs(self, basis_set):
        with open(basis_set, 'r') as bsfile:
            data = bsfile.read().split('****')
            atom = [x.split() for x in data[self.z].split('\n') if x]
            count = 0
            NewOrb = False
            bases = []
            for i in range(1, len(atom)):
                if NewOrb == False:
                    momentum  = atom[i][0]
                    primNum = int(atom[i][1])
                    NewOrb = True
                    mec = []
                    mecSP = []
                elif NewOrb:
                    if momentum == 'SP':
                        count += 1
                        mec.append((float(atom[i][0]), float(atom[i][1])))
                        mecSP.append((float(atom[i][0]), float(atom[i][2])))
                        if count == primNum:
                            bases.append(('S', mec))
                            bases.append(('P', mecSP))
                            NewOrb = False
                            count = 0
                    else:
                        mec.append((float(atom[i][0]), float(atom[i][1])))
                        count += 1
                        if count == primNum:
                            bases.append((momentum, mec))
                            NewOrb = False
                            count = 0
        return bases

    def populate_BasisFunctions(self, basis_set):
        basis = self.__readbs(basis_set)
        for i in basis:
            momentum = i[0]
            exps = [val[0] for val in i[1]]
            coefs = [val[1] for val in i[1]]
            if momentum == 'S':
                self.basisfunctions.append(BasisFunction(self.z, self.pos, (0,0,0), exps, coefs))
            elif momentum == 'P':
                self.basisfunctions.append(BasisFunction(self.z, self.pos, (1,0,0), exps, coefs))
                self.basisfunctions.append(BasisFunction(self.z, self.pos, (0,1,0), exps, coefs))
                self.basisfunctions.append(BasisFunction(self.z, self.pos, (0,0,1), exps, coefs))
            elif momentum == 'D':
                self.basisfunctions.append(BasisFunction(self.z, self.pos, (2,0,0), exps, coefs))
                self.basisfunctions.append(BasisFunction(self.z, self.pos, (1,1,0), exps, coefs))
                self.basisfunctions.append(BasisFunction(self.z, self.pos, (1,0,1), exps, coefs))
                self.basisfunctions.append(BasisFunction(self.z, self.pos, (0,2,0), exps, coefs))
                self.basisfunctions.append(BasisFunction(self.z, self.pos, (0,1,1), exps, coefs))
                self.basisfunctions.append(BasisFunction(self.z, self.pos, (0,0,2), exps, coefs))
            else:
                print("Only S, P and D orbitals are accounted for")

    def print_Atom(self):
        ''' Prints attributes of the Atom object '''
        print("Atomic Number: " + str(self.z))
        print("Position")
        print(self.pos)
        print("Basis Function")
        for bf in self.basisfunctions:
            bf.print_BF()
        print("Number of Orbitals: " + str(len(self.basisfunctions)))

class Molecule(object):
    '''
    Molecule Object
    | atoms - list of Atom objects ([Atom])
    | natoms - Number of atoms in the molecule (int)
    '''
    def __init__(self, natoms = 0, atoms=[], basis_set='basis_sets/sto-3g.dat'):
        self.atoms = atoms
        self.natoms = natoms
        self.basis_set = basis_set

    def read_Molecule(self, input_file):
        '''
        Reads in a molecule from a .xyz file of format:
        No. Atoms
        Molecule_Name
        Z_1 x_1 y_1 z_1
        ...
        Z_n x_n y_n z_n
        '''
        with open(input_file, 'r') as inmol:
            self.natoms = inmol.readline()
            inmol.readline()
            for line in inmol:
                line = line.split()
                self.atoms.append(Atom(int(float(line[0])), [float(line[1]), float(line[2]), float(line[3])]))
                print(line)

        for atom in self.atoms:
            atom.populate_BasisFunctions(self.basis_set)
            for bf in atom.basisfunctions:
                bf.normalise()

    def print_Molecule(self):
        '''
        Prints attributes of Molecule object
        '''
        print("Number of Atoms: " + str(self.natoms))
        print("Atoms: ")
        for i in range(len(self.atoms)):
            self.atoms[i].print_Atom()

class MMDEvaluator(object):
    '''
    Builds molecular integrals using McMurchie-Davidson Method
    '''
    def __init__(self, molecule):
        self.molecule = molecule
        self.S()
        self.T()
        self.V()
        self.ERI()

    def __E(self,i,j,t,Qx,a,b):
        '''
        Recursive definition of expansion coefficients for Hermite Gaussians
        '''
        p = a + b
        q = a*b / p
        if (t < 0) or (t > (i + j)):
            return 0
        elif i == j == t == 0:
            return np.exp(-q*Qx*Qx)
        elif j == 0:
            return (1 / (2*p))*self.__E(i-1,j,t-1,Qx,a,b) - (q*Qx/a)*self.__E(i-1,j,t,Qx,a,b) + (t+1)*self.__E(i-1,j,t+1,Qx,a,b)
        else:
            return (1 / (2*p))*self.__E(i,j-1,t-1,Qx,a,b) + (q*Qx/b)*self.__E(i,j-1,t,Qx,a,b) + (t+1)*self.__E(i,j-1,t+1,Qx,a,b)
    
    def __overlap(self,a,lmn1,A,b,lmn2,B):
        '''
        Calculates overlap integral between two primitive gaussians
        '''
        l1,m1,n1 = lmn1
        l2,m2,n2 = lmn2
        S1 = self.__E(l1,l2,0,A[0]-B[0],a,b)
        S2 = self.__E(m1,m2,0,A[1]-B[1],a,b)
        S3 = self.__E(n1,n2,0,A[2]-B[2],a,b)
        return S1*S2*S3*np.power(np.pi / (a+b), 1.5)

    def __s(self, a, b):
        '''
        Calculates overlap integral between two contracted gaussians
        '''
        s = 0.0
        for ia, ca in enumerate(a.coefs):
            for ib, cb in enumerate(b.coefs):
                s += a.norm[ia]*b.norm[ib]*ca*cb*self.__overlap(a.exps[ia], a.shell, a.origin, b.exps[ib], b.shell, b.origin)
        return s

    def __orbitals(self):
        '''
        Take all basisfunction objects and place them in to one list
        '''
        all_orbitals = []
        for atom in self.molecule.atoms:
            for bf in atom.basisfunctions:
                all_orbitals.append(bf)
        return all_orbitals

    def S(self):
        '''
        Build Overlap Matrix
        '''
        aos = self.__orbitals()
        nrorbs = len(aos)
        S_mat = np.zeros((nrorbs, nrorbs), float)
        for i in range(nrorbs):
            for j in range(nrorbs):
                S_mat[i][j] = self.__s(aos[i], aos[j])
        return S_mat

    def __kinetic(self,a,lmn1,A,b,lmn2,B):

        l1,m1,n1 = lmn1
        l2,m2,n2 = lmn2

        term0 = b*(2*(l2+m2+n2)+3)*(self.__overlap(a, (l1,m1,n1), A, b, (l2,m2,n2), B))
        term1 = -2*np.power(b,2)*(self.__overlap(a, (l1,m1,n1), A, b, (l2+2,m2,n2), B)+self.__overlap(a, (l1,m1,n1), A, b, (l2,m2+2,n2), B)+self.__overlap(a, (l1,m1,n1), A, b, (l2,m2,n2+2), B))       
        term2 = -0.5*(l2*(l2-1)*self.__overlap(a,(l1,m1,n1),A,b,(l2-2,m2,n2),B)+m2*(m2-1)*self.__overlap(a,(l1,m1,n1),A,b,(l2,m2-2,n2),B)+n2*(n2-1)*self.__overlap(a,(l1,m1,n1),A,b,(l2,m2,n2-2),B))

        return term0+term1+term2
    
    def __t(self, a, b):

        t = 0.0
        for ia, ca in enumerate(a.coefs):
            for ib, cb in enumerate(b.coefs):
                t+=a.norm[ia]*b.norm[ib]*ca*cb*self.__kinetic(a.exps[ia],a.shell,a.origin,b.exps[ib],b.shell,b.origin)
        return t

    def T(self):
        aos = self.__orbitals()
        nrorbs = len(aos)
        T_mat = np.zeros((nrorbs, nrorbs), float)
        for i in range(nrorbs):
            for j in range(nrorbs):
                T_mat[i][j] = self.__t(aos[i], aos[j])
        return T_mat

    def __boys(self, n, T):
        return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0)

    def __gpc(self,a,A,b,B):
        return (a*A+b*B)/(a+b)

    def __R(self,t,u,v,n,p,PCx,PCy,PCz,RPC):
        T = p*RPC*RPC
        val = 0.0
        if t == u == v == 0:
            val += np.power(-2*p,n)*self.__boys(n,T)
        elif t == u == 0:
            if v > 1:
                val += (v-1)*self.__R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
            val += PCz*self.__R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
        elif t == 0:
            if u > 1:
                val += (u-1)*self.__R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
            val += PCy*self.__R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
        else:
            if t > 1:
                val += (t-1)*self.__R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
            val += PCx*self.__R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
        return val
        
    def __nuclear(self,a,lmn1,A,b,lmn2,B,C):
      
        l1,m1,n1 = lmn1
        l2,m2,n2 = lmn2
        p = a + b
        P = np.asarray(self.__gpc(a,A,b,B))
        RPC = np.linalg.norm(P-C)

        val = 0.0
        for t in range(l1+l2+1):
            for u in range(m1+m2+1):
                 for v in range(n1+n2+1):
                     val += self.__E(l1,l2,t,A[0]-B[0],a,b) * \
                            self.__E(m1,m2,u,A[1]-B[1],a,b) * \
                            self.__E(n1,n2,v,A[2]-B[2],a,b) * \
                            self.__R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
        val *= 2*np.pi/p
        return val
    
    def __v(self, a, b, atom):
        v = 0.0
        for ia, ca in enumerate(a.coefs):
            for ib, cb in enumerate(b.coefs):
                v += a.norm[ia]*b.norm[ib]*ca*cb*\
                      self.__nuclear(a.exps[ia],a.shell,a.origin,b.exps[ib],b.shell,b.origin,atom.pos)
        return v 

    def V(self):
        aos = self.__orbitals()
        nrorbs = len(aos)
        V_mat = np.zeros((nrorbs, nrorbs), float)
        for i in range(nrorbs):
            for j in range(nrorbs):
                for atom in self.molecule.atoms:
                    V_mat[i][j] += -atom.z*self.__v(aos[i],aos[j],atom)
        return V_mat

    def __electron(self,a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D):
        l1,m1,n1 = lmn1
        l2,m2,n2 = lmn2
        l3,m3,n3 = lmn3
        l4,m4,n4 = lmn4
        p = a+b
        q = c+d
        alpha = p*q/(p+q)
        P = np.asarray(self.__gpc(a,A,b,B))
        Q = np.asarray(self.__gpc(c,C,d,D))
        RPQ = np.linalg.norm(P-Q)

        val = 0.0
        for t in range(l1+l2+1):
            for u in range(m1+m2+1):
                for v in range(n1+n2+1):
                    for tau in range(l3+l4+1):
                        for nu in range(m3+m4+1):
                            for phi in range(n3+n4+1):
                                val += self.__E(l1,l2,t,A[0]-B[0],a,b) * \
                                        self.__E(m1,m2,u,A[1]-B[1],a,b) * \
                                        self.__E(n1,n2,v,A[2]-B[2],a,b) * \
                                        self.__E(l3,l4,tau,C[0]-D[0],c,d) * \
                                        self.__E(m3,m4,nu ,C[1]-D[1],c,d) * \
                                        self.__E(n3,n4,phi,C[2]-D[2],c,d) * \
                                        np.power(-1,tau+nu+phi) * \
                                        self.__R(t+tau,u+nu,v+phi,0,\
                                        alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ)

        val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
        return val

    def __eri(self, a, b, c, d):
        eri = 0.0
        for ja, ca in enumerate(a.coefs):
            for jb, cb in enumerate(b.coefs):
                for jc, cc in enumerate(c.coefs):
                    for jd, cd in enumerate(d.coefs):
                        eri += a.norm[ja]*b.norm[jb]*c.norm[jc]*d.norm[jd]*\
                                ca*cb*cc*cd*\
                                self.__electron(a.exps[ja],a.shell,a.origin,\
                                                b.exps[jb],b.shell,b.origin,\
                                                c.exps[jc],c.shell,c.origin,\
                                                d.exps[jd],d.shell,d.origin)
        return eri

    def ERI(self):
        aos = self.__orbitals()
        nrorbs = len(aos)
        ERI_mat = np.zeros((nrorbs, nrorbs, nrorbs, nrorbs), float)
        for i in range(nrorbs):
            for j in range(nrorbs):
                for k in range(nrorbs):
                    for l in range(nrorbs):
                        ERI_mat[i,j,k,l] = self.__eri(aos[i], aos[j], aos[k], aos[l])
        return ERI_mat


class SCF(object):
    
    def __init__(self, integrals):
        self.integrals = integrals

    def initialise(self):
        pass
"""
def main():
    molecule = Molecule()
    molecule.read_Molecule("h2o.dat")
    molints = MMDEvaluator(molecule)
    molecule.print_Molecule()
    print(molints.S())
    print(molints.T())
    print(molints.V())
    print(molints.ERI())
start_time = time.time()
main()
"""
#print ("%s seconds", (time.time() - start_time))
