import numpy as np
from scipy.special import hyp1f1

#pythran export E(int, int, int, float, float, float)
def E(i, j, t, Qx, a, b):
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
        return (1 / (2*p))*E(i-1,j,t-1,Qx,a,b) - (q*Qx/a)*E(i-1,j,t,Qx,a,b) + (t+1)*E(i-1,j,t+1,Qx,a,b)
    else:
        return (1 / (2*p))*E(i,j-1,t-1,Qx,a,b) + (q*Qx/b)*E(i,j-1,t,Qx,a,b) + (t+1)*E(i,j-1,t+1,Qx,a,b)

#pythran export overlap(float, (int), float[], float, (int), float[])
def overlap(a, lmn1, A, b, lmn2, B):
    '''
    Calculates overlap integral between two primitive gaussians
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    S1 = E(l1,l2,0,A[0]-B[0],a,b)
    S2 = E(m1,m2,0,A[1]-B[1],a,b)
    S3 = E(n1,n2,0,A[2]-B[2],a,b)
    return S1*S2*S3*np.power(np.pi / (a+b), 1.5)

#pythran export kinetic(float, (int), float[], float, (int), float[])
def kinetic(a, lmn1, A, b, lmn2, B):
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    term0 = b*(2*(l2+m2+n2)+3)*(overlap(a, (l1,m1,n1), A, b, (l2,m2,n2), B))
    term1 = -2*np.power(b,2)*(overlap(a, (l1,m1,n1), A, b, (l2+2,m2,n2), B)+overlap(a, (l1,m1,n1), A, b, (l2,m2+2,n2), B)+overlap(a, (l1,m1,n1), A, b, (l2,m2,n2+2), B))       
    term2 = -0.5*(l2*(l2-1)*overlap(a,(l1,m1,n1),A,b,(l2-2,m2,n2),B)+m2*(m2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2-2,n2),B)+n2*(n2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2,n2-2),B))
    return term0+term1+term2

def boys(n, T):
    return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0)

def gpc(a, A, b, B):
    return (a*A+b*B)/(a+b)

#pythran export R(int, int, int, int, float, float, float, float)
def R(t, u, v, n, p, PCx, PCy, PCz, RPC):
    T = p*RPC*RPC
    val = 0.0
    if t == u == v == 0:
        val += np.power(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val

#pythran export nuclear(float, (int), float[], float, (int), float[], float[])
def nuclear(a, lmn1, A, b, lmn2, B, C):      
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    p = a + b
    P = np.asarray(self.__gpc(a,A,b,B))
    RPC = np.linalg.norm(P-C)
    val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                val +=  self.__E(l1,l2,t,A[0]-B[0],a,b) * \
                        self.__E(m1,m2,u,A[1]-B[1],a,b) * \
                        self.__E(n1,n2,v,A[2]-B[2],a,b) * \
                        self.__R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
    val *= 2*np.pi/p
    return val

#pythran export two_elec_cGTO(str, float[], float[], float[], float[], (int), (int), float[], float[], float[], float[], float[], float)
def two_elec_cGTO(integral_type, a_norm, b_norm, a_origin, b_origin, a_shell, b_shell, a_exps, b_exps, a_coefs, b_coefs, atom_position, atom_charge):
    if integral_type == "overlap":
        s = 0.0
        for ia, ca in enumerate(a_coefs):
            for ib, cb in enumerate(b_coefs):
              s += a_norm[ia]*b_norm[ib]*ca*cb*overlap(a_exps[ia], a_shell, a_origin, b_exps[ib], b_shell, b_origin)
        return s
    elif integral_type == "kinetic":
        v = 0.0
        for ia, ca in enumerate(a_coefs):
            for ib, cb in enumerate(b_coefs):
              v += a_norm[ia]*b_norm[ib]*ca*cb*kinetic(a_exps[ia], a_shell, a_origin, b_exps[ib], b_shell, b_origin)
        return v
    elif integral_type == "nuclear":
        n = 0.0
        for ia, ca in enumerate(a_coefs):
            for ib, cb in enumerate(b_coefs):
              v += a_norm[ia]*b_norm[ib]*ca*cb*kinetic(a_exps[ia], a_shell, a_origin, b_exps[ib], b_shell, b_origin)
        return v