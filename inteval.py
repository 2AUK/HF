#pythran export E(int, int, int, float, float, float)
def E(i,j,t,Qx,a,b):
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

#pythran export (float, set, float, float, set, float)
 def overlap(a,lmn1,A,b,lmn2,B):
        '''
        Calculates overlap integral between two primitive gaussians
        '''
        l1,m1,n1 = lmn1
        l2,m2,n2 = lmn2
        S1 = E(l1,l2,0,A[0]-B[0],a,b)
        S2 = E(m1,m2,0,A[1]-B[1],a,b)
        S3 = E(n1,n2,0,A[2]-B[2],a,b)
        return S1*S2*S3*np.power(np.pi / (a+b), 1.5)