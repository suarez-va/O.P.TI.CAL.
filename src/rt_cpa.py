import numpy as np

# Additinal term arising from the moving nuclei in the classical path approximation to be added to the fock matrix

def get_fock_pert(rt_cpa):
    mol = rt_cpa._scf.mol
    Rdot = rt_cpa.nuc.vel
    X = rt_cpa.orth
    Xinv = np.linalg.inv(X)
    dS = -mol.intor('int1e_ipovlp', comp=3)

    RdSX = np.zeros(X.shape)
    result = np.zeros((2, X.shape[0], X.shape[0]))
    aoslices = mol.aoslice_by_atom()
    for i in range(mol.natm):
        p0, p1 = aoslices[i,2:]
        RdSX += np.einsum('x,xij,ik->jk', Rdot[i], dS[:,p0:p1,:], X[p0:p1,:])
    
    result[0] = np.matmul(RdSX, Xinv)
    result[1] = result[0]
    return -1j * result


#def get_fock_pert(rt_cpa):
#    mol = rt_cpa.rt_scf._scf.mol
#    Rdot = rt_cpa.nuc.vel
#    X = rt_cpa.rt_scf.orth
#    Xinv = np.linalg.inv(X)
#    dS = mol.intor('int1e_ipovlp', comp=3)
#
#    RdSX = np.zeros(X.shape)
#    aoslices = mol.aoslice_by_atom()
#    for i in range(mol.natm):
#        p0, p1 = aoslices[i,2:]
#        RdSX += np.einsum('x,xij,ik->jk', Rdot[i], dS[:,p0:p1,:], X[p0:p1,:])
#
#    return -1j * np.matmul(RdSX, Xinv)


